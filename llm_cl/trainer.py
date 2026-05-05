from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.optim import Optimizer

from .data import ContextDatasetAdapter
from .ewc import EWCRegularizer


@dataclass
class TrainerConfig:
    epochs_per_context: int = 1
    train_batch_size: int = 1
    eval_batch_size: int = 1
    fisher_batch_size: int = 1
    fisher_max_batches: Optional[int] = 200
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    reg_strength: float = 1.0
    use_bf16: bool = True
    num_workers: int = 0
    log_every: int = 20


class ContextTrainer:
    """Context-by-context training loop with EWC consolidation."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        adapter: ContextDatasetAdapter,
        ewc: EWCRegularizer,
        config: TrainerConfig,
        data_root: Path,
        output_dir: Path,
        use_lora: bool,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.adapter = adapter
        self.ewc = ewc
        self.config = config
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.use_lora = use_lora
        self.device = next(model.parameters()).device

    def train(self) -> List[Dict[str, float]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        contexts = self.adapter.discover_contexts(self.data_root)
        all_metrics: List[Dict[str, float]] = []

        print(
            f"[INFO] Mode: {'LoRA' if self.use_lora else 'Full/Subset FT'}; "
            f"contexts={len(contexts)}, tracked_params={len(self.ewc.tracked_param_names)}"
        )

        for context in contexts:
            loaders = self.adapter.build_context_loaders(
                context=context,
                train_batch_size=self.config.train_batch_size,
                eval_batch_size=self.config.eval_batch_size,
                fisher_batch_size=self.config.fisher_batch_size,
                fisher_max_batches=self.config.fisher_max_batches,
                num_workers=self.config.num_workers,
            )

            train_metrics = self._train_context(loaders.train_loader, context.context_id)
            val_ppl = self._evaluate_perplexity(loaders.val_loader)

            self.ewc.estimate_fisher(
                self.model,
                loaders.fisher_loader,
                device=self.device,
                max_batches=self.config.fisher_max_batches,
                use_bf16=self.config.use_bf16,
            )

            context_metrics = {
                "context_id": context.context_id,
                "train_loss": train_metrics["train_loss"],
                "ewc_penalty": train_metrics["ewc_penalty"],
                "val_perplexity": val_ppl,
                "ewc_context_count": self.ewc.context_count,
            }
            all_metrics.append(context_metrics)
            self._save_context_artifacts(context.context_id, context_metrics)
            print(
                f"[INFO] Finished context={context.context_id} "
                f"train_loss={context_metrics['train_loss']:.4f} "
                f"val_ppl={context_metrics['val_perplexity']:.3f} "
                f"ewc_ctx={context_metrics['ewc_context_count']}"
            )

        return all_metrics

    def _train_context(self, train_loader, context_id: int) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_task_loss = 0.0
        total_penalty = 0.0
        total_steps = 0

        for epoch in range(self.config.epochs_per_context):
            for step, batch in enumerate(train_loader, 1):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=self.config.use_bf16 and self.device.type == "cuda",
                ):
                    outputs = self.model(**batch)
                    task_loss = outputs.loss
                    ewc_penalty = self.ewc.penalty(self.model)
                    loss_total = task_loss + self.config.reg_strength * ewc_penalty
                    loss_total = loss_total / self.config.grad_accum_steps

                loss_total.backward()
                if step % self.config.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                total_task_loss += task_loss.detach().float().item()
                total_penalty += ewc_penalty.detach().float().item()
                total_steps += 1

                if step % self.config.log_every == 0:
                    print(
                        f"[Train][ctx={context_id}][epoch={epoch+1}] "
                        f"step={step} task_loss={task_loss.item():.4f} "
                        f"ewc={ewc_penalty.item():.4f}"
                    )

        if total_steps == 0:
            raise RuntimeError("No training steps executed for context.")
        return {
            "train_loss": total_task_loss / total_steps,
            "ewc_penalty": total_penalty / total_steps,
        }

    def _evaluate_perplexity(self, val_loader) -> float:
        self.model.eval()
        losses: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=self.config.use_bf16 and self.device.type == "cuda",
                ):
                    outputs = self.model(**batch)
                losses.append(outputs.loss.detach().float().item())

        self.model.train()
        if not losses:
            return float("inf")
        avg_loss = sum(losses) / len(losses)
        return math.exp(min(avg_loss, 20.0))

    def _save_context_artifacts(self, context_id: int, metrics: Dict[str, float]) -> None:
        context_dir = self.output_dir / f"task_{context_id}"
        model_dir = context_dir / "model"
        fisher_dir = context_dir / "fisher"
        context_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        fisher_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(model_dir)
        else:
            torch.save(self.model.state_dict(), model_dir / "pytorch_model.bin")

        serializable = {
            "mode": "lora" if self.use_lora else "non_lora",
            "metrics": metrics,
            "config": asdict(self.config),
        }
        with (context_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2)

        latest_state = self.ewc.latest_state_dict()
        if latest_state is not None:
            torch.save(latest_state, fisher_dir / "state.pt")
