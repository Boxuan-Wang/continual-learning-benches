from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn


@dataclass
class EWCState:
    means: Dict[str, torch.Tensor]
    fisher: Dict[str, torch.Tensor]


class EWCRegularizer:
    """Diagonal-Fisher EWC for arbitrary subsets of model parameters."""

    def __init__(
        self,
        tracked_param_names: Iterable[str],
        gamma: float = 1.0,
        offline: bool = False,
    ) -> None:
        tracked = sorted(set(tracked_param_names))
        if not tracked:
            raise ValueError("EWCRegularizer requires at least one tracked parameter.")
        self.tracked_param_names: List[str] = tracked
        self.gamma = gamma
        self.offline = offline
        self._states: List[EWCState] = []

    @property
    def context_count(self) -> int:
        return len(self._states)

    def estimate_fisher(
        self,
        model: nn.Module,
        fisher_loader,
        *,
        device: torch.device,
        max_batches: Optional[int] = None,
        use_bf16: bool = True,
    ) -> None:
        named_params = dict(model.named_parameters())
        tracked_params = {
            name: named_params[name]
            for name in self.tracked_param_names
            if name in named_params and named_params[name].requires_grad
        }
        if not tracked_params:
            raise RuntimeError("No tracked trainable parameters available for Fisher estimation.")

        est_fisher = {name: torch.zeros_like(param, dtype=torch.float32) for name, param in tracked_params.items()}

        was_training = model.training
        model.eval()
        processed = 0

        for batch_idx, batch in enumerate(fisher_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            model.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16 and device.type == "cuda",
            ):
                outputs = model(**batch)
                neg_log_likelihood = outputs.loss

            neg_log_likelihood.backward()

            for name, param in tracked_params.items():
                if param.grad is not None:
                    est_fisher[name] += (param.grad.detach().float() ** 2)
            processed += 1

        if processed == 0:
            raise RuntimeError("Fisher estimation received zero batches.")
        est_fisher = {name: value / processed for name, value in est_fisher.items()}

        means = {name: param.detach().clone() for name, param in tracked_params.items()}
        new_state = EWCState(means=means, fisher=est_fisher)

        if self.offline or len(self._states) == 0:
            self._states.append(new_state)
        else:
            # Online EWC: keep one running precision matrix and MAP mean.
            prev = self._states[-1]
            merged_fisher: Dict[str, torch.Tensor] = {}
            for name in est_fisher:
                old = prev.fisher.get(name)
                if old is None:
                    merged_fisher[name] = est_fisher[name]
                else:
                    merged_fisher[name] = est_fisher[name] + self.gamma * old
            self._states[-1] = EWCState(means=means, fisher=merged_fisher)

        model.train(was_training)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if not self._states:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        named_params = dict(model.named_parameters())
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        states = self._states if self.offline else [self._states[-1]]

        for state in states:
            for name in self.tracked_param_names:
                if name not in state.means or name not in state.fisher or name not in named_params:
                    continue
                param = named_params[name]
                fisher = state.fisher[name].to(param.device)
                mean = state.means[name].to(param.device)
                weight = 1.0 if self.offline else self.gamma
                loss = loss + (weight * fisher * (param - mean).pow(2)).sum()
        return 0.5 * loss

    def latest_state_dict(self) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        if not self._states:
            return None
        latest = self._states[-1]
        return {
            "means": {k: v.detach().cpu() for k, v in latest.means.items()},
            "fisher": {k: v.detach().cpu() for k, v in latest.fisher.items()},
        }
