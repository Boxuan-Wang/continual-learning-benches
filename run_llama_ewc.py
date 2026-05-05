#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import torch
from torch.optim import AdamW

from llm_cl import EWCRegularizer, JsonlQADatasetAdapter
from llm_cl.trainer import ContextTrainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Context-by-context Llama EWC training.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing task_*/ data folders.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for task artifacts.")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B")
    parser.add_argument(
        "--use-lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LoRA adapters. Default is enabled.",
    )
    parser.add_argument("--epochs-per-context", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--fisher-batch-size", type=int, default=1)
    parser.add_argument("--fisher-max-batches", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--reg-strength", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=1.0, help="Online-EWC decay for old Fisher statistics.")
    parser.add_argument(
        "--offline-ewc",
        action="store_true",
        help="Keep separate Fisher penalties for all past contexts.",
    )
    parser.add_argument(
        "--malformed-policy",
        type=str,
        choices=["error", "skip"],
        default="error",
        help="How JSONL loader handles malformed records.",
    )
    parser.add_argument(
        "--train-on-prompt",
        action="store_true",
        help="If set, LM loss is applied to prompt tokens as well.",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated target modules for LoRA.",
    )
    parser.add_argument(
        "--trainable-param-patterns",
        type=str,
        default="lm_head",
        help="Comma-separated substrings for trainable params when --no-use-lora is set.",
    )
    return parser.parse_args()


def _split_csv(value: str) -> List[str]:
    return [chunk.strip() for chunk in value.split(",") if chunk.strip()]


def _configure_non_lora_trainable_params(model: torch.nn.Module, patterns: Iterable[str]) -> List[str]:
    patterns = [p for p in patterns if p]
    if not patterns:
        raise ValueError("At least one --trainable-param-patterns value is required when LoRA is disabled.")

    for _, param in model.named_parameters():
        param.requires_grad = False

    selected: List[str] = []
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            param.requires_grad = True
            selected.append(name)

    if not selected:
        raise ValueError(
            "No parameters matched --trainable-param-patterns. "
            "Provide explicit, memory-safe target patterns."
        )
    return selected


def _attach_lora(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA requested but `peft` is not installed. Install with `pip install peft`."
        ) from exc

    target_modules = _split_csv(args.lora_target_modules)
    if not target_modules:
        raise ValueError("LoRA is enabled but --lora-target-modules is empty.")

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    return model


def main() -> None:
    args = parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "This entrypoint requires transformers. Install with `pip install transformers`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    if args.use_lora:
        model = _attach_lora(model, args)
        tracked_param_names = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and ("lora_" in name or "modules_to_save" in name)
        ]
        if not tracked_param_names:
            raise RuntimeError("LoRA mode enabled but no LoRA trainable parameters were found.")
        mode = "lora"
    else:
        patterns = _split_csv(args.trainable_param_patterns)
        tracked_param_names = _configure_non_lora_trainable_params(model, patterns=patterns)
        mode = "non_lora"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if len(optimizer.param_groups[0]["params"]) == 0:
        raise RuntimeError("No trainable parameters are selected.")

    adapter = JsonlQADatasetAdapter(
        tokenizer=tokenizer,
        max_length=args.max_length,
        malformed_policy=args.malformed_policy,
        train_on_prompt=args.train_on_prompt,
    )
    ewc = EWCRegularizer(
        tracked_param_names=tracked_param_names,
        gamma=args.gamma,
        offline=args.offline_ewc,
    )
    config = TrainerConfig(
        epochs_per_context=args.epochs_per_context,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        fisher_batch_size=args.fisher_batch_size,
        fisher_max_batches=args.fisher_max_batches,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        reg_strength=args.reg_strength,
        use_bf16=torch.cuda.is_available(),
        num_workers=args.num_workers,
        log_every=args.log_every,
    )

    print(f"[INFO] Starting run with mode={mode}")
    trainer = ContextTrainer(
        model=model,
        optimizer=optimizer,
        adapter=adapter,
        ewc=ewc,
        config=config,
        data_root=args.data_root,
        output_dir=args.output_dir,
        use_lora=args.use_lora,
    )
    trainer.train()


if __name__ == "__main__":
    main()
