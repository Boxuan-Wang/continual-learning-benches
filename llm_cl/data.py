import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ContextSpec:
    """Filesystem contract for one sequential training context."""

    context_id: int
    context_dir: Path
    train_jsonl: Path
    val_jsonl: Path


@dataclass
class ContextDataLoaders:
    """Dataloaders consumed by context-by-context trainers."""

    context: ContextSpec
    train_loader: DataLoader
    fisher_loader: DataLoader
    val_loader: DataLoader


class ContextDatasetAdapter(ABC):
    """Pluggable API for mapping arbitrary datasets to sequential contexts."""

    @abstractmethod
    def discover_contexts(self, data_root: Path) -> List[ContextSpec]:
        """Return contexts in training order (task_0, task_1, ...)."""

    @abstractmethod
    def build_context_loaders(
        self,
        context: ContextSpec,
        train_batch_size: int,
        eval_batch_size: int,
        fisher_batch_size: int,
        fisher_max_batches: Optional[int],
        num_workers: int = 0,
    ) -> ContextDataLoaders:
        """Create train/fisher/validation loaders for one context."""


class PromptCompletionDataset(Dataset):
    """In-memory dataset with `question` / `answer` examples from JSONL."""

    def __init__(self, records: Sequence[Dict[str, str]]) -> None:
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.records[idx]


class SupervisedLMDataCollator:
    """Converts question/answer samples into causal-LM tensors."""

    def __init__(
        self,
        tokenizer,
        max_length: int,
        train_on_prompt: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_on_prompt = train_on_prompt

    def _to_prompt(self, question: str, answer: str) -> tuple[str, str]:
        prompt = f"Question:\n{question.strip()}\n\nAnswer:\n"
        completion = answer.strip()
        return prompt, completion

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        if len(batch) == 0:
            raise ValueError("Cannot collate empty batch.")

        prompts: List[str] = []
        full_texts: List[str] = []
        for sample in batch:
            prompt, completion = self._to_prompt(sample["question"], sample["answer"])
            prompts.append(prompt)
            full_texts.append(f"{prompt}{completion}")

        encoded = self.tokenizer(
            full_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        labels[encoded["attention_mask"] == 0] = -100

        if not self.train_on_prompt:
            prompt_only = self.tokenizer(
                prompts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            prompt_lengths = prompt_only["attention_mask"].sum(dim=1)
            for row_idx, prompt_len in enumerate(prompt_lengths.tolist()):
                labels[row_idx, :prompt_len] = -100

        encoded["labels"] = labels
        return encoded


class JsonlQADatasetAdapter(ContextDatasetAdapter):
    """Default adapter for the plan's task_*/{train_mix,val}/data.jsonl layout."""

    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        malformed_policy: str = "error",
        train_on_prompt: bool = False,
    ) -> None:
        if malformed_policy not in {"error", "skip"}:
            raise ValueError("malformed_policy must be 'error' or 'skip'.")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.malformed_policy = malformed_policy
        self.train_on_prompt = train_on_prompt

    def discover_contexts(self, data_root: Path) -> List[ContextSpec]:
        root = Path(data_root)
        if not root.exists():
            raise FileNotFoundError(f"data_root does not exist: {root}")

        contexts: List[ContextSpec] = []
        context_dirs = sorted(root.glob("task_*"), key=self._task_index)
        for context_dir in context_dirs:
            context_id = self._task_index(context_dir)
            train_jsonl = context_dir / "train_mixed" / "data.jsonl"
            val_jsonl = context_dir / "val" / "data.jsonl"
            if not train_jsonl.exists() or not val_jsonl.exists():
                raise FileNotFoundError(
                    f"Missing expected files in {context_dir}: "
                    f"{train_jsonl.relative_to(root)} and {val_jsonl.relative_to(root)}"
                )
            contexts.append(
                ContextSpec(
                    context_id=context_id,
                    context_dir=context_dir,
                    train_jsonl=train_jsonl,
                    val_jsonl=val_jsonl,
                )
            )

        if not contexts:
            raise ValueError(f"No contexts discovered under {root} (expected task_* dirs).")
        return contexts

    def build_context_loaders(
        self,
        context: ContextSpec,
        train_batch_size: int,
        eval_batch_size: int,
        fisher_batch_size: int,
        fisher_max_batches: Optional[int],
        num_workers: int = 0,
    ) -> ContextDataLoaders:
        train_records = self._read_jsonl(context.train_jsonl)
        val_records = self._read_jsonl(context.val_jsonl)

        if len(train_records) == 0:
            raise ValueError(f"No valid train records in {context.train_jsonl}")
        if len(val_records) == 0:
            raise ValueError(f"No valid val records in {context.val_jsonl}")

        train_ds = PromptCompletionDataset(train_records)
        val_ds = PromptCompletionDataset(val_records)

        fisher_ds = train_ds
        if fisher_max_batches is not None:
            fisher_n = fisher_batch_size * fisher_max_batches
            fisher_n = max(1, min(len(train_ds), fisher_n))
            fisher_ds = torch.utils.data.Subset(train_ds, range(fisher_n))

        collator = SupervisedLMDataCollator(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            train_on_prompt=self.train_on_prompt,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
        )
        fisher_loader = DataLoader(
            fisher_ds,
            batch_size=fisher_batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collator,
        )

        return ContextDataLoaders(
            context=context,
            train_loader=train_loader,
            fisher_loader=fisher_loader,
            val_loader=val_loader,
        )

    @staticmethod
    def _task_index(task_path: Path) -> int:
        try:
            return int(task_path.name.split("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Invalid task directory format: {task_path.name}") from exc

    def _read_jsonl(self, path: Path) -> List[Dict[str, str]]:
        records: List[Dict[str, str]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._handle_malformed(path, line_no, f"invalid JSON: {exc}")
                    continue

                if not isinstance(parsed, dict):
                    self._handle_malformed(path, line_no, "expected JSON object")
                    continue
                if "question" not in parsed or "answer" not in parsed:
                    self._handle_malformed(path, line_no, "missing required keys: question/answer")
                    continue
                question = str(parsed["question"])
                answer = str(parsed["answer"])
                records.append({"question": question, "answer": answer})
        return records

    def _handle_malformed(self, path: Path, line_no: int, reason: str) -> None:
        msg = f"Malformed record at {path}:{line_no} ({reason})"
        if self.malformed_policy == "error":
            raise ValueError(msg)
        print(f"[WARN] {msg}; skipping")
