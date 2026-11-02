#!/usr/bin/env python3
"""Type-specific binary BioMetaNER experiments.

Each experiment trains four independent 3-label NER models:
O, B-<TYPE>, I-<TYPE>. Final evaluation uses span-level partial match
(same type and any character overlap). Golden evaluation merges predictions
from the four models by confidence before scoring.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import Trainer
from transformers import TrainingArguments
from transformers import EarlyStoppingCallback
from transformers import set_seed
from transformers import get_linear_schedule_with_warmup

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_softcrf import BinaryNERWithCRF, extract_spans

ENTITY_TYPES = ("MIC", "DIS", "CHE", "HAB")
LABELS_BY_TYPE = {
    t: {"O": 0, f"B-{t}": 1, f"I-{t}": 2} for t in ENTITY_TYPES
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("BIOMETANER_DATA_ROOT", PROJECT_ROOT / "data/private"))
BRONZE_TYPE_ROOT = DATA_ROOT / "bronze_type_by_source_80_10_10_filtered_v1"
SILVER_JSONL = DATA_ROOT / "silver_new.jsonl"
GOLDEN_JSONL = DATA_ROOT / "golden_metasigndb_v2_for_eval.jsonl"
RESULT_ROOT = Path(os.environ.get("BIOMETANER_OUTPUT_ROOT", PROJECT_ROOT / "outputs/training"))

MODEL_PATHS = {
    "bert": "google-bert/bert-base-uncased",
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
    "biomegatron": "EMBO/BioMegatron345mUncased",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "scibert": "allenai/scibert_scivocab_uncased",
}

EXPERIMENTS = {
    "E1": "bronze_lastlayer",
    "E2": "bronze_full",
    "E3": "silver_full",
    "E4": "bronze_full_silver_full",
    "E5": "bronze_full_silver_lastlayer",
    "E6": "silver_full_bronze_lastlayer",
    "E7": "silver_lastlayer",
    "E8": "silver_full_bronze_full",
    "E9": "bronze_lastlayer_silver_full",
    # CRF variants: same data strategy, BinaryNERWithCRF decoder
    "E1_CRF": "bronze_lastlayer",
    "E2_CRF": "bronze_full",
    "E3_CRF": "silver_full",
    "E4_CRF": "bronze_full_silver_full",
    "E5_CRF": "bronze_full_silver_lastlayer",
    "E6_CRF": "silver_full_bronze_lastlayer",
    "E7_CRF": "silver_lastlayer",
}

EXPERIMENT_DESCRIPTIONS = {
    "E1": "Bronze train, classifier-only fine-tuning; bronze dev.",
    "E2": "Bronze train, full-parameter fine-tuning; bronze dev.",
    "E3": "Silver train, full-parameter fine-tuning; silver dev.",
    "E4": "Stage1 bronze full (bronze dev), then stage2 silver full (silver dev).",
    "E5": "Stage1 bronze full (bronze dev), then stage2 silver classifier-only (silver dev).",
    "E6": "Stage1 silver full (silver dev), then stage2 bronze classifier-only (bronze dev).",
    "E7": "Silver train, classifier-only fine-tuning; silver dev.",
    "E8": "Stage1 silver full (silver dev), then stage2 bronze full fine-tuning (bronze dev).",
    "E9": "Stage1 bronze classifier-only (bronze dev), then stage2 silver full fine-tuning (silver dev). "
          "Order-swap control for E6: same per-stage trainable-parameter scope (bronze=lastlayer, "
          "silver=full), only the stage order is reversed.",
    "E1_CRF": "CRF: Bronze train, lastlayer; bronze dev.",
    "E2_CRF": "CRF: Bronze train, full; bronze dev.",
    "E3_CRF": "CRF: Silver train, full; silver dev.",
    "E4_CRF": "CRF: Stage1 bronze full, stage2 silver full.",
    "E5_CRF": "CRF: Stage1 bronze full, stage2 silver lastlayer.",
    "E6_CRF": "CRF: Stage1 silver full, stage2 bronze lastlayer.",
    "E7_CRF": "CRF: Silver train, lastlayer; silver dev.",
}

# Stage/entity-specific training caps from seed42 diagnostics. These do not
# change E1-E7 semantics; they tune optimization for source, freeze mode, and
# entity sparsity. Keys are (source, mode, phase), where phase is single/stage1/stage2.
ENTITY_TUNING = {
    "MIC": {"epochs_cap": 80, "patience": 12},
    "DIS": {"epochs_cap": 100, "patience": 15},
    "CHE": {"epochs_cap": 100, "patience": 15},
    "HAB": {"epochs_cap": 80, "patience": 10, "warmup_ratio": 0.08, "weight_decay": 0.05},
}

STAGE_TUNING = {
    ("bronze", "full", "single"): {
        "MIC": {"epochs_cap": 80, "patience": 12},
        "DIS": {"epochs_cap": 100, "patience": 15},
        "CHE": {"epochs_cap": 100, "patience": 15},
        "HAB": {"epochs_cap": 60, "patience": 8, "warmup_ratio": 0.08, "weight_decay": 0.05},
    },
    ("silver", "full", "single"): {
        "MIC": {"epochs_cap": 80, "patience": 12},
        "DIS": {"epochs_cap": 100, "patience": 15},
        "CHE": {"epochs_cap": 100, "patience": 15},
        "HAB": {"epochs_cap": 80, "patience": 10, "warmup_ratio": 0.08, "weight_decay": 0.05},
    },
    ("bronze", "full", "stage1"): {
        "MIC": {"epochs_cap": 80, "patience": 12},
        "DIS": {"epochs_cap": 100, "patience": 15},
        "CHE": {"epochs_cap": 100, "patience": 15},
        "HAB": {"epochs_cap": 60, "patience": 8, "warmup_ratio": 0.08, "weight_decay": 0.05},
    },
    ("silver", "full", "stage1"): {
        "MIC": {"epochs_cap": 80, "patience": 12},
        "DIS": {"epochs_cap": 100, "patience": 15},
        "CHE": {"epochs_cap": 100, "patience": 15},
        "HAB": {"epochs_cap": 80, "patience": 10, "warmup_ratio": 0.08, "weight_decay": 0.05},
    },
    ("silver", "full", "stage2"): {
        "MIC": {"epochs_cap": 50, "patience": 8},
        "DIS": {"epochs_cap": 60, "patience": 10},
        "CHE": {"epochs_cap": 60, "patience": 10},
        "HAB": {"epochs_cap": 60, "patience": 8, "warmup_ratio": 0.08, "weight_decay": 0.05},
    },
    ("bronze", "lastlayer", "single"): {
        "MIC": {"epochs_cap": 80, "patience": 12},
        "DIS": {"epochs_cap": 80, "patience": 12},
        "CHE": {"epochs_cap": 80, "patience": 12},
        "HAB": {"epochs_cap": 40, "patience": 6, "lr": 5e-5},
    },
    ("silver", "lastlayer", "single"): {
        "MIC": {"epochs_cap": 80, "patience": 12},
        "DIS": {"epochs_cap": 80, "patience": 12},
        "CHE": {"epochs_cap": 80, "patience": 12},
        "HAB": {"epochs_cap": 40, "patience": 6, "lr": 5e-5},
    },
    ("silver", "lastlayer", "stage2"): {
        "MIC": {"epochs_cap": 50, "patience": 8, "lr": 1e-4},
        "DIS": {"epochs_cap": 50, "patience": 8, "lr": 1e-4},
        "CHE": {"epochs_cap": 50, "patience": 8, "lr": 1e-4},
        "HAB": {"epochs_cap": 30, "patience": 5, "lr": 5e-5},
    },
    ("bronze", "lastlayer", "stage2"): {
        "MIC": {"epochs_cap": 60, "patience": 8, "lr": 1e-4},
        "DIS": {"epochs_cap": 60, "patience": 8, "lr": 1e-4},
        "CHE": {"epochs_cap": 60, "patience": 8, "lr": 1e-4},
        "HAB": {"epochs_cap": 25, "patience": 4, "lr": 3e-5},
    },
    # E9 only: bronze lastlayer as stage1, starting from the vanilla pretrained
    # checkpoint (not an already-adapted one) -- mirrors the "single" budget,
    # not "stage2" (which assumes a prior full-tune and uses a higher LR/fewer
    # epochs).
    ("bronze", "lastlayer", "stage1"): {
        "MIC": {"epochs_cap": 80, "patience": 12},
        "DIS": {"epochs_cap": 80, "patience": 12},
        "CHE": {"epochs_cap": 80, "patience": 12},
        "HAB": {"epochs_cap": 40, "patience": 6, "lr": 5e-5},
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("train_ner_binary")


@dataclass
class RunCfg:
    epochs: float = 200
    early_stopping_patience: int = 20
    lr: float = 2e-5
    batch_size: int = 32
    max_len: int = 512
    stride: int = 256
    warmup_ratio: float = 0.05
    weight_decay: float = 0.1
    label_smoothing_start: float = 0.1   # linearly decays to label_smoothing_end
    label_smoothing_end: float = 0.0
    seed: int = 42
    save_total_limit: int = 3
    use_class_weight: bool = True  # HAB xN B/I weight; disabled for lastlayer to prevent divergence
    hab_weight: float = 3.0  # sensitivity sweep override


class SpanNERDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer: Any,
        entity_type: str,
        max_len: int,
        stride: int = 256,
    ) -> None:
        self.items: list[dict[str, Any]] = []
        label2id = LABELS_BY_TYPE[entity_type]
        for rec in records:
            text = rec["text"]
            ents = filter_entities(rec, entity_type)
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_len,
                return_offsets_mapping=True,
                return_overflowing_tokens=True,
                stride=stride,
                padding=False,
            )
            for i in range(len(enc["input_ids"])):
                offsets = enc["offset_mapping"][i]
                labels = align_labels(offsets, ents, label2id, entity_type)
                self.items.append(
                    {
                        "input_ids": enc["input_ids"][i],
                        "attention_mask": enc["attention_mask"][i],
                        "labels": labels,
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def filter_entities(record: dict[str, Any], entity_type: str | None = None) -> list[dict[str, Any]]:
    entities = []
    for ent in record.get("entities", []):
        if entity_type is not None and ent.get("type") != entity_type:
            continue
        try:
            start, end = int(ent["start"]), int(ent["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        item = dict(ent)
        item["start"], item["end"] = start, end
        entities.append(item)
    return entities


def align_labels(
    offsets: list[tuple[int, int]],
    entities: list[dict[str, Any]],
    label2id: dict[str, int],
    entity_type: str,
) -> list[int]:
    labels: list[int] = []
    seen_entity_starts: set[tuple[int, int]] = set()
    for start, end in offsets:
        if start == end:
            labels.append(-100)
            continue
        label = label2id["O"]
        for ent in entities:
            if start < ent["end"] and end > ent["start"]:
                key = (ent["start"], ent["end"])
                if key not in seen_entity_starts:
                    label = label2id[f"B-{entity_type}"]
                    seen_entity_starts.add(key)
                else:
                    label = label2id[f"I-{entity_type}"]
                break
        labels.append(label)
    return labels


def load_jsonl(path: Path, split: str | None = None) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if split is None or rec.get("split") == split:
                records.append(rec)
    return records


def stable_split(records: list[dict[str, Any]], dev_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_dev = max(1, round(len(shuffled) * dev_ratio))
    return shuffled[n_dev:], shuffled[:n_dev]


def bronze_path(entity_type: str, split: str) -> Path:
    return BRONZE_TYPE_ROOT / entity_type / f"{split}.jsonl"


def load_train_dev(source: str, entity_type: str, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if source == "bronze":
        # bronze dev matches training distribution; silver dev shows near-0 F1 → early stopping fires too soon
        train = load_jsonl(bronze_path(entity_type, "train"))
        dev = load_jsonl(bronze_path(entity_type, "dev"))
        return train, dev
    if source == "silver":
        records = load_jsonl(SILVER_JSONL)
        train, dev = stable_split(records, 0.2, seed)
        return train, dev
    raise ValueError(f"Unknown source: {source}")


def decode_token_metrics(predictions: np.ndarray, labels: np.ndarray, id2label: dict[int, str]) -> tuple[list[list[str]], list[list[str]]]:
    pred_ids = np.argmax(predictions, axis=-1)
    y_true, y_pred = [], []
    for pred_row, label_row in zip(pred_ids, labels):
        true_seq, pred_seq = [], []
        for pred_id, label_id in zip(pred_row, label_row):
            if int(label_id) == -100:
                continue
            true_seq.append(id2label[int(label_id)])
            pred_seq.append(id2label[int(pred_id)])
        y_true.append(true_seq)
        y_pred.append(pred_seq)
    return y_true, y_pred


def bio_to_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    spans = set()
    start, typ = None, None
    for idx, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-"):
            if typ is not None:
                spans.add((start, idx, typ))
            start, typ = idx, tag[2:]
        elif tag.startswith("I-") and typ == tag[2:]:
            continue
        else:
            if typ is not None:
                spans.add((start, idx, typ))
            start, typ = None, None
    return spans


def token_f1(y_true: list[list[str]], y_pred: list[list[str]]) -> dict[str, float]:
    tp = fp = fn = 0
    for true_tags, pred_tags in zip(y_true, y_pred):
        true_spans = bio_to_spans(true_tags)
        pred_spans = bio_to_spans(pred_tags)
        tp += len(true_spans & pred_spans)
        fp += len(pred_spans - true_spans)
        fn += len(true_spans - pred_spans)
    p, r, f1 = prf(tp, fp, fn)
    return {"precision": p, "recall": r, "f1": f1}


def make_compute_metrics(id2label: dict[int, str]):
    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        y_true, y_pred = decode_token_metrics(eval_pred.predictions, eval_pred.label_ids, id2label)
        return token_f1(y_true, y_pred)

    return compute_metrics


class BioNERTrainer(Trainer):
    """Trainer with dynamic label smoothing and optional per-class loss weighting."""

    def __init__(self, *args, entity_type: str = "", cfg: RunCfg | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.entity_type = entity_type
        self.cfg = cfg or RunCfg()
        # Class weights: upweight B/I for HAB only in full fine-tune (lastlayer freezes encoder,
        # high weight + high LR causes loss divergence on sparse HAB data)
        use_hab_weight = entity_type == "HAB" and getattr(cfg, "use_class_weight", True)
        w = self.cfg.hab_weight
        self.class_weight = torch.tensor([1.0, w, w]) if use_hab_weight else torch.tensor([1.0, 1.0, 1.0])

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (B, L, 3)

        # Dynamic label smoothing: linear decay from start → end over training
        progress = self.state.global_step / max(self.state.max_steps, 1)
        ls = self.cfg.label_smoothing_start + (
            self.cfg.label_smoothing_end - self.cfg.label_smoothing_start
        ) * progress

        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weight.to(logits.device),
            ignore_index=-100,
            label_smoothing=ls,
        )
        loss = loss_fct(logits.view(-1, 3), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def training_args_kwargs(output_dir: Path, cfg: RunCfg) -> dict[str, Any]:
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": cfg.epochs,
        "per_device_train_batch_size": cfg.batch_size,
        "per_device_eval_batch_size": cfg.batch_size,
        "learning_rate": cfg.lr,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "lr_scheduler_type": "cosine",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "logging_steps": 50,
        "save_total_limit": cfg.save_total_limit,
        "report_to": "none",
        "seed": cfg.seed,
        "fp16": torch.cuda.is_available(),
    }
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return kwargs


def train_one_type(
    model_path: str,
    entity_type: str,
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    output_dir: Path,
    cfg: RunCfg,
    mode: str,
) -> tuple[Path, list[dict[str, Any]]]:
    label2id = LABELS_BY_TYPE[entity_type]
    id2label = {v: k for k, v in label2id.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    if mode == "lastlayer":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    train_ds = SpanNERDataset(train_records, tokenizer, entity_type, cfg.max_len, cfg.stride)
    dev_ds = SpanNERDataset(dev_records, tokenizer, entity_type, cfg.max_len, cfg.stride)
    LOGGER.info("%s train windows=%d dev windows=%d", entity_type, len(train_ds), len(dev_ds))
    trainer = BioNERTrainer(
        model=model,
        args=TrainingArguments(**training_args_kwargs(output_dir / "all_checkpoints", cfg)),
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=make_compute_metrics(id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
        entity_type=entity_type,
        cfg=cfg,
    )
    train_result = trainer.train()
    best_dir = output_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    log = extract_epoch_log(trainer.state.log_history)
    with (output_dir / "train_log.json").open("w", encoding="utf-8") as f:
        json.dump({"train_result": train_result.metrics, **log}, f, indent=2)
    return best_dir, log["epochs"]


def extract_epoch_log(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Return per-step train losses and per-epoch dev metrics."""
    step_losses = [
        {"step": h.get("step"), "epoch": h.get("epoch"), "loss": h["loss"]}
        for h in history if "loss" in h and "eval_loss" not in h
    ]
    train_entries = [h for h in history if "loss" in h and "eval_loss" not in h]
    epoch_rows = []
    for ev in [h for h in history if "eval_f1" in h]:
        epoch = float(ev.get("epoch", 0.0))
        prior = [h["loss"] for h in train_entries if float(h.get("epoch", 0.0)) <= epoch]
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": prior[-1] if prior else None,
                "dev_loss": ev.get("eval_loss"),
                "dev_partial_proxy_f1": ev.get("eval_f1"),
                "dev_precision": ev.get("eval_precision"),
                "dev_recall": ev.get("eval_recall"),
            }
        )
    return {"step_losses": step_losses, "epochs": epoch_rows}


def collate_crf(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    max_len = max(len(b["input_ids"]) for b in batch)
    bsz = len(batch)
    input_ids = torch.zeros(bsz, max_len, dtype=torch.long)
    attn_mask  = torch.zeros(bsz, max_len, dtype=torch.long)
    labels     = torch.full((bsz, max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = torch.tensor(b["input_ids"])
        attn_mask[i, :n]  = torch.tensor(b["attention_mask"])
        labels[i, :n]     = torch.tensor(b["labels"])
    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}


def dev_span_f1_crf(
    model: BinaryNERWithCRF,
    tokenizer: Any,
    records: list[dict[str, Any]],
    entity_type: str,
    device: torch.device,
) -> float:
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for rec in records:
            enc = tokenizer(rec["text"], truncation=True, max_length=512,
                            return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(device) for k, v in enc.items()}
            preds = model(**enc)
            pred_seq = preds[0] if preds and isinstance(preds[0], list) else preds
            pred_spans = {(s["start"], s["end"])
                         for s in extract_spans(pred_seq, offsets, entity_type)}
            gold_spans = {(e["start"], e["end"])
                         for e in rec.get("entities", []) if e.get("type") == entity_type}
            tp += len(pred_spans & gold_spans)
            fp += len(pred_spans - gold_spans)
            fn += len(gold_spans - pred_spans)
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    return 2 * p * r / (p + r + 1e-9)


def train_one_type_crf(
    model_path: str,
    entity_type: str,
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    output_dir: Path,
    cfg: RunCfg,
    mode: str,
    stage1_ckpt: Path | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Train BinaryNERWithCRF for one entity type (E1-E7 CRF variants)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    train_ds = SpanNERDataset(train_records, tokenizer, entity_type, cfg.max_len, cfg.stride)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_crf)
    LOGGER.info("[CRF] %s train_windows=%d dev_records=%d", entity_type, len(train_ds), len(dev_records))

    model = BinaryNERWithCRF(model_path, num_labels=3, use_crf=True).to(device)
    if stage1_ckpt is not None:
        model.load_state_dict(torch.load(stage1_ckpt, map_location=device))
        LOGGER.info("[CRF] %s loaded stage1 from %s", entity_type, stage1_ckpt)

    if mode == "lastlayer":
        for name, param in model.named_parameters():
            if name.startswith("encoder."):
                param.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_dl) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total_steps), total_steps)

    best_f1 = 0.0
    best_ep = 0
    no_improve = 0
    log: list[dict[str, Any]] = []
    best_ckpt = output_dir / "best_model.pt"

    for ep in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        f1 = dev_span_f1_crf(model, tokenizer, dev_records, entity_type, device)
        avg_loss = total_loss / max(1, len(train_dl))
        log.append({"epoch": ep, "loss": round(avg_loss, 4), "dev_span_f1": round(f1, 4)})
        LOGGER.info("[CRF] %s Ep %d/%d loss=%.4f dev_f1=%.4f", entity_type, ep, cfg.epochs, avg_loss, f1)

        if f1 >= best_f1:
            best_f1, best_ep, no_improve = f1, ep, 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            no_improve += 1
            if no_improve >= cfg.early_stopping_patience:
                LOGGER.info("[CRF] %s early stop at ep %d", entity_type, ep)
                break

    (output_dir / "train_log.json").write_text(
        json.dumps({"best_epoch": best_ep, "best_dev_f1": best_f1, "epochs": log}, indent=2),
        encoding="utf-8",
    )
    return best_ckpt, log


def extract_spans_from_logits(
    text: str,
    logits: np.ndarray,
    offsets: list[tuple[int, int]],
    entity_type: str,
) -> list[dict[str, Any]]:
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_ids = probs.argmax(axis=-1)
    spans = []
    active: dict[str, Any] | None = None
    for idx, (label_id, (start, end)) in enumerate(zip(pred_ids, offsets)):
        if start == end:
            continue
        score = float(probs[idx, label_id])
        if label_id == 1:
            if active:
                spans.append(finalize_span(text, active))
            active = {"start": start, "end": end, "scores": [score], "type": entity_type}
        elif label_id == 2 and active:
            active["end"] = max(active["end"], end)
            active["scores"].append(score)
        else:
            if active:
                spans.append(finalize_span(text, active))
                active = None
    if active:
        spans.append(finalize_span(text, active))
    return spans


def finalize_span(text: str, span: dict[str, Any]) -> dict[str, Any]:
    start, end = int(span["start"]), int(span["end"])
    return {
        "start": start,
        "end": end,
        "text": text[start:end],
        "type": span["type"],
        "score": float(np.mean(span["scores"])),
    }


INFERENCE_STRIDE = 256  # must match training stride for consistent span coverage

def predict_type(model_dir: str | Path, records: list[dict[str, Any]], entity_type: str, max_len: int, batch_size: int) -> list[list[dict[str, Any]]]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_record_spans = []
    for rec in records:
        enc = tokenizer(
            rec["text"],
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            stride=INFERENCE_STRIDE,
            padding=True,
            return_tensors="pt",
        )
        offsets_by_window = enc.pop("offset_mapping").tolist()
        enc.pop("overflow_to_sample_mapping", None)
        spans = []
        n = enc["input_ids"].shape[0]
        for start_idx in range(0, n, batch_size):
            batch = {k: v[start_idx : start_idx + batch_size].to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**batch).logits.detach().cpu().numpy()
            for j in range(logits.shape[0]):
                spans.extend(
                    extract_spans_from_logits(
                        rec["text"], logits[j], offsets_by_window[start_idx + j], entity_type
                    )
                )
        all_record_spans.append(merge_predictions({entity_type: spans}))
    return all_record_spans


def merge_predictions(preds_by_type: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    all_spans = []
    for entity_type, spans in preds_by_type.items():
        for span in spans:
            item = dict(span)
            item["type"] = entity_type
            all_spans.append(item)
    all_spans.sort(key=lambda s: (-float(s.get("score", 0.0)), s["start"], s["end"]))
    final = []
    for span in all_spans:
        if not any(overlaps(span, kept) for kept in final):
            final.append(span)
    return sorted(final, key=lambda s: (s["start"], s["end"], s["type"]))


def overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return int(a["start"]) < int(b["end"]) and int(a["end"]) > int(b["start"])


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def partial_counts(gold: list[dict[str, Any]], pred: list[dict[str, Any]], entity_type: str | None = None) -> tuple[int, int, int]:
    if entity_type:
        gold = [g for g in gold if g.get("type") == entity_type]
        pred = [p for p in pred if p.get("type") == entity_type]
    matched_gold: set[int] = set()
    tp = 0
    for p in sorted(pred, key=lambda s: -float(s.get("score", 0.0))):
        best_idx = None
        for idx, g in enumerate(gold):
            if idx in matched_gold or p.get("type") != g.get("type"):
                continue
            if overlaps(p, g):
                best_idx = idx
                break
        if best_idx is not None:
            matched_gold.add(best_idx)
            tp += 1
    fp = len(pred) - tp
    fn = len(gold) - tp
    return tp, fp, fn


def score_records(records: list[dict[str, Any]], predictions: list[list[dict[str, Any]]]) -> dict[str, Any]:
    totals = defaultdict(lambda: [0, 0, 0])
    for rec, pred in zip(records, predictions):
        gold = filter_entities(rec)
        for entity_type in ENTITY_TYPES:
            tp, fp, fn = partial_counts(gold, pred, entity_type)
            bucket = totals[entity_type]
            bucket[0] += tp
            bucket[1] += fp
            bucket[2] += fn
        tp, fp, fn = partial_counts(gold, pred)
        bucket = totals["overall"]
        bucket[0] += tp
        bucket[1] += fp
        bucket[2] += fn
    out = {}
    for key, (tp, fp, fn) in totals.items():
        p, r, f1 = prf(tp, fp, fn)
        out[key] = {"P": p, "R": r, "F1": f1, "TP": tp, "FP": fp, "FN": fn}
    macro = np.mean([out[t]["F1"] for t in ENTITY_TYPES]) if all(t in out for t in ENTITY_TYPES) else 0.0
    out["macro_F1"] = float(macro)
    return out


def evaluate_golden(model_dirs: dict[str, str], output_json: Path, cfg: RunCfg) -> dict[str, Any]:
    records = load_jsonl(GOLDEN_JSONL)
    per_type_preds = {}
    for entity_type, model_dir in model_dirs.items():
        per_type_preds[entity_type] = predict_type(model_dir, records, entity_type, cfg.max_len, cfg.batch_size)
    merged = []
    for idx in range(len(records)):
        merged.append(merge_predictions({t: per_type_preds[t][idx] for t in ENTITY_TYPES}))
    result = score_records(records, merged)
    result["evaluation"] = "golden_merged_partial_span"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate_bronze(model_dirs: dict[str, str], output_json: Path, cfg: RunCfg) -> dict[str, Any]:
    result = {"evaluation": "type_specific_bronze_test_partial_span", "per_type": {}}
    total_tp = total_fp = total_fn = 0
    for entity_type, model_dir in model_dirs.items():
        records = load_jsonl(bronze_path(entity_type, "test"))
        preds = predict_type(model_dir, records, entity_type, cfg.max_len, cfg.batch_size)
        metrics = score_records(records, preds).get(entity_type, {"P": 0, "R": 0, "F1": 0, "TP": 0, "FP": 0, "FN": 0})
        result["per_type"][entity_type] = metrics
        total_tp += metrics["TP"]
        total_fp += metrics["FP"]
        total_fn += metrics["FN"]
    p, r, f1 = prf(total_tp, total_fp, total_fn)
    result["overall"] = {"P": p, "R": r, "F1": f1, "TP": total_tp, "FP": total_fp, "FN": total_fn}
    result["macro_F1"] = float(np.mean([result["per_type"][t]["F1"] for t in ENTITY_TYPES]))
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def cfg_for_stage(mode: str, model_key: str = "", seed: int = 42, epochs: int | None = None,
                  entity_type: str = "", batch_size: int = 32, source: str = "",
                  phase: str = "single", use_crf: bool = False, hab_weight: float = 3.0) -> RunCfg:
    tuning = dict(ENTITY_TUNING.get(entity_type, {}))
    tuning.update(STAGE_TUNING.get((source, mode, phase), {}).get(entity_type, {}))

    if model_key == "biomegatron" and mode == "full":
        lr = 1e-5
    elif mode == "lastlayer":
        lr = float(tuning.get("lr", 3e-4))
    else:
        lr = float(tuning.get("lr", 2e-5))

    use_class_weight = mode != "lastlayer"
    ep = epochs if epochs is not None else 100
    ep = min(ep, int(tuning.get("epochs_cap", ep)))
    patience = int(tuning.get("patience", RunCfg.early_stopping_patience))

    # lastlayer: encoder frozen, sparse entities have weak gradient signal;
    # label smoothing would suppress it further, so disable it entirely.
    ls_start = 0.0 if (mode == "lastlayer" or use_crf) else float(tuning.get("label_smoothing_start", 0.1))
    # CRF lastlayer: cap LR at 1e-4 to prevent CRF transition divergence
    if use_crf and mode == "lastlayer":
        lr = min(lr, 1e-4)
    # CRF: disable HAB class weighting (CRF uses global sequence score, not per-token CE)
    if use_crf:
        use_class_weight = False
    return RunCfg(
        lr=lr,
        seed=seed,
        use_class_weight=use_class_weight,
        epochs=ep,
        early_stopping_patience=patience,
        batch_size=batch_size,
        warmup_ratio=float(tuning.get("warmup_ratio", RunCfg.warmup_ratio)),
        weight_decay=float(tuning.get("weight_decay", RunCfg.weight_decay)),
        label_smoothing_start=ls_start,
        hab_weight=hab_weight,
    )


def run_experiment(model_key: str, exp_id: str, seed: int = 42, force: bool = False, batch_size: int = 32, suffix: str = "", hab_weight: float = 3.0) -> None:
    is_crf = exp_id.endswith("_CRF")
    base_id = exp_id[:-4] if is_crf else exp_id  # strip _CRF to get E1..E7

    exp_name = EXPERIMENTS[exp_id]
    out_dir = RESULT_ROOT / f"{exp_id}_{model_key}_{exp_name}{suffix}_seed{seed}"
    done_file = out_dir / "DONE"
    if done_file.exists() and not force:
        LOGGER.info("Skip completed %s", out_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_PATHS[model_key]
    metadata = {
        "experiment_id": exp_id,
        "experiment_name": exp_name,
        "description": EXPERIMENT_DESCRIPTIONS[exp_id],
        "model": model_key,
        "model_path": model_path,
        "seed": seed,
        "use_crf": is_crf,
        "entity_types": list(ENTITY_TYPES),
        "metric": "span-based partial match: same type plus any character overlap",
        "batch_size": batch_size,
    }
    (out_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logs: dict[str, Any] = {}
    model_dirs: dict[str, str] = {}

    for entity_type in ENTITY_TYPES:
        LOGGER.info("=== %s %s %s seed=%d crf=%s ===", exp_id, model_key, entity_type, seed, is_crf)

        if base_id in {"E1", "E2", "E3", "E7"}:
            source = "silver" if base_id in {"E3", "E7"} else "bronze"
            mode   = "lastlayer" if base_id in {"E1", "E7"} else "full"
            train, dev = load_train_dev(source, entity_type, seed)
            cfg = cfg_for_stage(mode, model_key, seed, entity_type=entity_type,
                                batch_size=batch_size, source=source, phase="single",
                                use_crf=is_crf, hab_weight=hab_weight)
            if is_crf:
                best, log = train_one_type_crf(model_path, entity_type, train, dev,
                                               out_dir / entity_type, cfg, mode)
            else:
                best, log = train_one_type(model_path, entity_type, train, dev,
                                           out_dir / entity_type, cfg, mode)
            logs[entity_type] = log
            model_dirs[entity_type] = str(best)

        else:  # E4, E5, E6, E8, E9
            s1_mode = "full"
            if base_id in {"E4", "E5"}:
                s1_source, s2_source = "bronze", "silver"
                s2_mode = "full" if base_id == "E4" else "lastlayer"
            elif base_id == "E9":
                # Order-swap control for E6: same per-stage scope (bronze=lastlayer,
                # silver=full), stage order reversed (bronze first, silver last).
                s1_source, s2_source = "bronze", "silver"
                s1_mode = "lastlayer"
                s2_mode = "full"
            else:
                s1_source, s2_source = "silver", "bronze"
                s2_mode = "full" if base_id == "E8" else "lastlayer"

            s1_train, s1_dev = load_train_dev(s1_source, entity_type, seed)
            s1_cfg = cfg_for_stage(s1_mode, model_key, seed, epochs=200, entity_type=entity_type,
                                   batch_size=batch_size, source=s1_source, phase="stage1",
                                   use_crf=is_crf, hab_weight=hab_weight)
            if is_crf:
                s1_best, s1_log = train_one_type_crf(model_path, entity_type, s1_train, s1_dev,
                                                     out_dir / "stage1" / entity_type, s1_cfg, s1_mode)
            else:
                s1_best, s1_log = train_one_type(model_path, entity_type, s1_train, s1_dev,
                                                 out_dir / "stage1" / entity_type, s1_cfg, s1_mode)

            s2_train, s2_dev = load_train_dev(s2_source, entity_type, seed)
            s2_epochs = 60 if s2_mode == "full" else 80
            s2_cfg = cfg_for_stage(s2_mode, model_key, seed, epochs=s2_epochs, entity_type=entity_type,
                                   batch_size=batch_size, source=s2_source, phase="stage2",
                                   use_crf=is_crf, hab_weight=hab_weight)
            if is_crf:
                s2_best, s2_log = train_one_type_crf(model_path, entity_type, s2_train, s2_dev,
                                                     out_dir / "stage2" / entity_type, s2_cfg, s2_mode,
                                                     stage1_ckpt=s1_best)
            else:
                s2_best, s2_log = train_one_type(str(s1_best), entity_type, s2_train, s2_dev,
                                                 out_dir / "stage2" / entity_type, s2_cfg, s2_mode)
            logs[entity_type] = {"stage1": s1_log, "stage2": s2_log}
            model_dirs[entity_type] = str(s2_best)

        (out_dir / "train_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")

    # Skip standard HF evaluation for CRF models (different checkpoint format)
    if not is_crf:
        eval_cfg = RunCfg(batch_size=batch_size)
        bronze = evaluate_bronze(model_dirs, out_dir / "eval_bronze_test.json", eval_cfg)
        golden = evaluate_golden(model_dirs, out_dir / "eval_golden.json", eval_cfg)
        (out_dir / "summary.json").write_text(
            json.dumps({"bronze_test": bronze, "golden": golden}, indent=2), encoding="utf-8")

    done_file.write_text("done\n", encoding="utf-8")
    LOGGER.info("DONE %s", out_dir)


def list_experiments() -> None:
    for exp_id, name in EXPERIMENTS.items():
        print(f"{exp_id}: {name} - {EXPERIMENT_DESCRIPTIONS[exp_id]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bert", "biobert", "biomegatron", "pubmedbert", "scibert", "all"], default="all")
    parser.add_argument("--experiments", nargs="*", default=list(EXPERIMENTS), choices=list(EXPERIMENTS) + ["E7"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--silver-file", type=str, default=None,
                        help="Override SILVER_JSONL path (e.g. datasets/unified/silver_combined_1000.jsonl)")
    parser.add_argument("--exp-suffix", type=str, default="",
                        help="Suffix appended to experiment output dir name (e.g. _silver1000)")
    parser.add_argument("--hab-weight", type=float, default=3.0,
                        help="HAB B/I class weight for full-fine-tune stages (sensitivity sweep)")
    args = parser.parse_args()
    if args.silver_file:
        global SILVER_JSONL
        SILVER_JSONL = Path(args.silver_file)
    if args.list_experiments:
        list_experiments()
        return
    models = list(MODEL_PATHS) if args.model == "all" else [args.model]
    for seed in args.seeds:
        for model_key in models:
            for exp_id in args.experiments:
                run_experiment(model_key, exp_id, seed=seed, force=args.force, batch_size=args.batch_size, suffix=args.exp_suffix, hab_weight=args.hab_weight)


if __name__ == "__main__":
    main()
