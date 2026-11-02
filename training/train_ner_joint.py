#!/usr/bin/env python3
"""
Joint multi-class NER ablation: shared encoder + 9-class BIO head.

Ablation against the Decoupled Binary Pipeline (train_ner_binary.py).
Runs strategies E3 and E6 with PubMedBERT, 3 seeds (42, 1, 2).

Label scheme (9 classes):
  0=O  1=B-MIC 2=I-MIC  3=B-DIS 4=I-DIS  5=B-CHE 6=I-CHE  7=B-HAB 8=I-HAB

Bronze data: per-type files are merged by text[:120] into unified joint records.
Results saved under experiments/result/JOINT_*/, eval written as eval_golden_v2.json
for direct comparison with binary pipeline results.
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

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ── Label scheme ──────────────────────────────────────────────────────────────
ENTITY_TYPES = ["MIC", "DIS", "CHE", "HAB"]
JOINT_LABELS = ["O", "B-MIC", "I-MIC", "B-DIS", "I-DIS", "B-CHE", "I-CHE", "B-HAB", "I-HAB"]
LABEL2ID = {l: i for i, l in enumerate(JOINT_LABELS)}
ID2LABEL  = {i: l for i, l in enumerate(JOINT_LABELS)}
NUM_LABELS = len(JOINT_LABELS)

# (B_idx, I_idx) per entity type
TYPE_INDICES = {"MIC": (1, 2), "DIS": (3, 4), "CHE": (5, 6), "HAB": (7, 8)}
# Lower value = higher priority when a token overlaps multiple entity spans
TYPE_PRIORITY = {"MIC": 0, "DIS": 1, "CHE": 2, "HAB": 3}

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parents[1]
DATA_ROOT        = Path(os.environ.get("BIOMETANER_DATA_ROOT", PROJECT_ROOT / "data/private"))
BRONZE_TYPE_ROOT = DATA_ROOT / "bronze_type_by_source_80_10_10_filtered_v1"
SILVER_JSONL     = DATA_ROOT / "silver_new.jsonl"
GOLDEN_JSONL     = DATA_ROOT / "golden_metasigndb_v2_for_eval.jsonl"
RESULT_ROOT      = Path(os.environ.get("BIOMETANER_OUTPUT_ROOT", PROJECT_ROOT / "outputs/training"))

MODEL_PATHS = {
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "biobert":    "dmis-lab/biobert-base-cased-v1.1",
    "scibert":    "allenai/scibert_scivocab_uncased",
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
LOGGER = logging.getLogger("train_ner_joint")


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class RunCfg:
    epochs: float = 100
    early_stopping_patience: int = 12
    lr: float = 2e-5
    batch_size: int = 32
    max_len: int = 512
    stride: int = 256
    warmup_ratio: float = 0.05
    weight_decay: float = 0.1
    label_smoothing_start: float = 0.1
    label_smoothing_end: float = 0.0
    seed: int = 42
    save_total_limit: int = 2
    use_hab_weight: bool = True


# ── Data loading ──────────────────────────────────────────────────────────────
def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stable_split(records: list[dict], dev_ratio: float, seed: int):
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_dev = max(1, round(len(shuffled) * dev_ratio))
    return shuffled[n_dev:], shuffled[:n_dev]


def load_bronze_joint(split: str) -> list[dict]:
    """
    Merge per-type Bronze split files into joint records.
    Documents shared across type files are merged (entities unioned).
    """
    by_key: dict[str, dict] = {}
    for et in ENTITY_TYPES:
        path = BRONZE_TYPE_ROOT / et / f"{split}.jsonl"
        if not path.exists():
            continue
        for rec in load_jsonl(path):
            # Key by first-100-char text hash to identify same document
            key = rec["text"][:120]
            if key not in by_key:
                by_key[key] = {k: v for k, v in rec.items() if k != "entities"}
                by_key[key]["entities"] = []
            existing = {(e["start"], e["end"], e.get("type")) for e in by_key[key]["entities"]}
            for e in rec.get("entities", []):
                k = (e["start"], e["end"], e.get("type"))
                if k not in existing:
                    by_key[key]["entities"].append(e)
                    existing.add(k)
    return list(by_key.values())


def load_train_dev_joint(source: str, seed: int):
    if source == "silver":
        records = load_jsonl(SILVER_JSONL)
        return stable_split(records, 0.2, seed)
    if source == "bronze":
        train = load_bronze_joint("train")
        dev   = load_bronze_joint("dev")
        return train, dev
    raise ValueError(f"Unknown source: {source}")


# ── Dataset ───────────────────────────────────────────────────────────────────
def align_labels_joint(offsets: list, entities: list) -> list[int]:
    """
    Assign a single BIO label per token for all entity types jointly.
    When a token overlaps multiple entity types, MIC > DIS > CHE > HAB wins.
    """
    labels: list[int] = []
    seen: set = set()  # (start, end) of spans whose B-label has been emitted

    for cs, ce in offsets:
        if cs == ce:
            labels.append(-100)
            continue

        # Collect overlapping entities, sorted by priority
        candidates = []
        for ent in entities:
            et = ent.get("type")
            if et not in TYPE_PRIORITY:
                continue
            try:
                es, ee = int(ent["start"]), int(ent["end"])
            except (KeyError, TypeError):
                continue
            if cs < ee and ce > es:
                candidates.append((TYPE_PRIORITY[et], es, ee, et))

        if not candidates:
            labels.append(0)  # O
            continue

        candidates.sort()
        _, es, ee, et = candidates[0]
        b_idx, i_idx = TYPE_INDICES[et]
        span_key = (es, ee)
        if span_key not in seen:
            labels.append(b_idx)
            seen.add(span_key)
        else:
            labels.append(i_idx)

    return labels


class JointSpanNERDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_len: int, stride: int = 256):
        self.items: list[dict] = []
        for rec in records:
            text = rec["text"]
            entities = [e for e in rec.get("entities", [])
                        if e.get("type") in TYPE_PRIORITY]
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
                labels  = align_labels_joint(offsets, entities)
                self.items.append({
                    "input_ids":      enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i],
                    "labels":         labels,
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ── Trainer ───────────────────────────────────────────────────────────────────
class JointBioNERTrainer(Trainer):
    """Trainer with dynamic label smoothing and HAB class upweighting."""

    def __init__(self, *args, cfg: RunCfg | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg or RunCfg()
        # Upweight B-HAB (idx 7) and I-HAB (idx 8); others weight=1
        if self.cfg.use_hab_weight:
            w = [1.0] * NUM_LABELS
            w[7] = 3.0
            w[8] = 3.0
            self.class_weight = torch.tensor(w)
        else:
            self.class_weight = torch.ones(NUM_LABELS)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        progress = self.state.global_step / max(self.state.max_steps, 1)
        ls = self.cfg.label_smoothing_start + (
            self.cfg.label_smoothing_end - self.cfg.label_smoothing_start
        ) * progress

        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weight.to(logits.device),
            ignore_index=-100,
            label_smoothing=ls,
        )
        loss = loss_fct(logits.view(-1, NUM_LABELS), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# ── Training ──────────────────────────────────────────────────────────────────
def training_kwargs(output_dir: Path, cfg: RunCfg) -> dict:
    kw = dict(
        output_dir=str(output_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=cfg.save_total_limit,
        report_to="none",
        seed=cfg.seed,
        fp16=torch.cuda.is_available(),
    )
    params = inspect.signature(TrainingArguments.__init__).parameters
    kw["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "epoch"
    return kw


def make_compute_metrics_joint():
    """Token-level span F1 proxy for early stopping (all entity types combined)."""
    def compute_metrics(eval_pred):
        pred_ids = np.argmax(eval_pred.predictions, axis=-1)
        labels   = eval_pred.label_ids
        tp = fp = fn = 0
        for pred_row, label_row in zip(pred_ids, labels):
            for p, l in zip(pred_row, label_row):
                if l == -100:
                    continue
                # Count B-label exact matches as TP proxy
                if l > 0 and l % 2 == 1:  # B-* labels are odd
                    if p == l:
                        tp += 1
                    else:
                        fn += 1
                        if p > 0 and p % 2 == 1:
                            fp += 1
        p  = tp / (tp + fp) if tp + fp else 0.0
        r  = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return {"f1": f1, "precision": p, "recall": r}
    return compute_metrics


def train_joint(
    model_path: str,
    train_records: list[dict],
    dev_records: list[dict],
    output_dir: Path,
    cfg: RunCfg,
    mode: str,  # "full" | "lastlayer"
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    if mode == "lastlayer":
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    train_ds = JointSpanNERDataset(train_records, tokenizer, cfg.max_len, cfg.stride)
    dev_ds   = JointSpanNERDataset(dev_records,   tokenizer, cfg.max_len, cfg.stride)
    LOGGER.info("joint train windows=%d dev windows=%d mode=%s", len(train_ds), len(dev_ds), mode)

    trainer = JointBioNERTrainer(
        model=model,
        args=TrainingArguments(**training_kwargs(output_dir / "ckpts", cfg)),
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=make_compute_metrics_joint(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
        cfg=cfg,
    )
    trainer.train()
    best_dir = output_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    return best_dir


# ── Evaluation ────────────────────────────────────────────────────────────────
def softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def extract_spans_joint(text: str, logits: np.ndarray, offsets: list) -> list[dict]:
    """Decode 9-class logit array into character-span predictions for all types."""
    spans = []
    for et, (b_idx, i_idx) in TYPE_INDICES.items():
        in_span = False
        s_start = s_end = 0
        scores: list[float] = []
        for i, (cs, ce) in enumerate(offsets):
            if cs == ce:
                continue
            probs = softmax_np(logits[i])
            label = int(np.argmax(logits[i]))
            if label == b_idx:
                if in_span:
                    spans.append({"start": s_start, "end": s_end, "type": et,
                                  "score": float(np.mean(scores)),
                                  "text": text[s_start:s_end]})
                in_span = True; s_start = cs; s_end = ce
                scores = [float(probs[b_idx])]
            elif label == i_idx and in_span:
                s_end = ce
                scores.append(float(probs[i_idx]))
            else:
                if in_span:
                    spans.append({"start": s_start, "end": s_end, "type": et,
                                  "score": float(np.mean(scores)),
                                  "text": text[s_start:s_end]})
                in_span = False
        if in_span:
            spans.append({"start": s_start, "end": s_end, "type": et,
                          "score": float(np.mean(scores)),
                          "text": text[s_start:s_end]})
    return spans


def predict_joint(model_dir: Path, records: list[dict], batch_size: int, max_len: int) -> list[list[dict]]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir), local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    all_preds = []
    for rec in records:
        enc = tokenizer(
            rec["text"],
            truncation=True,
            max_length=max_len,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            stride=256,
            padding=True,
            return_tensors="pt",
        )
        offsets_all = enc.pop("offset_mapping").tolist()
        enc.pop("overflow_to_sample_mapping", None)
        raw_spans = []
        n = enc["input_ids"].shape[0]
        for start in range(0, n, batch_size):
            batch = {k: v[start:start+batch_size].to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**batch).logits.detach().cpu().numpy()
            for j in range(logits.shape[0]):
                raw_spans.extend(
                    extract_spans_joint(rec["text"], logits[j], offsets_all[start+j])
                )
        # Greedy de-overlap by score descending (same as binary pipeline)
        raw_spans.sort(key=lambda s: -s.get("score", 0.0))
        final: list[dict] = []
        for s in raw_spans:
            if not any(int(s["start"]) < int(k["end"]) and int(s["end"]) > int(k["start"])
                       for k in final):
                final.append(s)
        final.sort(key=lambda s: s["start"])
        all_preds.append(final)
    return all_preds


def overlaps(a: dict, b: dict) -> bool:
    return int(a["start"]) < int(b["end"]) and int(a["end"]) > int(b["start"])


def prf(tp, fp, fn):
    p  = tp / (tp + fp) if tp + fp else 0.0
    r  = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def partial_counts(gold: list, pred: list, entity_type: str | None = None):
    if entity_type:
        gold = [g for g in gold if g.get("type") == entity_type]
        pred = [p for p in pred if p.get("type") == entity_type]
    matched: set[int] = set()
    tp = 0
    for p in sorted(pred, key=lambda s: -float(s.get("score", 0.0))):
        for idx, g in enumerate(gold):
            if idx in matched or p.get("type") != g.get("type"):
                continue
            if overlaps(p, g):
                matched.add(idx); tp += 1; break
    return tp, len(pred) - tp, len(gold) - tp


def score_records(records: list[dict], predictions: list[list[dict]]) -> dict:
    totals: dict[str, list] = defaultdict(lambda: [0, 0, 0])
    for rec, pred in zip(records, predictions):
        gold = rec.get("entities", [])
        for et in ENTITY_TYPES:
            tp, fp, fn = partial_counts(gold, pred, et)
            totals[et][0] += tp; totals[et][1] += fp; totals[et][2] += fn
        tp, fp, fn = partial_counts(gold, pred)
        totals["overall"][0] += tp; totals["overall"][1] += fp; totals["overall"][2] += fn
    out: dict[str, Any] = {}
    for key, (tp, fp, fn) in totals.items():
        p, r, f1 = prf(tp, fp, fn)
        out[key] = {"P": round(p,4), "R": round(r,4), "F1": round(f1,4),
                    "TP": tp, "FP": fp, "FN": fn}
    out["macro_F1"] = round(float(np.mean([out[t]["F1"] for t in ENTITY_TYPES])), 4)
    out["evaluation"] = "new_golden_partial_span"
    return out


def evaluate_golden_joint(model_dir: Path, out_path: Path, cfg: RunCfg) -> dict:
    records  = load_jsonl(GOLDEN_JSONL)
    preds    = predict_joint(model_dir, records, cfg.batch_size, cfg.max_len)
    result   = score_records(records, preds)
    out_path.write_text(json.dumps(result, indent=2))
    return result


# ── Experiment runner ──────────────────────────────────────────────────────────
def run_experiment_joint(
    model_key: str,
    strategy: str,   # "E3" | "E6"
    seed: int = 42,
    batch_size: int = 32,
    force: bool = False,
) -> None:
    assert strategy in ("E3", "E6"), f"Joint ablation only supports E3 and E6, got {strategy}"
    model_path = MODEL_PATHS[model_key]
    set_seed(seed)

    if strategy == "E3":
        exp_name = "silver_full"
    else:
        exp_name = "silver_full_bronze_lastlayer"

    out_dir  = RESULT_ROOT / f"JOINT_{strategy}_{model_key}_{exp_name}_seed{seed}"
    done_file = out_dir / "DONE"
    if done_file.exists() and not force:
        LOGGER.info("Skip completed %s", out_dir.name)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("=== JOINT %s %s seed=%d ===", strategy, model_key, seed)

    if strategy == "E3":
        # Single-stage: Silver full FT
        train, dev = load_train_dev_joint("silver", seed)
        cfg = RunCfg(
            seed=seed, batch_size=batch_size, epochs=100,
            early_stopping_patience=12, lr=2e-5,
            warmup_ratio=0.05, weight_decay=0.1,
            label_smoothing_start=0.1, use_hab_weight=True,
        )
        LOGGER.info("Silver train=%d dev=%d", len(train), len(dev))
        best_dir = train_joint(model_path, train, dev, out_dir / "stage", cfg, "full")

    else:  # E6
        # Stage 1: Silver full FT
        s1_train, s1_dev = load_train_dev_joint("silver", seed)
        s1_cfg = RunCfg(
            seed=seed, batch_size=batch_size, epochs=100,
            early_stopping_patience=12, lr=2e-5,
            warmup_ratio=0.05, weight_decay=0.1,
            label_smoothing_start=0.1, use_hab_weight=True,
        )
        LOGGER.info("Stage1 Silver train=%d dev=%d", len(s1_train), len(s1_dev))
        s1_best = train_joint(model_path, s1_train, s1_dev,
                              out_dir / "stage1", s1_cfg, "full")

        # Stage 2: Bronze last-layer (encoder frozen)
        s2_train, s2_dev = load_train_dev_joint("bronze", seed)
        s2_cfg = RunCfg(
            seed=seed, batch_size=batch_size, epochs=60,
            early_stopping_patience=8, lr=1e-4,   # last-layer needs higher LR
            warmup_ratio=0.05, weight_decay=0.1,
            label_smoothing_start=0.0,             # disabled for lastlayer (same as binary)
            use_hab_weight=False,                  # disabled for lastlayer (same as binary)
        )
        LOGGER.info("Stage2 Bronze train=%d dev=%d (lastlayer)", len(s2_train), len(s2_dev))
        best_dir = train_joint(str(s1_best), s2_train, s2_dev,
                               out_dir / "stage2", s2_cfg, "lastlayer")

    # Evaluate on Golden V2
    LOGGER.info("Evaluating on Golden V2 …")
    eval_cfg = RunCfg(batch_size=batch_size)
    result = evaluate_golden_joint(best_dir, out_dir / "eval_golden_v2.json", eval_cfg)
    LOGGER.info("macro_F1=%.4f  %s", result["macro_F1"],
                "  ".join(f"{t}={result[t]['F1']:.4f}" for t in ENTITY_TYPES))

    done_file.write_text("done\n")
    LOGGER.info("DONE %s", out_dir.name)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_PATHS), default="pubmedbert")
    parser.add_argument("--strategies", nargs="+", choices=["E3", "E6"], default=["E3", "E6"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for seed in args.seeds:
        for strategy in args.strategies:
            run_experiment_joint(
                args.model, strategy, seed=seed,
                batch_size=args.batch_size, force=args.force,
            )


if __name__ == "__main__":
    main()
