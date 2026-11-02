#!/usr/bin/env python3
"""
Binary NER with optional CRF decoder and source-conditioned soft labels.

Examples:
  CUDA_VISIBLE_DEVICES=0 python train_softcrf.py --model pubmedbert --gpu_id 0
  CUDA_VISIBLE_DEVICES=1 python train_softcrf.py --model biobert --gpu_id 1
"""

import argparse
import glob
import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from torchcrf import CRF
except ImportError:
    from TorchCRF import CRF


MODEL_PATHS = {
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
}

_DATA_ROOT = Path(os.environ.get("BIOMETANER_DATA_ROOT", Path(__file__).resolve().parents[1] / "data/private"))
BRONZE_PATH = str(_DATA_ROOT / "bronze_type_by_source_80_10_10")
BRONZE_CORRECTED_PATH = str(_DATA_ROOT / "bronze_corrected.jsonl")
SILVER_PATH = str(_DATA_ROOT / "silver.jsonl")
GOLDEN_PATH = str(_DATA_ROOT / "golden_metasigndb_v2_for_eval.jsonl")
MINT_PATH = str(_DATA_ROOT / "mint_predictions.jsonl")
OUTPUT_ROOT = str(Path(os.environ.get("BIOMETANER_OUTPUT_ROOT", Path(__file__).resolve().parents[1] / "outputs/training")))

ENTITY_TYPES = ["MIC", "DIS", "CHE", "HAB"]

FP_RATE = {
    "BC5CDR": {"MIC": 1.0, "DIS": 0.387, "CHE": 0.549, "HAB": 1.0},
    "BioRED": {"MIC": 0.99, "DIS": 0.459, "CHE": 0.593, "HAB": 1.0},
    "NCBI_Disease": {"MIC": 1.0, "DIS": 0.368, "CHE": 1.0, "HAB": 1.0},
    "S800": {"MIC": 0.623, "DIS": 1.0, "CHE": 1.0, "HAB": 1.0},
    "GutBrainIE": {"MIC": 0.134, "DIS": 0.341, "CHE": 0.333, "HAB": 0.333},
    "BB2019": {"MIC": 0.236, "DIS": 1.0, "CHE": 1.0, "HAB": 1.0},
}
FN_RATE = {
    "BC5CDR": {"MIC": 1.0, "DIS": 0.028, "CHE": 0.0, "HAB": 1.0},
    "BioRED": {"MIC": 0.0, "DIS": 0.019, "CHE": 0.522, "HAB": 1.0},
    "NCBI_Disease": {"MIC": 1.0, "DIS": 0.0, "CHE": 1.0, "HAB": 1.0},
    "S800": {"MIC": 0.052, "DIS": 1.0, "CHE": 1.0, "HAB": 1.0},
    "GutBrainIE": {"MIC": 0.094, "DIS": 0.049, "CHE": 0.127, "HAB": 0.653},
    "BB2019": {"MIC": 0.276, "DIS": 1.0, "CHE": 1.0, "HAB": 1.0},
}
MINT_P = {"CHE": 0.8675, "DIS": 0.9342, "HAB": 0.572, "MIC": 0.9043}
MINT_R = {"CHE": 0.731, "DIS": 0.8173, "HAB": 0.8297, "MIC": 0.8551}


def resolve_encoder_path(model_name):
    path = MODEL_PATHS[model_name]
    if model_name == "biobert":
        snapshots = sorted(glob.glob(path.rstrip("/") + "/*"))
        snapshots = [p for p in snapshots if Path(p).is_dir()]
        complete = [
            p for p in snapshots
            if (Path(p) / "config.json").exists()
            and (Path(p) / "vocab.txt").exists()
            and ((Path(p) / "pytorch_model.bin").exists() or (Path(p) / "model.safetensors").exists())
        ]
        if complete:
            return complete[-1]
        if snapshots:
            return snapshots[-1]
        return "dmis-lab/biobert-base-cased-v1.1"
    return path


def pick_device(gpu_id):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    count = torch.cuda.device_count()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and count <= gpu_id:
        print(f"CUDA_VISIBLE_DEVICES={visible}; using visible cuda:0 for requested gpu_id={gpu_id}")
        return torch.device("cuda:0")
    return torch.device(f"cuda:{gpu_id}")


def rate(table, source, entity_type, default=0.5):
    if not source:
        return default
    if source in table:
        return table[source].get(entity_type, default)
    parts = [p for p in source.split("+") if p in table]
    vals = [table[p].get(entity_type, default) for p in parts]
    return sum(vals) / len(vals) if vals else default


def compute_soft_prob(bronze_label, source, mint_pred, entity_type, use_mint=False):
    fp = rate(FP_RATE, source, entity_type)
    fn = rate(FN_RATE, source, entity_type)
    bronze_pos = bronze_label == entity_type
    mint_pos = mint_pred == entity_type if use_mint else False
    if not use_mint:
        return float((1 - fp) if bronze_pos else fn)

    mp = MINT_P[entity_type]
    mr = MINT_R[entity_type]
    if bronze_pos and mint_pos:
        p = 1 - fp * (1 - mp)
    elif bronze_pos:
        p = (1 - fp) * (1 - mr)
    elif mint_pos:
        p = mp
    else:
        p = fn * (1 - mr)
    return float(max(0.0, min(1.0, p)))


class NERDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def split_file(path, entity_type, split):
    path = Path(path)
    if path.is_dir():
        candidate = path / entity_type / f"{split}.jsonl"
        if candidate.exists():
            return candidate
    return path


def load_jsonl(path, split=None, entity_type=None):
    if split and entity_type:
        p = split_file(path, entity_type, split)
        if p != Path(path):
            return list(iter_jsonl(p))
    data = []
    for d in iter_jsonl(path):
        if split is None or d.get("split") == split:
            data.append(d)
    return data


def encode_sample(text, entities, entity_type, tokenizer, source=None, mint_entities=None, use_mint=False, max_length=512):
    enc = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    char_labels = {}
    for ent in entities:
        if ent.get("type") != entity_type:
            continue
        for c in range(ent["start"], ent["end"]):
            char_labels[c] = "B" if c == ent["start"] else "I"

    mint_char = set()
    if use_mint and mint_entities:
        for ent in mint_entities:
            if ent.get("type") == entity_type:
                mint_char.update(range(ent["start"], ent["end"]))

    hard_labels = []
    soft_probs = []
    for s, e in offsets:
        if s == e:
            hard_labels.append(-100)
            soft_probs.append([1.0, 0.0, 0.0])
            continue
        chars_in_tok = range(s, e)
        if any(c in char_labels for c in chars_in_tok):
            bio = char_labels.get(s, "I")
            label_idx = 1 if bio == "B" else 2
            hard_labels.append(label_idx)
            p = compute_soft_prob(entity_type, source, entity_type if s in mint_char else "O", entity_type, use_mint)
            soft_probs.append([1 - p, p, 0.0] if label_idx == 1 else [1 - p, 0.0, p])
        else:
            hard_labels.append(0)
            p_fn = compute_soft_prob("O", source, entity_type if s in mint_char else "O", entity_type, use_mint)
            soft_probs.append([1 - p_fn, p_fn * 0.5, p_fn * 0.5])
    return {
        "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        "labels": torch.tensor(hard_labels, dtype=torch.long),
        "soft_probs": torch.tensor(soft_probs, dtype=torch.float),
    }


def build_dataset(data_path, entity_type, tokenizer, use_mint, split="train", mint_index=None):
    records = load_jsonl(data_path, split, entity_type)
    samples = []
    for d in records:
        mint_ents = (mint_index or {}).get(str(d.get("pmid")), [])
        source = d.get("source") or d.get("bronze_source")
        samples.append(encode_sample(d["text"], d.get("entities", []), entity_type, tokenizer, source, mint_ents, use_mint))
    return NERDataset(samples)


def collate_fn(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attn_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    soft_probs = torch.zeros(len(batch), max_len, 3)
    for i, b in enumerate(batch):
        length = b["input_ids"].size(0)
        input_ids[i, :length] = b["input_ids"]
        attn_mask[i, :length] = b["attention_mask"]
        labels[i, :length] = b["labels"]
        soft_probs[i, :length] = b["soft_probs"]
    return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels, "soft_probs": soft_probs}


class BinaryNERWithCRF(nn.Module):
    def __init__(self, encoder_path, num_labels=3, use_crf=True):
        super().__init__()
        self.use_crf = use_crf
        self.encoder = AutoModel.from_pretrained(encoder_path)
        hidden = self.encoder.config.hidden_size
        self.emission = nn.Linear(hidden, num_labels)
        if use_crf:
            self.crf = CRF(num_labels, batch_first=True)
            with torch.no_grad():
                self.crf.transitions[0, 2] = -1e4

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, soft_probs=None):
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**encoder_kwargs).last_hidden_state
        emissions = self.emission(hidden)
        mask = attention_mask.bool()

        if soft_probs is not None:
            if self.use_crf:
                return self._expected_crf_loss(emissions, soft_probs, mask)
            log_p = torch.log_softmax(emissions, dim=-1)
            loss = -(soft_probs * log_p).sum(-1)
            valid = mask.float()
            return (loss * valid).sum() / valid.sum().clamp_min(1.0)

        if labels is not None:
            if self.use_crf:
                crf_labels = labels.clone()
                crf_labels[crf_labels == -100] = 0
                # torchcrf requires the first timestep to be unmasked. Keep
                # special tokens in the CRF path as O while evaluation still
                # ignores them through offset_mapping.
                crf_mask = mask
                return -self.crf(emissions, crf_labels, mask=crf_mask, reduction="mean")
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            return loss_fn(emissions.view(-1, emissions.size(-1)), labels.view(-1))

        if self.use_crf:
            return self.crf.decode(emissions, mask=mask)
        return emissions.argmax(-1).tolist()

    def _expected_crf_loss(self, emissions, soft_probs, mask):
        log_z = self._log_norm(emissions, mask)
        exp_e = (soft_probs * emissions).sum(-1)
        exp_e = (exp_e * mask.float()).sum(-1)

        transitions = self.crf.transitions
        batch_size, seq_len, _ = emissions.shape
        exp_t = torch.zeros(batch_size, device=emissions.device)
        for t in range(seq_len - 1):
            valid = (mask[:, t] & mask[:, t + 1]).float()
            joint = soft_probs[:, t, :].unsqueeze(2) * soft_probs[:, t + 1, :].unsqueeze(1)
            exp_t += (joint * transitions.unsqueeze(0)).sum((-1, -2)) * valid
        return (log_z - exp_e - exp_t).mean()

    def _log_norm(self, emissions, mask):
        e_t = emissions.transpose(0, 1)
        m_t = mask.transpose(0, 1)
        score = self.crf.start_transitions + e_t[0]
        for i in range(1, e_t.size(0)):
            score_i = score.unsqueeze(2) + self.crf.transitions.unsqueeze(0) + e_t[i].unsqueeze(1)
            score_i = torch.logsumexp(score_i, dim=1)
            score = torch.where(m_t[i].unsqueeze(1), score_i, score)
        score = score + self.crf.end_transitions.unsqueeze(0)
        return torch.logsumexp(score, dim=1)


def extract_spans(token_preds, offsets, entity_type):
    spans = []
    in_ent = False
    start_char = None
    last_end = None
    for pred, (cs, ce) in zip(token_preds, offsets):
        if cs == ce:
            if in_ent and start_char is not None and last_end is not None:
                spans.append({"start": start_char, "end": last_end, "type": entity_type})
            in_ent = False
            start_char = None
            last_end = None
            continue
        if pred == 1:
            if in_ent and start_char is not None and last_end is not None:
                spans.append({"start": start_char, "end": last_end, "type": entity_type})
            start_char = cs
            in_ent = True
        elif pred == 2 and in_ent:
            pass
        else:
            if in_ent and start_char is not None and last_end is not None:
                spans.append({"start": start_char, "end": last_end, "type": entity_type})
            in_ent = False
            start_char = None
        last_end = ce
    if in_ent and start_char is not None and last_end is not None:
        spans.append({"start": start_char, "end": last_end, "type": entity_type})
    return spans


def evaluate(model, tokenizer, data_path, entity_type, split, device):
    model.eval()
    records = load_jsonl(data_path, split, entity_type)
    tp = fp = fn = 0
    with torch.no_grad():
        for d in records:
            enc = tokenizer(d["text"], truncation=True, max_length=512, return_offsets_mapping=True, return_tensors="pt")
            offsets = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(device) for k, v in enc.items()}
            preds = model(**enc)
            pred_seq = preds[0] if preds and isinstance(preds[0], list) else preds
            pred_spans = {(s["start"], s["end"]) for s in extract_spans(pred_seq, offsets, entity_type)}
            gold_spans = {(e["start"], e["end"]) for e in d.get("entities", []) if e.get("type") == entity_type}
            tp += len(pred_spans & gold_spans)
            fp += len(pred_spans - gold_spans)
            fn += len(gold_spans - pred_spans)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def train_one_type(entity_type, train_data_path, label_mode, use_crf, use_mint, exp_name, device, encoder_path, epochs=5, lr=2e-5, batch_size=16, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)

    mint_index = {}
    if use_mint and os.path.exists(MINT_PATH):
        for d in iter_jsonl(MINT_PATH):
            mint_index[str(d.get("pmid"))] = d.get("entities", [])
    elif use_mint:
        print(f"  MINT missing: {MINT_PATH}; falling back to source-only soft labels")

    train_ds = build_dataset(train_data_path, entity_type, tokenizer, use_mint, split="train", mint_index=mint_index)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = BinaryNERWithCRF(encoder_path, num_labels=3, use_crf=use_crf).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = max(1, len(train_dl) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    out_dir = Path(OUTPUT_ROOT) / exp_name / entity_type
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_ep = 0
    log = []
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            if label_mode == "soft":
                loss = model(batch["input_ids"], batch["attention_mask"], soft_probs=batch["soft_probs"])
            else:
                loss = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        dev_res = evaluate(model, tokenizer, train_data_path, entity_type, "dev", device)
        avg_loss = total_loss / max(1, len(train_dl))
        print(f"  Ep {ep}: loss={avg_loss:.4f} dev F1={dev_res['f1']:.4f}", flush=True)
        log.append({"epoch": ep, "loss": avg_loss, **dev_res})
        if dev_res["f1"] >= best_f1:
            best_f1 = dev_res["f1"]
            best_ep = ep
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    test_res = evaluate(model, tokenizer, BRONZE_PATH, entity_type, "test", device)
    golden_res = evaluate(model, tokenizer, GOLDEN_PATH, entity_type, None, device)
    for name, payload in [
        ("eval_bronze_test.json", test_res),
        ("eval_golden.json", golden_res),
        ("train_log.json", log),
        ("config.json", {"entity_type": entity_type, "label_mode": label_mode, "use_crf": use_crf, "use_mint": use_mint, "best_epoch": best_ep, "best_dev_f1": best_f1, "encoder_path": encoder_path}),
    ]:
        with open(out_dir / name, "w") as f:
            json.dump(payload, f, indent=2)
    print(f"  [{entity_type}] golden F1={golden_res['f1']:.4f} test F1={test_res['f1']:.4f}", flush=True)
    return golden_res


def get_experiments(model_tag):
    return [
        (f"G2_{model_tag}_linear_hard", BRONZE_PATH, "hard", False, False),
        (f"G2_{model_tag}_CRF_hard", BRONZE_PATH, "hard", True, False),
        (f"G5_{model_tag}_linear_soft", BRONZE_CORRECTED_PATH, "soft", False, False),
        (f"G5_{model_tag}_CRF_soft", BRONZE_CORRECTED_PATH, "soft", True, False),
        (f"G5_{model_tag}_CRF_soft_MINT", BRONZE_CORRECTED_PATH, "soft", True, True),
    ]


def completed_result(exp_name, entity_type):
    result_path = Path(OUTPUT_ROOT) / exp_name / entity_type / "eval_golden.json"
    if not result_path.exists():
        return None
    with open(result_path) as f:
        return json.load(f)


def run_silver_stage2(base_exp, out_exp, device, encoder_path, epochs=3, lr=2e-5):
    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    for entity_type in ENTITY_TYPES:
        print(f"\n=== silver stage2 {entity_type} ===", flush=True)
        ckpt = Path(OUTPUT_ROOT) / base_exp / entity_type / "best_model.pt"
        if not ckpt.exists():
            print(f"  {base_exp} not complete; skipping")
            continue
        model = BinaryNERWithCRF(encoder_path, use_crf=True).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        silver_ds = build_dataset(SILVER_PATH, entity_type, tokenizer, False, split=None)
        dl = DataLoader(silver_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        out_dir = Path(OUTPUT_ROOT) / out_exp / entity_type
        out_dir.mkdir(parents=True, exist_ok=True)
        best_f1 = -1.0
        for ep in range(1, epochs + 1):
            model.train()
            for batch in dl:
                optimizer.zero_grad()
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            res = evaluate(model, tokenizer, BRONZE_PATH, entity_type, "dev", device)
            print(f"  Ep {ep}: dev F1={res['f1']:.4f}", flush=True)
            if res["f1"] >= best_f1:
                best_f1 = res["f1"]
                torch.save(model.state_dict(), out_dir / "best_model.pt")
        model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
        test_res = evaluate(model, tokenizer, BRONZE_PATH, entity_type, "test", device)
        golden_res = evaluate(model, tokenizer, GOLDEN_PATH, entity_type, None, device)
        with open(out_dir / "eval_bronze_test.json", "w") as f:
            json.dump(test_res, f, indent=2)
        with open(out_dir / "eval_golden.json", "w") as f:
            json.dump(golden_res, f, indent=2)
        print(f"  golden F1={golden_res['f1']:.4f}", flush=True)


def main():
    global BRONZE_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pubmedbert", "biobert"], required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--bronze_path", default=BRONZE_PATH)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    BRONZE_PATH = args.bronze_path
    device = pick_device(args.gpu_id)
    encoder_path = resolve_encoder_path(args.model)
    model_tag = "PubMedBERT" if args.model == "pubmedbert" else "BioBERT"
    if args.tag:
        model_tag = f"{model_tag}_{args.tag}"
    print(f"Using model={args.model} encoder_path={encoder_path} device={device}", flush=True)

    try:
        import torchcrf  # noqa: F401
    except ImportError:
        os.system("pip install pytorch-crf -q")

    for exp_name, data_path, label_mode, use_crf, use_mint in get_experiments(model_tag):
        if "G5" in exp_name and not os.path.exists(BRONZE_CORRECTED_PATH):
            pending_dir = Path(OUTPUT_ROOT) / exp_name
            pending_dir.mkdir(parents=True, exist_ok=True)
            with open(pending_dir / "PENDING.txt", "w") as f:
                f.write(f"Waiting for {BRONZE_CORRECTED_PATH}\n")
            print(f"{exp_name}: waiting for bronze_corrected.jsonl; skipping", flush=True)
            continue
        print(f"\n{'=' * 60}\nExperiment: {exp_name} use_crf={use_crf} label_mode={label_mode} use_mint={use_mint}\n{'=' * 60}", flush=True)
        results = {}
        for entity_type in ENTITY_TYPES:
            print(f"\n--- {entity_type} ---", flush=True)
            existing = completed_result(exp_name, entity_type)
            if existing is not None:
                print(f"  found existing eval_golden.json; skipping {exp_name}/{entity_type}", flush=True)
                results[entity_type] = existing
                continue
            results[entity_type] = train_one_type(
                entity_type=entity_type,
                train_data_path=data_path,
                label_mode=label_mode,
                use_crf=use_crf,
                use_mint=use_mint,
                exp_name=exp_name,
                device=device,
                encoder_path=encoder_path,
                epochs=3 if args.debug else 5,
            )
        avg_f1 = sum(r["f1"] for r in results.values()) / len(results)
        print(f"\n{exp_name} avg golden F1 = {avg_f1:.4f}", flush=True)

    print(f"\n=== G5_{model_tag}_CRF_soft_MINT_silver stage 2 ===", flush=True)
    run_silver_stage2(
        base_exp=f"G5_{model_tag}_CRF_soft_MINT",
        out_exp=f"G5_{model_tag}_CRF_soft_MINT_silver",
        device=device,
        encoder_path=encoder_path,
    )
    print("\nAll experiments complete", flush=True)


if __name__ == "__main__":
    main()
