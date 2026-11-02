"""Original linear-head Golden v2 inference and raw-logit span fusion."""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

ENTITY_TYPES = ["MIC", "DIS", "CHE", "HAB"]
MAX_LEN = 512
STRIDE = 256
BATCH = 16

def overlaps(a, b):
    return int(a["start"]) < int(b["end"]) and int(a["end"]) > int(b["start"])


def extract_spans_hf(text, logits, offsets, entity_type):
    """Extract char spans from HF logit array (3-label: O/B/I)."""
    spans = []
    in_span = False
    s_start = s_end = 0
    scores = []
    for i, (cs, ce) in enumerate(offsets):
        if cs == ce: continue  # special token
        label = int(np.argmax(logits[i]))
        conf  = float(np.max(logits[i]))
        if label == 1:  # B
            if in_span:
                spans.append({"start": s_start, "end": s_end,
                              "score": float(np.mean(scores)), "type": entity_type})
            in_span = True; s_start = cs; s_end = ce; scores = [conf]
        elif label == 2 and in_span:  # I
            s_end = ce; scores.append(conf)
        else:
            if in_span:
                spans.append({"start": s_start, "end": s_end,
                              "score": float(np.mean(scores)), "type": entity_type})
            in_span = False
    if in_span:
        spans.append({"start": s_start, "end": s_end,
                      "score": float(np.mean(scores)), "type": entity_type})
    return spans


def predict_hf(model_dir, records, entity_type, device):
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    mdl = AutoModelForTokenClassification.from_pretrained(str(model_dir), local_files_only=True).to(device)
    mdl.eval()
    all_preds = []
    for rec in records:
        enc = tok(rec["text"], truncation=True, max_length=MAX_LEN,
                  return_offsets_mapping=True, return_overflowing_tokens=True,
                  stride=STRIDE, padding=True, return_tensors="pt")
        offsets_list = enc.pop("offset_mapping").tolist()
        enc.pop("overflow_to_sample_mapping", None)
        spans = []
        n = enc["input_ids"].shape[0]
        for start in range(0, n, BATCH):
            batch = {k: v[start:start+BATCH].to(device) for k, v in enc.items()}
            with torch.no_grad():
                logits = mdl(**batch).logits.detach().cpu().numpy()
            for j in range(logits.shape[0]):
                spans.extend(extract_spans_hf(rec["text"], logits[j],
                                               offsets_list[start+j], entity_type))
        all_preds.append(spans)
    return all_preds


def merge_preds(per_type_preds, n_records):
    merged = []
    for i in range(n_records):
        all_spans = []
        for et in ENTITY_TYPES:
            all_spans.extend(per_type_preds[et][i])
        # de-overlap by score desc
        all_spans.sort(key=lambda s: -s.get("score", 0))
        final = []
        for s in all_spans:
            if not any(overlaps(s, k) for k in final):
                final.append(s)
        merged.append(final)
    return merged
