"""One metric definition for every CliniBench system.

Encoders and generative LLMs were scored by two different code paths in the
paper (``src/encoder_models/model/metrics.py`` via Lightning, and
``src/generative_models/metrics.py`` via ``map_and_evaluate.py``). Those paths
agree on macro recall/precision but not on MAP, F1 or ranking. This module is
the single definition used for the corrected numbers in the README.

Protocol (identical for every system, per data partition):

* Label space: the short codes of the released test gold. CliniBench keeps only
  codes that occur in all three splits, so this is the same set for every
  system on a partition (``check_shared_label_space`` enforces this).
* Gold: short codes of a note, de-duplicated in annotation order. The first
  code is the main diagnosis.
* Predictions: a ranked list, truncated to short codes, de-duplicated by first
  occurrence, cut to the top ``n`` (20). Encoders rank by score; generative
  LLMs rank by generation order.
* Macro recall / precision / F1: per class over the label space, 0 when
  undefined, then the unweighted mean. F1 is the mean of per-class F1.
* Micro recall / precision / F1: pooled over all (note, code) pairs.
* MAP@n: per-note average precision over the top-n list, normalised by
  ``min(|gold|, n)``, averaged over notes.
* MD Acc: share of notes whose main diagnosis is in the top-n list.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

TOP_N = 20


def short_code(code: str, icd_version: int) -> str:
    """CliniBench short code: 3 characters, 4 for ICD-9 E-codes."""
    code = str(code).strip().replace(".", "")
    if icd_version == 9 and code[:1].upper() == "E":
        return code[:4]
    return code[:3]


def dedup(codes: Iterable[str]) -> list[str]:
    seen, out = set(), []
    for code in codes:
        if code and code not in seen:
            seen.add(code)
            out.append(code)
    return out


def gold_short_codes(codes: Iterable[str], icd_version: int) -> list[str]:
    return dedup(short_code(c, icd_version) for c in codes)


def ranked_short_codes(codes: Iterable[str], icd_version: int,
                       top_n: int = TOP_N) -> list[str]:
    return dedup(short_code(c, icd_version) for c in codes if str(c).strip())[:top_n]


def check_shared_label_space(label_spaces: dict[str, set[str]]) -> set[str]:
    """All systems of one partition must be scored on the same label space."""
    reference_name, reference = next(iter(label_spaces.items()))
    for name, labels in label_spaces.items():
        if labels != reference:
            raise AssertionError(
                f"label space of {name} ({len(labels)} codes) differs from "
                f"{reference_name} ({len(reference)} codes); systems would be "
                "macro-averaged over different classes")
    return reference


def average_precision_at_n(pred: Sequence[str], gold: set[str], top_n: int) -> float:
    hits, total = 0, 0.0
    for rank, code in enumerate(pred[:top_n], start=1):
        if code in gold:
            hits += 1
            total += hits / rank
    return total / min(len(gold), top_n)


def evaluate(preds: Sequence[Sequence[str]], golds: Sequence[Sequence[str]],
             label_space: set[str], top_n: int = TOP_N) -> dict[str, float]:
    """Score ranked short-code lists against short-code gold lists."""
    if len(preds) != len(golds):
        raise AssertionError(f"{len(preds)} predictions vs {len(golds)} gold rows")
    if any(not g for g in golds):
        raise AssertionError("notes without gold codes must be removed upstream")

    index = {code: i for i, code in enumerate(sorted(label_space))}
    tp = np.zeros(len(index))
    fp = np.zeros(len(index))
    fn = np.zeros(len(index))
    micro_tp = micro_fp = micro_fn = 0
    out_of_space = 0
    aps, main_hits, n_pred = [], 0, 0

    for pred, gold in zip(preds, golds):
        pred = list(pred)[:top_n]
        pred_set, gold_set = set(pred), set(gold)
        if not gold_set <= index.keys():
            raise AssertionError(f"gold codes outside label space: {gold_set - index.keys()}")
        for code in pred_set & gold_set:
            tp[index[code]] += 1
        for code in pred_set - gold_set:
            if code in index:
                fp[index[code]] += 1
            else:
                out_of_space += 1
        for code in gold_set - pred_set:
            fn[index[code]] += 1
        micro_tp += len(pred_set & gold_set)
        micro_fp += len(pred_set - gold_set)
        micro_fn += len(gold_set - pred_set)
        aps.append(average_precision_at_n(pred, gold_set, top_n))
        main_hits += gold[0] in pred_set
        n_pred += len(pred_set)

    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    f1 = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp),
                   where=(2 * tp + fp + fn) > 0)
    micro_p = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0

    n = len(golds)
    return {
        "macro_recall": 100 * float(recall.mean()),
        "macro_precision": 100 * float(precision.mean()),
        "macro_f1": 100 * float(f1.mean()),
        "micro_recall": 100 * micro_r,
        "micro_precision": 100 * micro_p,
        "micro_f1": 100 * micro_f1,
        f"map@{top_n}": 100 * float(np.mean(aps)),
        "md_acc": 100 * main_hits / n,
        "avg_predicted_codes": n_pred / n,
        "avg_gold_codes": float(np.mean([len(set(g)) for g in golds])),
        "n_notes": n,
        "n_labels": len(index),
        "predictions_outside_label_space": out_of_space,
    }
