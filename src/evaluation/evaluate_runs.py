"""Score every CliniBench run with the shared protocol in ``unified_metrics``.

Inputs (all positionally aligned, one row per test note with gold codes):

* Encoder dumps ``Encoder_<model>_<tuned|untuned>_<icu|hosp>-icd-<9|10>.json``
  with ``predicted_codes@20`` (score-ranked) and ``annotated_codes``.
* Generative runs ``<model>-<icu|hosp>-icd-<9|10>-<setting>.{json,parquet}``
  as written by ``map_and_evaluate.py --save_mapped_results`` with
  ``predicted_codes`` (generation order) and ``annotated_codes``. A boolean
  ``in_json`` column, if present, marks the rows that belong to the run.

Usage:
    python src/evaluation/evaluate_runs.py score --results_dir <dir> --out_csv results/unified_metrics.csv
    python src/evaluation/evaluate_runs.py tables --metrics_csv results/unified_metrics.csv --all_runs_md results/all_runs.md
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from unified_metrics import (TOP_N, check_shared_label_space, evaluate,
                             gold_short_codes, ranked_short_codes)

ENCODER_RE = re.compile(r"^Encoder_(?P<model>.+)_(?P<setting>tuned|untuned)_"
                        r"(?P<unit>icu|hosp)-icd-(?P<icd>9|10)\.json$")
GENERATIVE_RE = re.compile(r"^(?P<model>.+?)-(?P<unit>icu|hosp)-icd-(?P<icd>9|10)-"
                           r"(?P<setting>zeroshot(?:-cot)?|fewshot-\d+(?:-[a-z0-9-]+)?)"
                           r"\.(?:json|parquet)$")

# File-name suffix -> retriever name used in the paper (verified by reproducing
# Table 3: recall/precision/MD Acc of each suffix match exactly one row).
RETRIEVERS = {"": "PubMedBERT-NeuML", "pubmed-clinibench": "BioMedBERT-ft",
              "bm25": "BM25", "gold": "Gold heuristic", "random": "Random"}


# ── discovery ────────────────────────────────────────────────────────────────
def discover_runs(results_dir: Path, exclude: str | None = None) -> list[dict]:
    runs = []
    for path in sorted(results_dir.rglob("*")):
        if exclude and re.search(exclude, path.name):
            continue
        if m := ENCODER_RE.match(path.name):
            runs.append({"family": "encoder", "path": path, **m.groupdict()})
        elif m := GENERATIVE_RE.match(path.name):
            runs.append({"family": "generative", "path": path, **m.groupdict()})
    keys = [(r["family"], r["model"], r["setting"], r["unit"], r["icd"]) for r in runs]
    duplicates = {k for k in keys if keys.count(k) > 1}
    if duplicates:
        raise AssertionError(f"same run found in several files: {sorted(duplicates)}")
    for run in runs:
        run["icd"] = int(run["icd"])
        run.update(describe_setting(run["family"], run["setting"]))
    return runs


def describe_setting(family: str, setting: str) -> dict:
    if family == "encoder":
        return {"prompting": "threshold-tuned" if setting == "tuned" else "untuned",
                "shots": 0, "retriever": ""}
    if setting in ("zeroshot", "zeroshot-cot"):
        return {"prompting": "zero-shot CoT" if setting.endswith("cot") else "zero-shot",
                "shots": 0, "retriever": ""}
    m = re.match(r"fewshot-(\d+)-?(.*)$", setting)
    return {"prompting": "few-shot", "shots": int(m.group(1)),
            "retriever": RETRIEVERS[m.group(2)]}


# ── loading ──────────────────────────────────────────────────────────────────
def load_run(run: dict) -> tuple[list[list[str]], list[list[str]]]:
    """Ranked short-code predictions and short-code gold, rows with gold only."""
    path, icd = run["path"], run["icd"]
    if run["family"] == "encoder":
        payload = json.loads(path.read_text())
        raw_preds, raw_gold = payload["predicted_codes@20"], payload["annotated_codes"]
    elif path.suffix == ".json":
        payload = json.loads(path.read_text())
        raw_preds, raw_gold = payload["predicted_codes"], payload["annotated_codes"]
    else:
        columns = ["predicted_codes", "annotated_codes"]
        if "in_json" in pq.read_schema(path).names:
            columns.append("in_json")
        table = pq.read_table(path, columns=columns).to_pydict()
        keep = table.get("in_json") or [True] * len(table["annotated_codes"])
        raw_preds = [p for p, k in zip(table["predicted_codes"], keep) if k]
        raw_gold = [g for g, k in zip(table["annotated_codes"], keep) if k]

    if len(raw_preds) != len(raw_gold):
        raise AssertionError(f"{path.name}: {len(raw_preds)} predictions vs {len(raw_gold)} gold rows")
    golds = [gold_short_codes(g, icd) for g in raw_gold]
    preds = [ranked_short_codes(p or [], icd, TOP_N) for p in raw_preds]
    keep = [i for i, g in enumerate(golds) if g]
    return [preds[i] for i in keep], [golds[i] for i in keep]


# ── scoring ──────────────────────────────────────────────────────────────────
def score(results_dir: Path, out_csv: Path, skip_existing: bool,
          time_budget_s: float | None, exclude: str | None) -> None:
    import time
    started = time.time()
    runs = discover_runs(results_dir, exclude)
    done = set()
    if skip_existing and out_csv.exists():
        done = set(pd.read_csv(out_csv)["file"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for unit_icd, group in pd.DataFrame(runs).groupby(["unit", "icd"]):
        label_space = partition_label_space(group.to_dict("records"))
        for run in group.to_dict("records"):
            if run["path"].name in done:
                continue
            if time_budget_s and time.time() - started > time_budget_s:
                print("time budget reached, rerun with --skip_existing to continue")
                return
            preds, golds = load_run(run)
            check_shared_label_space({run["path"].name: {c for g in golds for c in g},
                                      "partition": label_space})
            row = {k: run[k] for k in ("family", "model", "setting", "prompting", "shots",
                                       "retriever", "unit", "icd")}
            row["file"] = run["path"].name
            row |= evaluate(preds, golds, label_space, TOP_N)
            pd.DataFrame([row]).to_csv(out_csv, mode="a", index=False,
                                       header=not out_csv.exists())
            print(f"{row['file']}: R {row['macro_recall']:.2f} P {row['macro_precision']:.2f} "
                  f"F1 {row['macro_f1']:.2f} MAP {row[f'map@{TOP_N}']:.2f} MD {row['md_acc']:.2f}",
                  flush=True)


def partition_label_space(runs: list[dict]) -> set[str]:
    """Label space of a partition, taken from an encoder dump's gold (cheap to load)."""
    reference = next((r for r in runs if r["family"] == "encoder"), runs[0])
    _, golds = load_run(reference)
    return {c for g in golds for c in g}


# ── tables ───────────────────────────────────────────────────────────────────
COLUMNS = {"macro_recall": "Rec.", "macro_precision": "Prec.", "map@20": "MAP@20",
           "md_acc": "MD Acc.", "macro_f1": "F1"}


def mean_over_icd(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    counts = df.groupby(keys)["icd"].nunique()
    if (counts != 2).any():
        raise AssertionError(f"missing ICD-9 or ICD-10 run for {counts[counts != 2].index.tolist()}")
    return df.groupby(keys, sort=False)[list(COLUMNS)].mean().reset_index()


MODEL_NAMES = {
    "BiomedBERT": "BioMedBERT-110M", "GatorTronS": "GatorTronS-345M", "M2-BERT": "M2-BERT-80M",
    "Llama3p1-8B": "Llama 3.1-8B", "Llama3p1-8B-Instruct": "Llama 3.1-8B-Instruct",
    "Llama3p1-70B": "Llama 3.1-70B", "Llama3p1-70B-Instruct": "Llama 3.1-70B-Instruct",
    "Mistral-7B": "Mistral-7B", "Mistral-7B-Instruct": "Mistral-7B-Instruct",
    "Mistral-Nemo": "Mistral-Nemo-13B", "Mistral-Nemo-Instruct": "Mistral-Nemo-13B-Instruct",
    "Qwen2-7B": "Qwen2-7B", "Qwen2-7B-Instruct": "Qwen2-7B-Instruct",
    "Qwen2-72B": "Qwen2-72B", "Qwen2-72B-Instruct": "Qwen2-72B-Instruct",
}
RETRIEVER_ORDER = ["Random", "BM25", "PubMedBERT-NeuML", "BioMedBERT-ft", "Gold heuristic"]


def aggregated(metrics_csv: Path) -> pd.DataFrame:
    """Unweighted mean over ICD-9 and ICD-10, as in the paper."""
    df = pd.read_csv(metrics_csv).fillna({"retriever": ""})
    keys = ["unit", "family", "model", "prompting", "shots", "retriever"]
    agg = mean_over_icd(df, keys)
    agg["Model"] = agg["model"].map(MODEL_NAMES)
    if agg["Model"].isna().any():
        raise AssertionError(f"no display name for {agg.loc[agg['Model'].isna(), 'model'].unique()}")
    agg["order"] = agg["model"].map(list(MODEL_NAMES).index)
    return agg.sort_values(["unit", "family", "order", "prompting", "shots"])


def setting_label(row) -> str:
    if row["family"] == "encoder":
        return row["prompting"]
    if row["shots"] == 0:
        return row["prompting"]
    return f"{row['shots']}-shot {row['retriever']}"


def main_tables(agg: pd.DataFrame) -> str:
    """README tables: encoders, then LLMs zero-shot / zero-shot CoT / 5-shot BM25."""
    parts = []
    for unit in ("hosp", "icu"):
        sub = agg[agg["unit"] == unit]
        enc = sub[sub["family"] == "encoder"].sort_values(
            ["order", "prompting"], ascending=[True, False]).copy()
        enc["Model"] = enc["Model"] + enc["prompting"].map(
            {"threshold-tuned": " (threshold-tuned)", "untuned": ""})
        parts.append(f"**{unit.upper()}: encoders**\n\n" + to_markdown(enc, ["Model"]))
        for title, mask in (
                ("zero-shot", (sub["shots"] == 0) & (sub["prompting"] == "zero-shot")),
                ("zero-shot CoT", sub["prompting"] == "zero-shot CoT"),
                ("5-shot BM25", (sub["shots"] == 5) & (sub["retriever"] == "BM25"))):
            gen = sub[(sub["family"] == "generative") & mask]
            parts.append(f"**{unit.upper()}: generative LLMs, {title}**\n\n"
                         + to_markdown(gen, ["Model"]))
    return "\n\n".join(parts)


def retriever_table(agg: pd.DataFrame, model: str = "Qwen2-72B-Instruct") -> str:
    """Counterpart of the paper's Tables 3 and 10."""
    parts = []
    for unit in ("hosp", "icu"):
        sub = agg[(agg["unit"] == unit) & (agg["model"] == model)
                  & agg["prompting"].isin(["zero-shot", "few-shot"])].copy()
        sub["Retriever"] = sub["retriever"].replace("", "Zero-shot")
        sub["order"] = sub["Retriever"].map((["Zero-shot"] + RETRIEVER_ORDER).index)
        sub = sub.sort_values(["order", "shots"]).rename(columns={"shots": "#FS"})
        parts.append(f"**{unit.upper()}: {MODEL_NAMES[model]}, few-shot retrievers**\n\n"
                     + to_markdown(sub, ["Retriever", "#FS"]))
    return "\n\n".join(parts)


def all_runs_table(agg: pd.DataFrame) -> str:
    parts = []
    for unit in ("hosp", "icu"):
        sub = agg[agg["unit"] == unit].copy()
        sub["Setting"] = sub.apply(setting_label, axis=1)
        sub["r"] = sub["retriever"].map(lambda r: ([""] + RETRIEVER_ORDER).index(r))
        sub = sub.sort_values(["family", "order", "r", "shots", "prompting"])
        parts.append(f"## {unit.upper()}\n\n" + to_markdown(sub, ["Model", "Setting"]))
    return "\n\n".join(parts)


def to_markdown(df: pd.DataFrame, keys: list[str]) -> str:
    out = df[keys + list(COLUMNS)].rename(columns=COLUMNS)
    lines = ["| " + " | ".join(map(str, out.columns)) + " |",
             "|" + "|".join([":---"] * len(keys) + ["---:"] * len(COLUMNS)) + "|"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v)
                                       for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--results_dir", type=Path, required=True)
    s.add_argument("--out_csv", type=Path, default=Path("results/unified_metrics.csv"))
    s.add_argument("--skip_existing", action="store_true")
    s.add_argument("--time_budget_s", type=float, default=None)
    s.add_argument("--exclude", default=None, help="regex on file names to skip")
    t = sub.add_parser("tables")
    t.add_argument("--metrics_csv", type=Path, default=Path("results/unified_metrics.csv"))
    t.add_argument("--all_runs_md", type=Path, default=Path("results/all_runs.md"))
    args = parser.parse_args()
    if args.command == "score":
        score(args.results_dir, args.out_csv, args.skip_existing, args.time_budget_s,
              args.exclude)
    else:
        agg = aggregated(args.metrics_csv)
        print(main_tables(agg) + "\n\n" + retriever_table(agg))
        args.all_runs_md.write_text(
            "# CliniBench: all runs (unified evaluation)\n\n"
            "Macro-averaged over the shared label space, top-20 predictions, unweighted "
            "mean of ICD-9 and ICD-10. Produced by `src/evaluation/evaluate_runs.py tables`; "
            "per-partition values in `unified_metrics.csv`.\n\n" + all_runs_table(agg) + "\n")


if __name__ == "__main__":
    main()
