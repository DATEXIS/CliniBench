"""Score every CliniBench run with the shared protocol in ``unified_metrics``.

Inputs (all positionally aligned, one row per test note with gold codes):

* Encoder dumps ``Encoder_<model>_<tuned|untuned>_<icu|hosp>-icd-<9|10>.json``
  with ``predicted_codes@20`` (score-ranked), ``predicted_codes`` (every label
  above its threshold: 0.5 untuned, per-class validation thresholds tuned) and
  ``annotated_codes``. Each dump is scored twice: ``budget=top-20`` (the paper's
  protocol, shared with the LLMs) and ``budget=threshold`` (the encoder's own
  operating point; no MAP because the set is unranked).
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
def load_run(run: dict, budget: str = "top-20") -> tuple[list[list[str]], list[list[str]]]:
    """Short-code predictions and short-code gold, rows with gold only."""
    path, icd = run["path"], run["icd"]
    if run["family"] == "encoder":
        payload = json.loads(path.read_text())
        key = "predicted_codes@20" if budget == "top-20" else "predicted_codes"
        raw_preds, raw_gold = payload[key], payload["annotated_codes"]
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
    top_n = TOP_N if budget == "top-20" else None
    preds = [ranked_short_codes(p or [], icd, top_n) for p in raw_preds]
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
        done = set(pd.read_csv(out_csv)[["file", "budget"]].itertuples(index=False, name=None))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for unit_icd, group in pd.DataFrame(runs).groupby(["unit", "icd"]):
        label_space = partition_label_space(group.to_dict("records"))
        for run in group.to_dict("records"):
            budgets = ["top-20", "threshold"] if run["family"] == "encoder" else ["top-20"]
            for budget in budgets:
                if (run["path"].name, budget) in done:
                    continue
                if time_budget_s and time.time() - started > time_budget_s:
                    print("time budget reached, rerun with --skip_existing to continue")
                    return
                preds, golds = load_run(run, budget)
                check_shared_label_space({run["path"].name: {c for g in golds for c in g},
                                          "partition": label_space})
                row = {k: run[k] for k in ("family", "model", "setting", "prompting",
                                           "shots", "retriever", "unit", "icd")}
                row |= {"budget": budget, "file": run["path"].name}
                row |= evaluate(preds, golds, label_space,
                                TOP_N if budget == "top-20" else None)
                pd.DataFrame([row]).to_csv(out_csv, mode="a", index=False,
                                           header=not out_csv.exists())
                print(f"{row['file']} [{budget}]: microF1 {row['micro_f1']:.2f} "
                      f"macroF1 {row['macro_f1']:.2f} MAP {row[f'map@{TOP_N}']:.2f} "
                      f"MD {row['md_acc']:.2f}", flush=True)


def partition_label_space(runs: list[dict]) -> set[str]:
    """Label space of a partition, taken from an encoder dump's gold (cheap to load)."""
    reference = next((r for r in runs if r["family"] == "encoder"), runs[0])
    _, golds = load_run(reference)
    return {c for g in golds for c in g}


# ── tables ───────────────────────────────────────────────────────────────────
ALL_COLUMNS = {"micro_recall": "Micro Rec.", "micro_precision": "Micro Prec.",
               "micro_f1": "Micro F1", "macro_recall": "Macro Rec.",
               "macro_precision": "Macro Prec.", "macro_f1": "Macro F1",
               "map@20": "MAP@20", "md_acc": "MD Acc."}


def mean_over_icd(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    counts = df.groupby(keys)["icd"].nunique()
    if (counts != 2).any():
        raise AssertionError(f"missing ICD-9 or ICD-10 run for {counts[counts != 2].index.tolist()}")
    columns = list(ALL_COLUMNS) + ["avg_predicted_codes"]
    return df.groupby(keys, sort=False)[columns].mean().reset_index()


MODEL_NAMES = {
    "BiomedBERT": "BioMedBERT-110M", "GatorTronS": "GatorTronS-345M", "M2-BERT": "M2-BERT-80M",
    "Llama3p1-8B": "Llama 3.1-8B", "Llama3p1-8B-Instruct": "Llama 3.1-8B-Instruct",
    "Llama3p1-70B": "Llama 3.1-70B", "Llama3p1-70B-Instruct": "Llama 3.1-70B-Instruct",
    "Mistral-7B": "Mistral-7B", "Mistral-7B-Instruct": "Mistral-7B-Instruct",
    "Mistral-Nemo": "Mistral-Nemo-13B", "Mistral-Nemo-Instruct": "Mistral-Nemo-13B-Instruct",
    "Qwen2-7B": "Qwen2-7B", "Qwen2-7B-Instruct": "Qwen2-7B-Instruct",
    "Qwen2-72B": "Qwen2-72B", "Qwen2-72B-Instruct": "Qwen2-72B-Instruct",
}
ENCODER_SETTINGS = {("untuned", "top-20"): "top-20, raw scores",
                    ("threshold-tuned", "top-20"): "top-20, tuned scores",
                    ("untuned", "threshold"): "threshold 0.5",
                    ("threshold-tuned", "threshold"): "tuned thresholds"}
RETRIEVER_ORDER = ["Random", "BM25", "PubMedBERT-NeuML", "BioMedBERT-ft", "Gold heuristic"]


def aggregated(metrics_csv: Path) -> pd.DataFrame:
    """Unweighted mean over ICD-9 and ICD-10, as in the paper."""
    df = pd.read_csv(metrics_csv).fillna({"retriever": ""})
    keys = ["unit", "family", "model", "prompting", "shots", "retriever", "budget"]
    agg = mean_over_icd(df, keys)
    agg["Model"] = agg["model"].map(MODEL_NAMES)
    if agg["Model"].isna().any():
        raise AssertionError(f"no display name for {agg.loc[agg['Model'].isna(), 'model'].unique()}")
    agg["order"] = agg["model"].map(list(MODEL_NAMES).index)
    return agg.sort_values(["unit", "family", "order", "prompting", "shots"])


def setting_label(row) -> str:
    if row["family"] == "encoder":
        return ENCODER_SETTINGS[(row["prompting"], row["budget"])]
    if row["shots"] == 0:
        return row["prompting"]
    return f"{row['shots']}-shot {row['retriever']}"


UNITS = ("hosp", "icu")
F1_COLUMNS = {"micro_f1": "Micro F1", "macro_f1": "Macro F1"}
MAIN_METRICS = {"micro_f1": "Micro F1", "macro_f1": "Macro F1", "map@20": "MAP@20"}
MAIN_SETTINGS = {
    "Zero-shot": lambda d: (d["family"] == "encoder") | ((d["shots"] == 0) & (d["prompting"] == "zero-shot")),
    "5-shot BM25": lambda d: (d["shots"] == 5) & (d["retriever"] == "BM25"),
}


def grid_markdown(df: pd.DataFrame, row: str, specs: list[tuple], bold_best: bool = False) -> str:
    """One line per value of ``row`` (in frame order); ``specs`` = [(header, mask_fn, metric)]."""
    labels = list(dict.fromkeys(df[row]))
    columns = []
    for header, mask, metric in specs:
        values = df[mask(df)].set_index(row)[metric]
        if values.index.duplicated().any():
            raise AssertionError(f"column {header!r} has several values per {row}")
        columns.append((header, values))
    lines = ["| " + " | ".join([row] + [h for h, _ in columns]) + " |",
             "|:---|" + "---:|" * len(columns)]
    for label in labels:
        cells = []
        for _, values in columns:
            v = values.get(label, float("nan"))
            cell = "–" if v != v else f"{v:.2f}"
            if bold_best and v == values.max():
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def unit_mask(unit: str, extra=None):
    return lambda d: (d["unit"] == unit) & (extra(d) if extra else True)


def unit_table(agg: pd.DataFrame, unit: str) -> str:
    """Paper-style table for one care unit: encoders, then LLMs; zero-shot and 5-shot BM25.

    Encoders (fine-tuned, top 20 by raw score) fill the zero-shot columns. Bold marks the
    best value per column, underline the best generative LLM where it is not the best overall.
    """
    sub = agg[(agg["unit"] == unit) & (agg["budget"] == "top-20")
              & ((agg["family"] == "generative") | (agg["prompting"] == "untuned"))]
    columns = [(f"{setting} {name}", sub[mask(sub)].set_index("Model")[metric])
               for setting, mask in MAIN_SETTINGS.items() for metric, name in MAIN_METRICS.items()]
    gen_models = set(sub.loc[sub["family"] == "generative", "Model"])
    lines = ["| Model | " + " | ".join(h for h, _ in columns) + " |",
             "|:---|" + "---:|" * len(columns)]
    for family, title in (("encoder", "Encoders"), ("generative", "Generative LLMs")):
        lines.append(f"| *{title}* |" + " |" * len(columns))
        for model in dict.fromkeys(sub.loc[sub["family"] == family, "Model"]):
            cells = []
            for _, values in columns:
                v = values.get(model, float("nan"))
                cell = "–" if v != v else f"{v:.2f}"
                gen_best = values[values.index.isin(gen_models)].max()
                if v == values.max():
                    cell = f"**{cell}**"
                elif model in gen_models and v == gen_best:
                    cell = f"<ins>{cell}</ins>"
                cells.append(cell)
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main_tables(agg: pd.DataFrame) -> str:
    return "\n\n".join(f"**{u.upper()}**\n\n" + unit_table(agg, u) for u in UNITS)


def threshold_table(agg: pd.DataFrame) -> str:
    """Encoders at both operating points: the shared top-20 budget and their own thresholds."""
    enc = agg[agg["family"] == "encoder"].copy()
    enc["s"] = enc.apply(lambda r: list(ENCODER_SETTINGS).index((r["prompting"], r["budget"])), axis=1)
    enc = enc.sort_values(["order", "s"])
    enc["Encoder, setting"] = enc["Model"] + ", " + enc.apply(setting_label, axis=1)
    specs = [(f"{u.upper()} {name}", unit_mask(u), m) for u in UNITS
             for m, name in ({"avg_predicted_codes": "Codes/note"} | F1_COLUMNS).items()]
    return grid_markdown(enc, "Encoder, setting", specs)


def retriever_table(agg: pd.DataFrame, model: str = "Qwen2-72B-Instruct") -> str:
    """Counterpart of the paper's Tables 3 and 10."""
    sub = agg[(agg["model"] == model) & agg["prompting"].isin(["zero-shot", "few-shot"])].copy()
    sub["o"] = sub["retriever"].map(([""] + RETRIEVER_ORDER).index)
    sub = sub.sort_values(["o", "shots"])
    sub["Demonstrations"] = sub.apply(
        lambda r: "none (zero-shot)" if r["shots"] == 0 else f"{r['retriever']}, {r['shots']}-shot", axis=1)
    specs = [(f"{u.upper()} {name}", unit_mask(u), m)
             for u in UNITS for m, name in F1_COLUMNS.items()]
    return grid_markdown(sub, "Demonstrations", specs)


def all_runs_table(agg: pd.DataFrame) -> str:
    parts = []
    for unit in ("hosp", "icu"):
        sub = agg[agg["unit"] == unit].copy()
        sub["Setting"] = sub.apply(setting_label, axis=1)
        sub["r"] = sub["retriever"].map(lambda r: ([""] + RETRIEVER_ORDER).index(r))
        sub["s"] = sub.apply(lambda r: list(ENCODER_SETTINGS).index((r["prompting"], r["budget"]))
                             if r["family"] == "encoder" else 0, axis=1)
        sub = sub.sort_values(["family", "order", "s", "r", "shots", "prompting"])
        parts.append(f"## {unit.upper()}\n\n" + to_markdown(
            sub, ["Model", "Setting"], ALL_COLUMNS | {"avg_predicted_codes": "Codes/note"}))
    return "\n\n".join(parts)


def to_markdown(df: pd.DataFrame, keys: list[str], columns: dict[str, str]) -> str:
    out = df[keys + list(columns)].rename(columns=columns)
    lines = ["| " + " | ".join(map(str, out.columns)) + " |",
             "|" + "|".join([":---"] * len(keys) + ["---:"] * len(columns)) + "|"]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(
            ("–" if v != v else f"{v:.2f}") if isinstance(v, float) else str(v)
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
        print("\n\n<!-- next table -->\n\n".join(
            [main_tables(agg), threshold_table(agg), retriever_table(agg)]))
        args.all_runs_md.write_text(
            "# CliniBench: all runs (unified evaluation)\n\n"
            "Micro- and macro-averaged over the shared label space, top-20 predictions, "
            "unweighted mean of ICD-9 and ICD-10. Produced by `src/evaluation/evaluate_runs.py tables`; "
            "per-partition values in `unified_metrics.csv`.\n\n" + all_runs_table(agg) + "\n")


if __name__ == "__main__":
    main()
