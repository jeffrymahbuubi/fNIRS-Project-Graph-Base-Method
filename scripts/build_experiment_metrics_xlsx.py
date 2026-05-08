"""Build per-architecture experiment_metrics.xlsx mirroring the format in
references/analysis/experiment_metrics.xlsx (minus BA, NPV, MCC, kappa).

For each architecture in {spatial_graph, spatial_temporal_graph}, walks
experiments/<arch>/{5-fold, 10-fold, loso}/<exp>/, reads per-fold/per-subject
pickles + the *_overall.pkl, and emits 6 sheets per workbook:
  - 5-Fold Summary, 5-Fold Detail
  - 10-Fold Summary, 10-Fold Detail
  - LOSO Summary, LOSO Detail

Metrics: Acc, Sens, Spec, Prec, F1 (no BA/NPV/MCC/kappa).
Macro = mean ± SD across folds/subjects (per-fold metric, then aggregated).
Micro = overall_* from the pooled-prediction *_overall.pkl.
95% CI = mean ± 1.96 * SD / sqrt(n).
"""

from __future__ import annotations

import glob
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = ROOT / "experiments"

ARCHITECTURES = ["spatial_graph", "spatial_temporal_graph"]

# Order matters: how rows appear in the spreadsheet.
METRIC_KEYS = ["accuracy", "sensitivity", "specificity", "precision", "f1"]
METRIC_LABELS = {
    "accuracy": "Acc",
    "sensitivity": "Sens",
    "specificity": "Spec",
    "precision": "Prec",
    "f1": "F1",
}

CV_DIRS = {
    "5-fold": ("5-fold", "5-Fold"),
    "10-fold": ("10-fold", "10-Fold"),
    "loso": ("loso", "LOSO"),
}


def to_scalar(v: Any) -> float:
    if hasattr(v, "shape"):
        return float(v)
    return float(v)


def load_pkl(p: Path) -> dict:
    with open(p, "rb") as f:
        return pickle.load(f)


_NAME_RE = re.compile(
    r"^(?:ST_)?GATv2_(?P<task>[A-Za-z0-9]+)_(?P<signal>hbo|hbr|hbt)_"
    r"(?P<val>kfold|loso)_mt(?P<mt>\d+)_"
)


def parse_exp_name(name: str) -> dict | None:
    m = _NAME_RE.match(name)
    if not m:
        return None
    return {
        "task": m["task"],
        "signal": m["signal"].upper(),
        "validation": m["val"],
        "mt": int(m["mt"]),
    }


def collect_per_fold(exp_dir: Path, validation: str) -> list[dict]:
    pattern = "*_fold_*.pkl" if validation == "kfold" else "*_subj_*.pkl"
    pkls = sorted(exp_dir.glob(pattern))
    rows = []
    for p in pkls:
        d = load_pkl(p)
        rows.append(
            {
                "accuracy": to_scalar(d["accuracy"]),
                "f1": to_scalar(d["f1_score"]),
                "precision": to_scalar(d["precision"]),
                "sensitivity": to_scalar(d["sensitivity"]),
                "specificity": to_scalar(d["specificity"]),
            }
        )
    return rows


def collect_overall(exp_dir: Path, validation: str) -> dict | None:
    suffix = "_kfold_overall.pkl" if validation == "kfold" else "_loso_overall.pkl"
    candidates = list(exp_dir.glob(f"*{suffix}"))
    if not candidates:
        return None
    d = load_pkl(candidates[0])
    return {
        "accuracy": to_scalar(d["overall_accuracy"]),
        "f1": to_scalar(d["overall_f1"]),
        "precision": to_scalar(d["overall_precision"]),
        "sensitivity": to_scalar(d["overall_sensitivity"]),
        "specificity": to_scalar(d["overall_specificity"]),
    }


def macro_stats(per_fold: list[dict]) -> dict:
    n = len(per_fold)
    out = {}
    for k in METRIC_KEYS:
        vals = np.array([r[k] for r in per_fold], dtype=float)
        m = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        se = sd / np.sqrt(n) if n > 1 else 0.0
        half = 1.96 * se
        out[k] = {
            "mean": m,
            "sd": sd,
            "ci_lo": m - half,
            "ci_hi": m + half,
            "n": n,
        }
    return out


def fmt_meansd(mean: float, sd: float) -> str:
    return f"{mean:.3f}±{sd:.3f}"


def fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"


def build_summary_row(meta: dict, macro: dict, micro: dict, n: int, cv_label: str) -> dict:
    row = {
        "Task": meta["task"],
        "Signal": meta["signal"],
        "MaxTrials": f"mt{meta['mt']}",
        "CV": cv_label,
        "N Folds/Subjects": n,
    }
    for k in METRIC_KEYS:
        L = METRIC_LABELS[k]
        s = macro[k]
        row[f"{L} Mean±SD"] = fmt_meansd(s["mean"], s["sd"])
    for k in METRIC_KEYS:
        L = METRIC_LABELS[k]
        row[f"Overall {L}"] = round(micro[k], 4)
    return row


def build_detail_row(meta: dict, macro: dict, micro: dict, n: int, cv_label: str) -> dict:
    row = {
        "Task": meta["task"],
        "Signal": meta["signal"],
        "MaxTrials": f"mt{meta['mt']}",
        "CV": cv_label,
        "N Folds/Subjects": n,
    }
    for k in METRIC_KEYS:
        L = METRIC_LABELS[k]
        s = macro[k]
        row[f"{L} Mean"] = round(s["mean"], 4)
        row[f"{L} SD"] = round(s["sd"], 4)
        row[f"{L} Mean±SD"] = fmt_meansd(s["mean"], s["sd"])
        row[f"{L} 95% CI"] = fmt_ci(s["ci_lo"], s["ci_hi"])
    for k in METRIC_KEYS:
        L = METRIC_LABELS[k]
        row[f"Overall {L}"] = round(micro[k], 4)
    return row


def process_arch(arch: str) -> dict[str, pd.DataFrame]:
    """Return dict[sheet_name -> DataFrame] for one architecture."""
    arch_dir = EXP_ROOT / arch
    sheets: dict[str, pd.DataFrame] = {}

    for cv_key, (cv_subdir, cv_label) in CV_DIRS.items():
        cv_path = arch_dir / cv_subdir
        if not cv_path.is_dir():
            continue

        summary_rows = []
        detail_rows = []
        for exp_dir in sorted(cv_path.iterdir()):
            if not exp_dir.is_dir():
                continue
            meta = parse_exp_name(exp_dir.name)
            if meta is None:
                print(f"  ! could not parse experiment name: {exp_dir.name}")
                continue

            validation = "kfold" if cv_key in ("5-fold", "10-fold") else "loso"
            per_fold = collect_per_fold(exp_dir, validation)
            if not per_fold:
                print(f"  ! no per-fold pickles in {exp_dir}")
                continue
            micro = collect_overall(exp_dir, validation)
            if micro is None:
                print(f"  ! no overall pickle in {exp_dir}")
                continue

            macro = macro_stats(per_fold)
            n = len(per_fold)
            summary_rows.append(build_summary_row(meta, macro, micro, n, cv_label))
            detail_rows.append(build_detail_row(meta, macro, micro, n, cv_label))

        if not summary_rows:
            continue

        # Order rows: signal (HBO, HBR, HBT) then mt (mt2, mt4)
        sort_key = lambda r: (
            ["HBO", "HBR", "HBT"].index(r["Signal"])
            if r["Signal"] in ("HBO", "HBR", "HBT")
            else 99,
            r["MaxTrials"],
        )
        summary_rows.sort(key=sort_key)
        detail_rows.sort(key=sort_key)

        sheets[f"{cv_label} Summary"] = pd.DataFrame(summary_rows)
        sheets[f"{cv_label} Detail"] = pd.DataFrame(detail_rows)

    return sheets


def write_workbook(arch: str, sheets: dict[str, pd.DataFrame]) -> Path:
    out = EXP_ROOT / arch / "experiment_metrics.xlsx"
    sheet_order = [
        "5-Fold Summary",
        "5-Fold Detail",
        "10-Fold Summary",
        "10-Fold Detail",
        "LOSO Summary",
        "LOSO Detail",
    ]
    # Columns that should display with fixed 4-decimal formatting (numeric).
    four_dp_suffixes = (" Mean", " SD")
    four_dp_prefixes = ("Overall ",)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for sheet in sheet_order:
            if sheet in sheets:
                sheets[sheet].to_excel(writer, sheet_name=sheet, index=False)

        for sheet in sheet_order:
            if sheet not in sheets:
                continue
            ws = writer.sheets[sheet]
            df = sheets[sheet]
            for i, col in enumerate(df.columns, start=1):
                col_letter = ws.cell(row=1, column=i).column_letter

                # Auto-fit column width.
                max_len = max(
                    [len(str(col))]
                    + [len(str(v)) for v in df[col].astype(str).tolist()]
                )
                ws.column_dimensions[col_letter].width = min(
                    max(max_len + 2, 8), 28
                )

                # Apply 4-decimal display to numeric metric columns.
                is_numeric_metric = col.startswith(four_dp_prefixes) or any(
                    col.endswith(suf) and not col.endswith("Mean±SD")
                    for suf in four_dp_suffixes
                )
                if is_numeric_metric:
                    for row in range(2, ws.max_row + 1):
                        ws.cell(row=row, column=i).number_format = "0.0000"
    return out


def main() -> None:
    for arch in ARCHITECTURES:
        print(f"\n=== {arch} ===")
        sheets = process_arch(arch)
        if not sheets:
            print(f"  no data found under experiments/{arch}/ — skipping")
            continue
        out = write_workbook(arch, sheets)
        print(f"  wrote {out}")
        for name, df in sheets.items():
            print(f"    [{name}] {len(df)} rows x {len(df.columns)} cols")


if __name__ == "__main__":
    main()
