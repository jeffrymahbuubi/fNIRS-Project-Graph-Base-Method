"""Run provenance — `run.json` writer (SPEC §7.3 / §10.5).

`write_run_json` bundles everything needed to reproduce a single XAI run:
- git SHA + dirty flag
- python / torch / torch_geometric / numpy / pandas / scipy / matplotlib versions
- ISO-8601 UTC timestamp
- the XAIRunConfig as a dict
- the PopulationResult counters and `extras` (estimator, head_reduce,
  layer_reduce, n_checkpoints, checkpoint labels, path_rebases, ...)

Lives next to the CSVs / NPY in `cfg.output_dir`.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.xai.aggregate import PopulationResult
from src.xai.config import XAIRunConfig


def _git_sha(repo_root: Path) -> Dict[str, Any]:
    """Returns {'sha', 'dirty', 'error'}; non-fatal — populates 'error' on failure."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return {"sha": sha, "dirty": bool(status), "error": None}
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        return {"sha": "unknown", "dirty": None, "error": repr(e)}


def _package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("torch", "torch_geometric", "numpy", "pandas", "scipy", "matplotlib"):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[name] = "not_installed"
    return versions


def _result_summary(result: PopulationResult) -> Dict[str, Any]:
    return {
        "arch": result.arch,
        "hb": result.hb,
        "regime": result.regime,
        "mt": result.mt,
        "n_trials": result.n_trials,
        "n_trials_total": result.n_trials_total,
        "n_subjects": result.n_subjects,
        "included_pct": result.included_pct,
        "per_subject_trial_counts": result.per_subject_trial_counts,
    }


def write_run_json(
    cfg: XAIRunConfig,
    result: PopulationResult,
    out_dir: str | Path | None = None,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist `run.json` with full provenance. Returns the absolute path.

    `out_dir` defaults to `cfg.output_dir`. `extra` is a free-form dict merged
    into the top level — use it to record one-off bits (e.g. notebook commit
    SHA when the notebook is what actually ran the cell).
    """
    out = Path(out_dir if out_dir is not None else cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    project_root = Path(cfg.project_root) if cfg.project_root else Path(__file__).resolve().parents[2]

    record: Dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_sha(project_root),
        "versions": _package_versions(),
        "config": asdict(cfg),
        "result": _result_summary(result),
        "extras": dict(result.extras),
    }
    if extra is not None:
        record["extra"] = extra

    path = out / "run.json"
    path.write_text(json.dumps(record, indent=2, default=str, sort_keys=False))
    return path


def read_run_json(in_dir: str | Path) -> Dict[str, Any]:
    """Inverse of `write_run_json`. Returns the parsed dict (no schema validation)."""
    p = Path(in_dir) / "run.json"
    return json.loads(p.read_text())
