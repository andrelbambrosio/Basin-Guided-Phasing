#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
23_analyze_autobuild_outcomes.py

Pipeline-integrated analysis of Phenix AutoBuild runs (AutoBuild_run_* layout),
restricted to user-specified methods.

Detects valid runs under:
  <phenix_root>/<method>/<pdb_id>/AutoBuild_run_*/overall_best.pdb
  <phenix_root>/<method>/<pdb_id>/AutoBuild_run_*/overall_best_refine_data.mtz
  plus a log tail containing "Citations for AutoBuild:".

Key outputs (default under <base_folder>/23_Autobuild_Analysis/):
  - 23_valid_jobs.csv                  (one row per valid job)
  - autobuild_rfree_raw.csv              (alias of 23_valid_jobs.csv)
  - all_metrics_by_pdb_id.csv            (wide pivot: method__metric columns)
  - outliers_long.csv                    (percentile outliers for multiple metrics)
  - 23_rfree_tails.csv                 (low-tail and high-tail lists per method)
  - plots: autobuild_rfree_by_method.png, autobuild_residues_placed_by_method.png,
           autobuild_delta_vs_*.png, autobuild_relative_residues_placed_vs_<ref>.png
  - logs/23_autobuild_outcomes_<timestamp>.log

NEW IN v02 (improvements requested earlier):
  - Progress logging for directory scanning (so it never “looks idle”).
  - Faster log-tail check (byte substring search; avoids decode/splitlines).
  - Avoids repeatedly globbing job_dir logs for each run (job-dir scan done once).
  - Optional parallel scanning of PDB job directories via ThreadPoolExecutor:
        --max_workers N  (default: min(32, os.cpu_count()+4))
  - Extra logging controls:
        --console_log_level INFO|DEBUG|WARNING...
        --file_log_level    INFO|DEBUG|WARNING...

NEW IN v02+ (this update):
  - Relative chain length proxy (residues placed ratio) vs ref method:
        rel_residues_placed_vs_<ref> = 100 * (method__residues_placed / ref__residues_placed)
    * Requires ref_method residues_placed >= --min_ref_residues_placed (default 20)
    * Plotted as jitter + box with annotated statistics, and added to all_metrics_by_pdb_id.csv.
  - Success report with user-tunable thresholds:
        * Free R <= --success_rfree_max
        * Relative residues placed (%) >= --success_relplaced_min
        evaluated on paired-complete subset (both method and ref present) and
        with ref residues_placed >= --min_ref_residues_placed.

Notes:
  - Space group (SG) is read from the PHIB input MTZ:
        <phenix_root>/<method>/<pdb_id>/<pdb_id>_<method>_PHIB_input.mtz
  - Total number of reflections is read from:
        overall_best_refine_data.mtz
    (Gemmi MTZ reflection count; i.e., number of HKL records in that MTZ.)
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shlex
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gemmi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Logging
# =============================================================================

def _parse_log_level(*, level_str: str) -> int:
    s = str(level_str).strip().upper()
    lvl = getattr(logging, s, None)
    if lvl is None:
        raise ValueError(f"Invalid log level: {level_str}. Use INFO, DEBUG, WARNING, ERROR, CRITICAL.")
    return int(lvl)


def configure_logging(
    *,
    out_dir: Path,
    console_level: int,
    file_level: int,
) -> Path:
    logs_dir = out_dir.parent / "LOGS"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"23_autobuild_outcomes_{ts}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # collect everything; handlers filter
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(filename=str(log_path), mode="w", encoding="utf-8")
    fh.setLevel(int(file_level))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(int(console_level))
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logging.info("Log file: %s", str(log_path))
    return log_path


# =============================================================================
# Plot style (uniform; matches your prefs)
# =============================================================================
MAJOR_TICK_SIZE: int = 10
MINOR_TICK_SIZE: int = 5
FS_NUMBERS: int = 14
FS_AXES: int = 14
FS_TITLE: int = 14
FS_LEGEND: int = 12
PLOT_DECIMALS: int = 1


def summarize_numeric(*, x: np.ndarray) -> Dict[str, float]:
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan, "q01": np.nan, "q99": np.nan}
    mean = float(np.mean(v))
    sd = float(np.std(v, ddof=1)) if n >= 2 else 0.0
    med = float(np.median(v))
    q01 = float(np.quantile(v, 0.01))
    q99 = float(np.quantile(v, 0.99))
    return {"n": n, "mean": mean, "sd": sd, "median": med, "q01": q01, "q99": q99}


def fmt_stats(*, stats: Dict[str, float], unit: str = "%", decimals: int = 3) -> str:
    if int(stats.get("n", 0)) <= 0 or not np.isfinite(float(stats.get("mean", np.nan))):
        return "n=0"
    f = f"{{:.{int(decimals)}f}}"
    u = f" {unit}".rstrip()
    return (
        f"n={int(stats['n'])} | "
        f"mean±sd={f.format(stats['mean'])}±{f.format(stats['sd'])}{u} | "
        f"median={f.format(stats['median'])}{u} | "
        f"q01={f.format(stats['q01'])}{u} | q99={f.format(stats['q99'])}{u}"
    )


def summarize_array(*, y: np.ndarray) -> Tuple[int, float, float, float]:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = int(y.size)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    mean = float(np.mean(y))
    sd = float(np.std(y, ddof=1)) if n > 1 else 0.0
    med = float(np.median(y))
    return n, mean, sd, med


def format_stats_label(*, method: str, y: np.ndarray, unit_suffix: str, decimals: int) -> str:
    n, mean, sd, med = summarize_array(y=y)
    if n == 0:
        return f"{method}\n n=0\n μ±σ=NA\n med=NA"
    fmt = f"{{:.{int(decimals)}f}}"
    suf = f" {unit_suffix}".rstrip()
    return (
        f"{method}\n"
        f" n={n}\n"
        f" μ±σ={fmt.format(mean)}±{fmt.format(sd)}{suf}\n"
        f" med={fmt.format(med)}{suf}"
    )


def jitter_boxplot_multimetric(
    *,
    df_long: pd.DataFrame,
    method_col: str,
    metric_col: str,
    value_col: str,
    metric_order: List[str],
    metric_labels: Dict[str, str],
    unit_suffix_by_metric: Dict[str, str],
    decimals_by_metric: Dict[str, int],
    title: str,
    ylabel: str,
    xlabel: str,
    out_png: Path,
    rng_seed: int,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> None:
    if df_long.empty:
        logging.info("[PLOT] No data to plot for %s", out_png.name)
        return

    methods_sorted = sorted(df_long[method_col].dropna().unique().tolist())
    if len(methods_sorted) == 0:
        logging.info("[PLOT] No methods to plot for %s", out_png.name)
        return

    offsets = {metric_order[0]: 0.0}
    box_width = 0.40
    jitter_span = 0.10

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)

    ax.minorticks_on()
    ax.grid(which="major", linestyle="--", linewidth=1.0, color="gray", alpha=0.6)
    ax.grid(which="minor", linestyle=":", linewidth=0.8, color="lightgray", alpha=0.8)
    ax.tick_params(axis="both", which="major", length=MAJOR_TICK_SIZE, labelsize=FS_NUMBERS)
    ax.tick_params(axis="both", which="minor", length=MINOR_TICK_SIZE)

    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_xlabel(xlabel, fontsize=FS_AXES)
    ax.set_ylabel(ylabel, fontsize=FS_AXES)

    base_positions = np.arange(len(methods_sorted), dtype=float)

    box_positions: List[float] = []
    box_data: List[np.ndarray] = []
    for method_index, method in enumerate(methods_sorted):
        metric = metric_order[0]
        y = df_long.loc[
            (df_long[method_col] == method) & (df_long[metric_col] == metric),
            value_col,
        ].dropna().to_numpy(dtype=float)
        box_positions.append(float(base_positions[method_index] + offsets[metric]))
        box_data.append(y)

    bp = ax.boxplot(
        box_data,
        positions=box_positions,
        widths=float(box_width),
        patch_artist=True,
        showfliers=True,
        medianprops={"linewidth": 1.5},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )
    for patch in bp["boxes"]:
        patch.set_alpha(0.25)

    rng = np.random.default_rng(seed=int(rng_seed))
    for method_index, method in enumerate(methods_sorted):
        metric = metric_order[0]
        y = df_long.loc[
            (df_long[method_col] == method) & (df_long[metric_col] == metric),
            value_col,
        ].dropna().to_numpy(dtype=float)
        if y.size == 0:
            continue
        x_center = float(base_positions[method_index] + offsets[metric])
        x = x_center + rng.uniform(low=-float(jitter_span), high=float(jitter_span), size=y.size)
        ax.scatter(x, y, s=24, alpha=0.8)

    xticklabels: List[str] = []
    for method in methods_sorted:
        metric = metric_order[0]
        y = df_long.loc[
            (df_long[method_col] == method) & (df_long[metric_col] == metric),
            value_col,
        ].dropna().to_numpy(dtype=float)
        xticklabels.append(
            format_stats_label(
                method=str(method),
                y=y,
                unit_suffix=str(unit_suffix_by_metric.get(metric, "")),
                decimals=int(decimals_by_metric.get(metric, PLOT_DECIMALS)),
            )
        )

    ax.set_xticks(base_positions)
    ax.set_xticklabels(xticklabels, fontsize=FS_NUMBERS)

    if (y_min is not None) or (y_max is not None):
        ax.set_ylim(
            bottom=float(y_min) if y_min is not None else None,
            top=float(y_max) if y_max is not None else None,
        )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=200)
    plt.close(fig)
    logging.info("[PLOT] Saved: %s", str(out_png))


# =============================================================================
# Validity detection: AutoBuild_run_* under <method>/<pdb_id>/
# =============================================================================

_CITATION_NEEDLE: bytes = b"Citations for AutoBuild:"


def _log_tail_has_citations(
    *,
    log_path: Path,
    tail_bytes: int = 65536,
) -> bool:
    """
    Fast tail scan: reads up to `tail_bytes` from the end of the file and searches for the byte needle.
    Avoids decode/splitlines overhead.
    """
    try:
        with open(file=log_path, mode="rb") as fh:
            fh.seek(0, os.SEEK_END)
            size: int = int(fh.tell())
            start: int = int(max(0, size - int(tail_bytes)))
            fh.seek(start, os.SEEK_SET)
            chunk: bytes = fh.read(int(tail_bytes))
        return _CITATION_NEEDLE in chunk
    except Exception:
        return False


def read_spacegroup_from_mtz(*, mtz_path: Path) -> str:
    """Read SG from MTZ via gemmi. Returns 'UNKNOWN' on failure."""
    try:
        mtz = gemmi.read_mtz_file(str(mtz_path))
        sg = mtz.spacegroup
        try:
            hm = sg.hm
            if hm:
                return str(hm)
        except Exception:
            pass
        return str(sg)
    except Exception:
        return "UNKNOWN"


def count_reflections_in_mtz(*, mtz_path: Path) -> Optional[int]:
    """Return number of MTZ reflections. None on failure."""
    try:
        mtz = gemmi.read_mtz_file(str(mtz_path))
        return int(mtz.nreflections)
    except Exception:
        return None


def _choose_valid_overall_best_from_job_dir(
    *,
    job_dir: Path,
    tail_bytes: int,
) -> Optional[Tuple[Path, Path, Path]]:
    """
    Select the most recent AutoBuild_run_* inside job_dir for which:
      - overall_best.pdb exists and non-empty
      - overall_best_refine_data.mtz exists and non-empty
      - some AutoBuild_run_*.log/LOG contains the citation marker near its end

    Improvements:
      - job_dir log globbing is performed once (not repeated per run_dir)
      - log tail scan uses a byte substring search
    """
    candidates: List[Tuple[float, Path, Path, Path]] = []

    run_dirs = [p for p in job_dir.glob("AutoBuild_run_*") if p.is_dir()]
    if not run_dirs:
        return None

    # Only scan job_dir logs once, as a fallback pool (some layouts place logs at the job root)
    job_level_logs: List[Path] = []
    job_level_logs.extend(job_dir.glob("AutoBuild_run_*.log"))
    job_level_logs.extend(job_dir.glob("AutoBuild_run_*.LOG"))
    job_level_logs = [lp for lp in job_level_logs if lp.is_file() and lp.stat().st_size > 0]

    for run_dir in sorted(run_dirs):
        best_pdb = run_dir / "overall_best.pdb"
        best_mtz = run_dir / "overall_best_refine_data.mtz"
        if not (best_pdb.is_file() and best_pdb.stat().st_size > 0):
            continue
        if not (best_mtz.is_file() and best_mtz.stat().st_size > 0):
            continue

        # Prefer logs inside the run directory
        logs: List[Path] = []
        logs.extend(run_dir.glob("AutoBuild_run_*.log"))
        logs.extend(run_dir.glob("AutoBuild_run_*.LOG"))
        logs = [lp for lp in logs if lp.is_file() and lp.stat().st_size > 0]

        # If none found, fall back to job-level logs (already collected once)
        if not logs:
            logs = list(job_level_logs)

        if not logs:
            continue

        chosen_log: Optional[Path] = None
        for lp in sorted(logs):
            if _log_tail_has_citations(log_path=lp, tail_bytes=int(tail_bytes)):
                chosen_log = lp
                break

        if chosen_log is None:
            continue

        candidates.append((float(run_dir.stat().st_mtime), best_pdb, best_mtz, chosen_log))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, best_pdb, best_mtz, chosen_log = candidates[0]
    return best_pdb, best_mtz, chosen_log


def _scan_one_pdb_dir(
    *,
    method: str,
    pdb_dir: Path,
    phenix_root: Path,
    tail_bytes: int,
) -> Optional[Tuple[str, str, str, Path, Path, Path]]:
    """
    Scan one PDB job directory and return a valid (method, pdb_id, sg, best_pdb, best_mtz, log_path) tuple.
    Returns None if invalid/no run.
    """
    best = _choose_valid_overall_best_from_job_dir(job_dir=pdb_dir, tail_bytes=int(tail_bytes))
    if best is None:
        return None
    best_pdb, best_mtz, log_path = best

    pdb_id = pdb_dir.name
    input_mtz = pdb_dir / f"{pdb_id}_{method}_PHIB_input.mtz"
    sg = read_spacegroup_from_mtz(mtz_path=input_mtz) if input_mtz.is_file() else "UNKNOWN"

    return (str(method), str(pdb_id), str(sg), best_pdb, best_mtz, log_path)


def find_valid_overall_best_for_methods(
    *,
    phenix_root: Path,
    methods: List[str],
    max_workers: int,
    progress_every: int,
    tail_bytes: int,
) -> List[Tuple[str, str, str, Path, Path, Path]]:
    """
    Returns list of:
      (method, pdb_id, SG, overall_best_pdb, overall_best_refine_data_mtz, log_path)
    """
    out: List[Tuple[str, str, str, Path, Path, Path]] = []

    for method in methods:
        method_dir = phenix_root / method
        if not method_dir.is_dir():
            logging.info("[SKIP] method dir missing: %s", str(method_dir))
            continue

        pdb_dirs = [p for p in method_dir.iterdir() if p.is_dir()]
        n_total = int(len(pdb_dirs))
        logging.info("[SCAN] method=%s | pdb_dirs=%d", str(method), n_total)
        if n_total == 0:
            continue

        # Parallel scan per pdb_dir (I/O-heavy; threads work well here)
        futures = []
        with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
            for pdb_dir in pdb_dirs:
                futures.append(
                    ex.submit(
                        _scan_one_pdb_dir,
                        method=str(method),
                        pdb_dir=Path(pdb_dir),
                        phenix_root=Path(phenix_root),
                        tail_bytes=int(tail_bytes),
                    )
                )

            done_count = 0
            for fut in as_completed(futures):
                done_count += 1
                if (done_count % int(progress_every)) == 0 or done_count == 1 or done_count == n_total:
                    logging.info("[SCAN] method=%s | %d/%d done", str(method), done_count, n_total)

                res = fut.result()
                if res is not None:
                    out.append(res)

        logging.info(
            "[SCAN] method=%s | valid_jobs=%d",
            str(method),
            int(sum(1 for t in out if t[0] == str(method))),
        )

    return out


# =============================================================================
# Parsing metrics
# =============================================================================

FREE_R_PATTERNS = [
    re.compile(r"^REMARK\s+3\s+FREE\s+R\s+VALUE\s*:?\s*([0-9]*\.?[0-9]+)\s*%?\s*$"),
    re.compile(r"^REMARK\s+3\s+R\s+FREE\s*:?\s*([0-9]*\.?[0-9]+)\s*%?\s*$"),
]


def parse_free_r_from_pdb(*, pdb_path: Path) -> Optional[float]:
    try:
        with open(file=pdb_path, mode="r", encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                line = raw.rstrip()
                for pat in FREE_R_PATTERNS:
                    m = pat.match(line)
                    if m:
                        val = float(m.group(1))
                        return val * 100.0 if val <= 1.0 else val
    except Exception:
        return None
    return None


BEST_CYCLE_RE = re.compile(r"Best solution on cycle:\s*(\d+)\b")
SOLUTION_HEADER_RE = re.compile(r"^\s*SOLUTION\s+CYCLE", re.IGNORECASE)
SOLUTION_LINE_RE = re.compile(r"^\s*\d+\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s+(\d+)\s*$")


def parse_best_cycle_and_residues(*, log_path: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return None, None, None

    best_cycle: Optional[int] = None
    for line in lines:
        m = BEST_CYCLE_RE.search(line)
        if m:
            try:
                best_cycle = int(m.group(1))
            except Exception:
                continue

    if best_cycle is None:
        return None, None, None

    in_table = False
    for line in lines:
        if not in_table:
            if SOLUTION_HEADER_RE.search(line):
                in_table = True
            continue
        if not line.strip():
            break
        m = SOLUTION_LINE_RE.match(line)
        if not m:
            continue
        try:
            cycle = int(m.group(1))
            built = int(m.group(4))
            placed = int(m.group(5))
        except Exception:
            continue
        if cycle == best_cycle:
            return best_cycle, built, placed

    return best_cycle, None, None


# =============================================================================
# Consolidation + outliers
# =============================================================================

def _wide_from_long(*, df_long: pd.DataFrame, index_col: str, method_col: str, metric_col: str, value_col: str) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame({index_col: []})
    df_piv = df_long.pivot_table(
        index=index_col,
        columns=[method_col, metric_col],
        values=value_col,
        aggfunc="first",
    )
    df_piv.columns = [f"{str(m)}__{str(k)}" for (m, k) in df_piv.columns.to_list()]
    return df_piv.reset_index(drop=False)


def compute_outliers_long(
    *,
    df_long: pd.DataFrame,
    outlier_percentage: float,
    index_col: str,
    method_col: str,
    metric_col: str,
    value_col: str,
) -> pd.DataFrame:
    p = float(outlier_percentage)
    if not (0.0 < p < 50.0):
        raise ValueError(f"--outlier_percentage must be in (0, 50). Got: {p}")

    rows: List[Dict[str, object]] = []
    for (method, metric), sub in df_long.groupby([method_col, metric_col]):
        vals = pd.to_numeric(sub[value_col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 3:
            continue

        low_thr = float(np.percentile(vals, p))
        high_thr = float(np.percentile(vals, 100.0 - p))

        sub2 = sub.copy()
        sub2[value_col] = pd.to_numeric(sub2[value_col], errors="coerce")
        sub2 = sub2.dropna(subset=[value_col])

        for _, r in sub2.iterrows():
            v = float(r[value_col])
            if v < low_thr:
                rows.append({
                    "method": str(method),
                    "metric": str(metric),
                    "pdb_id": str(r[index_col]),
                    "SG": str(r.get("SG", "UNKNOWN")),
                    "value": v,
                    "side": "low",
                    "low_threshold": low_thr,
                    "high_threshold": high_thr,
                    "outlier_percentage": p,
                })
            elif v > high_thr:
                rows.append({
                    "method": str(method),
                    "metric": str(metric),
                    "pdb_id": str(r[index_col]),
                    "SG": str(r.get("SG", "UNKNOWN")),
                    "value": v,
                    "side": "high",
                    "low_threshold": low_thr,
                    "high_threshold": high_thr,
                    "outlier_percentage": p,
                })
    return pd.DataFrame(rows)


# =============================================================================
# Rfree tail lists
# =============================================================================

def build_rfree_tails_table(
    *,
    df_ab: pd.DataFrame,
    methods: List[str],
    low_tail_pct: float,
    high_tail_pct: float,
    high_rfree_min: Optional[float],
) -> pd.DataFrame:
    """
    For each method:
      - lowest low_tail_pct percent
      - highest high_tail_pct percent AND (if provided) >= high_rfree_min

    Allow pct in (0, 100]. If pct==100: keep everything on that side before applying high_rfree_min.
    """
    low_tail_pct = float(low_tail_pct)
    high_tail_pct = float(high_tail_pct)

    if not (0.0 < low_tail_pct <= 100.0):
        raise ValueError(f"--low_tail_pct must be in (0,100]. Got: {low_tail_pct}")
    if not (0.0 < high_tail_pct <= 100.0):
        raise ValueError(f"--high_tail_pct must be in (0,100]. Got: {high_tail_pct}")

    rows: List[Dict[str, object]] = []

    for m in methods:
        sub = df_ab.loc[(df_ab["Method"] == m)].copy()
        sub["Rfree_percent"] = pd.to_numeric(sub["Rfree_percent"], errors="coerce")
        sub = sub.dropna(subset=["Rfree_percent"]).copy()

        n_method = int(sub.shape[0])
        if n_method == 0:
            continue

        vals = sub["Rfree_percent"].to_numpy(dtype=float)

        low_cut = float(np.percentile(vals, low_tail_pct))
        high_cut = float(np.percentile(vals, max(0.0, 100.0 - high_tail_pct)))

        low_sel = sub.loc[sub["Rfree_percent"] <= low_cut].copy()
        low_sel = low_sel.sort_values(by="Rfree_percent", ascending=True, kind="mergesort").reset_index(drop=True)
        for i, r in low_sel.iterrows():
            rows.append({
                "Method": str(m),
                "Side": "lowest",
                "Pct": float(low_tail_pct),
                "Threshold": float(low_cut),
                "Rank": int(i + 1),
                "PDB_ID": str(r["PDB_ID"]),
                "SG": str(r.get("SG", "UNKNOWN")),
                "N_refl_total": int(r["N_refl_total"]) if pd.notna(r.get("N_refl_total", np.nan)) else np.nan,
                "Rfree_percent": float(r["Rfree_percent"]),
                "n_method": int(n_method),
                "high_rfree_min": float(high_rfree_min) if high_rfree_min is not None else np.nan,
            })

        high_sel = sub.loc[sub["Rfree_percent"] >= high_cut].copy()
        if high_rfree_min is not None:
            high_sel = high_sel.loc[high_sel["Rfree_percent"] >= float(high_rfree_min)].copy()

        high_sel = high_sel.sort_values(by="Rfree_percent", ascending=False, kind="mergesort").reset_index(drop=True)
        for i, r in high_sel.iterrows():
            rows.append({
                "Method": str(m),
                "Side": "highest",
                "Pct": float(high_tail_pct),
                "Threshold": float(high_cut),
                "Rank": int(i + 1),
                "PDB_ID": str(r["PDB_ID"]),
                "SG": str(r.get("SG", "UNKNOWN")),
                "N_refl_total": int(r["N_refl_total"]) if pd.notna(r.get("N_refl_total", np.nan)) else np.nan,
                "Rfree_percent": float(r["Rfree_percent"]),
                "n_method": int(n_method),
                "high_rfree_min": float(high_rfree_min) if high_rfree_min is not None else np.nan,
            })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["Method", "Side", "Rank"], ascending=[True, True, True], kind="mergesort").reset_index(drop=True)
    return out


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="23: Analyze AutoBuild outcomes, paired method comparisons, outliers, and success rates.")
    ap.add_argument("--base_folder", type=str, required=True)
    ap.add_argument("--phenix_root", type=str, default="./09_Phenix_Autobuild_from_csv/")
    ap.add_argument("--out_dir", type=str, default="")

    ap.add_argument(
        "--only-suffixes",
        dest="only_suffixes",
        type=str,
        required=True,
        help='Comma-separated methods to analyze, e.g. "iREDO,K2_atten"',
    )
    ap.add_argument(
        "--only_suffixes",
        dest="only_suffixes",
        type=str,
        required=False,
        help="Alias for --only-suffixes",
    )

    ap.add_argument("--ref-method", type=str, default="iREDO")
    ap.add_argument("--outlier_percentage", type=float, default=5.0)

    # Tail extraction
    ap.add_argument("--low_tail_pct", type=float, default=1.0,
                    help="Percent (0-100] of lowest Free-R values to list per method (default 1.0).")
    ap.add_argument("--high_tail_pct", type=float, default=40.0,
                    help="Percent (0-100] of highest Free-R values to list per method (default 40.0).")
    ap.add_argument("--high_rfree_min", type=float, default=None,
                    help="Absolute cutoff for HIGH side: require Free-R >= this value (e.g., 40.0). Default: disabled.")

    # Plot limits
    ap.add_argument("--rfree_ymin", type=float, default=None)
    ap.add_argument("--rfree_ymax", type=float, default=None)
    ap.add_argument("--delta_ymin", type=float, default=None)
    ap.add_argument("--delta_ymax", type=float, default=None)

    # Relative residues placed plot limits (percent)
    ap.add_argument("--relplaced_ymin", type=float, default=0.0)
    ap.add_argument("--relplaced_ymax", type=float, default=140.0)

    # Relative residues placed denominator robustness
    ap.add_argument(
        "--min_ref_residues_placed",
        type=float,
        default=20.0,
        help="For relative residues_placed vs ref, require ref_method residues_placed >= this value. Default: 20.",
    )

    # Success definition (paired completeness, using ref for denominator)
    ap.add_argument(
        "--success_rfree_max",
        type=float,
        default=30.0,
        help="Success criterion: require method Free R <= this value (percent). Default: 30.",
    )
    ap.add_argument(
        "--success_relplaced_min",
        type=float,
        default=70.0,
        help="Success criterion: require relative residues placed vs ref >= this value (percent). Default: 80.",
    )

    # Scanning performance + progress
    ap.add_argument(
        "--max_workers",
        type=int,
        default=max(1, min(32, (os.cpu_count() or 4) + 4)),
        help="Thread workers used for scanning PDB dirs (I/O-heavy). Default: min(32, cpu+4).",
    )
    ap.add_argument(
        "--progress_every",
        type=int,
        default=50,
        help="Emit progress line every N completed PDB dirs per method. Default: 50.",
    )
    ap.add_argument(
        "--log_tail_bytes",
        type=int,
        default=65536,
        help="Bytes read from the end of AutoBuild logs when checking for citation marker. Default: 65536.",
    )

    # Logging levels
    ap.add_argument(
        "--console_log_level",
        type=str,
        default="INFO",
        help="Console log level: DEBUG, INFO, WARNING, ERROR, CRITICAL. Default: INFO.",
    )
    ap.add_argument(
        "--file_log_level",
        type=str,
        default="DEBUG",
        help="File log level: DEBUG, INFO, WARNING, ERROR, CRITICAL. Default: DEBUG.",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    base = Path(args.base_folder).expanduser().resolve()
    phenix_root = Path(args.phenix_root)
    if not phenix_root.is_absolute():
        phenix_root = (base / phenix_root).resolve()

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (base / "23_Autobuild_Analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    console_level = _parse_log_level(level_str=str(args.console_log_level))
    file_level = _parse_log_level(level_str=str(args.file_log_level))
    log_path = configure_logging(out_dir=out_dir, console_level=int(console_level), file_level=int(file_level))

    methods = [s.strip() for s in str(args.only_suffixes).split(",") if s.strip()]
    if not methods:
        raise SystemExit("[ERROR] --only-suffixes must contain at least one method name.")
    ref_method = str(args.ref_method).strip()

    logging.info("Run prefix   : %s", "23_analyze_autobuild_outcomes")
    logging.info("Command line : %s", " ".join(shlex.quote(arg) for arg in sys.argv))
    logging.info("base_folder  : %s", str(base))
    logging.info("phenix_root  : %s", str(phenix_root))
    logging.info("out_dir      : %s", str(out_dir))
    logging.info("methods      : %s", methods)
    logging.info("ref_method   : %s", str(ref_method))
    logging.info("scan         : max_workers=%d | progress_every=%d | log_tail_bytes=%d",
                 int(args.max_workers), int(args.progress_every), int(args.log_tail_bytes))
    logging.info("tails        : low_tail_pct=%.3f | high_tail_pct=%.3f | high_rfree_min=%s",
                 float(args.low_tail_pct), float(args.high_tail_pct),
                 "None" if args.high_rfree_min is None else f"{float(args.high_rfree_min):.3f}")
    logging.info("relplaced    : min_ref_residues_placed=%.1f", float(args.min_ref_residues_placed))
    logging.info("success      : rfree<=%.3f%% AND relplaced>=%.3f%% (vs %s; ref>=%.1f)",
                 float(args.success_rfree_max), float(args.success_relplaced_min),
                 str(ref_method), float(args.min_ref_residues_placed))

    triples = find_valid_overall_best_for_methods(
        phenix_root=phenix_root,
        methods=methods,
        max_workers=int(args.max_workers),
        progress_every=int(args.progress_every),
        tail_bytes=int(args.log_tail_bytes),
    )
    if not triples:
        logging.info("No valid AutoBuild_run_* results found for requested methods.")
        print(f"[INFO] Summary log written to: {log_path}")
        return

    rows_ab: List[Dict[str, object]] = []
    for method, pdb_id, sg, best_pdb, best_mtz, logp in triples:
        rfree = parse_free_r_from_pdb(pdb_path=best_pdb)
        best_cycle, built, placed = parse_best_cycle_and_residues(log_path=logp)
        n_refl_total = count_reflections_in_mtz(mtz_path=best_mtz)

        rows_ab.append({
            "PDB_ID": str(pdb_id),
            "Method": str(method),
            "SG": str(sg),
            "N_refl_total": float(n_refl_total) if n_refl_total is not None else np.nan,
            "Rfree_percent": float(rfree) if rfree is not None else np.nan,
            "Residues_built": float(built) if built is not None else np.nan,
            "Residues_placed": float(placed) if placed is not None else np.nan,
            "Best_cycle": float(best_cycle) if best_cycle is not None else np.nan,
            "overall_best_pdb": str(best_pdb),
            "overall_best_refine_data_mtz": str(best_mtz),
            "autobuild_log": str(logp),
        })

    df_ab = pd.DataFrame(rows_ab)
    df_ab["N_refl_total"] = pd.to_numeric(df_ab["N_refl_total"], errors="coerce")
    df_ab["Rfree_percent"] = pd.to_numeric(df_ab["Rfree_percent"], errors="coerce")
    df_ab["Residues_placed"] = pd.to_numeric(df_ab["Residues_placed"], errors="coerce")

    df_ab.to_csv(out_dir / "23_valid_jobs.csv", index=False)
    df_ab.to_csv(out_dir / "autobuild_rfree_raw.csv", index=False)

    logging.info("================================================================================")
    logging.info("SUMMARY: successful runs per method (valid AutoBuild_run_* runs)")
    logging.info("================================================================================")
    for m in methods:
        vals = df_ab.loc[df_ab["Method"] == m, "Rfree_percent"].to_numpy(dtype=float)
        stats = summarize_numeric(x=vals)
        logging.info("[Rfree] %s | %s", str(m), fmt_stats(stats=stats, unit="%", decimals=3))

    # Wide pivots for paired analyses
    wide_rfree = df_ab.pivot_table(index="PDB_ID", columns="Method", values="Rfree_percent", aggfunc="min")
    wide_placed = df_ab.pivot_table(index="PDB_ID", columns="Method", values="Residues_placed", aggfunc="max")

    if ref_method in wide_rfree.columns:
        logging.info("--------------------------------------------------------------------------------")
        logging.info("SUMMARY: ΔRfree vs reference method (%s) [paired by PDB_ID]", str(ref_method))
        logging.info("--------------------------------------------------------------------------------")
        for m in methods:
            if m == ref_method:
                continue
            if m not in wide_rfree.columns:
                logging.info("[ΔRfree] %s - %s | n=0 (method missing)", str(m), str(ref_method))
                continue
            sub = wide_rfree.dropna(subset=[ref_method, m]).copy()
            delta = (sub[m] - sub[ref_method]).to_numpy(dtype=float)
            stats = summarize_numeric(x=delta)
            logging.info("[ΔRfree] %s - %s | %s", str(m), str(ref_method), fmt_stats(stats=stats, unit="%", decimals=3))
    else:
        logging.warning("Reference method '%s' not present in valid results; delta summaries skipped.", str(ref_method))

    # Relative residues placed vs ref_method (%), with robust denominator filter
    relplaced_by_method: Dict[str, np.ndarray] = {}
    if ref_method in wide_placed.columns:
        logging.info("--------------------------------------------------------------------------------")
        logging.info("SUMMARY: Relative residues placed vs reference method (%s) [paired by PDB_ID]", str(ref_method))
        logging.info("--------------------------------------------------------------------------------")
        min_ref = float(args.min_ref_residues_placed)
        for m in methods:
            if m == ref_method:
                continue
            if m not in wide_placed.columns:
                logging.info("[RelPlaced] %s / %s | n=0 (method missing)", str(m), str(ref_method))
                continue

            sub = wide_placed.dropna(subset=[ref_method, m]).copy()
            sub[ref_method] = pd.to_numeric(sub[ref_method], errors="coerce")
            sub[m] = pd.to_numeric(sub[m], errors="coerce")
            sub = sub.dropna(subset=[ref_method, m]).copy()
            sub = sub.loc[sub[ref_method] >= min_ref].copy()
            sub = sub.loc[sub[m] >= 1.0].copy()

            rel_pct = (100.0 * (sub[m] / sub[ref_method])).to_numpy(dtype=float)
            relplaced_by_method[str(m)] = rel_pct

            # Also report >=80% and >=100% as useful diagnostics
            success80 = float(np.mean(rel_pct >= 80.0) * 100.0) if rel_pct.size else float("nan")
            success100 = float(np.mean(rel_pct >= 100.0) * 100.0) if rel_pct.size else float("nan")
            stats = summarize_numeric(x=rel_pct)
            logging.info(
                "[RelPlaced] %s / %s | %s | >=80%%: %.1f%% | >=100%%: %.1f%% | min_ref=%.1f",
                str(m), str(ref_method), fmt_stats(stats=stats, unit="%", decimals=3),
                success80, success100, min_ref
            )
    else:
        logging.warning("Reference method '%s' not present for residues_placed; relative-chain analysis skipped.", str(ref_method))

    # SUCCESS REPORT (paired subset): rfree<=thr AND relplaced>=thr
    # For each non-ref method, compute on paired-complete subset and with ref residues>=min_ref.
    if (ref_method in wide_rfree.columns) and (ref_method in wide_placed.columns):
        logging.info("--------------------------------------------------------------------------------")
        logging.info("SUCCESS REPORT (paired subset vs %s)", str(ref_method))
        logging.info("  Criteria: Free R <= %.3f%% AND rel_residues_placed >= %.3f%% (ref residues >= %.1f)",
                     float(args.success_rfree_max), float(args.success_relplaced_min), float(args.min_ref_residues_placed))
        logging.info("--------------------------------------------------------------------------------")
        rfree_thr = float(args.success_rfree_max)
        rel_thr = float(args.success_relplaced_min)
        min_ref = float(args.min_ref_residues_placed)

        for m in methods:
            if m == ref_method:
                continue
            if (m not in wide_rfree.columns) or (m not in wide_placed.columns):
                logging.info("[SUCCESS] %s | n_paired=0 (missing columns)", str(m))
                continue

            sub_r = wide_rfree.dropna(subset=[ref_method, m]).copy()
            sub_p = wide_placed.dropna(subset=[ref_method, m]).copy()

            # Intersect paired PDB_IDs where both rfree and placed are available for both methods
            common = sub_r.index.intersection(sub_p.index)
            if common.size == 0:
                logging.info("[SUCCESS] %s | n_paired=0 (no overlap)", str(m))
                continue

            rref = pd.to_numeric(wide_rfree.loc[common, ref_method], errors="coerce")
            rm = pd.to_numeric(wide_rfree.loc[common, m], errors="coerce")
            pref = pd.to_numeric(wide_placed.loc[common, ref_method], errors="coerce")
            pm = pd.to_numeric(wide_placed.loc[common, m], errors="coerce")

            df_pair = pd.DataFrame({
                "rfree_ref": rref,
                "rfree_m": rm,
                "placed_ref": pref,
                "placed_m": pm,
            }).dropna()

            # Robust denominator filter
            df_pair = df_pair.loc[df_pair["placed_ref"] >= min_ref].copy()
            df_pair = df_pair.loc[df_pair["placed_m"] >= 1.0].copy()

            n_paired = int(df_pair.shape[0])
            if n_paired == 0:
                logging.info("[SUCCESS] %s | n_paired=0 (after filters: ref>=%.1f)", str(m), min_ref)
                continue

            rel_pct = 100.0 * (df_pair["placed_m"] / df_pair["placed_ref"])
            ok = (df_pair["rfree_m"] <= rfree_thr) & (rel_pct >= rel_thr)

            n_ok = int(np.sum(ok.to_numpy(dtype=bool)))
            frac_ok = float(n_ok / n_paired) * 100.0

            # Also provide marginal pass rates (diagnostic)
            ok_r = (df_pair["rfree_m"] <= rfree_thr)
            ok_rel = (rel_pct >= rel_thr)
            n_ok_r = int(np.sum(ok_r.to_numpy(dtype=bool)))
            n_ok_rel = int(np.sum(ok_rel.to_numpy(dtype=bool)))

            logging.info(
                "[SUCCESS] %s | n_paired=%d | both=%d (%.1f%%) | rfree=%d (%.1f%%) | rel=%d (%.1f%%)",
                str(m),
                n_paired,
                n_ok, frac_ok,
                n_ok_r, float(n_ok_r / n_paired) * 100.0,
                n_ok_rel, float(n_ok_rel / n_paired) * 100.0,
            )
    else:
        logging.warning("SUCCESS REPORT skipped: need both rfree and residues_placed for ref_method '%s'.", str(ref_method))

    print(f"[INFO] Summary log written to: {log_path}")

    # Tail lists
    df_tails = build_rfree_tails_table(
        df_ab=df_ab,
        methods=methods,
        low_tail_pct=float(args.low_tail_pct),
        high_tail_pct=float(args.high_tail_pct),
        high_rfree_min=float(args.high_rfree_min) if args.high_rfree_min is not None else None,
    )
    tails_csv = out_dir / "23_rfree_tails.csv"
    df_tails.to_csv(tails_csv, index=False)
    logging.info("[CSV] Saved: %s (n=%d)", str(tails_csv), int(df_tails.shape[0]))

    # Long + wide metrics and outliers
    master_long_rows: List[Dict[str, object]] = []

    for _, r in df_ab.dropna(subset=["Rfree_percent"]).iterrows():
        master_long_rows.append({
            "PDB_ID": r["PDB_ID"],
            "Method": r["Method"],
            "SG": r["SG"],
            "Metric": "rfree_percent",
            "Value": float(r["Rfree_percent"]),
        })

    for _, r in df_ab.dropna(subset=["Residues_placed"]).iterrows():
        master_long_rows.append({
            "PDB_ID": r["PDB_ID"],
            "Method": r["Method"],
            "SG": r["SG"],
            "Metric": "residues_placed",
            "Value": float(r["Residues_placed"]),
        })

    # ΔFreeR vs ref
    if ref_method in wide_rfree.columns:
        for m in [c for c in wide_rfree.columns if c != ref_method]:
            sub = wide_rfree.dropna(subset=[ref_method, m]).copy()
            if sub.empty:
                continue
            delta = sub[m] - sub[ref_method]
            for pid, val in delta.items():
                sg_val = df_ab.loc[(df_ab["PDB_ID"] == str(pid)) & (df_ab["Method"] == str(m)), "SG"]
                sg_str = str(sg_val.iloc[0]) if not sg_val.empty else "UNKNOWN"
                master_long_rows.append({
                    "PDB_ID": str(pid),
                    "Method": str(m),
                    "SG": str(sg_str),
                    "Metric": f"delta_rfree_vs_{ref_method}",
                    "Value": float(val),
                })

    # Relative residues placed vs ref (%), with robust denominator filter
    if ref_method in wide_placed.columns:
        min_ref = float(args.min_ref_residues_placed)
        for m in [c for c in wide_placed.columns if c != ref_method]:
            sub = wide_placed.dropna(subset=[ref_method, m]).copy()
            sub[ref_method] = pd.to_numeric(sub[ref_method], errors="coerce")
            sub[m] = pd.to_numeric(sub[m], errors="coerce")
            sub = sub.dropna(subset=[ref_method, m]).copy()
            sub = sub.loc[sub[ref_method] >= min_ref].copy()
            sub = sub.loc[sub[m] >= 1.0].copy()
            if sub.empty:
                continue
            rel_pct = 100.0 * (sub[m] / sub[ref_method])
            for pid, val in rel_pct.items():
                sg_val = df_ab.loc[(df_ab["PDB_ID"] == str(pid)) & (df_ab["Method"] == str(m)), "SG"]
                sg_str = str(sg_val.iloc[0]) if not sg_val.empty else "UNKNOWN"
                master_long_rows.append({
                    "PDB_ID": str(pid),
                    "Method": str(m),
                    "SG": str(sg_str),
                    "Metric": f"rel_residues_placed_vs_{ref_method}",
                    "Value": float(val),
                })

    df_master_long = pd.DataFrame(master_long_rows).dropna(subset=["Value"])
    df_master_long = df_master_long.sort_values(by=["PDB_ID", "Method", "Metric"]).reset_index(drop=True)

    df_master_wide = _wide_from_long(
        df_long=df_master_long,
        index_col="PDB_ID",
        method_col="Method",
        metric_col="Metric",
        value_col="Value",
    ).sort_values(by="PDB_ID").reset_index(drop=True)

    all_csv = out_dir / "all_metrics_by_pdb_id.csv"
    df_master_wide.to_csv(all_csv, index=False)
    logging.info("[CSV] Saved: %s", str(all_csv))

    # Compact Basin Score calibration target table
    target_cols = ["PDB_ID", "K2_atten__rfree_percent", "K2_atten__rel_residues_placed_vs_iREDO"]
    target_df = df_master_wide.copy()
    for col in target_cols:
        if col not in target_df.columns:
            target_df[col] = np.nan
    target_df = target_df[target_cols].copy()
    target_csv = out_dir / "23_AutoBuild_basin_score_targets.csv"
    target_df.to_csv(target_csv, index=False)
    logging.info("[CSV] Saved: %s", str(target_csv))

    outliers_df = compute_outliers_long(
        df_long=df_master_long.rename(columns={"Method": "method", "Metric": "metric", "Value": "value", "PDB_ID": "pdb_id", "SG": "SG"}),
        outlier_percentage=float(args.outlier_percentage),
        index_col="pdb_id",
        method_col="method",
        metric_col="metric",
        value_col="value",
    )
    outliers_csv = out_dir / "outliers_long.csv"
    outliers_df.to_csv(outliers_csv, index=False)
    logging.info("[CSV] Saved: %s", str(outliers_csv))

    # Plots
    df_plot = df_ab.dropna(subset=["Rfree_percent"]).copy()
    if not df_plot.empty:
        df_plot = df_plot[df_plot["Method"].isin(methods)].rename(columns={"Method": "method", "Rfree_percent": "value"})
        df_plot["metric"] = "rfree"
        jitter_boxplot_multimetric(
            df_long=df_plot[["method", "metric", "value"]],
            method_col="method",
            metric_col="metric",
            value_col="value",
            metric_order=["rfree"],
            metric_labels={"rfree": "Free R"},
            unit_suffix_by_metric={"rfree": "%"},
            decimals_by_metric={"rfree": PLOT_DECIMALS},
            title="AutoBuild — Free R (%) by method (valid AutoBuild_run_* runs)",
            ylabel="Free R (%)",
            xlabel="Method",
            out_png=out_dir / "autobuild_rfree_by_method.png",
            rng_seed=42,
            y_min=args.rfree_ymin,
            y_max=args.rfree_ymax,
        )

    df_plot = df_ab.dropna(subset=["Residues_placed"]).copy()
    if not df_plot.empty:
        df_plot = df_plot[df_plot["Method"].isin(methods)].rename(columns={"Method": "method", "Residues_placed": "value"})
        df_plot["metric"] = "placed"
        jitter_boxplot_multimetric(
            df_long=df_plot[["method", "metric", "value"]],
            method_col="method",
            metric_col="metric",
            value_col="value",
            metric_order=["placed"],
            metric_labels={"placed": "Placed"},
            unit_suffix_by_metric={"placed": "res"},
            decimals_by_metric={"placed": PLOT_DECIMALS},
            title="AutoBuild — Residues placed (best cycle) by method (valid AutoBuild_run_* runs)",
            ylabel="Residues placed",
            xlabel="Method",
            out_png=out_dir / "autobuild_residues_placed_by_method.png",
            rng_seed=42,
        )

    if ref_method in wide_rfree.columns:
        delta_rows = []
        for m in [c for c in wide_rfree.columns if c != ref_method and c in methods]:
            sub = wide_rfree.dropna(subset=[ref_method, m]).copy()
            if sub.empty:
                continue
            delta = sub[m] - sub[ref_method]
            for _, val in delta.items():
                delta_rows.append({"method": str(m), "metric": "delta", "value": float(val)})

        df_delta = pd.DataFrame(delta_rows)
        if not df_delta.empty:
            jitter_boxplot_multimetric(
                df_long=df_delta,
                method_col="method",
                metric_col="metric",
                value_col="value",
                metric_order=["delta"],
                metric_labels={"delta": f"ΔFree R vs {ref_method}"},
                unit_suffix_by_metric={"delta": "%"},
                decimals_by_metric={"delta": PLOT_DECIMALS},
                title=f"AutoBuild — ΔFree R vs {ref_method} by method (paired PDB_ID)",
                ylabel=f"ΔFree R vs {ref_method} (%)",
                xlabel="Method",
                out_png=out_dir / f"autobuild_delta_vs_{ref_method}_by_method.png",
                rng_seed=42,
                y_min=args.delta_ymin,
                y_max=args.delta_ymax,
            )

    # Relative residues placed vs ref (%), jitter + box + annotated stats
    if ref_method in wide_placed.columns:
        rel_rows = []
        min_ref = float(args.min_ref_residues_placed)
        for m in [c for c in wide_placed.columns if c != ref_method and c in methods]:
            sub = wide_placed.dropna(subset=[ref_method, m]).copy()
            sub[ref_method] = pd.to_numeric(sub[ref_method], errors="coerce")
            sub[m] = pd.to_numeric(sub[m], errors="coerce")
            sub = sub.dropna(subset=[ref_method, m]).copy()
            sub = sub.loc[sub[ref_method] >= min_ref].copy()
            sub = sub.loc[sub[m] >= 1.0].copy()
            if sub.empty:
                continue
            rel_pct = 100.0 * (sub[m] / sub[ref_method])
            for _, val in rel_pct.items():
                rel_rows.append({"method": str(m), "metric": "relplaced", "value": float(val)})

        df_rel = pd.DataFrame(rel_rows)
        if not df_rel.empty:
            jitter_boxplot_multimetric(
                df_long=df_rel,
                method_col="method",
                metric_col="metric",
                value_col="value",
                metric_order=["relplaced"],
                metric_labels={"relplaced": f"Rel. residues placed vs {ref_method}"},
                unit_suffix_by_metric={"relplaced": "%"},
                decimals_by_metric={"relplaced": PLOT_DECIMALS},
                title=f"AutoBuild — Relative residues placed vs {ref_method} (paired PDB_ID; ref>= {min_ref:.0f})",
                ylabel=f"Relative residues placed vs {ref_method} (%)",
                xlabel="Method",
                out_png=out_dir / f"autobuild_relative_residues_placed_vs_{ref_method}.png",
                rng_seed=42,
                y_min=float(args.relplaced_ymin) if args.relplaced_ymin is not None else None,
                y_max=float(args.relplaced_ymax) if args.relplaced_ymax is not None else None,
            )

    logging.info("[DONE] Stage-23 AutoBuild outcome analysis complete.")


if __name__ == "__main__":
    main()