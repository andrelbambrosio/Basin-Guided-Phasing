#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
43_analyze_autobuild_degradation_results.py

Direct analysis of Phenix AutoBuild runs produced from degradation-state folders, including paired anti-phase runs from the latest PC33 scheduler.

Primary use case:
  <degradation_root>/<source_pdb>/<mode>/mode_<mode>__frac_<fraction>__round_<round>/
  <degradation_root>/<source_pdb>/<mode>/mode_<mode>__frac_<fraction>__round_<round>__anti/

Example:
  python PC34_autobuild_rfree_analysis__v04.py \
    --degradation_root /media/alba/ALPHA_ALBA/Attenuated_Signed_Amplitudes-22Mar2026/14_Run_AutoBuild_from_Degradation \
    --source_pdb 2bkf \
    --modes uniform_random

Outputs (default under <degradation_root>/PC34_Degradation_Analysis/):
  - pc34_degradation_valid_jobs.csv
  - pc34_degradation_summary_by_fraction.csv
  - pc34_degradation_summary_by_fraction_wide.csv
  - pc34_degradation_seed_vs_anti_by_fraction.csv
  - degradation_rfree_by_fraction__<mode>.png
  - degradation_residues_placed_by_fraction__<mode>.png
  - degradation_success_fraction_by_fraction__<mode>.png
  - degradation_rfree_jitter_box__<mode>.png
  - degradation_residues_placed_jitter_box__<mode>.png

A run is considered valid only if:
  - some AutoBuild_run_*/overall_best.pdb exists
  - some AutoBuild_run_*/overall_best_refine_data.mtz exists
  - a corresponding log tail contains "Citations for AutoBuild:"
"""

from __future__ import annotations

import argparse
import csv
import logging
import shlex
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gemmi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MAJOR_TICK_SIZE: int = 10
MINOR_TICK_SIZE: int = 5
FS_NUMBERS: int = 14
FS_AXES: int = 14
FS_TITLE: int = 14
FS_LEGEND: int = 12


RUN_PREFIX = "43_analyze_autobuild_degradation_results"



PLOT_DECIMALS: int = 1

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


def format_stats_label(*, label: str, y: np.ndarray, unit_suffix: str, decimals: int) -> str:
    n, mean, sd, med = summarize_array(y=y)
    if n == 0:
        return f"{label}\n n=0\n μ±σ=NA\n med=NA"
    fmt = f"{{:.{int(decimals)}f}}"
    suf = f" {unit_suffix}".rstrip()
    return (
        f"{label}\n"
        f" n={n}\n"
        f" μ±σ={fmt.format(mean)}±{fmt.format(sd)}{suf}\n"
        f" med={fmt.format(med)}{suf}"
    )


def make_plot_subdir(*, out_dir: Path, src_pdb: str, degr_mode: str, phase_variant: str) -> Path:
    subdir = out_dir / sanitize_token(str(src_pdb)) / sanitize_token(str(degr_mode)) / sanitize_token(str(phase_variant))
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def sanitize_token(value: str) -> str:
    s = str(value).strip()
    s = re.sub(r"[^0-9A-Za-z_.-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    s = s.strip("._")
    return s if s else "unknown"


_CITATION_NEEDLE: bytes = b"Citations for AutoBuild:"
STATE_RE = re.compile(
    r"^mode_(?P<mode>.+?)__frac_(?P<fraction>[0-9]+(?:\.[0-9]+)?)__round_(?P<round>[0-9]+?)(?P<anti>__anti)?$"
)

FREE_R_PATTERNS = [
    re.compile(r"^REMARK\s+3\s+FREE\s+R\s+VALUE\s*:?\s*([0-9]*\.?[0-9]+)\s*%?\s*$"),
    re.compile(r"^REMARK\s+3\s+R\s+FREE\s*:?\s*([0-9]*\.?[0-9]+)\s*%?\s*$"),
]
BEST_CYCLE_RE = re.compile(r"Best solution on cycle:\s*(\d+)\b")
SOLUTION_HEADER_RE = re.compile(r"^\s*SOLUTION\s+CYCLE", re.IGNORECASE)
SOLUTION_LINE_RE = re.compile(r"^\s*\d+\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s+(\d+)\s*$")


def _parse_log_level(level_str: str) -> int:
    s = str(level_str).strip().upper()
    lvl = getattr(logging, s, None)
    if lvl is None:
        raise ValueError(f"Invalid log level: {level_str}")
    return int(lvl)


def configure_logging(out_dir: Path, console_level: int, file_level: int, run_prefix: str = RUN_PREFIX) -> Path:
    logs_dir = out_dir.parent / "LOGS"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{run_prefix}__{ts}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
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


def _log_tail_has_citations(log_path: Path, tail_bytes: int = 65536) -> bool:
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            start = max(0, size - int(tail_bytes))
            fh.seek(start, os.SEEK_SET)
            chunk = fh.read(int(tail_bytes))
        return _CITATION_NEEDLE in chunk
    except Exception:
        return False


def parse_free_r_from_pdb(pdb_path: Path) -> Optional[float]:
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fh:
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


def parse_best_cycle_and_residues(log_path: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
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


def count_reflections_in_mtz(mtz_path: Path) -> Optional[int]:
    try:
        mtz = gemmi.read_mtz_file(str(mtz_path))
        return int(mtz.nreflections)
    except Exception:
        return None


def read_spacegroup_from_mtz(mtz_path: Path) -> str:
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


def _find_best_outputs_in_state_dir(state_dir: Path, tail_bytes: int) -> Optional[Tuple[Path, Path, Path]]:
    candidates: List[Tuple[float, Path, Path, Path]] = []

    run_dirs = [p for p in state_dir.glob("AutoBuild_run_*") if p.is_dir()]
    if not run_dirs:
        return None

    job_level_logs = []
    job_level_logs.extend(state_dir.glob("AutoBuild_run_*.log"))
    job_level_logs.extend(state_dir.glob("AutoBuild_run_*.LOG"))
    job_level_logs = [lp for lp in job_level_logs if lp.is_file() and lp.stat().st_size > 0]

    for run_dir in sorted(run_dirs):
        best_pdb = run_dir / "overall_best.pdb"
        best_mtz = run_dir / "overall_best_refine_data.mtz"
        if not (best_pdb.is_file() and best_pdb.stat().st_size > 0):
            continue
        if not (best_mtz.is_file() and best_mtz.stat().st_size > 0):
            continue

        logs: List[Path] = []
        logs.extend(run_dir.glob("AutoBuild_run_*.log"))
        logs.extend(run_dir.glob("AutoBuild_run_*.LOG"))
        logs = [lp for lp in logs if lp.is_file() and lp.stat().st_size > 0]
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


def _extract_input_mtz_from_state_dir(state_dir: Path) -> Optional[Path]:
    mtzs = sorted(state_dir.glob("*_PHIB_input.mtz"))
    if len(mtzs) >= 1:
        return mtzs[0]
    return None


def parse_state_metadata(state_dir: Path) -> Optional[Dict[str, object]]:
    m = STATE_RE.match(state_dir.name)
    if m is None:
        return None
    anti_suffix = m.group("anti")
    phase_variant = "anti" if anti_suffix else "seed"
    paired_state_name = re.sub(r"__anti$", "", str(state_dir.name))
    return {
        "Degradation_mode": str(m.group("mode")),
        "Degradation_fraction": float(m.group("fraction")),
        "Degradation_round": int(m.group("round")),
        "Phase_variant": str(phase_variant),
        "State_name": str(state_dir.name),
        "Paired_state_name": str(paired_state_name),
    }


def discover_degradation_state_dirs(
    degradation_root: Path,
    source_pdb: Optional[str],
    modes: List[str],
) -> List[Tuple[str, str, Path]]:
    """
    Returns list of:
      (source_pdb, mode, state_dir)
    """
    out: List[Tuple[str, str, Path]] = []

    source_dirs = [p for p in degradation_root.iterdir() if p.is_dir()]
    if source_pdb is not None:
        source_dirs = [p for p in source_dirs if p.name.lower() == str(source_pdb).lower()]

    for src_dir in sorted(source_dirs):
        mode_dirs = [p for p in src_dir.iterdir() if p.is_dir()]
        if modes:
            wanted = {m.lower() for m in modes}
            mode_dirs = [p for p in mode_dirs if p.name.lower() in wanted]

        for mode_dir in sorted(mode_dirs):
            state_dirs = [p for p in mode_dir.iterdir() if p.is_dir()]
            for state_dir in sorted(state_dirs):
                meta = parse_state_metadata(state_dir=state_dir)
                if meta is None:
                    continue
                out.append((str(src_dir.name), str(mode_dir.name), state_dir))

    return out


def scan_degradation_runs(
    degradation_root: Path,
    source_pdb: Optional[str],
    modes: List[str],
    tail_bytes: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    discovered = discover_degradation_state_dirs(
        degradation_root=degradation_root,
        source_pdb=source_pdb,
        modes=modes,
    )
    logging.info("[SCAN] discovered state dirs: %d", int(len(discovered)))

    done = 0
    for src_pdb, mode_dir_name, state_dir in discovered:
        done += 1
        if (done == 1) or (done % 100 == 0) or (done == len(discovered)):
            logging.info("[SCAN] %d/%d state dirs done", done, len(discovered))

        meta = parse_state_metadata(state_dir=state_dir)
        if meta is None:
            continue

        best = _find_best_outputs_in_state_dir(state_dir=state_dir, tail_bytes=int(tail_bytes))
        if best is None:
            continue
        best_pdb, best_mtz, log_path = best

        input_mtz = _extract_input_mtz_from_state_dir(state_dir=state_dir)
        sg = read_spacegroup_from_mtz(input_mtz) if input_mtz is not None else "UNKNOWN"
        n_input_refl = count_reflections_in_mtz(input_mtz) if input_mtz is not None else None
        n_output_refl = count_reflections_in_mtz(best_mtz)

        rfree = parse_free_r_from_pdb(best_pdb)
        best_cycle, built, placed = parse_best_cycle_and_residues(log_path)

        row = {
            "Source_PDB_ID": str(src_pdb),
            "Method": str(mode_dir_name),
            "Phase_variant": str(meta["Phase_variant"]),
            "State_name": str(meta["State_name"]),
            "Paired_state_name": str(meta["Paired_state_name"]),
            "Is_degradation": True,
            "Degradation_mode": str(meta["Degradation_mode"]),
            "Degradation_fraction": float(meta["Degradation_fraction"]),
            "Degradation_round": int(meta["Degradation_round"]),
            "SG": str(sg),
            "N_refl_input": float(n_input_refl) if n_input_refl is not None else np.nan,
            "N_refl_total": float(n_output_refl) if n_output_refl is not None else np.nan,
            "Rfree_percent": float(rfree) if rfree is not None else np.nan,
            "Residues_built": float(built) if built is not None else np.nan,
            "Residues_placed": float(placed) if placed is not None else np.nan,
            "Best_cycle": float(best_cycle) if best_cycle is not None else np.nan,
            "workdir": str(state_dir),
            "input_mtz": str(input_mtz) if input_mtz is not None else "",
            "overall_best_pdb": str(best_pdb),
            "overall_best_refine_data_mtz": str(best_mtz),
            "autobuild_log": str(log_path),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Degradation_fraction"] = pd.to_numeric(df["Degradation_fraction"], errors="coerce")
        df["Degradation_round"] = pd.to_numeric(df["Degradation_round"], errors="coerce")
        df["Rfree_percent"] = pd.to_numeric(df["Rfree_percent"], errors="coerce")
        df["Residues_placed"] = pd.to_numeric(df["Residues_placed"], errors="coerce")
        df["Best_cycle"] = pd.to_numeric(df["Best_cycle"], errors="coerce")
        df["N_refl_input"] = pd.to_numeric(df["N_refl_input"], errors="coerce")
        df["N_refl_total"] = pd.to_numeric(df["N_refl_total"], errors="coerce")
        df = df.sort_values(
            by=["Source_PDB_ID", "Method", "Phase_variant", "Degradation_fraction", "Degradation_round"],
            ascending=[True, True, True, True, True],
        ).reset_index(drop=True)
    return df


def agg_stats(values: np.ndarray) -> Dict[str, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    n = int(v.size)
    if n == 0:
        return {
            "n": 0, "mean": np.nan, "std": np.nan, "sem": np.nan,
            "ci95_low": np.nan, "ci95_high": np.nan,
            "median": np.nan, "q25": np.nan, "q75": np.nan,
        }
    mean = float(np.mean(v))
    std = float(np.std(v, ddof=1)) if n >= 2 else 0.0
    sem = float(std / math.sqrt(n)) if n >= 2 else 0.0
    half = float(1.96 * sem) if n >= 2 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "median": float(np.median(v)),
        "q25": float(np.quantile(v, 0.25)),
        "q75": float(np.quantile(v, 0.75)),
    }


def build_degradation_summary_by_fraction(
    df_valid: pd.DataFrame,
    success_rfree_max: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if df_valid.empty:
        return pd.DataFrame()

    group_cols = ["Source_PDB_ID", "Method", "Phase_variant", "Degradation_mode", "Degradation_fraction"]
    for keys, sub in df_valid.groupby(group_cols, dropna=False):
        src_pdb, method, phase_variant, degr_mode, frac = keys
        row: Dict[str, object] = {
            "Source_PDB_ID": str(src_pdb),
            "Method": str(method),
            "Phase_variant": str(phase_variant),
            "Degradation_mode": str(degr_mode),
            "Degradation_fraction": float(frac),
            "n_runs": int(sub.shape[0]),
            "n_unique_rounds": int(sub["Degradation_round"].nunique()) if "Degradation_round" in sub.columns else np.nan,
        }

        for metric in ["Rfree_percent", "Residues_placed", "Best_cycle", "N_refl_total"]:
            st = agg_stats(sub[metric].to_numpy(dtype=float))
            for k, v in st.items():
                row[f"{metric}__{k}"] = v

        rfree = pd.to_numeric(sub["Rfree_percent"], errors="coerce")
        good = rfree.notna()
        success = good & (rfree <= float(success_rfree_max))
        row["success_rfree_max"] = float(success_rfree_max)
        row["success_n"] = int(success.sum())
        row["success_frac_percent"] = float(success.mean() * 100.0) if int(good.sum()) > 0 else np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            by=["Source_PDB_ID", "Method", "Phase_variant", "Degradation_mode", "Degradation_fraction"],
            ascending=[True, True, True, True, True],
        ).reset_index(drop=True)
    return out


def build_degradation_summary_wide(df_summary: pd.DataFrame) -> pd.DataFrame:
    if df_summary.empty:
        return pd.DataFrame()

    metrics = [
        "Rfree_percent__mean",
        "Rfree_percent__ci95_low",
        "Rfree_percent__ci95_high",
        "Residues_placed__mean",
        "Residues_placed__ci95_low",
        "Residues_placed__ci95_high",
        "success_frac_percent",
        "n_runs",
    ]
    rows: List[Dict[str, object]] = []
    id_cols = ["Source_PDB_ID", "Method", "Phase_variant", "Degradation_mode"]

    for keys, sub in df_summary.groupby(id_cols, dropna=False):
        src_pdb, method, phase_variant, degr_mode = keys
        for metric in metrics:
            row: Dict[str, object] = {
                "Source_PDB_ID": str(src_pdb),
                "Method": str(method),
                "Phase_variant": str(phase_variant),
                "Degradation_mode": str(degr_mode),
                "Metric": str(metric),
            }
            for _, r in sub.sort_values("Degradation_fraction").iterrows():
                frac = float(r["Degradation_fraction"])
                row[f"frac_{frac:0.4f}"] = r.get(metric, np.nan)
            rows.append(row)

    out = pd.DataFrame(rows)
    return out


def _style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel=xlabel, fontsize=FS_AXES)
    ax.set_ylabel(ylabel=ylabel, fontsize=FS_AXES)
    ax.set_title(label=title, fontsize=FS_TITLE)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="--", color="gray", alpha=0.7)
    ax.grid(which="minor", linestyle=":", color="lightgray", alpha=0.8)
    ax.tick_params(axis="both", which="major", labelsize=FS_NUMBERS, length=MAJOR_TICK_SIZE)
    ax.tick_params(axis="both", which="minor", length=MINOR_TICK_SIZE)


def plot_metric_by_fraction(
    df_summary: pd.DataFrame,
    metric_prefix: str,
    ylabel: str,
    out_png: Path,
    title: str,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> None:
    if df_summary.empty:
        logging.info("[PLOT] No data to plot for %s", str(out_png))
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    x = pd.to_numeric(df_summary["Degradation_fraction"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df_summary[f"{metric_prefix}__mean"], errors="coerce").to_numpy(dtype=float)
    ylo = pd.to_numeric(df_summary[f"{metric_prefix}__ci95_low"], errors="coerce").to_numpy(dtype=float)
    yhi = pd.to_numeric(df_summary[f"{metric_prefix}__ci95_high"], errors="coerce").to_numpy(dtype=float)

    ax.plot(x, y, marker="o", linewidth=1.8)
    if np.any(np.isfinite(ylo) & np.isfinite(yhi)):
        ax.fill_between(x, ylo, yhi, alpha=0.20)

    _style_axes(ax=ax, xlabel="Degradation fraction", ylabel=ylabel, title=title)
    if (y_min is not None) or (y_max is not None):
        ax.set_ylim(
            bottom=float(y_min) if y_min is not None else None,
            top=float(y_max) if y_max is not None else None,
        )

    fig.tight_layout()
    fig.savefig(str(out_png), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("[PLOT] Saved: %s", str(out_png))


def plot_success_fraction_by_fraction(
    df_summary: pd.DataFrame,
    out_png: Path,
    title: str,
) -> None:
    if df_summary.empty:
        logging.info("[PLOT] No data to plot for %s", str(out_png))
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    x = pd.to_numeric(df_summary["Degradation_fraction"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df_summary["success_frac_percent"], errors="coerce").to_numpy(dtype=float)

    ax.plot(x, y, marker="o", linewidth=1.8)
    _style_axes(ax=ax, xlabel="Degradation fraction", ylabel="Success fraction (%)", title=title)
    ax.set_ylim(bottom=0.0, top=100.0)

    fig.tight_layout()
    fig.savefig(str(out_png), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("[PLOT] Saved: %s", str(out_png))


def plot_jitter_box_by_fraction(
    df_valid: pd.DataFrame,
    value_col: str,
    ylabel: str,
    out_png: Path,
    title: str,
    unit_suffix: str = "",
    decimals: int = PLOT_DECIMALS,
) -> None:
    if df_valid.empty:
        logging.info("[PLOT] No data to plot for %s", str(out_png))
        return

    fracs = sorted(df_valid["Degradation_fraction"].dropna().unique().tolist())
    data: List[np.ndarray] = []
    positions: List[float] = []
    labels: List[str] = []

    for i, frac in enumerate(fracs):
        vals = pd.to_numeric(
            df_valid.loc[df_valid["Degradation_fraction"] == frac, value_col],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        data.append(vals)
        positions.append(float(i + 1))
        frac_label = f"{float(frac):0.2f}"
        labels.append(
            format_stats_label(
                label=frac_label,
                y=vals,
                unit_suffix=str(unit_suffix),
                decimals=int(decimals),
            )
        )

    if len(data) == 0:
        logging.info("[PLOT] No finite data to plot for %s", str(out_png))
        return

    fig, ax = plt.subplots(figsize=(max(10, 1.15 * len(data)), 7), dpi=150)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=True,
        medianprops={"linewidth": 1.5},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )
    for patch in bp["boxes"]:
        patch.set_alpha(0.25)

    rng = np.random.default_rng(seed=1234)
    for pos, vals in zip(positions, data):
        x = pos + rng.uniform(low=-0.10, high=0.10, size=vals.size)
        ax.scatter(x, vals, s=20, alpha=0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=FS_NUMBERS)
    _style_axes(ax=ax, xlabel="Degradation fraction", ylabel=ylabel, title=title)

    fig.tight_layout()
    fig.savefig(str(out_png), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logging.info("[PLOT] Saved: %s", str(out_png))





def print_degradation_statistics_to_terminal(*, df_summary: pd.DataFrame, df_pair: pd.DataFrame) -> None:
    if df_summary is None or df_summary.empty:
        logging.info("[STATS] No degradation summary rows to print.")
        return

    logging.info("================================================================================")
    logging.info("DEGRADATION SUMMARY STATISTICS BY MODE AND PHASE VARIANT")
    logging.info("================================================================================")

    group_cols = ["Source_PDB_ID", "Method", "Phase_variant", "Degradation_mode"]
    for keys, sub in df_summary.groupby(group_cols, dropna=False):
        src_pdb, method, phase_variant, degr_mode = keys
        logging.info("[GROUP] source_pdb=%s | method=%s | phase_variant=%s | degradation_mode=%s",
                     str(src_pdb), str(method), str(phase_variant), str(degr_mode))
        sub = sub.sort_values("Degradation_fraction").reset_index(drop=True)

        for _, r in sub.iterrows():
            logging.info(
                "  frac=%.4f | n=%d | FreeR mean±sd=%.3f±%.3f %% | median=%.3f %% | "
                "Placed mean±sd=%.3f±%.3f | median=%.3f | Success=%.1f%%",
                float(r.get("Degradation_fraction", float("nan"))),
                int(r.get("n_runs", 0)),
                float(r.get("Rfree_percent__mean", float("nan"))),
                float(r.get("Rfree_percent__std", float("nan"))),
                float(r.get("Rfree_percent__median", float("nan"))),
                float(r.get("Residues_placed__mean", float("nan"))),
                float(r.get("Residues_placed__std", float("nan"))),
                float(r.get("Residues_placed__median", float("nan"))),
                float(r.get("success_frac_percent", float("nan"))),
            )

    if df_pair is not None and (not df_pair.empty):
        logging.info("================================================================================")
        logging.info("SEED VS ANTI SUMMARY BY DEGRADATION FRACTION")
        logging.info("================================================================================")
        pair_group_cols = ["Source_PDB_ID", "Method", "Degradation_mode", "Degradation_fraction"]
        for keys, sub in df_pair.groupby(pair_group_cols, dropna=False):
            src_pdb, method, degr_mode, frac = keys
            d_r = pd.to_numeric(sub["delta_rfree_anti_minus_seed"], errors="coerce").dropna().to_numpy(dtype=float)
            d_p = pd.to_numeric(sub["delta_residues_placed_anti_minus_seed"], errors="coerce").dropna().to_numpy(dtype=float)

            if d_r.size > 0:
                n_r, mean_r, sd_r, med_r = summarize_array(y=d_r)
            else:
                n_r, mean_r, sd_r, med_r = 0, float("nan"), float("nan"), float("nan")

            if d_p.size > 0:
                n_p, mean_p, sd_p, med_p = summarize_array(y=d_p)
            else:
                n_p, mean_p, sd_p, med_p = 0, float("nan"), float("nan"), float("nan")

            logging.info(
                "[PAIR] source_pdb=%s | method=%s | degradation_mode=%s | frac=%.4f | "
                "ΔRfree anti-seed: n=%d mean±sd=%.3f±%.3f %% median=%.3f %% | "
                "ΔPlaced anti-seed: n=%d mean±sd=%.3f±%.3f median=%.3f",
                str(src_pdb), str(method), str(degr_mode), float(frac),
                int(n_r), float(mean_r), float(sd_r), float(med_r),
                int(n_p), float(mean_p), float(sd_p), float(med_p),
            )

def build_seed_vs_anti_summary(df_valid: pd.DataFrame) -> pd.DataFrame:
    if df_valid.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    group_cols = ["Source_PDB_ID", "Method", "Degradation_mode", "Degradation_fraction", "Paired_state_name"]
    for keys, sub in df_valid.groupby(group_cols, dropna=False):
        src_pdb, method, degr_mode, frac, paired_state = keys
        seed = sub.loc[sub["Phase_variant"] == "seed"]
        anti = sub.loc[sub["Phase_variant"] == "anti"]
        row: Dict[str, object] = {
            "Source_PDB_ID": str(src_pdb),
            "Method": str(method),
            "Degradation_mode": str(degr_mode),
            "Degradation_fraction": float(frac),
            "Paired_state_name": str(paired_state),
            "has_seed": bool(seed.shape[0] > 0),
            "has_anti": bool(anti.shape[0] > 0),
        }
        for prefix, frame in [("seed", seed), ("anti", anti)]:
            if frame.shape[0] > 0:
                rr = pd.to_numeric(frame["Rfree_percent"], errors="coerce").dropna().to_numpy(dtype=float)
                rp = pd.to_numeric(frame["Residues_placed"], errors="coerce").dropna().to_numpy(dtype=float)
                row[f"{prefix}_rfree"] = float(rr[0]) if rr.size else np.nan
                row[f"{prefix}_residues_placed"] = float(rp[0]) if rp.size else np.nan
            else:
                row[f"{prefix}_rfree"] = np.nan
                row[f"{prefix}_residues_placed"] = np.nan
        if np.isfinite(row["seed_rfree"]) and np.isfinite(row["anti_rfree"]):
            row["delta_rfree_anti_minus_seed"] = float(row["anti_rfree"] - row["seed_rfree"])
        else:
            row["delta_rfree_anti_minus_seed"] = np.nan
        if np.isfinite(row["seed_residues_placed"]) and np.isfinite(row["anti_residues_placed"]):
            row["delta_residues_placed_anti_minus_seed"] = float(row["anti_residues_placed"] - row["seed_residues_placed"])
        else:
            row["delta_residues_placed_anti_minus_seed"] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["Source_PDB_ID","Method","Degradation_mode","Degradation_fraction","Paired_state_name"]).reset_index(drop=True)
    return out
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage 43: analyze AutoBuild results from degraded and anti-phase MTZ states.")
    ap.add_argument("--degradation_root", type=str, required=True,
                    help="Root folder containing <source_pdb>/<mode>/mode_<mode>__frac_<f>__round_<r>/")
    ap.add_argument("--source_pdb", type=str, default=None,
                    help="Optional source structure to restrict, e.g. 2bkf")
    ap.add_argument("--modes", type=str, default="",
                    help='Optional comma-separated degradation modes to restrict, e.g. "uniform_random"')
    ap.add_argument("--out_dir", type=str, default="",
                    help="Output folder. Default: <degradation_root>/43_AutoBuild_Degradation_Analysis")
    ap.add_argument("--success_rfree_max", type=float, default=30.0,
                    help="Success threshold for pooled summary fraction. Default 30.0")
    ap.add_argument("--log_tail_bytes", type=int, default=65536)
    ap.add_argument("--console_log_level", type=str, default="INFO")
    ap.add_argument("--file_log_level", type=str, default="DEBUG")
    ap.add_argument("--rfree_ymin", type=float, default=None)
    ap.add_argument("--rfree_ymax", type=float, default=None)
    ap.add_argument("--placed_ymin", type=float, default=None)
    ap.add_argument("--placed_ymax", type=float, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    degradation_root = Path(args.degradation_root).expanduser().resolve()
    if not degradation_root.is_dir():
        raise SystemExit(f"[ERROR] degradation_root not found: {degradation_root}")

    out_dir = Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else (degradation_root / "43_AutoBuild_Degradation_Analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    console_level = _parse_log_level(str(args.console_log_level))
    file_level = _parse_log_level(str(args.file_log_level))
    log_path = configure_logging(out_dir=out_dir, console_level=console_level, file_level=file_level)

    logging.info("Run prefix         : %s", RUN_PREFIX)
    logging.info("Command line       : %s", " ".join(shlex.quote(arg) for arg in sys.argv))
    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    logging.info("degradation_root  : %s", str(degradation_root))
    logging.info("out_dir           : %s", str(out_dir))
    logging.info("source_pdb        : %s", str(args.source_pdb) if args.source_pdb else "ALL")
    logging.info("modes             : %s", modes if modes else "ALL")
    logging.info("success_rfree_max : %.2f", float(args.success_rfree_max))

    df_valid = scan_degradation_runs(
        degradation_root=degradation_root,
        source_pdb=str(args.source_pdb) if args.source_pdb else None,
        modes=modes,
        tail_bytes=int(args.log_tail_bytes),
    )
    if df_valid.empty:
        logging.info("No valid degradation AutoBuild runs found.")
        print(f"[INFO] Summary log written to: {log_path}")
        return

    raw_csv = out_dir / "pc34_degradation_valid_jobs.csv"
    df_valid.to_csv(raw_csv, index=False)
    logging.info("[CSV] Saved: %s (n=%d)", str(raw_csv), int(df_valid.shape[0]))

    df_summary = build_degradation_summary_by_fraction(
        df_valid=df_valid,
        success_rfree_max=float(args.success_rfree_max),
    )
    summary_csv = out_dir / "pc34_degradation_summary_by_fraction.csv"
    df_summary.to_csv(summary_csv, index=False)
    logging.info("[CSV] Saved: %s (n=%d)", str(summary_csv), int(df_summary.shape[0]))

    df_summary_wide = build_degradation_summary_wide(df_summary=df_summary)
    summary_wide_csv = out_dir / "pc34_degradation_summary_by_fraction_wide.csv"
    df_summary_wide.to_csv(summary_wide_csv, index=False)
    logging.info("[CSV] Saved: %s (n=%d)", str(summary_wide_csv), int(df_summary_wide.shape[0]))

    df_pair = build_seed_vs_anti_summary(df_valid=df_valid)
    pair_csv = out_dir / "pc34_degradation_seed_vs_anti_by_fraction.csv"
    df_pair.to_csv(pair_csv, index=False)
    logging.info("[CSV] Saved: %s (n=%d)", str(pair_csv), int(df_pair.shape[0]))

    for (src_pdb, method, phase_variant, degr_mode), sub_sum in df_summary.groupby(["Source_PDB_ID", "Method", "Phase_variant", "Degradation_mode"], dropna=False):
        prefix = f"{src_pdb} | {degr_mode} | {phase_variant}"
        safe = f"{src_pdb}__{degr_mode}__{phase_variant}"
        plot_subdir = make_plot_subdir(out_dir=out_dir, src_pdb=str(src_pdb), degr_mode=str(degr_mode), phase_variant=str(phase_variant))

        plot_metric_by_fraction(
            df_summary=sub_sum.sort_values("Degradation_fraction"),
            metric_prefix="Rfree_percent",
            ylabel="Free R (%)",
            out_png=plot_subdir / f"degradation_rfree_by_fraction__{safe}.png",
            title=f"AutoBuild Free R by degradation fraction | {prefix}",
            y_min=args.rfree_ymin,
            y_max=args.rfree_ymax,
        )
        plot_metric_by_fraction(
            df_summary=sub_sum.sort_values("Degradation_fraction"),
            metric_prefix="Residues_placed",
            ylabel="Residues placed",
            out_png=plot_subdir / f"degradation_residues_placed_by_fraction__{safe}.png",
            title=f"AutoBuild residues placed by degradation fraction | {prefix}",
            y_min=args.placed_ymin,
            y_max=args.placed_ymax,
        )
        plot_success_fraction_by_fraction(
            df_summary=sub_sum.sort_values("Degradation_fraction"),
            out_png=plot_subdir / f"degradation_success_fraction_by_fraction__{safe}.png",
            title=f"Success fraction by degradation factor | {prefix} | Free R <= {float(args.success_rfree_max):.1f}%",
        )

        sub_valid = df_valid.loc[
            (df_valid["Source_PDB_ID"] == src_pdb)
            & (df_valid["Method"] == method)
            & (df_valid["Phase_variant"] == phase_variant)
            & (df_valid["Degradation_mode"] == degr_mode)
        ].copy()

        plot_jitter_box_by_fraction(
            df_valid=sub_valid,
            value_col="Rfree_percent",
            ylabel="Free R (%)",
            out_png=plot_subdir / f"degradation_rfree_jitter_box__{safe}.png",
            title=f"AutoBuild Free R pooled by degradation factor | {prefix}",
            unit_suffix="%",
            decimals=PLOT_DECIMALS,
        )
        plot_jitter_box_by_fraction(
            df_valid=sub_valid,
            value_col="Residues_placed",
            ylabel="Residues placed",
            out_png=plot_subdir / f"degradation_residues_placed_jitter_box__{safe}.png",
            title=f"AutoBuild residues placed pooled by degradation factor | {prefix}",
            unit_suffix="res",
            decimals=PLOT_DECIMALS,
        )

    print_degradation_statistics_to_terminal(df_summary=df_summary, df_pair=df_pair)

    logging.info("[DONE] Degradation analysis complete.")
    logging.info("Log file          : %s", str(log_path))
    print(f"[INFO] Summary log written to: {log_path}")


if __name__ == "__main__":
    main()
