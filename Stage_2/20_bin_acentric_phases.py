#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20_bin_acentric_phases.py

python 20_bin_acentric_phases.py     --source_csv_dir ./03_Run_Ecalc/     --source_csv_pattern "*_rs_ecalc.csv"     --phase_col PHIC_ALL     --fom_col FOM     --bins "4,3,2"     --max_workers 30     --logs_dir ./LOGS     --skip_write_if_exists

(Repository-integrated stage-20 acentric compression script)

Example:
  python 20_bin_acentric_phases.py \
    --source_csv_dir ./03_Run_Ecalc/ \
    --source_csv_pattern "*_rs_ecalc.csv" \
    --phase_col PHIC_ALL \
    --fom_col FOM \
    --bins "4,3,2" \
    --max_workers 30 \
    --logs_dir ./LOGS \
    --skip_write_if_exists

Key features
- Robust logging to BOTH terminal and --logs_dir (default: ./LOGS)
- The .log file stays directly in --logs_dir
- ALL other artifacts (CSV/TXT/PNG) are written into a dedicated subfolder under --logs_dir,
  named after the log file stem (chronological + parameter-rich), e.g.:

  LOGS/
    PC30_phase_binning__v01__YYYYMMDD_HHMMSS__nXXXX__k4-3-2__w0030__phasePHIC_ALL__fomFOM.log
    PC30_phase_binning__v01__YYYYMMDD_HHMMSS__nXXXX__k4-3-2__w0030__phasePHIC_ALL__fomFOM/
      <same-stem>__dataset_counts.csv
      <same-stem>__per_dataset_per_k.csv
      <same-stem>__global_per_k.csv
      <same-stem>__summary.txt
      <same-stem>__rms_delta_by_K.png
      <same-stem>__rms_delta_table.csv
      <same-stem>__centrosymmetric_hits.csv   (if any)
      ...

- “analysis-only if already binned exists” behavior:
    If a binned CSV already exists and --skip_write_if_exists is set, it does NOT rewrite it,
    but it still computes ALL requested analyses/statistics.

New analyses added (per request)
- Detect any centrosymmetric space group in the dataset set (report PDB_ID + SG; write CSV + log warnings)
- Detect whether acentric phases are close to uniform around the circle using mean resultant length R
  (R≈0 uniform; R→1 clustered). These metrics are saved per dataset and summarized in logs.
- Plot RMS(Δφ) per dataset grouped by K, and save a pivoted RMS table CSV including mean resultant length R.

Output folder for binned CSVs:
- Default: derived from --source_csv_dir, e.g. 03_Run_Ecalc -> 04_Run_Ecalc_Binned (OK as-is for pipeline)
"""

from __future__ import annotations

import argparse
import glob
import logging
import math
import os
import re
import shlex
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional: crystal system classification via gemmi (best-effort)
try:
    import gemmi  # type: ignore
    _HAS_GEMMI = True
except Exception:
    gemmi = None  # type: ignore
    _HAS_GEMMI = False


# =========================
# Plotting style (user prefs)
# =========================

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

MAJOR_TICK_SIZE = 10
MINOR_TICK_SIZE = 5
GRID_MAJOR_KW: Dict[str, object] = {
    "which": "major",
    "linestyle": (0, (5, 5)),
    "color": "0.5",
    "linewidth": 0.8,
}
GRID_MINOR_KW: Dict[str, object] = {
    "which": "minor",
    "linestyle": (0, (1, 4)),
    "color": "0.7",
    "linewidth": 0.6,
}


# =========================
# Logging helper
# =========================

def configure_logging_pc30(
    *,
    logs_dir: str,
    script_tag: str,
    n_csv: int,
    k_values: List[int],
    max_workers: int,
    phase_col: str,
    fom_col: str,
) -> Tuple[str, str]:
    """Configure logging to stdout and an information-rich timestamped log file; return (timestamp, log_path)."""
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    k_slug = "-".join([str(int(k)) for k in k_values]) if k_values else "none"

    log_filename = (
        f"{script_tag}__{timestamp}"
        f"__n{int(n_csv):04d}"
        f"__k{k_slug}"
        f"__w{int(max_workers):04d}"
        f"__phase{str(phase_col)}"
        f"__fom{str(fom_col)}.log"
    )
    log_path = os.path.join(logs_dir, log_filename)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(filename=log_path, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logging.info("Log file: %s", log_path)
    return timestamp, log_path


# =========================
# Utilities
# =========================

def ensure_dir(*, path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_csv_list(*, s: str | None) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def discover_csvs(*, folder: str, patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pat in patterns:
        full = os.path.join(folder, pat)
        recursive = ("**" in pat)
        matches = glob.glob(full, recursive=recursive)
        files.extend(matches)
    return sorted(set(files))


def pdb_id_from_filename(*, path: str) -> str:
    base = os.path.basename(path)
    head = base.split("_")[0]
    return head if head else base[:4]


def dynamic_output_dir_from_source(*, source_dir: str) -> str:
    """
    Given a folder like '03_Run_Ecalc', create a sibling folder
    '04_Run_Ecalc_Binned' by incrementing the numeric prefix and appending '_Binned' to the suffix.

    If the name does not start with an integer, fall back to '<basename>_Binned'.
    """
    parent = os.path.dirname(os.path.normpath(source_dir))
    basename = os.path.basename(os.path.normpath(source_dir))
    m = re.match(pattern=r"^(\d+)(?:[_-])?(.*)$", string=basename)
    if m:
        num_str, rest = m.groups()
        width = len(num_str)
        next_num = str(int(num_str) + 1).zfill(width)
        rest_clean = rest.lstrip("_-") or "Output"
        if not rest_clean.endswith("_Binned"):
            rest_clean = f"{rest_clean}_Binned"
        out_name = f"{next_num}_{rest_clean}"
    else:
        rest_clean = basename
        if not rest_clean.endswith("_Binned"):
            rest_clean = f"{rest_clean}_Binned"
        out_name = rest_clean
    return os.path.join(parent, out_name)


def get_spacegroup_from_df(*, df: pd.DataFrame) -> str:
    """Best-effort extraction of space group from the ECALC CSV (column 'SG')."""
    if "SG" not in df.columns:
        return "<MISSING>"
    try:
        sg_val = str(df["SG"].iloc[0]).strip()
    except Exception:
        sg_val = ""
    return sg_val if sg_val else "<MISSING>"


def classify_crystal_system(*, sg_str: str) -> str:
    """
    Best-effort classification of crystal system from Hermann–Mauguin SG string.

    Output: triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic, unknown
    """
    if not sg_str or sg_str == "<MISSING>":
        return "unknown"

    if _HAS_GEMMI and gemmi is not None:
        try:
            sg = gemmi.SpaceGroup(sg_str)
            cs = sg.crystal_system()
            cs_str = str(cs).lower()
            if "triclinic" in cs_str:
                return "triclinic"
            if "monoclinic" in cs_str:
                return "monoclinic"
            if "orthorhombic" in cs_str:
                return "orthorhombic"
            if "tetragonal" in cs_str:
                return "tetragonal"
            if "trigonal" in cs_str:
                return "trigonal"
            if "hexagonal" in cs_str:
                return "hexagonal"
            if "cubic" in cs_str:
                return "cubic"
        except Exception:
            return "unknown"

    return "unknown"


def safe_slug(*, s: str) -> str:
    s0 = str(s).strip().lower()
    s0 = re.sub(pattern=r"\s+", repl="_", string=s0)
    s0 = re.sub(pattern=r"[^a-z0-9_]+", repl="", string=s0)
    return s0 if s0 else "unknown"


def log_stem_from_log_path(*, log_path: str) -> str:
    base = os.path.basename(log_path)
    stem, _ = os.path.splitext(base)
    return stem


def make_run_artifact_dir(*, logs_dir: str, log_path: str) -> str:
    stem = log_stem_from_log_path(log_path=log_path)
    run_dir = os.path.join(os.path.abspath(logs_dir), stem)
    ensure_dir(path=run_dir)
    return run_dir


def is_centrosymmetric_sg(*, sg_str: str) -> bool:
    """Return True if sg_str is centrosymmetric (best-effort via gemmi)."""
    if (not sg_str) or (sg_str == "<MISSING>"):
        return False
    if (not _HAS_GEMMI) or (gemmi is None):
        return False
    try:
        sg = gemmi.SpaceGroup(str(sg_str))
        if hasattr(sg, "is_centrosymmetric"):
            return bool(sg.is_centrosymmetric())
    except Exception:
        return False
    return False


# =========================
# Phase math
# =========================

def wrap_deg_m180_180(*, a: np.ndarray) -> np.ndarray:
    """Wrap degrees to [-180, 180)."""
    x = np.asarray(a, dtype=float)
    return (x + 180.0) % 360.0 - 180.0


def canonical_phase_deg(*, a: np.ndarray) -> np.ndarray:
    """Canonicalize phases into (-180, 180] with -180° ≡ +180°."""
    x = wrap_deg_m180_180(a=a)
    x = np.where(np.isclose(x, -180.0, atol=1e-9), 180.0, x)
    return x


def circ_diff_deg(*, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal signed circular difference a - b in degrees, mapped to (-180, 180] with -180° ≡ +180°."""
    diff = wrap_deg_m180_180(a=(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))
    return canonical_phase_deg(a=diff)


def acentric_phase_uniformity_metrics(*, phase_deg: np.ndarray) -> Dict[str, float]:
    """
    Circular uniformity metrics using mean resultant length R (0≈uniform, 1≈clustered).
    Returns NaNs if too few points.
    """
    ph = np.asarray(phase_deg, dtype=float)
    ph = ph[np.isfinite(ph)]
    n = int(ph.size)
    if n < 10:
        return {"n": float(n), "R": np.nan, "circ_var": np.nan, "rayleigh_z": np.nan}

    theta = np.deg2rad(ph)
    c = float(np.mean(np.cos(theta)))
    s = float(np.mean(np.sin(theta)))
    R = float(np.sqrt(c * c + s * s))
    circ_var = float(1.0 - R)
    rayleigh_z = float(n * (R ** 2))
    return {"n": float(n), "R": R, "circ_var": circ_var, "rayleigh_z": rayleigh_z}


# =========================
# Binning logic
# =========================

def bin_centers_deg(*, phase_canon_deg: np.ndarray, k: int) -> np.ndarray:
    """Compute nearest bin center for each phase (acentric-only logic applied later)."""
    delta = 360.0 / float(k)
    phase_mod = np.mod(phase_canon_deg, 360.0)
    bin_index = (np.floor(phase_mod / delta + 0.5).astype(int)) % int(k)
    phase_bin = bin_index.astype(float) * delta
    return canonical_phase_deg(a=phase_bin)


@dataclass(frozen=True)
class DatasetCounts:
    pdb_id: str
    sg: str
    crystal_system: str
    n_total: int
    n_acentric: int
    n_centric: int
    pct_acentric: float
    pct_centric: float
    binned_csv_exists: bool
    binned_csv_written: bool
    status: str  # "ok" or error tag


def compute_theory(*, k: int) -> Dict[str, float]:
    delta_deg = 360.0 / float(k)
    rms_theory = delta_deg / math.sqrt(12.0)
    half_delta_rad = math.radians(delta_deg / 2.0)
    att_theory = math.sin(half_delta_rad) / half_delta_rad if half_delta_rad > 0 else 1.0
    return {
        "delta_deg": float(delta_deg),
        "rms_theory_deg": float(rms_theory),
        "attenuation_theory": float(att_theory),
    }


def compute_dataset_stats_for_k(
    *,
    phase_canon_deg: np.ndarray,
    phase_binned_deg: np.ndarray,
    fom: np.ndarray,
    is_acentric: np.ndarray,
    k: int,
) -> Dict[str, Any]:
    """Compute per-dataset stats for one K (acentric-only)."""
    delta_phi = circ_diff_deg(a=phase_canon_deg, b=phase_binned_deg)

    sel = is_acentric & np.isfinite(delta_phi) & np.isfinite(fom)
    n_ac = int(np.sum(sel))
    th = compute_theory(k=k)

    if n_ac <= 0:
        return {
            "K": int(k),
            "n_acentric": 0,
            "rms_delta_deg": np.nan,
            "mean_abs_delta_deg": np.nan,
            "p95_abs_delta_deg": np.nan,
            "max_abs_delta_deg": np.nan,
            "mean_fom_before": np.nan,
            "mean_fom_after": np.nan,
            "mean_fom_ratio": np.nan,
            **th,
        }

    delta_ac = delta_phi[sel]
    abs_delta_ac = np.abs(delta_ac)

    rms_delta = float(np.sqrt(np.mean(delta_ac ** 2)))
    mean_abs = float(np.mean(abs_delta_ac))
    p95 = float(np.percentile(abs_delta_ac, 95.0))
    max_abs = float(np.max(abs_delta_ac))

    delta_rad = np.radians(delta_ac)
    atten = np.cos(delta_rad)
    fom_before = fom[sel]
    fom_after = np.clip(fom_before * atten, 0.0, 1.0)

    mean_before = float(np.mean(fom_before))
    mean_after = float(np.mean(fom_after))
    ratio = (mean_after / mean_before) if mean_before > 1e-12 else np.nan

    return {
        "K": int(k),
        "n_acentric": n_ac,
        "rms_delta_deg": rms_delta,
        "mean_abs_delta_deg": mean_abs,
        "p95_abs_delta_deg": p95,
        "max_abs_delta_deg": max_abs,
        "mean_fom_before": mean_before,
        "mean_fom_after": mean_after,
        "mean_fom_ratio": ratio,
        **th,
    }


def bin_phases_for_dataset(
    *,
    df: pd.DataFrame,
    phase_col: str,
    fom_col: str,
    k_values: List[int],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Add binned phase columns and attenuated FOM columns to df."""
    df_out = df.copy()

    if "CENTRIC" not in df_out.columns:
        raise ValueError("Input CSV must contain a 'CENTRIC' column (0 = acentric, 1 = centric).")
    if phase_col not in df_out.columns:
        raise ValueError(f"Phase column '{phase_col}' not found in CSV.")
    if fom_col not in df_out.columns:
        raise ValueError(f"FOM column '{fom_col}' not found in CSV.")

    centric_flag = df_out["CENTRIC"].to_numpy(dtype=int, copy=False)
    is_acentric = (centric_flag == 0)

    phase_deg = df_out[phase_col].to_numpy(dtype=float, copy=False)
    phase_canon = canonical_phase_deg(a=phase_deg)
    fom = df_out[fom_col].to_numpy(dtype=float, copy=False)

    per_k_stats: List[Dict[str, Any]] = []

    for k in k_values:
        phase_bin_canon = bin_centers_deg(phase_canon_deg=phase_canon, k=int(k))

        # Centrics copied unchanged; acentrics binned
        phase_out = phase_canon.copy()
        phase_out[is_acentric] = phase_bin_canon[is_acentric]

        out_phase_col = f"{phase_col}_K{int(k)}"
        out_fom_col = f"{fom_col}_K{int(k)}"

        df_out[out_phase_col] = np.rint(phase_out).astype(int)

        delta_phi = circ_diff_deg(a=phase_canon, b=phase_out)
        atten_factor = np.cos(np.radians(delta_phi))
        fom_new = fom.copy()
        fom_new[is_acentric] = fom[is_acentric] * atten_factor[is_acentric]
        df_out[out_fom_col] = np.clip(fom_new, 0.0, 1.0)

        stats_k = compute_dataset_stats_for_k(
            phase_canon_deg=phase_canon,
            phase_binned_deg=phase_out,
            fom=fom,
            is_acentric=is_acentric,
            k=int(k),
        )
        per_k_stats.append(stats_k)

    return df_out, per_k_stats


# =========================
# Plotting helpers
# =========================

def jitter_boxplot_categories(
    *,
    values_by_category: Dict[str, np.ndarray],
    ylabel: str,
    title: str,
    out_png: str,
    rotate_xticks: bool = True,
) -> None:
    """Jitter + boxplot with the requested geometry; mean marker overlaid on each box."""
    cats_all = list(values_by_category.keys())

    cats: List[str] = []
    data: List[np.ndarray] = []
    for c in cats_all:
        v = np.asarray(values_by_category[c], dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        cats.append(str(c))
        data.append(v)

    if not cats:
        logging.warning("[PLOT] No categories with data for: %s", title)
        return

    jitter_width = 0.20
    box_width = 0.28
    gap = 0.04
    box_offset = (jitter_width / 2.0) + (box_width / 2.0) + gap

    x = np.arange(len(cats), dtype=float)
    box_pos = x + box_offset

    fig_w = max(8.0, 0.8 * len(cats))
    fig, ax = plt.subplots(figsize=(fig_w, 6.0))
    rng = np.random.default_rng(seed=123)

    for i, vals in enumerate(data):
        jitter = (rng.random(vals.size) - 0.5) * jitter_width
        ax.scatter(x[i] + jitter, vals, alpha=0.25, s=10, edgecolor="none")

    ax.boxplot(
        x=data,
        positions=box_pos,
        widths=box_width,
        patch_artist=True,
        showfliers=True,
        manage_ticks=False,
        boxprops=dict(facecolor="white", edgecolor="black"),
        medianprops=dict(linewidth=1.8),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )

    means = np.array([float(np.mean(v)) for v in data], dtype=float)
    ax.scatter(
        box_pos,
        means,
        marker="o",
        s=80,
        facecolor="white",
        edgecolor="black",
        linewidth=1.5,
        zorder=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    if rotate_xticks and len(cats) > 6:
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")

    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.tick_params(axis="both", which="major", length=MAJOR_TICK_SIZE)
    ax.tick_params(axis="both", which="minor", length=MINOR_TICK_SIZE)
    ax.minorticks_on()
    ax.grid(**GRID_MAJOR_KW)
    ax.grid(**GRID_MINOR_KW)

    fig.tight_layout()
    ensure_dir(path=os.path.dirname(out_png))
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def jitter_boxplot_two_categories(
    *,
    values_a: np.ndarray,
    values_b: np.ndarray,
    label_a: str,
    label_b: str,
    ylabel: str,
    title: str,
    out_png: str,
) -> None:
    values_by_category = {
        str(label_a): np.asarray(values_a, dtype=float),
        str(label_b): np.asarray(values_b, dtype=float),
    }
    jitter_boxplot_categories(
        values_by_category=values_by_category,
        ylabel=ylabel,
        title=title,
        out_png=out_png,
        rotate_xticks=False,
    )


# =========================
# Aggregation helpers
# =========================

def aggregate_global_per_k(*, per_dataset_per_k: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted aggregation across datasets.
    IMPORTANT: filters out non-finite per-dataset metrics to avoid NaN propagation.
    """
    rows: List[Dict[str, Any]] = []
    if per_dataset_per_k.empty:
        return pd.DataFrame(rows)

    for k in sorted(per_dataset_per_k["K"].unique()):
        sub = per_dataset_per_k[per_dataset_per_k["K"] == k].copy()
        if sub.empty:
            continue

        sel = (
            np.isfinite(sub["n_acentric"].to_numpy(dtype=float)) &
            (sub["n_acentric"].to_numpy(dtype=float) > 0) &
            np.isfinite(sub["rms_delta_deg"].to_numpy(dtype=float)) &
            np.isfinite(sub["mean_abs_delta_deg"].to_numpy(dtype=float)) &
            np.isfinite(sub["p95_abs_delta_deg"].to_numpy(dtype=float)) &
            np.isfinite(sub["mean_fom_before"].to_numpy(dtype=float)) &
            np.isfinite(sub["mean_fom_after"].to_numpy(dtype=float))
        )
        sub = sub.loc[sel].copy()
        if sub.empty:
            continue

        n_tot = int(sub["n_acentric"].sum())
        if n_tot <= 0:
            continue

        weights = sub["n_acentric"].to_numpy(dtype=float)
        w = weights / float(np.sum(weights))

        rms_per = sub["rms_delta_deg"].to_numpy(dtype=float)
        rms_global = float(np.sqrt(np.sum(w * (rms_per ** 2))))

        mean_abs_global = float(np.sum(w * sub["mean_abs_delta_deg"].to_numpy(dtype=float)))
        p95_proxy = float(np.sum(w * sub["p95_abs_delta_deg"].to_numpy(dtype=float)))

        mean_before = float(np.sum(w * sub["mean_fom_before"].to_numpy(dtype=float)))
        mean_after = float(np.sum(w * sub["mean_fom_after"].to_numpy(dtype=float)))
        ratio = (mean_after / mean_before) if mean_before > 1e-12 else np.nan

        th = compute_theory(k=int(k))
        rows.append({
            "K": int(k),
            "n_acentric_total": int(n_tot),
            "delta_deg": th["delta_deg"],
            "rms_delta_global_deg": rms_global,
            "rms_theory_deg": th["rms_theory_deg"],
            "mean_abs_delta_global_deg": mean_abs_global,
            "p95_abs_delta_weighted_proxy_deg": p95_proxy,
            "mean_fom_before_global": mean_before,
            "mean_fom_after_global": mean_after,
            "mean_fom_ratio_global": ratio,
            "attenuation_theory": th["attenuation_theory"],
            "ratio_minus_theory": (ratio - th["attenuation_theory"]) if np.isfinite(ratio) else np.nan,
            "ratio_over_theory": (ratio / th["attenuation_theory"]) if (np.isfinite(ratio) and th["attenuation_theory"] > 0) else np.nan,
            "n_datasets_used": int(sub.shape[0]),
        })

    return pd.DataFrame(rows)


def summarize_by_group(
    *,
    counts_df_ok: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    if counts_df_ok.empty:
        return pd.DataFrame()

    g = counts_df_ok.groupby(by=group_col, sort=False, dropna=False)

    out = g.agg(
        n_datasets=pd.NamedAgg(column="PDB_ID", aggfunc="count"),
        total_reflections=pd.NamedAgg(column="n_total", aggfunc="sum"),
        total_acentric=pd.NamedAgg(column="n_acentric", aggfunc="sum"),
        total_centric=pd.NamedAgg(column="n_centric", aggfunc="sum"),
        mean_pct_acentric=pd.NamedAgg(column="pct_acentric", aggfunc="mean"),
        std_pct_acentric=pd.NamedAgg(column="pct_acentric", aggfunc="std"),
        median_pct_acentric=pd.NamedAgg(column="pct_acentric", aggfunc="median"),
        mean_pct_centric=pd.NamedAgg(column="pct_centric", aggfunc="mean"),
        std_pct_centric=pd.NamedAgg(column="pct_centric", aggfunc="std"),
        median_pct_centric=pd.NamedAgg(column="pct_centric", aggfunc="median"),
        median_acentric_phase_R=pd.NamedAgg(column="acentric_phase_R", aggfunc="median"),
    ).reset_index()

    out["pct_acentric_global"] = np.where(
        out["total_reflections"].to_numpy(dtype=float) > 0,
        100.0 * out["total_acentric"].to_numpy(dtype=float) / out["total_reflections"].to_numpy(dtype=float),
        np.nan,
    )
    out["pct_centric_global"] = np.where(
        out["total_reflections"].to_numpy(dtype=float) > 0,
        100.0 * out["total_centric"].to_numpy(dtype=float) / out["total_reflections"].to_numpy(dtype=float),
        np.nan,
    )

    out = out.sort_values(by=["n_datasets", "total_reflections"], ascending=[False, False], kind="mergesort")
    return out


def write_summary_txt(
    *,
    out_txt: str,
    args: argparse.Namespace,
    counts_df: pd.DataFrame,
    global_k_df: pd.DataFrame,
    log_path: str,
    out_dir: str,
    n_csv_discovered: int,
    centrosym_hits: int,
) -> None:
    lines: List[str] = []
    lines.append("Phase binning summary")
    lines.append("====================")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("Run parameters")
    lines.append("--------------")
    lines.append(f"source_csv_dir       : {os.path.abspath(args.source_csv_dir)}")
    lines.append(f"source_csv_pattern   : {args.source_csv_pattern}")
    lines.append(f"phase_col            : {args.phase_col}")
    lines.append(f"fom_col              : {args.fom_col}")
    lines.append(f"K values             : {args.bins}")
    lines.append(f"max_workers          : {args.max_workers}")
    lines.append(f"output_dir           : {os.path.abspath(out_dir)}")
    lines.append(f"skip_write_if_exists : {bool(args.skip_write_if_exists)}")
    lines.append(f"logs_dir             : {os.path.abspath(args.logs_dir)}")
    lines.append(f"log_file             : {log_path}")
    lines.append(f"csvs_discovered      : {int(n_csv_discovered)}")
    lines.append(f"gemmi_available      : {bool(_HAS_GEMMI)}")
    lines.append("")

    ok_df = counts_df[counts_df["status"] == "ok"].copy()
    n_ok = int(ok_df.shape[0])
    n_all = int(counts_df.shape[0])

    lines.append("Dataset accounting")
    lines.append("------------------")
    lines.append(f"datasets discovered                 : {n_all}")
    lines.append(f"datasets successfully analyzed      : {n_ok}")
    lines.append(f"datasets with existing binned CSV   : {int(np.sum(counts_df['binned_csv_exists']))}")
    lines.append(f"datasets newly written binned CSV   : {int(np.sum(counts_df['binned_csv_written']))}")
    lines.append(f"centrosymmetric SG hits             : {int(centrosym_hits)}")

    if n_ok > 0:
        total_ref = int(ok_df["n_total"].sum())
        total_ac = int(ok_df["n_acentric"].sum())
        total_c = int(ok_df["n_centric"].sum())
        pct_ac = (100.0 * total_ac / total_ref) if total_ref > 0 else np.nan
        pct_c = (100.0 * total_c / total_ref) if total_ref > 0 else np.nan
        lines.append(f"total reflections (all datasets)    : {total_ref}")
        lines.append(f"acentric reflections (all datasets) : {total_ac} ({pct_ac:.2f}%)")
        lines.append(f"centric reflections (all datasets)  : {total_c} ({pct_c:.2f}%)")

        r_vals = ok_df["acentric_phase_R"].to_numpy(dtype=float)
        r_vals = r_vals[np.isfinite(r_vals)]
        if r_vals.size > 0:
            lines.append("")
            lines.append("Acentric phase uniformity (mean resultant length R)")
            lines.append("--------------------------------------------------")
            lines.append(f"R median = {float(np.median(r_vals)):.4f}")
            lines.append(f"R mean   = {float(np.mean(r_vals)):.4f}")
            lines.append(f"R q05    = {float(np.quantile(r_vals, 0.05)):.4f}")
            lines.append(f"R q95    = {float(np.quantile(r_vals, 0.95)):.4f}")
            lines.append("Interpretation: R≈0 uniform; R→1 clustered")
    lines.append("")

    lines.append("Global K statistics (acentric only)")
    lines.append("----------------------------------")
    if global_k_df.empty:
        lines.append("No global K statistics available (after filtering non-finite per-dataset metrics).")
    else:
        for _, row in global_k_df.iterrows():
            k = int(row["K"])
            lines.append(
                f"K={k:>2d} | N_ac={int(row['n_acentric_total']):>10d} | "
                f"RMS={row['rms_delta_global_deg']:.2f}° (theory {row['rms_theory_deg']:.2f}°) | "
                f"FOM ratio={row['mean_fom_ratio_global']:.3f} (theory {row['attenuation_theory']:.3f}) | "
                f"ratio/theory={row['ratio_over_theory']:.3f} | n_datasets_used={int(row['n_datasets_used'])}"
            )
    lines.append("")

    ensure_dir(path=os.path.dirname(out_txt))
    with open(file=out_txt, mode="w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


# =========================
# Worker
# =========================

def process_one_csv(
    *,
    csv_path: str,
    out_dir: str,
    phase_col: str,
    fom_col: str,
    k_values: List[int],
    skip_write_if_exists: bool,
) -> Tuple[DatasetCounts, List[Dict[str, Any]], Dict[str, float]]:
    """
    For one dataset:
      - read source ECALC CSV
      - compute per-dataset CENTRIC/ACENTRIC counts
      - compute acentric phase uniformity (R, etc.) from original phase_col
      - compute per-K stats (always)
      - write binned CSV if missing or if skip_write_if_exists=False
    Returns:
      (DatasetCounts, per_k_stats_rows, uniformity_metrics)
    """
    pdb_id = pdb_id_from_filename(path=csv_path)
    base = os.path.basename(csv_path)
    root, ext = os.path.splitext(base)
    out_csv = os.path.join(out_dir, f"{root}_binned{ext}")

    binned_exists = bool(os.path.exists(out_csv) and os.path.getsize(out_csv) > 0)
    wrote = False

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return (
            DatasetCounts(
                pdb_id=pdb_id,
                sg="<UNKNOWN>",
                crystal_system="unknown",
                n_total=0,
                n_acentric=0,
                n_centric=0,
                pct_acentric=np.nan,
                pct_centric=np.nan,
                binned_csv_exists=binned_exists,
                binned_csv_written=False,
                status=f"read_error: {exc}",
            ),
            [],
            {"n": np.nan, "R": np.nan, "circ_var": np.nan, "rayleigh_z": np.nan},
        )

    sg_val = get_spacegroup_from_df(df=df)
    cs_val = classify_crystal_system(sg_str=sg_val)

    required_cols = ["H", "K", "L", "CENTRIC", phase_col, fom_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return (
            DatasetCounts(
                pdb_id=pdb_id,
                sg=sg_val,
                crystal_system=cs_val,
                n_total=int(df.shape[0]),
                n_acentric=0,
                n_centric=0,
                pct_acentric=np.nan,
                pct_centric=np.nan,
                binned_csv_exists=binned_exists,
                binned_csv_written=False,
                status=f"missing_cols: {','.join(missing)}",
            ),
            [],
            {"n": np.nan, "R": np.nan, "circ_var": np.nan, "rayleigh_z": np.nan},
        )

    centric_flag = df["CENTRIC"].to_numpy(dtype=int, copy=False)
    n_total = int(df.shape[0])
    n_acentric = int(np.sum(centric_flag == 0))
    n_centric = int(np.sum(centric_flag == 1))
    pct_acentric = (100.0 * n_acentric / n_total) if n_total > 0 else np.nan
    pct_centric = (100.0 * n_centric / n_total) if n_total > 0 else np.nan

    # Acentric phase uniformity (mean resultant length R) from original phase_col, acentric-only
    ac_mask = (centric_flag == 0)
    phase_raw = df[phase_col].to_numpy(dtype=float, copy=False)
    phase_ac = canonical_phase_deg(a=phase_raw[ac_mask]) if np.any(ac_mask) else np.asarray([], dtype=float)
    u = acentric_phase_uniformity_metrics(phase_deg=phase_ac)

    df_binned, per_k_stats = bin_phases_for_dataset(
        df=df,
        phase_col=phase_col,
        fom_col=fom_col,
        k_values=k_values,
    )

    if (not binned_exists) or (not skip_write_if_exists):
        try:
            df_binned.to_csv(path_or_buf=out_csv, index=False)
            wrote = True
        except Exception as exc:
            return (
                DatasetCounts(
                    pdb_id=pdb_id,
                    sg=sg_val,
                    crystal_system=cs_val,
                    n_total=n_total,
                    n_acentric=n_acentric,
                    n_centric=n_centric,
                    pct_acentric=pct_acentric,
                    pct_centric=pct_centric,
                    binned_csv_exists=binned_exists,
                    binned_csv_written=False,
                    status=f"write_error: {exc}",
                ),
                per_k_stats,
                u,
            )

    return (
        DatasetCounts(
            pdb_id=pdb_id,
            sg=sg_val,
            crystal_system=cs_val,
            n_total=n_total,
            n_acentric=n_acentric,
            n_centric=n_centric,
            pct_acentric=pct_acentric,
            pct_centric=pct_centric,
            binned_csv_exists=binned_exists,
            binned_csv_written=wrote,
            status="ok",
        ),
        per_k_stats,
        u,
    )


# =========================
# CLI / main
# =========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase binning on ECALC CSV outputs with persistent logs + summary outputs."
    )
    parser.add_argument("--source_csv_dir", type=str, required=True, help="Folder containing ECALC CSV files.")
    parser.add_argument(
        "--source_csv_pattern",
        type=str,
        required=True,
        help='Comma-separated glob pattern(s) under source_csv_dir (e.g. "*_rs_ecalc.csv").',
    )
    parser.add_argument("--phase_col", type=str, default="PHIC_ALL", help="Phase column in degrees to be binned.")
    parser.add_argument("--fom_col", type=str, default="FOM", help="FOM column.")
    parser.add_argument("--bins", type=str, required=True, help='Comma-separated K values (e.g. "4,3,2").')
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for binned CSVs.")
    parser.add_argument("--max_workers", type=int, default=8, help="Max parallel worker processes.")
    parser.add_argument("--logs_dir", type=str, default="LOGS", help="Folder for logs (log) and run artifacts (subdir).")
    parser.add_argument("--skip_write_if_exists", action="store_true", help="Do not rewrite binned CSVs if they exist.")
    parser.add_argument("--top_sg_n", type=int, default=10, help="Top space groups to show in SG plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    patterns = parse_csv_list(s=args.source_csv_pattern)
    if not patterns:
        raise SystemExit("--source_csv_pattern must contain at least one glob pattern.")

    try:
        k_values = [int(x) for x in parse_csv_list(s=args.bins)]
    except ValueError:
        raise SystemExit("--bins must be a comma-separated list of integers (e.g. '4,3,2').")
    if not k_values:
        raise SystemExit("No valid K values parsed from --bins.")

    source_dir = os.path.abspath(args.source_csv_dir)

    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
    else:
        out_dir = os.path.abspath(dynamic_output_dir_from_source(source_dir=source_dir))
    ensure_dir(path=out_dir)

    csv_paths = discover_csvs(folder=source_dir, patterns=patterns)
    if not csv_paths:
        print(f"[ERROR] No CSV files found in {source_dir} with patterns {patterns}")
        return

    script_tag = "20_bin_acentric_phases"
    timestamp, log_path = configure_logging_pc30(
        logs_dir=str(args.logs_dir),
        script_tag=str(script_tag),
        n_csv=int(len(csv_paths)),
        k_values=k_values,
        max_workers=int(args.max_workers),
        phase_col=str(args.phase_col),
        fom_col=str(args.fom_col),
    )

    run_dir = make_run_artifact_dir(logs_dir=str(args.logs_dir), log_path=str(log_path))
    log_stem = log_stem_from_log_path(log_path=str(log_path))
    logging.info("Run prefix: %s", script_tag)
    logging.info("Command line: %s", " ".join(shlex.quote(arg) for arg in sys.argv))
    logging.info("[RUN-DIR] Artifacts will be written to: %s", run_dir)

    logging.info("[INIT] %s", script_tag)
    logging.info("[ARGS] source_csv_dir=%s", source_dir)
    logging.info("[ARGS] source_csv_pattern=%s", patterns)
    logging.info("[ARGS] phase_col=%s", args.phase_col)
    logging.info("[ARGS] fom_col=%s", args.fom_col)
    logging.info("[ARGS] bins=%s", k_values)
    logging.info("[ARGS] output_dir=%s", out_dir)
    logging.info("[ARGS] max_workers=%d", int(args.max_workers))
    logging.info("[ARGS] logs_dir=%s", os.path.abspath(args.logs_dir))
    logging.info("[ARGS] skip_write_if_exists=%s", bool(args.skip_write_if_exists))
    logging.info("[ARGS] top_sg_n=%d", int(args.top_sg_n))
    logging.info("[ENV] gemmi_available=%s", bool(_HAS_GEMMI))
    logging.info("[DISCOVER] %d CSV files discovered", len(csv_paths))

    counts_rows: List[Dict[str, Any]] = []
    per_dataset_per_k_rows: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=int(args.max_workers)) as executor:
        futures = {
            executor.submit(
                process_one_csv,
                csv_path=csv_path,
                out_dir=out_dir,
                phase_col=str(args.phase_col),
                fom_col=str(args.fom_col),
                k_values=k_values,
                skip_write_if_exists=bool(args.skip_write_if_exists),
            ): csv_path
            for csv_path in csv_paths
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Phase binning + analysis", unit="file"):
            csv_path = futures[fut]
            try:
                counts_obj, per_k_stats, u = fut.result()
            except Exception as exc:
                pdb_id = pdb_id_from_filename(path=csv_path)
                logging.error("[WORKER] Error processing %s (%s): %s", os.path.basename(csv_path), pdb_id, exc)
                counts_obj = DatasetCounts(
                    pdb_id=pdb_id,
                    sg="<UNKNOWN>",
                    crystal_system="unknown",
                    n_total=0,
                    n_acentric=0,
                    n_centric=0,
                    pct_acentric=np.nan,
                    pct_centric=np.nan,
                    binned_csv_exists=False,
                    binned_csv_written=False,
                    status=f"worker_error: {exc}",
                )
                per_k_stats = []
                u = {"n": np.nan, "R": np.nan, "circ_var": np.nan, "rayleigh_z": np.nan}

            counts_rows.append({
                "PDB_ID": counts_obj.pdb_id,
                "SG": counts_obj.sg,
                "crystal_system": counts_obj.crystal_system,
                "n_total": counts_obj.n_total,
                "n_acentric": counts_obj.n_acentric,
                "n_centric": counts_obj.n_centric,
                "pct_acentric": counts_obj.pct_acentric,
                "pct_centric": counts_obj.pct_centric,
                "binned_csv_exists": counts_obj.binned_csv_exists,
                "binned_csv_written": counts_obj.binned_csv_written,
                "status": counts_obj.status,
                # Uniformity metrics (acentric-only)
                "acentric_phase_n": float(u.get("n", np.nan)),
                "acentric_phase_R": float(u.get("R", np.nan)),
                "acentric_phase_circ_var": float(u.get("circ_var", np.nan)),
                "acentric_phase_rayleigh_z": float(u.get("rayleigh_z", np.nan)),
            })

            for row in per_k_stats:
                row_out = dict(row)
                row_out["PDB_ID"] = counts_obj.pdb_id
                row_out["SG"] = counts_obj.sg
                row_out["crystal_system"] = counts_obj.crystal_system
                per_dataset_per_k_rows.append(row_out)

    counts_df = pd.DataFrame(counts_rows).sort_values(by="PDB_ID", kind="mergesort")
    per_dataset_per_k_df = pd.DataFrame(per_dataset_per_k_rows)

    ok_counts_df = counts_df[counts_df["status"] == "ok"].copy()

    if (not per_dataset_per_k_df.empty) and (not ok_counts_df.empty):
        ok_perk_df = per_dataset_per_k_df.merge(
            ok_counts_df[["PDB_ID"]],
            on="PDB_ID",
            how="inner",
            validate="many_to_one",
        )
    else:
        ok_perk_df = pd.DataFrame()

    global_k_df = aggregate_global_per_k(per_dataset_per_k=ok_perk_df) if not ok_perk_df.empty else pd.DataFrame()

    # =========================
    # Centrosymmetric SG detection
    # =========================
    centrosym_hits_df = ok_counts_df[ok_counts_df["SG"].apply(lambda x: is_centrosymmetric_sg(sg_str=str(x)))].copy()
    centrosym_hits = int(centrosym_hits_df.shape[0])
    out_centro_csv = os.path.join(run_dir, f"{log_stem}__centrosymmetric_hits.csv")
    if centrosym_hits > 0:
        centrosym_hits_df = centrosym_hits_df[["PDB_ID", "SG", "crystal_system", "n_total", "pct_acentric", "pct_centric"]].copy()
        centrosym_hits_df.to_csv(out_centro_csv, index=False)
        logging.warning("[CENTROSYM] Found %d dataset(s) in centrosymmetric space groups. Wrote: %s", centrosym_hits, out_centro_csv)
        for _, r in centrosym_hits_df.head(25).iterrows():
            logging.warning("[CENTROSYM] %s  SG=%s", r["PDB_ID"], r["SG"])
    else:
        logging.info("[CENTROSYM] No centrosymmetric space groups detected in analyzed datasets.")

    # =========================
    # Save core artifacts to run_dir
    # =========================
    out_counts_csv = os.path.join(run_dir, f"{log_stem}__dataset_counts.csv")
    out_perk_csv = os.path.join(run_dir, f"{log_stem}__per_dataset_per_k.csv")
    out_global_csv = os.path.join(run_dir, f"{log_stem}__global_per_k.csv")
    out_summary_txt = os.path.join(run_dir, f"{log_stem}__summary.txt")

    counts_df.to_csv(path_or_buf=out_counts_csv, index=False)
    logging.info("[WRITE] %s", out_counts_csv)

    per_dataset_per_k_df.to_csv(path_or_buf=out_perk_csv, index=False)
    logging.info("[WRITE] %s", out_perk_csv)

    if not global_k_df.empty:
        global_k_df.to_csv(path_or_buf=out_global_csv, index=False)
        logging.info("[WRITE] %s", out_global_csv)
    else:
        logging.warning("[WRITE] global_k_df is empty; not writing %s", out_global_csv)

    # =========================
    # Uniformity summary logging
    # =========================
    if not ok_counts_df.empty:
        r_vals = ok_counts_df["acentric_phase_R"].to_numpy(dtype=float)
        r_vals = r_vals[np.isfinite(r_vals)]
        if r_vals.size > 0:
            logging.info(
                "[PHASE-UNIF] Acentric phase R summary across datasets: median=%.4f mean=%.4f q05=%.4f q95=%.4f",
                float(np.median(r_vals)),
                float(np.mean(r_vals)),
                float(np.quantile(r_vals, 0.05)),
                float(np.quantile(r_vals, 0.95)),
            )
            logging.info(
                "[PHASE-UNIF] Near-uniform counts: R<0.02: %d | R<0.05: %d",
                int(np.sum(r_vals < 0.02)),
                int(np.sum(r_vals < 0.05)),
            )

    # =========================
    # RMS plot by K and RMS table (with mean resultant length R)
    # =========================
    out_rms_png = os.path.join(run_dir, f"{log_stem}__rms_delta_by_K.png")
    out_rms_table_csv = os.path.join(run_dir, f"{log_stem}__rms_delta_table.csv")

    if not ok_perk_df.empty and "rms_delta_deg" in ok_perk_df.columns:
        values_by_k: Dict[str, np.ndarray] = {}
        for k in sorted(ok_perk_df["K"].unique()):
            vals = ok_perk_df.loc[ok_perk_df["K"] == k, "rms_delta_deg"].to_numpy(dtype=float)
            values_by_k[f"K={int(k)}"] = vals

        jitter_boxplot_categories(
            values_by_category=values_by_k,
            ylabel="RMS(Δφ) per dataset (degrees), acentric only",
            title="Per-dataset RMS phase error after K-binning (acentric only)",
            out_png=out_rms_png,
            rotate_xticks=False,
        )
        logging.info("[PLOT] %s", out_rms_png)

        # Pivot to wide RMS table: one row per dataset, columns RMS_K2, RMS_K3, ...
        rms_wide = ok_perk_df.pivot_table(
            index=["PDB_ID", "SG", "crystal_system"],
            columns="K",
            values="rms_delta_deg",
            aggfunc="first",
        ).reset_index()

        # Rename K columns
        new_cols: List[str] = []
        for c in rms_wide.columns:
            if isinstance(c, (int, np.integer)):
                new_cols.append(f"RMS_K{int(c)}")
            else:
                new_cols.append(str(c))
        rms_wide.columns = new_cols

        # Attach R (mean resultant length) and a few useful counts from ok_counts_df
        attach_cols = [
            "PDB_ID",
            "n_total",
            "n_acentric",
            "n_centric",
            "pct_acentric",
            "pct_centric",
            "acentric_phase_R",
            "acentric_phase_circ_var",
            "acentric_phase_rayleigh_z",
            "acentric_phase_n",
        ]
        attach_df = ok_counts_df[attach_cols].copy()
        rms_table = rms_wide.merge(attach_df, on="PDB_ID", how="left", validate="one_to_one")

        # Reorder: identifiers + counts + R + RMS columns
        rms_cols = [c for c in rms_table.columns if c.startswith("RMS_K")]
        front = [
            "PDB_ID", "SG", "crystal_system",
            "n_total", "n_acentric", "n_centric", "pct_acentric", "pct_centric",
            "acentric_phase_n", "acentric_phase_R", "acentric_phase_circ_var", "acentric_phase_rayleigh_z",
        ]
        remaining = [c for c in rms_table.columns if (c not in front) and (c not in rms_cols)]
        rms_table = rms_table[front + rms_cols + remaining]

        rms_table.to_csv(out_rms_table_csv, index=False)
        logging.info("[WRITE] %s", out_rms_table_csv)
    else:
        logging.warning("[RMS] ok_perk_df empty; skipping RMS plot and RMS table.")

    # =========================
    # Summary tables and plots
    # =========================
    out_pct_png = os.path.join(run_dir, f"{log_stem}__centric_acentric_pct.png")
    out_cs_png = os.path.join(run_dir, f"{log_stem}__pct_acentric_by_crystalsystem.png")
    out_sg_png = os.path.join(run_dir, f"{log_stem}__pct_acentric_by_top_spacegroups.png")

    out_sg_summary_csv = os.path.join(run_dir, f"{log_stem}__summary_by_spacegroup.csv")
    out_cs_summary_csv = os.path.join(run_dir, f"{log_stem}__summary_by_crystalsystem.csv")

    if not ok_counts_df.empty:
        jitter_boxplot_two_categories(
            values_a=ok_counts_df["pct_acentric"].to_numpy(dtype=float),
            values_b=ok_counts_df["pct_centric"].to_numpy(dtype=float),
            label_a="% acentric",
            label_b="% centric",
            ylabel="Percentage of reflections per dataset (%)",
            title="CENTRIC vs ACENTRIC fractions across datasets",
            out_png=out_pct_png,
        )
        logging.info("[PLOT] %s", out_pct_png)

        # Top-N SG plot + summary
        sg_summary_df = summarize_by_group(counts_df_ok=ok_counts_df, group_col="SG")
        if not sg_summary_df.empty:
            sg_summary_df.to_csv(path_or_buf=out_sg_summary_csv, index=False)
            logging.info("[WRITE] %s", out_sg_summary_csv)

            top_n = int(max(1, args.top_sg_n))
            top_sgs = sg_summary_df["SG"].head(top_n).tolist()
            sg_plot_df = ok_counts_df.copy()
            sg_plot_df["SG_plot"] = np.where(sg_plot_df["SG"].isin(top_sgs), sg_plot_df["SG"], "Other")

            values_by_sg: Dict[str, np.ndarray] = {}
            for sg_name, sub in sg_plot_df.groupby(by="SG_plot", sort=False):
                values_by_sg[str(sg_name)] = sub["pct_acentric"].to_numpy(dtype=float)

            jitter_boxplot_categories(
                values_by_category=values_by_sg,
                ylabel="% acentric per dataset (%)",
                title=f"% acentric grouped by top {top_n} space groups (others merged)",
                out_png=out_sg_png,
                rotate_xticks=True,
            )
            logging.info("[PLOT] %s", out_sg_png)

        # Crystal system plot + summary
        cs_summary_df = summarize_by_group(counts_df_ok=ok_counts_df, group_col="crystal_system")
        if not cs_summary_df.empty:
            cs_summary_df.to_csv(path_or_buf=out_cs_summary_csv, index=False)
            logging.info("[WRITE] %s", out_cs_summary_csv)

            cs_order = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic", "unknown"]
            values_by_cs: Dict[str, np.ndarray] = {}
            for cs_name in cs_order:
                sub = ok_counts_df[ok_counts_df["crystal_system"] == cs_name]
                if not sub.empty:
                    values_by_cs[cs_name] = sub["pct_acentric"].to_numpy(dtype=float)

            jitter_boxplot_categories(
                values_by_category=values_by_cs,
                ylabel="% acentric per dataset (%)",
                title="% acentric grouped by crystal system",
                out_png=out_cs_png,
                rotate_xticks=False,
            )
            logging.info("[PLOT] %s", out_cs_png)

            # One plot per crystal system: SG categories within that system
            for cs_name in sorted(ok_counts_df["crystal_system"].unique().tolist()):
                sub_cs = ok_counts_df[ok_counts_df["crystal_system"] == cs_name].copy()
                if sub_cs.empty:
                    continue

                sg_counts = sub_cs["SG"].value_counts(dropna=False).to_dict()
                sg_order = sorted(sg_counts.keys(), key=lambda x: (-int(sg_counts.get(x, 0)), str(x)))

                values_by_sg_in_cs: Dict[str, np.ndarray] = {}
                for sg_name in sg_order:
                    vals = sub_cs[sub_cs["SG"] == sg_name]["pct_acentric"].to_numpy(dtype=float)
                    if vals.size > 0:
                        values_by_sg_in_cs[str(sg_name)] = vals

                out_cs_sg_png = os.path.join(
                    run_dir,
                    f"{log_stem}__pct_acentric_by_sg_within_{safe_slug(s=cs_name)}.png",
                )

                jitter_boxplot_categories(
                    values_by_category=values_by_sg_in_cs,
                    ylabel="% acentric per dataset (%)",
                    title=f"% acentric grouped by space group within crystal system: {cs_name}",
                    out_png=out_cs_sg_png,
                    rotate_xticks=True,
                )
                logging.info("[PLOT] %s", out_cs_sg_png)

    # =========================
    # Write summary TXT last (includes global K + uniformity + centrosym counts)
    # =========================
    write_summary_txt(
        out_txt=out_summary_txt,
        args=args,
        counts_df=counts_df,
        global_k_df=global_k_df,
        log_path=log_path,
        out_dir=out_dir,
        n_csv_discovered=int(len(csv_paths)),
        centrosym_hits=int(centrosym_hits),
    )
    logging.info("[WRITE] %s", out_summary_txt)

    # =========================
    # Terminal summary
    # =========================
    logging.info("=== Global phase-binning statistics (acentric only) ===")
    print("\n=== Global phase-binning statistics (acentric only) ===")
    if global_k_df.empty:
        logging.info("No global K statistics available (after filtering non-finite per-dataset metrics).")
        print("No global K statistics available (after filtering non-finite per-dataset metrics).")
    else:
        for _, row in global_k_df.iterrows():
            k = int(row["K"])
            summary_line = (
                f"K={k:>2d} | N_ac={int(row['n_acentric_total']):>10d} | "
                f"RMS={row['rms_delta_global_deg']:.2f}° (theory {row['rms_theory_deg']:.2f}°) | "
                f"FOM ratio={row['mean_fom_ratio_global']:.3f} (theory {row['attenuation_theory']:.3f}) | "
                f"ratio/theory={row['ratio_over_theory']:.3f} | n_datasets_used={int(row['n_datasets_used'])}"
            )
            logging.info("%s", summary_line)
            print(summary_line)

    logging.info("Artifacts written to:")
    logging.info("  Log file: %s", log_path)
    logging.info("  Run folder: %s", run_dir)
    logging.info("  Counts CSV: %s", out_counts_csv)
    logging.info("  Per-K CSV: %s", out_perk_csv)
    print("\nArtifacts written to:")
    print(f"  Log file:      {log_path}")
    print(f"  Run folder:    {run_dir}")
    print(f"  Counts CSV:    {out_counts_csv}")
    print(f"  Per-K CSV:     {out_perk_csv}")
    if not global_k_df.empty:
        logging.info("  Global K CSV: %s", out_global_csv)
        print(f"  Global K CSV:  {out_global_csv}")
    if os.path.exists(out_rms_table_csv):
        logging.info("  RMS table CSV: %s", out_rms_table_csv)
        print(f"  RMS table CSV: {out_rms_table_csv}")
    if os.path.exists(out_rms_png):
        logging.info("  RMS plot PNG: %s", out_rms_png)
        print(f"  RMS plot PNG:  {out_rms_png}")
    if centrosym_hits > 0:
        logging.info("  Centrosym CSV: %s", out_centro_csv)
        print(f"  Centrosym CSV: {out_centro_csv}")

    logging.info("[DONE] Phase binning analysis complete.")
    print("\n[DONE] Phase binning analysis complete.")


if __name__ == "__main__":
    main()
