#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_plot_symmetry_complexity_vs_centric_burden.py

Repository-ready stage-11 script for relating centric-burden quantities to
symmetry-derived complexity measures.

This script:
- loads the per-dataset centric-burden table from stage 10
- computes, for each space group:
    - n_rev_families : number of distinct reversing families in full reciprocal space
    - n_asu_traces   : number of distinct ASU-restricted centric traces
    - point_group    : point-group label for annotation
- summarizes centric-burden quantities at the space-group level using weighted means
  across datasets (weights = centric_completeness)
- plots the resulting SG-level summaries against either:
    - number of reversing families
    - number of ASU traces
    - both

Model fitting has been intentionally removed in this repository version.
All outputs are descriptive only.

Example:
    python 11_plot_symmetry_complexity_vs_centric_burden.py \
        --project_dir /path/to/project \
        --counts_dir 10_Empirical_Centric_Burden/01_Centric_Acentric_Counts_By_SG \
        --out_dir 11_Symmetry_vs_Centric_Burden \
        --x_axis_mode asu_traces \
        --asu_trace_max_index 6
"""

from __future__ import annotations

import argparse
import logging
import math
import shlex
import sys
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gemmi


# -------------------------
# Styling
# -------------------------

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

MAJOR_TICK_SIZE = 10
MINOR_TICK_SIZE = 5
GRID_MAJOR_KW: dict[str, object] = {
    "which": "major",
    "linestyle": (0, (5, 5)),
    "color": "0.5",
    "linewidth": 0.8,
}
GRID_MINOR_KW: dict[str, object] = {
    "which": "minor",
    "linestyle": (0, (1, 4)),
    "color": "0.7",
    "linewidth": 0.6,
}

RUN_PREFIX = "11_plot_symmetry_complexity_vs_centric_burden"


# -------------------------
# Logging
# -------------------------

def setup_logger(*, logs_dir: Path, run_prefix: str = RUN_PREFIX) -> Path:
    """
    Configure logging to both a LOGS file and stdout.

    The log captures:
    - timestamped INFO messages
    - the reconstructed command line
    - all main progress/status messages also shown in the terminal
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{run_prefix}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(filename=log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt=formatter)
    logger.addHandler(hdlr=file_handler)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(fmt=formatter)
    logger.addHandler(hdlr=stream_handler)

    return log_path


# -------------------------
# Integer 3x3 helpers
# -------------------------

Mat3 = tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
Row3 = tuple[int, int, int]


def det3(m: Mat3) -> int:
    (a, b, c), (d, e, f), (g, h, i) = m
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def adjugate3(m: Mat3) -> Mat3:
    (a, b, c), (d, e, f), (g, h, i) = m
    c00 = (e * i - f * h)
    c01 = -(d * i - f * g)
    c02 = (d * h - e * g)

    c10 = -(b * i - c * h)
    c11 = (a * i - c * g)
    c12 = -(a * h - b * g)

    c20 = (b * f - c * e)
    c21 = -(a * f - c * d)
    c22 = (a * e - b * d)

    return (
        (c00, c10, c20),
        (c01, c11, c21),
        (c02, c12, c22),
    )


def inv_int3(m: Mat3) -> Mat3:
    determinant = det3(m)
    if determinant not in (1, -1):
        raise ValueError(f"Rotation det not ±1: det={determinant}, m={m}")
    adj = adjugate3(m)
    if determinant == 1:
        return adj
    return tuple(tuple(-x for x in row) for row in adj)


def transpose3(m: Mat3) -> Mat3:
    return (
        (m[0][0], m[1][0], m[2][0]),
        (m[0][1], m[1][1], m[2][1]),
        (m[0][2], m[1][2], m[2][2]),
    )


def mat_add_I(m: Mat3) -> Mat3:
    return (
        (m[0][0] + 1, m[0][1],     m[0][2]),
        (m[1][0],     m[1][1] + 1, m[1][2]),
        (m[2][0],     m[2][1],     m[2][2] + 1),
    )


def mat_vec_mul(m: Mat3, v: tuple[int, int, int]) -> tuple[int, int, int]:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


# -------------------------
# Rotation normalization
# -------------------------

def normalize_rot_from_op(op: gemmi.Op) -> Mat3:
    den = int(op.DEN)
    r = op.rot
    out: list[list[int]] = []
    for i in range(3):
        row: list[int] = []
        for j in range(3):
            value = int(r[i][j])
            if value % den != 0:
                raise ValueError(f"rot entry not divisible by DEN={den}: {value}")
            row.append(value // den)
        out.append(row)
    return (tuple(out[0]), tuple(out[1]), tuple(out[2]))


# -------------------------
# Canonical row-space key
# -------------------------

def lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a // gcd(a, b) * b)


def row_gcd(row: Row3) -> int:
    g = 0
    for x in row:
        g = gcd(g, abs(int(x)))
    return g


def normalize_int_row(row: Row3) -> Row3:
    g = row_gcd(row)
    if g == 0:
        return (0, 0, 0)
    r = (row[0] // g, row[1] // g, row[2] // g)
    for x in r:
        if x != 0:
            if x < 0:
                r = (-r[0], -r[1], -r[2])
            break
    return r


def row_space_basis_key(A: Mat3) -> tuple[Row3, ...]:
    matrix: list[list[Fraction]] = [[Fraction(int(A[i][j]), 1) for j in range(3)] for i in range(3)]

    rank_row = 0
    for c in range(3):
        pivot = None
        for rr in range(rank_row, 3):
            if matrix[rr][c] != 0:
                pivot = rr
                break
        if pivot is None:
            continue

        if pivot != rank_row:
            matrix[rank_row], matrix[pivot] = matrix[pivot], matrix[rank_row]

        piv = matrix[rank_row][c]
        matrix[rank_row] = [x / piv for x in matrix[rank_row]]

        for rr in range(3):
            if rr == rank_row:
                continue
            factor = matrix[rr][c]
            if factor != 0:
                matrix[rr] = [matrix[rr][j] - factor * matrix[rank_row][j] for j in range(3)]

        rank_row += 1
        if rank_row == 3:
            break

    basis_rows: list[Row3] = []
    for rr in range(3):
        row = matrix[rr]
        if all(x == 0 for x in row):
            continue

        den = 1
        for x in row:
            den = lcm(den, int(x.denominator))
        ints = tuple(int(x * den) for x in row)
        ints_norm = normalize_int_row(ints)
        if ints_norm != (0, 0, 0):
            basis_rows.append(ints_norm)

    basis_rows = sorted(set(basis_rows))
    return tuple(basis_rows)


def enumerate_reversing_rotations(spacegroup_hm: str) -> list[Mat3]:
    """
    Return unique normalized rotations R for which det((R^{-1})^T + I) == 0.
    """
    sg = gemmi.SpaceGroup(spacegroup_hm)
    ops = sg.operations()

    rotations: list[Mat3] = []
    seen: set[Mat3] = set()
    for op in ops.sym_ops:
        R = normalize_rot_from_op(op)
        if R in seen:
            continue
        seen.add(R)

        try:
            rinvt = transpose3(inv_int3(R))
        except Exception:
            continue
        A = mat_add_I(rinvt)
        if det3(A) != 0:
            continue

        rotations.append(R)

    return rotations


def count_distinct_reversing_families(spacegroup_hm: str) -> int:
    """
    Count distinct reversing families in full reciprocal space.
    """
    family_keys: set[tuple[Row3, ...]] = set()
    for R in enumerate_reversing_rotations(spacegroup_hm=spacegroup_hm):
        rinvt = transpose3(inv_int3(R))
        A = mat_add_I(rinvt)
        key = row_space_basis_key(A)
        if len(key) > 0:
            family_keys.add(key)
    return int(len(family_keys))


# -------------------------
# ASU trace counting
# -------------------------

def asu_map_hkl(
    *,
    reciprocal_asu: gemmi.ReciprocalAsu,
    operations: gemmi.GroupOps,
    hkl: tuple[int, int, int],
) -> tuple[int, int, int]:
    h_asu, _ = reciprocal_asu.to_asu([int(hkl[0]), int(hkl[1]), int(hkl[2])], operations)
    return (int(h_asu[0]), int(h_asu[1]), int(h_asu[2]))


def asu_trace_signature(
    *,
    spacegroup_hm: str,
    R: Mat3,
    max_index: int,
    max_points_keep: int = 400,
) -> tuple[tuple[int, int, int], ...]:
    """
    Build a stable signature for the ASU trace of the reversing family induced by R.
    """
    sg = gemmi.SpaceGroup(spacegroup_hm)
    operations = sg.operations()
    reciprocal_asu = gemmi.ReciprocalAsu(sg)

    rinvt = transpose3(inv_int3(R))
    A = mat_add_I(rinvt)

    N = int(max_index)
    points: set[tuple[int, int, int]] = set()

    for h in range(-N, N + 1):
        for k in range(-N, N + 1):
            for l in range(-N, N + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                v = (int(h), int(k), int(l))
                Av = mat_vec_mul(A, v)
                if Av != (0, 0, 0):
                    continue
                points.add(asu_map_hkl(reciprocal_asu=reciprocal_asu, operations=operations, hkl=v))

    points_sorted = sorted(points)
    n_total = len(points_sorted)
    head = points_sorted[: int(max_points_keep)]
    head.append((10**9, int(n_total), 0))
    return tuple(head)


def count_distinct_asu_traces(
    *,
    spacegroup_hm: str,
    max_index: int,
) -> int:
    """
    Count distinct ASU-restricted traces for all reversing rotations.
    """
    signatures: set[tuple[tuple[int, int, int], ...]] = set()
    rotations = enumerate_reversing_rotations(spacegroup_hm=spacegroup_hm)
    for R in rotations:
        signature = asu_trace_signature(spacegroup_hm=spacegroup_hm, R=R, max_index=max_index)
        signatures.add(signature)
    return int(len(signatures))


# -------------------------
# Point-group label
# -------------------------

def point_group_label(spacegroup_hm: str) -> str:
    sg = gemmi.SpaceGroup(spacegroup_hm)
    if hasattr(sg, "point_group_hm"):
        return str(sg.point_group_hm())
    if hasattr(sg, "point_group"):
        pg = sg.point_group()
        if hasattr(pg, "hm"):
            return str(pg.hm())
        return str(pg)
    return "?"


# -------------------------
# Weighted summaries
# -------------------------

def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if not np.any(mask):
        return float("nan")
    return float(np.sum(w[mask] * v[mask]) / np.sum(w[mask]))


def sd_ddof1(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return float("nan")
    return float(np.std(x, ddof=1))


# -------------------------
# Dataset loading
# -------------------------

def load_dataset_table(*, project_dir: Path, counts_dir_rel: str) -> pd.DataFrame:
    """
    Load the per-space-group counts tables from stage 10 and concatenate them.
    """
    counts_dir = project_dir / counts_dir_rel
    paths = sorted(counts_dir.glob("SG__*.csv"))
    if len(paths) == 0:
        raise FileNotFoundError(f"No SG__*.csv files found under: {counts_dir}")

    df = pd.concat([pd.read_csv(filepath_or_buffer=path) for path in paths], ignore_index=True)

    for col in ["pdb_id", "spacegroup", "crystal_system"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "centric_completeness" in df.columns:
        df["centric_completeness"] = pd.to_numeric(df["centric_completeness"], errors="coerce")
    return df


def add_symmetry_counts(
    *,
    df: pd.DataFrame,
    asu_trace_max_index: int,
) -> pd.DataFrame:
    """
    Add:
      - n_rev_families
      - n_asu_traces
      - point_group
    """
    df_out = df.copy()
    sgs = sorted(df_out["spacegroup"].dropna().astype(str).unique().tolist())

    sg_to_families: dict[str, int] = {}
    sg_to_asu: dict[str, int] = {}
    sg_to_pg: dict[str, str] = {}

    for sg in sgs:
        logging.info("Computing symmetry complexity for SG=%s", sg)
        sg_to_families[sg] = count_distinct_reversing_families(spacegroup_hm=sg)
        sg_to_asu[sg] = count_distinct_asu_traces(spacegroup_hm=sg, max_index=int(asu_trace_max_index))
        sg_to_pg[sg] = point_group_label(spacegroup_hm=sg)

    df_out["n_rev_families"] = df_out["spacegroup"].astype(str).map(sg_to_families).astype(int)
    df_out["n_asu_traces"] = df_out["spacegroup"].astype(str).map(sg_to_asu).astype(int)
    df_out["point_group"] = df_out["spacegroup"].astype(str).map(sg_to_pg).astype(str)
    return df_out


# -------------------------
# Per-SG summarization
# -------------------------

def summarize_by_spacegroup(
    *,
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
) -> pd.DataFrame:
    req = {"pdb_id", "spacegroup", "crystal_system", "centric_completeness", "point_group", x_col, y_col}
    missing = sorted(list(req - set(df.columns)))
    if missing:
        raise ValueError(f"Dataset table missing required columns: {missing}")

    rows: list[dict[str, object]] = []
    for sg, group in df.groupby("spacegroup", sort=True):
        crystal_system = str(group["crystal_system"].iloc[0])
        point_group = str(group["point_group"].iloc[0])

        x_value = int(pd.to_numeric(group[x_col], errors="coerce").iloc[0])
        n_datasets = int(group["pdb_id"].nunique())

        y = pd.to_numeric(group[y_col], errors="coerce").to_numpy(dtype=float)
        centric_completeness = pd.to_numeric(group["centric_completeness"], errors="coerce").to_numpy(dtype=float)

        weights = np.clip(centric_completeness, 0.0, 1.0)
        weights[~np.isfinite(weights)] = 0.0

        y_mean = float(np.nanmean(y))
        y_sd = sd_ddof1(y)
        y_wmean = weighted_mean(y, weights)
        if not np.isfinite(y_wmean) and np.isfinite(y_mean):
            y_wmean = y_mean

        cc_mean = float(np.nanmean(centric_completeness)) if np.any(np.isfinite(centric_completeness)) else float("nan")
        cc_sd = sd_ddof1(centric_completeness)
        weight_sum = float(np.nansum(weights))

        rows.append(
            {
                "spacegroup": str(sg),
                "crystal_system": crystal_system,
                "point_group": point_group,
                x_col: x_value,
                "n_datasets": n_datasets,
                f"{y_col}_mean": y_mean,
                f"{y_col}_sd": y_sd,
                f"{y_col}_wmean": y_wmean,
                "centric_completeness_mean": cc_mean,
                "centric_completeness_sd": cc_sd,
                "weight_sum": weight_sum,
            }
        )

    out = pd.DataFrame(rows)
    out[x_col] = pd.to_numeric(out[x_col], errors="coerce").fillna(0).astype(int)
    return out


# -------------------------
# Plot helpers
# -------------------------

def infer_percent_mode(*, y_col: str) -> bool:
    return y_col.startswith("frac_") or y_col.startswith("delta_")


def annotate_point_groups_per_x(
    *,
    ax: plt.Axes,
    sg_df: pd.DataFrame,
    x_col: str,
    color_map: dict[str, str],
    annot_dx: float,
    annot_fontsize: int,
    annot_y_base_frac: float,
    annot_y_spacing_frac: float,
) -> None:
    x_vals = sg_df[x_col].to_numpy(dtype=int)
    systems = sg_df["crystal_system"].astype(str).tolist()
    point_groups = sg_df["point_group"].astype(str).tolist()

    y0, y1 = ax.get_ylim()
    yspan = max(1e-9, y1 - y0)
    y_base = y1 - float(annot_y_base_frac) * yspan
    dy = float(annot_y_spacing_frac) * yspan

    by_x: dict[int, list[tuple[str, str]]] = {}
    for x, pg, system in zip(x_vals, point_groups, systems):
        by_x.setdefault(int(x), [])
        if (pg, system) not in by_x[int(x)]:
            by_x[int(x)].append((pg, system))

    for x in sorted(by_x.keys()):
        items = sorted(by_x[x], key=lambda t: (t[1], t[0]))
        for j, (pg, system) in enumerate(items):
            ax.text(
                float(x) + float(annot_dx),
                float(y_base) - float(j) * dy,
                pg,
                fontsize=int(annot_fontsize),
                ha="left",
                va="center",
                color=color_map.get(system, "black"),
                alpha=0.95,
            )


def make_plot(
    *,
    sg_df: pd.DataFrame,
    x_col: str,
    x_label: str,
    y_col: str,
    y_is_percent: bool,
    out_png: Path,
    out_pdf: Path,
    annot_dx: float,
    annot_fontsize: int,
    annot_y_base_frac: float,
    annot_y_spacing_frac: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.8))

    systems = sorted(sg_df["crystal_system"].unique().tolist())
    color_map = {system: f"C{i}" for i, system in enumerate(systems)}

    sizes = 40.0 + 25.0 * np.sqrt(sg_df["n_datasets"].to_numpy(dtype=float))

    for system in systems:
        mask = sg_df["crystal_system"].to_numpy(dtype=object) == system
        sub = sg_df.loc[mask].copy()

        y_sub = sub[f"{y_col}_wmean"].to_numpy(dtype=float)
        if y_is_percent:
            y_sub = 100.0 * y_sub

        ax.scatter(
            sub[x_col].to_numpy(dtype=float),
            y_sub,
            s=sizes[mask],
            alpha=0.78,
            label=system,
            edgecolors="black",
            linewidths=0.6,
            c=color_map[system],
        )

    ax.set_xlabel(x_label)
    y_unit = " (%)" if y_is_percent else ""
    ax.set_ylabel(f"{y_col} — weighted mean across datasets{y_unit}\n(weights = centric completeness; NaN → 0)")
    ax.set_title(f"{x_label} vs. {y_col} (by space group)")

    xmin = int(sg_df[x_col].min())
    xmax = int(sg_df[x_col].max())
    ax.set_xlim(xmin - 0.5, xmax + 1.2)
    ax.set_xticks(np.arange(xmin, xmax + 1, 1, dtype=int))

    ax.tick_params(axis="both", which="major", length=MAJOR_TICK_SIZE)
    ax.tick_params(axis="both", which="minor", length=MINOR_TICK_SIZE)
    ax.minorticks_on()
    ax.grid(**GRID_MAJOR_KW)
    ax.grid(**GRID_MINOR_KW)

    ax.legend(frameon=True, loc="best", title="Crystal system")

    annotate_point_groups_per_x(
        ax=ax,
        sg_df=sg_df,
        x_col=x_col,
        color_map=color_map,
        annot_dx=float(annot_dx),
        annot_fontsize=int(annot_fontsize),
        annot_y_base_frac=float(annot_y_base_frac),
        annot_y_spacing_frac=float(annot_y_spacing_frac),
    )

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=250, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# SG report
# -------------------------

def build_sg_report_text(*, sg_df: pd.DataFrame, x_col: str, y_col: str, y_is_percent: bool) -> str:
    unit = "%" if y_is_percent else ""
    lines: list[str] = []
    lines.append(f"[STAGE-11] x={x_col} vs {y_col} ({'percent' if y_is_percent else 'raw'})\n")
    lines.append("Columns:\n")
    lines.append(
        f"  SG | system | pg | {x_col} | n_datasets | {y_col}_mean±sd{unit} | "
        f"{y_col}_wmean{unit} | centric_completeness_mean±sd | weight_sum\n"
    )

    tmp = sg_df.copy().sort_values(by=[x_col, "crystal_system", "spacegroup"], ascending=[True, True, True])
    for _, row in tmp.iterrows():
        sg = str(row["spacegroup"])
        crystal_system = str(row["crystal_system"])
        point_group = str(row["point_group"])
        x_value = int(row[x_col])
        n_datasets = int(row["n_datasets"])
        weight_sum = float(row["weight_sum"])

        mean_value = float(row.get(f"{y_col}_mean", np.nan))
        sd_value = float(row.get(f"{y_col}_sd", np.nan))
        wmean_value = float(row.get(f"{y_col}_wmean", np.nan))

        if y_is_percent:
            mean_value *= 100.0
            wmean_value *= 100.0
            if np.isfinite(sd_value):
                sd_value *= 100.0

        cc_mean = float(row.get("centric_completeness_mean", np.nan))
        cc_sd = float(row.get("centric_completeness_sd", np.nan))

        mean_sd_str = f"{mean_value:7.3f} ± {sd_value:7.3f}" if np.isfinite(sd_value) else f"{mean_value:7.3f} ±   nan"
        cc_str = f"{cc_mean:6.3f} ± {cc_sd:6.3f}" if np.isfinite(cc_sd) else f"{cc_mean:6.3f} ±   nan"

        lines.append(
            f"{sg:<12s} | {crystal_system:<10s} | {point_group:<4s} | {x_value:2d} | {n_datasets:4d} | "
            f"{mean_sd_str}{unit:1s} | {wmean_value:7.3f}{unit:1s} | {cc_str} | {weight_sum:8.3f}\n"
        )

    return "".join(lines)


# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot symmetry complexity versus centric-burden quantities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project_dir", required=True, type=Path)
    parser.add_argument(
        "--counts_dir",
        type=str,
        default="10_Empirical_Centric_Burden/01_Centric_Acentric_Counts_By_SG",
        help="Relative path under project_dir to per-SG counts CSVs.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="11_Symmetry_vs_Centric_Burden",
        help="Relative path under project_dir for outputs.",
    )

    parser.add_argument(
        "--x_axis_mode",
        type=str,
        default="asu_traces",
        choices=["families", "asu_traces", "both"],
        help="Which x-axis to use: full families, ASU traces, or both.",
    )
    parser.add_argument(
        "--asu_trace_max_index",
        type=int,
        default=6,
        help="Maximum |H|,|K|,|L| used to build ASU-trace signatures.",
    )

    parser.add_argument("--annot_dx", type=float, default=0.12)
    parser.add_argument("--annot_fontsize", type=int, default=10)
    parser.add_argument("--annot_y_base_frac", type=float, default=0.06)
    parser.add_argument("--annot_y_spacing_frac", type=float, default=0.05)

    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    out_dir = project_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = setup_logger(logs_dir=project_dir / "LOGS")
    logging.info("Run prefix: %s", RUN_PREFIX)
    logging.info("Command line: %s", " ".join(shlex.quote(arg) for arg in sys.argv))
    logging.info("project_dir=%s", project_dir)
    logging.info("counts_dir=%s", args.counts_dir)
    logging.info("out_dir=%s", out_dir)
    logging.info("x_axis_mode=%s", args.x_axis_mode)
    logging.info("asu_trace_max_index=%d", int(args.asu_trace_max_index))
    logging.info("annot_dx=%.3f", float(args.annot_dx))
    logging.info("annot_fontsize=%d", int(args.annot_fontsize))
    logging.info("annot_y_base_frac=%.3f", float(args.annot_y_base_frac))
    logging.info("annot_y_spacing_frac=%.3f", float(args.annot_y_spacing_frac))
    logging.info("Log file: %s", log_path)

    df_ds = load_dataset_table(project_dir=project_dir, counts_dir_rel=args.counts_dir)
    if not {"pdb_id", "spacegroup", "crystal_system"}.issubset(df_ds.columns):
        raise RuntimeError("Input per-SG CSVs must contain at least pdb_id, spacegroup, crystal_system columns.")
    logging.info("Loaded dataset table with %d rows", int(df_ds.shape[0]))

    df_ds = add_symmetry_counts(df=df_ds, asu_trace_max_index=int(args.asu_trace_max_index))
    logging.info("Added symmetry complexity counts to dataset table")

    dataset_table_path = out_dir / "dataset_symmetry_complexity_vs_centric_burden.csv"
    df_ds.to_csv(path_or_buf=dataset_table_path, index=False)
    logging.info("Wrote annotated dataset table: %s", dataset_table_path)

    y_cols = ["frac_centric_unique_obs", "frac_centric_unique_th", "delta_frac_unique", "frac_centric"]

    x_modes: list[tuple[str, str]] = []
    if args.x_axis_mode in ("families", "both"):
        x_modes.append(("n_rev_families", "Number of distinct reversing families in SG"))
    if args.x_axis_mode in ("asu_traces", "both"):
        x_modes.append(("n_asu_traces", "Number of distinct ASU traces  F(R)∩ASU"))

    for x_col, x_label in x_modes:
        logging.info("Processing x-axis mode: %s", x_col)
        for y_col in y_cols:
            if y_col not in df_ds.columns:
                logging.warning("Skipping missing y_col=%s", y_col)
                continue

            df_y = df_ds.copy()
            df_y[y_col] = pd.to_numeric(df_y[y_col], errors="coerce")
            df_y = df_y.dropna(subset=[y_col])
            logging.info("y_col=%s -> %d finite dataset rows", y_col, int(df_y.shape[0]))

            sg_df = summarize_by_spacegroup(df=df_y, y_col=y_col, x_col=x_col)
            y_is_percent = infer_percent_mode(y_col=y_col)

            tag = f"x_{x_col}"
            out_csv = out_dir / f"{tag}__vs_{y_col}__by_sg.csv"
            sg_df.to_csv(path_or_buf=out_csv, index=False)
            logging.info("Wrote SG summary table: %s", out_csv)

            out_png = out_dir / f"{tag}__vs_{y_col}__scatter.png"
            out_pdf = out_dir / f"{tag}__vs_{y_col}__scatter.pdf"

            make_plot(
                sg_df=sg_df,
                x_col=x_col,
                x_label=x_label,
                y_col=y_col,
                y_is_percent=y_is_percent,
                out_png=out_png,
                out_pdf=out_pdf,
                annot_dx=float(args.annot_dx),
                annot_fontsize=int(args.annot_fontsize),
                annot_y_base_frac=float(args.annot_y_base_frac),
                annot_y_spacing_frac=float(args.annot_y_spacing_frac),
            )
            logging.info("Wrote plot: %s", out_png)
            logging.info("Wrote plot: %s", out_pdf)

            sg_report = build_sg_report_text(sg_df=sg_df, x_col=x_col, y_col=y_col, y_is_percent=y_is_percent)
            report_path = out_dir / f"{tag}__vs_{y_col}__report.txt"
            with report_path.open(mode="w", encoding="utf-8") as file_handle:
                file_handle.write(sg_report)
            logging.info("Wrote report: %s", report_path)

            logging.info("[OK] %s vs %s", tag, y_col)

    logging.info("Done.")
    logging.info("Dataset-level annotated table: %s", dataset_table_path)
    logging.info("Notes:")
    logging.info(" - n_rev_families counts distinct families in full reciprocal space.")
    logging.info(" - n_asu_traces counts distinct ASU-restricted traces F(R)∩ASU; this can be smaller.")
    logging.info(" - ASU-trace signature depends mildly on ASU convention and on --asu_trace_max_index.")

if __name__ == "__main__":
    main()
