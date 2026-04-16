#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_analyze_trace_resolved_centric_phase_supports.py

Repository-ready stage-12 script for space-group-specific, trace-resolved
centric phase-support analysis within the empirical centric-burden stage.

Core idea
---------
For a chosen space group, the script:

1. Enumerates reversing operations (R|t) such that (I + R^{-T}) is singular.
2. Groups the resulting reciprocal-space centric loci into ASU traces.
3. For each centric HKL with a finite observed phase, determines whether the
   observed phase is compatible with one of the translation-consistent
   two-point centric supports implied by a reversing operation that fixes HKL:
       phi0(h) = 180° * frac(h·t)
       allowed = {phi0, phi0 + 180} wrapped to (-180, 180]
4. Summarizes trace-level occupancy and optionally computes parity partitions
   within traces to support actionable trace-conditioned priors.

This is an SG-specific mechanistic follow-up to the stage-10 empirical
centric-burden analysis.

Outputs
-------
- traces_index.json
- trace_table.csv
- dataset_summaries.csv
- invalid_centric_phase_dataset_ids.csv
- invalid_rows__all_traces.csv (optional)
- actionable_priors__CLEAN.csv/.md (optional)
- actionable_rules__CLEAN_unique.csv (optional)

Example
-------
python 12_analyze_trace_resolved_centric_phase_supports.py \
  --project_dir /path/to/project \
  --input_dir   /path/to/project/03_Run_Ecalc \
  --pattern     "*_rs_ecalc.csv" \
  --spacegroup  "P 43 21 2" \
  --trace_signature_index 8 \
  --write_invalid_rows_csv \
  --write_actionable_priors_report
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gemmi


# -------------------------
# Plot style
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
GRID_MAJOR_KW: dict[str, object] = {"which": "major", "linestyle": (0, (5, 5)), "color": "0.5", "linewidth": 0.8}
GRID_MINOR_KW: dict[str, object] = {"which": "minor", "linestyle": (0, (1, 4)), "color": "0.7", "linewidth": 0.6}

RUN_PREFIX = "12_analyze_trace_resolved_centric_phase_supports"


# -------------------------
# Logging
# -------------------------

def setup_logger(*, logs_dir: Path, run_prefix: str = RUN_PREFIX, sg_key: str) -> Path:
    """
    Configure logging to both a LOGS file and stdout.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{run_prefix}__{sanitize_for_filename(text=sg_key)}__{timestamp}.log"

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
# Helpers
# -------------------------

def sanitize_for_filename(*, text: str) -> str:
    s = str(text).strip()
    s = re.sub(pattern=r"\s+", repl="_", string=s)
    s = re.sub(pattern=r"[^A-Za-z0-9_\-\.]+", repl="_", string=s)
    return s


def normalize_sg_key(*, sg: str) -> str:
    return re.sub(pattern=r"\s+", repl="", string=str(sg).strip()).replace(":", "")


def extract_pdb_id_from_filename(*, csv_path: Path) -> str:
    base = csv_path.name
    if base.endswith("_rs_ecalc.csv"):
        return base.replace("_rs_ecalc.csv", "")
    if base.endswith("_rs.csv"):
        return base.replace("_rs.csv", "")
    return csv_path.stem


def validate_csv_minimal(*, df: pd.DataFrame, csv_path: Path, phase_col: str) -> None:
    required = ["H", "K", "L", "CENTRIC", phase_col, "SG"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"[{csv_path}] Missing required columns: {missing}")

    centric_values = pd.to_numeric(df["CENTRIC"], errors="coerce")
    if centric_values.isna().any():
        raise RuntimeError(f"[{csv_path}] CENTRIC has NaNs/non-numeric.")
    unique_values = set(centric_values.astype(int).unique().tolist())
    if not unique_values.issubset({0, 1}):
        raise RuntimeError(f"[{csv_path}] CENTRIC values outside {{0,1}}: {sorted(unique_values)}")

    phases = pd.to_numeric(df[phase_col], errors="coerce").dropna().to_numpy(dtype=float)
    if phases.size == 0:
        raise RuntimeError(f"[{csv_path}] {phase_col} has no finite values.")


def wrap_deg_pm180(*, phi_deg: float) -> int:
    x = float(phi_deg)
    x = ((x + 180.0) % 360.0) - 180.0
    if abs(x + 180.0) < 1e-9:
        x = 180.0
    return int(round(x))


# -------------------------
# Integer matrix utilities
# -------------------------

Mat3 = tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
Vec3 = tuple[int, int, int]


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


def mat_vec_mul(m: Mat3, v: Vec3) -> Vec3:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def normalize_rot_from_op(*, op: gemmi.Op) -> Mat3:
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


def enumerate_reversing_rotations(*, spacegroup_hm: str) -> list[Mat3]:
    sg = gemmi.SpaceGroup(spacegroup_hm)
    ops = sg.operations()

    rotations: list[Mat3] = []
    seen: set[Mat3] = set()
    for op in ops.sym_ops:
        R = normalize_rot_from_op(op=op)
        if R in seen:
            continue
        seen.add(R)
        rinvt = transpose3(inv_int3(R))
        A = mat_add_I(rinvt)
        if det3(A) != 0:
            continue
        rotations.append(R)
    return rotations


def asu_map_hkl(*, rasu: gemmi.ReciprocalAsu, ops: gemmi.GroupOps, hkl: Vec3) -> Vec3:
    h_asu, _ = rasu.to_asu([int(hkl[0]), int(hkl[1]), int(hkl[2])], ops)
    return (int(h_asu[0]), int(h_asu[1]), int(h_asu[2]))


def asu_trace_signature(
    *,
    spacegroup_hm: str,
    R: Mat3,
    max_index: int,
    max_points_keep: int = 400,
) -> tuple[tuple[int, int, int], ...]:
    sg = gemmi.SpaceGroup(spacegroup_hm)
    ops = sg.operations()
    rasu = gemmi.ReciprocalAsu(sg)

    rinvt = transpose3(inv_int3(R))
    A = mat_add_I(rinvt)

    N = int(max_index)
    points: set[Vec3] = set()

    for h in range(-N, N + 1):
        for k in range(-N, N + 1):
            for l in range(-N, N + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                v = (int(h), int(k), int(l))
                if mat_vec_mul(A, v) != (0, 0, 0):
                    continue
                points.add(asu_map_hkl(rasu=rasu, ops=ops, hkl=v))

    points_sorted = sorted(points)
    n_total = len(points_sorted)
    head = points_sorted[: int(max_points_keep)]
    head.append((10**9, int(n_total), 0))
    return tuple(head)


def short_hash_trace(*, sig: tuple[tuple[int, int, int], ...]) -> str:
    s = ";".join([f"{h},{k},{l}" for (h, k, l) in sig])
    h = 0
    for ch in s:
        h = (h * 1315423911 + ord(ch)) & 0xFFFFFFFF
    return f"{h:08x}"


# -------------------------
# Translation-layer support
# -------------------------

@dataclass(frozen=True)
class RevOp:
    R: Mat3
    t: tuple[Fraction, Fraction, Fraction]


def op_translation_fraction(*, op: gemmi.Op) -> tuple[Fraction, Fraction, Fraction]:
    den = int(op.DEN)
    tr = op.tran
    return (Fraction(int(tr[0]), den), Fraction(int(tr[1]), den), Fraction(int(tr[2]), den))


def enumerate_reversing_ops(*, spacegroup_hm: str) -> list[RevOp]:
    sg = gemmi.SpaceGroup(spacegroup_hm)
    ops = sg.operations()

    out: list[RevOp] = []
    for op in ops.sym_ops:
        R = normalize_rot_from_op(op=op)
        rinvt = transpose3(inv_int3(R))
        A = mat_add_I(rinvt)
        if det3(A) != 0:
            continue
        t = op_translation_fraction(op=op)
        out.append(RevOp(R=R, t=t))
    return out


def allowed_labels_for_hkl_and_op(*, hkl: Vec3, t: tuple[Fraction, Fraction, Fraction]) -> tuple[int, int]:
    H, K, L = int(hkl[0]), int(hkl[1]), int(hkl[2])
    ht = Fraction(H, 1) * t[0] + Fraction(K, 1) * t[1] + Fraction(L, 1) * t[2]
    frac_part = ht % 1
    phi0 = 180.0 * float(frac_part)
    a = wrap_deg_pm180(phi_deg=phi0)
    b = wrap_deg_pm180(phi_deg=phi0 + 180.0)
    return (a, b)


# -------------------------
# Trace labeling
# -------------------------

def normalize_int_triple(*, v: Vec3) -> Vec3:
    a, b, c = v
    g = abs(a)
    g = np.gcd(g, abs(b))
    g = np.gcd(g, abs(c))
    if g == 0:
        return (0, 0, 0)
    a //= int(g)
    b //= int(g)
    c //= int(g)

    for x in (a, b, c):
        if x < 0:
            a, b, c = (-a, -b, -c)
            break
        if x > 0:
            break
    return (int(a), int(b), int(c))


def cross(*, u: Vec3, v: Vec3) -> Vec3:
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


def infer_plane_normal_from_signature(*, sig: tuple[tuple[int, int, int], ...]) -> Optional[Vec3]:
    pts = [p for p in sig if p[0] != 10**9]
    if len(pts) < 3:
        return None

    pts = pts[:80]
    normals: dict[Vec3, int] = {}

    for i in range(min(25, len(pts))):
        for j in range(i + 1, min(25, len(pts))):
            n = cross(u=pts[i], v=pts[j])
            n = normalize_int_triple(v=n)
            if n == (0, 0, 0):
                continue
            normals[n] = normals.get(n, 0) + 1

    if not normals:
        return None

    best = sorted(normals.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return best


def label_trace_from_normal(*, n: Optional[Vec3]) -> str:
    if n is None:
        return "unlabeled"

    n = normalize_int_triple(v=n)
    if n == (1, 0, 0):
        return "H=0 (0KL)"
    if n == (0, 1, 0):
        return "K=0 (H0L)"
    if n == (0, 0, 1):
        return "L=0 (HK0)"
    if n == (1, -1, 0):
        return "H=K (diagonal)"
    if n == (1, 1, 0):
        return "H=-K (anti-diagonal)"
    if n == (1, 0, -1):
        return "H=L"
    if n == (0, 1, -1):
        return "K=L"
    a, b, c = n
    return f"{a}H+{b}K+{c}L=0"


# -------------------------
# Priors & unique rules
# -------------------------

@dataclass(frozen=True)
class PriorItem:
    item_index: int
    trace_index: int
    trace_label: str
    trace_hash: str
    condition: str
    phase_labels_deg: list[int]
    weights_mean: list[float]
    weights_sd: list[float]
    n_datasets_total: int


def mean_sd_percent(*, x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    if x.size == 1:
        return (float(x[0]), 0.0)
    return (float(np.mean(x)), float(np.std(x, ddof=1)))


def compute_overall_centric_fractions(*, df_sum: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df_sum.copy()
    for col in ["n_total_reflections", "n_centric_flagged", "n_centric_with_phase"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["n_total_reflections", "n_centric_flagged", "n_centric_with_phase"]).copy()
    df["n_total_reflections"] = df["n_total_reflections"].astype(int)
    df["n_centric_flagged"] = df["n_centric_flagged"].astype(int)
    df["n_centric_with_phase"] = df["n_centric_with_phase"].astype(int)

    df["pct_centric_flagged_total"] = 100.0 * df["n_centric_flagged"] / df["n_total_reflections"].replace({0: np.nan})
    df["pct_centric_with_phase_total"] = 100.0 * df["n_centric_with_phase"] / df["n_total_reflections"].replace({0: np.nan})

    per_ds = df[[
        "dataset_id",
        "n_total_reflections",
        "n_centric_flagged",
        "n_centric_with_phase",
        "pct_centric_flagged_total",
        "pct_centric_with_phase_total",
    ]].sort_values(by=["dataset_id"], ascending=True)

    m1, s1 = mean_sd_percent(x=per_ds["pct_centric_flagged_total"].to_numpy(dtype=float))
    m2, s2 = mean_sd_percent(x=per_ds["pct_centric_with_phase_total"].to_numpy(dtype=float))

    pooled = pd.DataFrame([{
        "n_datasets": int(per_ds.shape[0]),
        "mean_pct_centric_flagged_total": float(m1),
        "sd_pct_centric_flagged_total": float(s1),
        "mean_pct_centric_with_phase_total": float(m2),
        "sd_pct_centric_with_phase_total": float(s2),
    }])

    return per_ds, pooled


PARITY_RULES: list[tuple[str, str]] = [
    ("H", "H"),
    ("K", "K"),
    ("L", "L"),
    ("HplusK", "H + K"),
    ("HplusL", "H + L"),
    ("KplusL", "K + L"),
]


def compute_trace_parity_partitions(*, df_all: pd.DataFrame, df_sum: pd.DataFrame) -> pd.DataFrame:
    denom = df_sum[["dataset_id", "n_centric_with_phase"]].copy()
    denom["n_centric_with_phase"] = pd.to_numeric(denom["n_centric_with_phase"], errors="coerce")
    denom = denom.dropna(subset=["n_centric_with_phase"]).copy()
    denom["n_centric_with_phase"] = denom["n_centric_with_phase"].astype(int)

    dfp = df_all.copy()
    H = dfp["H_asu"].to_numpy(dtype=int)
    K = dfp["K_asu"].to_numpy(dtype=int)
    L = dfp["L_asu"].to_numpy(dtype=int)

    rows_out: list[pd.DataFrame] = []
    for rule_id, expr in PARITY_RULES:
        code = compile(expr, "<parity_expr>", "eval")
        val = eval(code, {"__builtins__": {}}, {"H": H, "K": K, "L": L, "np": np})
        val = np.asarray(val, dtype=int)
        bucket = np.where((val % 2) == 0, "even", "odd")

        tmp = dfp.copy()
        tmp["parity_rule"] = str(rule_id)
        tmp["parity_bucket"] = bucket

        g = (
            tmp.groupby(["dataset_id", "trace_hash", "trace_label", "parity_rule", "parity_bucket"])
            .size()
            .rename("n_in_bucket")
            .reset_index()
        )
        rows_out.append(g)

    if not rows_out:
        return pd.DataFrame()

    out = pd.concat(rows_out, ignore_index=True)
    out = out.merge(denom, on=["dataset_id"], how="left", validate="many_to_one")
    out["pct_of_centric_with_phase"] = 100.0 * out["n_in_bucket"] / out["n_centric_with_phase"].replace({0: np.nan})
    return out


def pool_trace_parity_partitions(*, df_tp: pd.DataFrame) -> pd.DataFrame:
    if df_tp is None or df_tp.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (rule, th, lbl, bucket), sub in df_tp.groupby(
        ["parity_rule", "trace_hash", "trace_label", "parity_bucket"], sort=False
    ):
        vals = sub["pct_of_centric_with_phase"].to_numpy(dtype=float)
        m, s = mean_sd_percent(x=vals)
        rows.append({
            "parity_rule": str(rule),
            "trace_hash": str(th),
            "trace_label": str(lbl),
            "parity_bucket": str(bucket),
            "n_datasets": int(sub["dataset_id"].nunique()),
            "mean_pct_of_centric_with_phase": float(m),
            "sd_pct_of_centric_with_phase": float(s),
        })
    return pd.DataFrame(rows)


def write_unique_actionable_rules_csv(
    *,
    out_root: Path,
    sg_key: str,
    pooled_overall: pd.DataFrame,
    pooled_tp: pd.DataFrame,
    prior_items: list[PriorItem],
    tracehash_to_index: dict[str, int],
) -> None:
    overall = pooled_overall.iloc[0].to_dict() if (pooled_overall is not None and not pooled_overall.empty) else {}

    pri_map: dict[tuple[str, str], PriorItem] = {}
    for item in prior_items:
        pri_map[(str(item.trace_hash), str(item.condition))] = item

    rows: list[dict[str, object]] = []

    if pooled_tp is not None and (not pooled_tp.empty):
        for _, row in pooled_tp.iterrows():
            th = str(row["trace_hash"])
            lbl = str(row["trace_label"])
            rule = str(row["parity_rule"])
            bucket = str(row["parity_bucket"])
            trace_index = int(tracehash_to_index.get(th, -1))

            condition = f"parity {rule} = {bucket}"
            prior = pri_map.get((th, condition), None)

            out_row = {
                "sg_key": str(sg_key),
                "rule_id": f"{sg_key}::T{trace_index:02d}::{lbl}::parity={rule}:{bucket}",
                "trace_index": int(trace_index),
                "trace_hash": str(th),
                "trace_label": str(lbl),
                "parity_rule": str(rule),
                "parity_bucket": str(bucket),
                "n_datasets": int(row["n_datasets"]),
                "mean_pct_centric_flagged_total": float(overall.get("mean_pct_centric_flagged_total", float("nan"))),
                "sd_pct_centric_flagged_total": float(overall.get("sd_pct_centric_flagged_total", float("nan"))),
                "mean_pct_centric_with_phase_total": float(overall.get("mean_pct_centric_with_phase_total", float("nan"))),
                "sd_pct_centric_with_phase_total": float(overall.get("sd_pct_centric_with_phase_total", float("nan"))),
                "mean_pct_of_centric_with_phase": float(row["mean_pct_of_centric_with_phase"]),
                "sd_pct_of_centric_with_phase": float(row["sd_pct_of_centric_with_phase"]),
            }

            if prior is not None:
                out_row["phase_labels_deg"] = ",".join(str(int(x)) for x in prior.phase_labels_deg)
                out_row["mean_weights"] = ",".join(f"{float(w):.6f}" for w in prior.weights_mean)
                out_row["sd_weights"] = ",".join(f"{float(w):.6f}" for w in prior.weights_sd)
            else:
                out_row["phase_labels_deg"] = ""
                out_row["mean_weights"] = ""
                out_row["sd_weights"] = ""

            rows.append(out_row)

    df_rules = pd.DataFrame(rows)
    df_rules.to_csv(out_root / "actionable_rules__CLEAN_unique.csv", index=False)


def write_actionable_priors_report(
    *,
    out_root: Path,
    sg_key: str,
    prior_items: list[PriorItem],
    invalid_dataset_ids: list[str],
    notes: list[str],
) -> None:
    rows_csv: list[dict[str, object]] = []
    for item in prior_items:
        rows_csv.append({
            "item": int(item.item_index),
            "trace_index": int(item.trace_index),
            "trace_label": item.trace_label,
            "trace_hash": item.trace_hash,
            "condition": item.condition,
            "phase_labels_deg": ",".join(str(int(x)) for x in item.phase_labels_deg),
            "mean_weights": ",".join(f"{w:.6f}" for w in item.weights_mean),
            "sd_weights": ",".join(f"{w:.6f}" for w in item.weights_sd),
            "n_datasets_total": int(item.n_datasets_total),
        })
    pd.DataFrame(rows_csv).to_csv(out_root / "actionable_priors__CLEAN.csv", index=False)

    lines: list[str] = []
    lines.append(f"# Actionable centric phase-label priors (CLEAN) for SG {sg_key}")
    lines.append("")
    for note in notes:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Priors")
    lines.append("")
    for item in prior_items:
        lines.append(f"({item.item_index}) Trace T{item.trace_index:02d} — {item.trace_label} | hash={item.trace_hash}")
        lines.append(f"- Condition: {item.condition}")
        lines.append(f"- Labels: {item.phase_labels_deg}")
        lines.append("")
    lines.append("## Invalid datasets (DIRTY outliers)")
    lines.append(", ".join(invalid_dataset_ids) if invalid_dataset_ids else "None")
    with (out_root / "actionable_priors__CLEAN.md").open(mode="w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(lines))


# -------------------------
# Main
# -------------------------

@dataclass(frozen=True)
class TraceInfo:
    trace_hash: str
    label: str
    n_rotations_merged: int
    rotations: list[Mat3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze trace-resolved centric phase supports for a chosen space group.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project_dir", required=True, type=Path)
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*_rs_ecalc.csv")
    parser.add_argument("--spacegroup", required=True)
    parser.add_argument("--out_root_subdir", default="12_Trace_Resolved_Centric_Phase_Supports")
    parser.add_argument("--trace_signature_index", type=int, default=8)
    parser.add_argument("--min_datasets_per_trace_plot", type=int, default=10)  # compatibility only
    parser.add_argument("--phase_col", default="PHIC_ALL")
    parser.add_argument("--dataset_id_mode", choices=["filename", "column"], default="filename")
    parser.add_argument("--dataset_id_col", default="pdb_id")
    parser.add_argument("--write_invalid_rows_csv", action="store_true")
    parser.add_argument("--write_actionable_priors_report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_dir = args.project_dir.resolve()
    input_dir = args.input_dir.resolve()
    all_paths = sorted(input_dir.glob(args.pattern))
    if not all_paths:
        raise RuntimeError(f"No files matched: {input_dir / args.pattern}")

    target_sg_key = normalize_sg_key(sg=args.spacegroup)

    log_path = setup_logger(logs_dir=project_dir / "LOGS", sg_key=target_sg_key)
    logging.info("Run prefix: %s", RUN_PREFIX)
    logging.info("Command line: %s", " ".join(shlex.quote(arg) for arg in sys.argv))
    logging.info("project_dir=%s", project_dir)
    logging.info("input_dir=%s", input_dir)
    logging.info("pattern=%s", args.pattern)
    logging.info("spacegroup_raw=%s", args.spacegroup)
    logging.info("spacegroup_key=%s", target_sg_key)
    logging.info("out_root_subdir=%s", args.out_root_subdir)
    logging.info("trace_signature_index=%d", int(args.trace_signature_index))
    logging.info("phase_col=%s", args.phase_col)
    logging.info("dataset_id_mode=%s", args.dataset_id_mode)
    logging.info("dataset_id_col=%s", args.dataset_id_col)
    logging.info("write_invalid_rows_csv=%s", bool(args.write_invalid_rows_csv))
    logging.info("write_actionable_priors_report=%s", bool(args.write_actionable_priors_report))
    logging.info("Log file: %s", log_path)

    out_root = project_dir / args.out_root_subdir / target_sg_key
    out_root.mkdir(parents=True, exist_ok=True)
    logging.info("Output root: %s", out_root)

    matched_paths: list[Path] = []
    for path in all_paths:
        try:
            dfsg = pd.read_csv(filepath_or_buffer=path, usecols=["SG"])
            sg = str(dfsg["SG"].dropna().iloc[0])
            if normalize_sg_key(sg=sg) == target_sg_key:
                matched_paths.append(path)
        except Exception:
            continue

    if not matched_paths:
        raise RuntimeError(f"No datasets found for spacegroup '{args.spacegroup}' (key={target_sg_key}).")

    logging.info("Total files matched SG: %d", len(matched_paths))

    sg_hm = args.spacegroup
    sg_obj = gemmi.SpaceGroup(sg_hm)
    ops = sg_obj.operations()
    rasu = gemmi.ReciprocalAsu(sg_obj)

    rotations = enumerate_reversing_rotations(spacegroup_hm=sg_hm)
    rev_ops = enumerate_reversing_ops(spacegroup_hm=sg_hm)
    logging.info("Reversing rotations: %d | Reversing ops (R|t): %d", len(rotations), len(rev_ops))

    sig_to_rotlist: dict[tuple[tuple[int, int, int], ...], list[Mat3]] = {}
    for R in rotations:
        sig = asu_trace_signature(spacegroup_hm=sg_hm, R=R, max_index=int(args.trace_signature_index))
        sig_to_rotlist.setdefault(sig, []).append(R)

    trace_infos: list[TraceInfo] = []
    tracehash_to_info: dict[str, TraceInfo] = {}
    rotation_to_tracehash: dict[Mat3, str] = {}

    for sig, rlist in sig_to_rotlist.items():
        trace_hash = short_hash_trace(sig=sig)
        normal = infer_plane_normal_from_signature(sig=sig)
        label = label_trace_from_normal(n=normal)
        for R in rlist:
            rotation_to_tracehash[R] = trace_hash
        info = TraceInfo(
            trace_hash=trace_hash,
            label=label,
            n_rotations_merged=int(len(rlist)),
            rotations=rlist,
        )
        trace_infos.append(info)
        tracehash_to_info[trace_hash] = info

    trace_infos = sorted(trace_infos, key=lambda x: (x.label, x.trace_hash))
    tracehash_to_index: dict[str, int] = {info.trace_hash: i + 1 for i, info in enumerate(trace_infos)}

    tracehash_to_revops: dict[str, list[RevOp]] = {info.trace_hash: [] for info in trace_infos}
    for op_item in rev_ops:
        trace_hash = rotation_to_tracehash.get(op_item.R, None)
        if trace_hash is not None:
            tracehash_to_revops[trace_hash].append(op_item)

    traces_index = {
        "spacegroup_raw": sg_hm,
        "spacegroup_key": target_sg_key,
        "trace_signature_index": int(args.trace_signature_index),
        "traces": [
            {
                "trace_index": int(tracehash_to_index[info.trace_hash]),
                "trace_hash": info.trace_hash,
                "label": info.label,
                "n_rotations_merged": info.n_rotations_merged,
                "n_reversing_ops_in_trace": int(len(tracehash_to_revops.get(info.trace_hash, []))),
            }
            for info in trace_infos
        ],
    }
    with (out_root / "traces_index.json").open(mode="w", encoding="utf-8") as file_handle:
        json.dump(traces_index, file_handle, indent=2)
    logging.info("Wrote traces index: %s", out_root / "traces_index.json")

    traceA_list: list[tuple[Mat3, str]] = []
    for info in trace_infos:
        for R in info.rotations:
            rinvt = transpose3(inv_int3(R))
            A = mat_add_I(rinvt)
            traceA_list.append((A, info.trace_hash))

    rows_all: list[dict[str, object]] = []
    dataset_summaries: list[dict[str, object]] = []
    invalid_dataset_ids: set[str] = set()
    invalid_rows_all: list[dict[str, object]] = []

    for path in matched_paths:
        try:
            df = pd.read_csv(filepath_or_buffer=path)
            validate_csv_minimal(df=df, csv_path=path, phase_col=args.phase_col)

            if args.dataset_id_mode == "filename":
                dataset_id = extract_pdb_id_from_filename(csv_path=path)
            else:
                dataset_id = str(df[args.dataset_id_col].dropna().iloc[0])

            sg_here = str(df["SG"].dropna().iloc[0])
            if normalize_sg_key(sg=sg_here) != target_sg_key:
                continue

            n_total_reflections = int(df.shape[0])
            centric = pd.to_numeric(df["CENTRIC"], errors="coerce").astype(int).to_numpy()
            n_centric_flagged = int(np.sum(centric == 1))

            phi_raw = pd.to_numeric(df[args.phase_col], errors="coerce").to_numpy(dtype=float)
            mask = (centric == 1) & np.isfinite(phi_raw)
            hkls = df.loc[mask, ["H", "K", "L"]].astype(int).to_numpy()
            phi_sel = phi_raw[mask]
            n_centric_with_phase = int(hkls.shape[0])

            n_unassigned = 0
            n_multi = 0
            n_invalid = 0

            for (H, K, L), ph in zip(hkls, phi_sel):
                v: Vec3 = (int(H), int(K), int(L))
                ph_obs = wrap_deg_pm180(phi_deg=float(ph))

                matched_hashes: list[str] = []
                for A, trace_hash in traceA_list:
                    if mat_vec_mul(A, v) == (0, 0, 0):
                        matched_hashes.append(trace_hash)

                if len(matched_hashes) == 0:
                    n_unassigned += 1
                    continue

                unique_hashes = sorted(set(matched_hashes))
                if len(unique_hashes) > 1:
                    n_multi += 1
                trace_hash_pick = unique_hashes[0]

                h_asu = asu_map_hkl(rasu=rasu, ops=ops, hkl=v)

                allowed_set: set[int] = set()
                allowed_ok = False

                for op_item in tracehash_to_revops.get(trace_hash_pick, []):
                    rinvt = transpose3(inv_int3(op_item.R))
                    Aop = mat_add_I(rinvt)
                    if mat_vec_mul(Aop, v) != (0, 0, 0):
                        continue
                    a, b = allowed_labels_for_hkl_and_op(hkl=v, t=op_item.t)
                    allowed_set.add(a)
                    allowed_set.add(b)
                    if ph_obs == a or ph_obs == b:
                        allowed_ok = True
                        break

                if not allowed_ok:
                    invalid_dataset_ids.add(dataset_id)
                    n_invalid += 1
                    invalid_rows_all.append({
                        "dataset_id": dataset_id,
                        "trace_hash": trace_hash_pick,
                        "trace_label": tracehash_to_info[trace_hash_pick].label,
                        "H": int(H), "K": int(K), "L": int(L),
                        "H_asu": int(h_asu[0]), "K_asu": int(h_asu[1]), "L_asu": int(h_asu[2]),
                        "phase_deg": int(ph_obs),
                        "allowed_labels": ",".join(str(x) for x in sorted(allowed_set)),
                    })

                rows_all.append({
                    "dataset_id": dataset_id,
                    "trace_hash": trace_hash_pick,
                    "trace_label": tracehash_to_info[trace_hash_pick].label,
                    "H_asu": int(h_asu[0]), "K_asu": int(h_asu[1]), "L_asu": int(h_asu[2]),
                    "phase_deg": int(ph_obs),
                    "phase_allowed": bool(allowed_ok),
                })

            dataset_summaries.append({
                "dataset_id": dataset_id,
                "csv_path": str(path),
                "spacegroup": sg_here,
                "n_total_reflections": n_total_reflections,
                "n_centric_flagged": n_centric_flagged,
                "n_centric_with_phase": n_centric_with_phase,
                "n_unassigned_trace": n_unassigned,
                "n_multi_match": n_multi,
                "n_invalid_phase": n_invalid,
            })
            logging.info(
                "[DATASET] %s | n_total=%d | n_centric_flagged=%d | n_centric_with_phase=%d | invalid=%d",
                dataset_id,
                n_total_reflections,
                n_centric_flagged,
                n_centric_with_phase,
                n_invalid,
            )

        except Exception as exc:
            logging.exception("FAILED %s: %s", path, str(exc))

    df_all = pd.DataFrame(rows_all)
    df_sum = pd.DataFrame(dataset_summaries)

    df_sum.to_csv(out_root / "dataset_summaries.csv", index=False)
    logging.info("Wrote dataset summaries: %s", out_root / "dataset_summaries.csv")

    pd.DataFrame({"dataset_id": sorted(invalid_dataset_ids)}).to_csv(
        out_root / "invalid_centric_phase_dataset_ids.csv", index=False
    )
    logging.info("Wrote invalid dataset IDs: %s", out_root / "invalid_centric_phase_dataset_ids.csv")

    if args.write_invalid_rows_csv:
        pd.DataFrame(invalid_rows_all).to_csv(out_root / "invalid_rows__all_traces.csv", index=False)
        logging.info("Wrote invalid rows CSV: %s", out_root / "invalid_rows__all_traces.csv")

    if df_all.empty:
        raise RuntimeError("No assigned centric reflections produced.")

    trace_table = (
        df_all.groupby(["trace_hash", "trace_label"])["phase_deg"]
        .size()
        .rename("n_centric")
        .reset_index()
        .sort_values(["n_centric"], ascending=False)
    )
    trace_table.to_csv(out_root / "trace_table.csv", index=False)
    logging.info("Wrote trace table: %s", out_root / "trace_table.csv")

    prior_items: list[PriorItem] = []
    item_idx = 0

    for (trace_hash, trace_label), sub in df_all[df_all["phase_allowed"] == True].groupby(["trace_hash", "trace_label"]):
        trace_index = int(tracehash_to_index.get(trace_hash, -1))
        g = sub.groupby(["dataset_id", "phase_deg"]).size().rename("count").reset_index()
        totals = g.groupby("dataset_id")["count"].sum().rename("total").reset_index()
        g = g.merge(totals, on="dataset_id", how="left")
        g["fraction"] = g["count"] / g["total"].replace({0: np.nan})

        piv = g.pivot_table(index="dataset_id", columns="phase_deg", values="fraction", aggfunc="sum", fill_value=0.0)
        phases = sorted([int(x) for x in piv.columns.tolist()])
        weights_mean = [float(np.mean(piv[p].values)) for p in phases]
        weights_sd = [float(np.std(piv[p].values, ddof=1)) if piv.shape[0] > 1 else 0.0 for p in phases]

        item_idx += 1
        prior_items.append(PriorItem(
            item_index=item_idx,
            trace_index=trace_index,
            trace_label=str(trace_label),
            trace_hash=str(trace_hash),
            condition="all (no parity split)",
            phase_labels_deg=phases,
            weights_mean=weights_mean,
            weights_sd=weights_sd,
            n_datasets_total=int(piv.shape[0]),
        ))

    if args.write_actionable_priors_report:
        notes = [
            f"Space group raw: {sg_hm}",
            f"Datasets matched: {len(matched_paths)}",
            f"Assigned centric reflections: {int(df_all.shape[0])}",
        ]
        write_actionable_priors_report(
            out_root=out_root,
            sg_key=target_sg_key,
            prior_items=prior_items,
            invalid_dataset_ids=sorted(invalid_dataset_ids),
            notes=notes,
        )
        logging.info("Wrote actionable priors report files")

        _, pooled_overall = compute_overall_centric_fractions(df_sum=df_sum)
        df_tp = compute_trace_parity_partitions(df_all=df_all, df_sum=df_sum)
        pooled_tp = pool_trace_parity_partitions(df_tp=df_tp)

        write_unique_actionable_rules_csv(
            out_root=out_root,
            sg_key=target_sg_key,
            pooled_overall=pooled_overall,
            pooled_tp=pooled_tp,
            prior_items=prior_items,
            tracehash_to_index=tracehash_to_index,
        )
        logging.info("Wrote actionable rules CSV")

    logging.info("DONE. Output root: %s", out_root)


if __name__ == "__main__":
    main()
