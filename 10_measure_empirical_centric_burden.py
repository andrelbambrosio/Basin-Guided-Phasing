#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_measure_empirical_centric_burden.py

Repository-ready stage-10 script for measuring empirical centric burden across
a cohort of reflection-table CSV files.

This script computes:
- empirical centric phase populations
- centric and acentric reflection counts
- observed unique-HKL centric fractions
- theoretical unique-HKL centric fractions
- overall and centric completeness metrics
- top-10 space-group reports

Inputs:
- per-dataset reflection CSV files, typically from stage 01
  or later stage outputs (for example, *_rs.csv or *_rs_ecalc.csv)

Outputs (relative to --project_dir / --out_subdir):
- 01_Centric_Acentric_Counts_By_SG/
- 02_Centric_Phase_Counts_By_SG/
- 03_Top10_SG_Reports/

Example:
    python 10_measure_empirical_centric_burden.py \
        --project_dir . \
        --input_dir ./03_Run_Ecalc \
        --pattern "*_rs_ecalc.csv" \
        --num_processes 16
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import math
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

try:
    import gemmi
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("This script requires gemmi (pip install gemmi).") from exc


RUN_PREFIX = "10_measure_empirical_centric_burden"


# ---------------------------
# Logging
# ---------------------------

def now_local() -> datetime:
    """Return current local time."""
    return datetime.now()


def setup_logger(*, log_path: Path) -> None:
    """Configure root logger to file plus stdout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s")
    console_formatter = logging.Formatter(fmt="%(levelname)s | %(message)s")

    file_handler = logging.FileHandler(filename=log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt=file_formatter)
    logger.addHandler(hdlr=file_handler)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt=console_formatter)
    logger.addHandler(hdlr=console_handler)


# ---------------------------
# Validation
# ---------------------------

def validate_required_columns(
    *,
    df: pd.DataFrame,
    csv_path: Path,
    required_cols: Sequence[str],
) -> list[str]:
    errors: list[str] = []
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"[{csv_path}] Missing required column: '{col}'")
    return errors


def validate_centric_column(
    *,
    df: pd.DataFrame,
    csv_path: Path,
    centric_col: str,
) -> list[str]:
    errors: list[str] = []
    try:
        values = pd.to_numeric(df[centric_col], errors="coerce")
    except Exception as exc:
        return [f"[{csv_path}] CENTRIC column '{centric_col}' not numeric: {exc}"]

    if values.isna().any():
        n_bad = int(values.isna().sum())
        errors.append(f"[{csv_path}] CENTRIC column '{centric_col}' has {n_bad} NaN/non-numeric values")

    unique_values = set(values.dropna().astype(int).unique().tolist())
    if not unique_values.issubset({0, 1}):
        errors.append(
            f"[{csv_path}] CENTRIC column '{centric_col}' contains values outside {{0,1}}: {sorted(unique_values)}"
        )

    return errors


def validate_phase_canonical(
    *,
    df: pd.DataFrame,
    csv_path: Path,
    phase_col: str,
) -> list[str]:
    """
    Require pre-canonicalized phases in (-180, 180] with -180 forbidden.
    """
    errors: list[str] = []
    try:
        phases = pd.to_numeric(df[phase_col], errors="coerce")
    except Exception as exc:
        return [f"[{csv_path}] Phase column '{phase_col}' not numeric: {exc}"]

    finite = phases.dropna()
    if finite.size == 0:
        errors.append(f"[{csv_path}] Phase column '{phase_col}' has no finite values")
        return errors

    arr = finite.to_numpy(dtype=float)

    forbidden_mask = np.isclose(arr, -180.0, atol=1e-12)
    if bool(np.any(forbidden_mask)):
        n_forbidden = int(np.sum(forbidden_mask))
        errors.append(f"[{csv_path}] Phase column '{phase_col}' contains {n_forbidden} forbidden value(s) == -180°")

    out_low = arr <= -180.0
    out_high = arr > 180.0
    if bool(np.any(out_low)):
        n = int(np.sum(out_low))
        errors.append(f"[{csv_path}] Phase column '{phase_col}' has {n} value(s) <= -180° (must be > -180°)")
    if bool(np.any(out_high)):
        n = int(np.sum(out_high))
        errors.append(f"[{csv_path}] Phase column '{phase_col}' has {n} value(s) > 180° (must be <= 180°)")

    return errors


def infer_spacegroup_from_df(
    *,
    df: pd.DataFrame,
    spacegroup_col_candidates: Sequence[str],
) -> Optional[str]:
    for col in spacegroup_col_candidates:
        if col in df.columns:
            series = df[col].dropna().astype(str)
            if series.size > 0:
                return str(series.value_counts().idxmax())
    return None


def validate_one_csv(
    *,
    csv_path: Path,
    phase_col: str,
    centric_col: str,
    h_col: str,
    k_col: str,
    l_col: str,
    spacegroup_cols: Sequence[str],
    cell_cols: Sequence[str],
) -> tuple[bool, list[str]]:
    try:
        df = pd.read_csv(filepath_or_buffer=csv_path)
    except Exception as exc:
        return False, [f"[{csv_path}] Failed to read CSV: {exc}"]

    errors: list[str] = []
    errors.extend(
        validate_required_columns(
            df=df,
            csv_path=csv_path,
            required_cols=[h_col, k_col, l_col, centric_col, phase_col, *cell_cols],
        )
    )
    if errors:
        return False, errors

    if infer_spacegroup_from_df(df=df, spacegroup_col_candidates=spacegroup_cols) is None:
        errors.append(f"[{csv_path}] Could not infer space group from columns: {list(spacegroup_cols)}")

    errors.extend(validate_centric_column(df=df, csv_path=csv_path, centric_col=centric_col))
    errors.extend(validate_phase_canonical(df=df, csv_path=csv_path, phase_col=phase_col))
    return (len(errors) == 0), errors


# ---------------------------
# Crystal system helper
# ---------------------------

def crystal_system_from_spacegroup(*, spacegroup_hm: str) -> str:
    sg = gemmi.SpaceGroup(spacegroup_hm)
    if hasattr(sg, "crystal_system_str"):
        return str(sg.crystal_system_str())

    num = int(sg.number)
    if num <= 2:
        return "triclinic"
    if num <= 15:
        return "monoclinic"
    if num <= 74:
        return "orthorhombic"
    if num <= 142:
        return "tetragonal"
    if num <= 167:
        return "trigonal"
    if num <= 194:
        return "hexagonal"
    return "cubic"


# ---------------------------
# Utilities
# ---------------------------

def extract_pdb_id_from_filename(*, csv_path: Path) -> str:
    base = csv_path.name
    for suffix in ("_rs_ecalc.csv", "_rs.csv"):
        if base.endswith(suffix):
            return base.replace(suffix, "")
    return csv_path.stem


def sanitize_for_filename(*, text: str) -> str:
    s = text.strip()
    s = re.sub(pattern=r"\s+", repl="_", string=s)
    s = re.sub(pattern=r"[^A-Za-z0-9_\-\.]+", repl="_", string=s)
    return s


def safe_ratio(*, num: float, den: float) -> float:
    if den == 0.0:
        return float("nan")
    return float(num / den)


def build_unit_cell_from_df(
    *,
    df: pd.DataFrame,
    a_col: str,
    b_col: str,
    c_col: str,
    alpha_col: str,
    beta_col: str,
    gamma_col: str,
) -> gemmi.UnitCell:
    return gemmi.UnitCell(
        float(df[a_col].iloc[0]),
        float(df[b_col].iloc[0]),
        float(df[c_col].iloc[0]),
        float(df[alpha_col].iloc[0]),
        float(df[beta_col].iloc[0]),
        float(df[gamma_col].iloc[0]),
    )


# ---------------------------
# Completeness
# ---------------------------

_THEORY_CACHE: dict[
    tuple[str, tuple[float, float, float, float, float, float], float, float],
    tuple[int, int]
] = {}


def cell_key(*, cell: gemmi.UnitCell) -> tuple[float, float, float, float, float, float]:
    return (
        round(float(cell.a), 6),
        round(float(cell.b), 6),
        round(float(cell.c), 6),
        round(float(cell.alpha), 6),
        round(float(cell.beta), 6),
        round(float(cell.gamma), 6),
    )


def theoretical_unique_counts_in_range(
    *,
    spacegroup_hm: str,
    cell: gemmi.UnitCell,
    d_min: float,
    d_max: float,
) -> tuple[int, int]:
    """
    Return (n_theory_unique_total, n_theory_unique_centric) for the given SG/cell/window,
    counting unique HKLs reduced to reciprocal ASU, excluding systematic absences.
    """
    key = (str(spacegroup_hm), cell_key(cell=cell), float(d_min), float(d_max))
    if key in _THEORY_CACHE:
        return _THEORY_CACHE[key]

    sg = gemmi.SpaceGroup(spacegroup_hm)
    operations = sg.operations()
    reciprocal_asu = gemmi.ReciprocalAsu(sg)

    max_h = int(math.ceil(cell.a / d_min)) + 1
    max_k = int(math.ceil(cell.b / d_min)) + 1
    max_l = int(math.ceil(cell.c / d_min)) + 1

    total_asu: set[tuple[int, int, int]] = set()
    centric_asu: set[tuple[int, int, int]] = set()

    for h in range(-max_h, max_h + 1):
        for k in range(-max_k, max_k + 1):
            for l in range(-max_l, max_l + 1):
                if h == 0 and k == 0 and l == 0:
                    continue

                d = float(cell.calculate_d([int(h), int(k), int(l)]))
                if not (float(d_min) <= d <= float(d_max)):
                    continue

                if operations.is_systematically_absent([int(h), int(k), int(l)]):
                    continue

                h_asu, _ = reciprocal_asu.to_asu([int(h), int(k), int(l)], operations)
                hku = (int(h_asu[0]), int(h_asu[1]), int(h_asu[2]))
                total_asu.add(hku)

                if operations.is_reflection_centric([int(h), int(k), int(l)]):
                    centric_asu.add(hku)

    n_total = int(len(total_asu))
    n_centric = int(len(centric_asu))
    _THEORY_CACHE[key] = (n_total, n_centric)
    return n_total, n_centric


def observed_unique_asu_sets(
    *,
    df: pd.DataFrame,
    spacegroup_hm: str,
    centric_col: str,
    h_col: str,
    k_col: str,
    l_col: str,
) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]]:
    """
    Return (obs_total_asu, obs_centric_asu) as unique HKLs in reciprocal ASU,
    excluding systematic absences.
    """
    sg = gemmi.SpaceGroup(spacegroup_hm)
    operations = sg.operations()
    reciprocal_asu = gemmi.ReciprocalAsu(sg)

    hkls = df[[h_col, k_col, l_col]].dropna().astype(int).to_numpy()
    centric_vals = pd.to_numeric(df[centric_col], errors="coerce").astype(int).to_numpy()

    obs_total_asu: set[tuple[int, int, int]] = set()
    obs_centric_asu: set[tuple[int, int, int]] = set()

    for (h, k, l), cflag in zip(hkls, centric_vals):
        if h == 0 and k == 0 and l == 0:
            continue
        if operations.is_systematically_absent([int(h), int(k), int(l)]):
            continue

        h_asu, _ = reciprocal_asu.to_asu([int(h), int(k), int(l)], operations)
        hku = (int(h_asu[0]), int(h_asu[1]), int(h_asu[2]))

        obs_total_asu.add(hku)
        if int(cflag) == 1:
            obs_centric_asu.add(hku)

    return obs_total_asu, obs_centric_asu


# ---------------------------
# Worker
# ---------------------------

def process_one_csv_worker(
    *,
    csv_path: Path,
    phase_col: str,
    centric_col: str,
    spacegroup_cols: Sequence[str],
    a_col: str,
    b_col: str,
    c_col: str,
    alpha_col: str,
    beta_col: str,
    gamma_col: str,
    d_min: float,
    d_max: float,
    h_col: str,
    k_col: str,
    l_col: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    df = pd.read_csv(filepath_or_buffer=csv_path)

    sg = infer_spacegroup_from_df(df=df, spacegroup_col_candidates=spacegroup_cols)
    if sg is None:
        raise RuntimeError(f"SG not found for {csv_path} (should not happen after validation).")

    crystal_system = crystal_system_from_spacegroup(spacegroup_hm=sg)
    pdb_id = extract_pdb_id_from_filename(csv_path=csv_path)

    cell = build_unit_cell_from_df(
        df=df,
        a_col=a_col,
        b_col=b_col,
        c_col=c_col,
        alpha_col=alpha_col,
        beta_col=beta_col,
        gamma_col=gamma_col,
    )

    n_total_rows = int(df.shape[0])
    centric_vals = pd.to_numeric(df[centric_col], errors="coerce").astype(int)
    n_centric_rows = int((centric_vals == 1).sum())
    n_acentric_rows = int((centric_vals == 0).sum())

    obs_total_asu, obs_centric_asu = observed_unique_asu_sets(
        df=df,
        spacegroup_hm=str(sg),
        centric_col=centric_col,
        h_col=h_col,
        k_col=k_col,
        l_col=l_col,
    )
    n_obs_unique_total = int(len(obs_total_asu))
    n_obs_unique_centric = int(len(obs_centric_asu))

    n_theory_total, n_theory_centric = theoretical_unique_counts_in_range(
        spacegroup_hm=str(sg),
        cell=cell,
        d_min=float(d_min),
        d_max=float(d_max),
    )

    overall_completeness = (float(n_obs_unique_total) / float(n_theory_total)) if n_theory_total > 0 else float("nan")
    centric_completeness = (float(n_obs_unique_centric) / float(n_theory_centric)) if n_theory_centric > 0 else float("nan")

    frac_centric_unique_obs = (float(n_obs_unique_centric) / float(n_obs_unique_total)) if n_obs_unique_total > 0 else float("nan")
    frac_centric_unique_th = (float(n_theory_centric) / float(n_theory_total)) if n_theory_total > 0 else float("nan")
    delta_frac_unique = (
        frac_centric_unique_obs - frac_centric_unique_th
        if (np.isfinite(frac_centric_unique_obs) and np.isfinite(frac_centric_unique_th))
        else float("nan")
    )

    df_centric = df.loc[centric_vals == 1].copy()
    phases = pd.to_numeric(df_centric[phase_col], errors="coerce").dropna()
    phase_int = phases.astype(int).replace(to_replace=-180, value=180)
    n_centric_with_phase_rows = int(phase_int.size)

    counts = phase_int.value_counts().sort_index()
    centric_hist_df = pd.DataFrame(
        {
            "pdb_id": pdb_id,
            "spacegroup": sg,
            "crystal_system": crystal_system,
            "phase_deg": counts.index.astype(int),
            "n": counts.values.astype(int),
        }
    )
    centric_hist_df["p_centric"] = (
        centric_hist_df["n"] / float(n_centric_with_phase_rows)
        if n_centric_with_phase_rows > 0 else np.nan
    )
    centric_hist_df["overall_completeness"] = overall_completeness
    centric_hist_df["centric_completeness"] = centric_completeness
    centric_hist_df["frac_centric_unique_obs"] = frac_centric_unique_obs
    centric_hist_df["frac_centric_unique_th"] = frac_centric_unique_th
    centric_hist_df["delta_frac_unique"] = delta_frac_unique

    counts_dict: dict[str, object] = {
        "pdb_id": pdb_id,
        "spacegroup": sg,
        "crystal_system": crystal_system,

        "n_total": n_total_rows,
        "n_centric": n_centric_rows,
        "n_acentric": n_acentric_rows,
        "frac_centric": (float(n_centric_rows) / float(n_total_rows)) if n_total_rows > 0 else np.nan,
        "ratio_centric_acentric": safe_ratio(num=float(n_centric_rows), den=float(n_acentric_rows)),

        "n_centric_with_phase": n_centric_with_phase_rows,

        "n_obs_unique_total": n_obs_unique_total,
        "n_obs_unique_centric": n_obs_unique_centric,
        "n_theory_unique_total": int(n_theory_total),
        "n_theory_unique_centric": int(n_theory_centric),

        "overall_completeness": overall_completeness,
        "centric_completeness": centric_completeness,

        "frac_centric_unique_obs": frac_centric_unique_obs,
        "frac_centric_unique_th": frac_centric_unique_th,
        "delta_frac_unique": delta_frac_unique,

        "source_file": csv_path.name,
    }

    return counts_dict, centric_hist_df


# ---------------------------
# Top-10 report
# ---------------------------

def build_top10_sg_report(
    *,
    counts_df: pd.DataFrame,
    all_phases: pd.DataFrame,
    sg: str,
) -> pd.DataFrame:
    """
    One row per dataset in this SG, plus wide phase count/% columns.
    """
    df_counts = counts_df.loc[counts_df["spacegroup"] == sg].copy()
    df_counts = df_counts[
        [
            "pdb_id", "spacegroup", "crystal_system",
            "n_total", "n_centric", "n_acentric", "frac_centric",
            "overall_completeness", "centric_completeness",
            "n_obs_unique_total", "n_obs_unique_centric",
            "n_theory_unique_total", "n_theory_unique_centric",
            "frac_centric_unique_obs", "frac_centric_unique_th", "delta_frac_unique",
        ]
    ].sort_values(by="pdb_id").reset_index(drop=True)

    df_phase = all_phases.loc[all_phases["spacegroup"] == sg].copy()
    if df_phase.empty:
        return df_counts

    piv_n = (
        df_phase.pivot_table(index="pdb_id", columns="phase_deg", values="n", aggfunc="sum", fill_value=0)
        .sort_index(axis=1)
    )
    denom = piv_n.sum(axis=1).astype(float).replace(to_replace=0.0, value=np.nan)
    piv_p = piv_n.div(denom, axis=0)

    phase_degs = [int(col) for col in piv_n.columns.tolist()]
    piv_n.columns = [f"phase_{deg}_n" for deg in phase_degs]
    piv_p.columns = [f"phase_{deg}_p" for deg in phase_degs]

    out = df_counts.set_index("pdb_id").join(piv_n, how="left").join(piv_p, how="left").reset_index()
    n_cols = [col for col in out.columns if col.startswith("phase_") and col.endswith("_n")]
    out[n_cols] = out[n_cols].fillna(0).astype(int)

    base_cols = list(df_counts.columns)
    phase_cols: list[str] = []
    for deg in sorted(phase_degs):
        phase_cols.extend([f"phase_{deg}_n", f"phase_{deg}_p"])
    keep = base_cols + [col for col in phase_cols if col in out.columns]
    return out[keep]


# ---------------------------
# Main
# ---------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure empirical centric burden and completeness across reflection CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project_dir", type=Path, default=Path("."), help="Project root.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Input directory containing reflection CSV files.")
    parser.add_argument("--pattern", type=str, default="*_rs_ecalc.csv", help="Input filename pattern.")
    parser.add_argument("--out_subdir", type=str, default="10_Empirical_Centric_Burden", help="Output subdirectory.")

    parser.add_argument("--phase_col", type=str, default="PHIC_ALL")
    parser.add_argument("--centric_col", type=str, default="CENTRIC")

    parser.add_argument("--h_col", type=str, default="H")
    parser.add_argument("--k_col", type=str, default="K")
    parser.add_argument("--l_col", type=str, default="L")

    parser.add_argument(
        "--spacegroup_cols",
        type=str,
        default="SG,SPACEGROUP,SPACE_GROUP,spacegroup,space_group,sg",
    )

    parser.add_argument("--a_col", type=str, default="LENGTH_A")
    parser.add_argument("--b_col", type=str, default="LENGTH_B")
    parser.add_argument("--c_col", type=str, default="LENGTH_C")
    parser.add_argument("--alpha_col", type=str, default="ANGLE_ALPHA")
    parser.add_argument("--beta_col", type=str, default="ANGLE_BETA")
    parser.add_argument("--gamma_col", type=str, default="ANGLE_GAMMA")

    parser.add_argument("--d_min", type=float, default=2.2)
    parser.add_argument("--d_max", type=float, default=20.0)

    parser.add_argument("--num_processes", type=int, default=0, help="0 means auto-detect CPU count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_dir = args.project_dir.resolve()
    logs_dir = project_dir / "LOGS"
    out_dir = project_dir / args.out_subdir
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = now_local().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{RUN_PREFIX}__{timestamp}.log"
    setup_logger(log_path=log_path)

    input_glob_pattern = args.input_dir.resolve() / args.pattern
    files = sorted(args.input_dir.resolve().glob(args.pattern))
    if len(files) == 0:
        raise RuntimeError(f"No files matched: {input_glob_pattern}")

    spacegroup_cols = [col.strip() for col in args.spacegroup_cols.split(",") if col.strip()]
    cell_cols = [args.a_col, args.b_col, args.c_col, args.alpha_col, args.beta_col, args.gamma_col]

    n_workers = int(args.num_processes)
    if n_workers <= 0:
        n_workers = os.cpu_count() or 1

    logging.info("Run prefix: %s", RUN_PREFIX)
    logging.info("Command line: %s", " ".join(shlex.quote(arg) for arg in sys.argv))
    logging.info("Input glob: %s", input_glob_pattern)
    logging.info("Output dir: %s", out_dir)
    logging.info("Log file: %s", log_path)
    logging.info("Workers: %d", n_workers)

    logging.info("Validating %d CSV files (hard-stop on any failure)...", len(files))
    all_errors: list[str] = []
    for path in files:
        ok, errors = validate_one_csv(
            csv_path=path,
            phase_col=args.phase_col,
            centric_col=args.centric_col,
            h_col=args.h_col,
            k_col=args.k_col,
            l_col=args.l_col,
            spacegroup_cols=spacegroup_cols,
            cell_cols=cell_cols,
        )
        if not ok:
            all_errors.extend(errors)

    if all_errors:
        logging.error("VALIDATION FAILED. Found %d issue(s):", len(all_errors))
        for error in all_errors:
            logging.error("%s", error)
        raise RuntimeError("Validation failed for one or more CSV files. See LOGS for details.")
    logging.info("Validation passed for all files.")

    out_counts_dir = out_dir / "01_Centric_Acentric_Counts_By_SG"
    out_phase_dir = out_dir / "02_Centric_Phase_Counts_By_SG"
    out_top10_dir = out_dir / "03_Top10_SG_Reports"
    out_counts_dir.mkdir(parents=True, exist_ok=True)
    out_phase_dir.mkdir(parents=True, exist_ok=True)
    out_top10_dir.mkdir(parents=True, exist_ok=True)

    counts_rows: list[dict[str, object]] = []
    phase_rows: list[pd.DataFrame] = []

    worker_kwargs = dict(
        phase_col=args.phase_col,
        centric_col=args.centric_col,
        spacegroup_cols=spacegroup_cols,
        a_col=args.a_col,
        b_col=args.b_col,
        c_col=args.c_col,
        alpha_col=args.alpha_col,
        beta_col=args.beta_col,
        gamma_col=args.gamma_col,
        d_min=float(args.d_min),
        d_max=float(args.d_max),
        h_col=args.h_col,
        k_col=args.k_col,
        l_col=args.l_col,
    )

    if n_workers == 1:
        logging.info("Processing in SERIAL mode (num_processes=1)...")
        for i, path in enumerate(files, start=1):
            try:
                counts_dict, phase_df = process_one_csv_worker(csv_path=path, **worker_kwargs)
                counts_rows.append(counts_dict)
                phase_rows.append(phase_df)
            except Exception as exc:
                logging.exception("FAILED processing %s: %s", path, str(exc))
            if i % 50 == 0:
                logging.info("Progress: %d/%d", i, len(files))
    else:
        logging.info("Processing in PARALLEL mode (ProcessPoolExecutor)...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(n_workers)) as executor:
            future_to_path: dict[concurrent.futures.Future, Path] = {}
            for path in files:
                future = executor.submit(process_one_csv_worker, csv_path=path, **worker_kwargs)
                future_to_path[future] = path

            done = 0
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    counts_dict, phase_df = future.result()
                    counts_rows.append(counts_dict)
                    phase_rows.append(phase_df)
                except Exception as exc:
                    logging.exception("FAILED processing %s: %s", path, str(exc))
                done += 1
                if done % 50 == 0:
                    logging.info("Progress: %d/%d", done, len(files))

    if not counts_rows:
        raise RuntimeError("No files were successfully processed. Check the log.")

    counts_df = pd.DataFrame(counts_rows)
    all_phases = pd.concat(phase_rows, ignore_index=True) if phase_rows else pd.DataFrame()

    for sg, df_sg in counts_df.groupby(by="spacegroup", sort=True):
        df_out = df_sg[
            [
                "pdb_id", "spacegroup", "crystal_system",
                "n_total", "n_centric", "n_acentric", "frac_centric", "ratio_centric_acentric",
                "n_obs_unique_total", "n_obs_unique_centric",
                "n_theory_unique_total", "n_theory_unique_centric",
                "overall_completeness", "centric_completeness",
                "frac_centric_unique_obs", "frac_centric_unique_th", "delta_frac_unique",
            ]
        ].sort_values(by="pdb_id", ascending=True).reset_index(drop=True)

        df_out.to_csv(out_counts_dir / f"SG__{sanitize_for_filename(text=str(sg))}.csv", index=False)

    for sg, df_sg in all_phases.groupby(by="spacegroup", sort=True):
        df_out = df_sg[
            [
                "pdb_id", "spacegroup", "crystal_system",
                "overall_completeness", "centric_completeness",
                "frac_centric_unique_obs", "frac_centric_unique_th", "delta_frac_unique",
                "phase_deg", "n", "p_centric",
            ]
        ].sort_values(by=["pdb_id", "phase_deg"], ascending=[True, True]).reset_index(drop=True)

        df_out.to_csv(out_phase_dir / f"SG__{sanitize_for_filename(text=str(sg))}.csv", index=False)

    sg_counts = counts_df.groupby("spacegroup")["pdb_id"].nunique().sort_values(ascending=False)
    top10_sg = sg_counts.head(10).index.tolist()
    for rank, sg in enumerate(top10_sg, start=1):
        wide = build_top10_sg_report(counts_df=counts_df, all_phases=all_phases, sg=str(sg))
        wide.to_csv(out_top10_dir / f"rank{rank:02d}__SG__{sanitize_for_filename(text=str(sg))}.csv", index=False)

    logging.info("DONE.")
    logging.info("Counts-by-SG dir: %s", out_counts_dir)
    logging.info("Phase-by-SG dir: %s", out_phase_dir)
    logging.info("Top10 SG reports dir: %s", out_top10_dir)
    logging.info("Processed CSV files successfully: %d", int(len(counts_rows)))


if __name__ == "__main__":
    main()
