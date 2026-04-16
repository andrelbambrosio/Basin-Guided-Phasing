#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
21_prepare_autobuild_mtz_inputs.py

Prepare minimal MTZ files for Phenix AutoBuild from a folder of per-dataset CSVs,
with pipeline integration and stratified random sampling across space groups.

UPDATED (this version)
----------------------
- Removes K2_SNR entirely (method + SNR_FOM computation + all associated CLI options).
- Keeps method "K2_atten": phase_col=PHIC_ALL_K2, fom_col=K2_ATTEN_FOM, suffix=K2_atten,
  where K2_ATTEN_FOM is a constant global prior = mean cosine attenuation for K=2:
      m2_prior = 2/pi ≈ 0.63662

Typical use (sample up to 1000 PDB IDs, balanced across SGs):
  python 21_prepare_autobuild_mtz_inputs.py \
    --input_csv_folder ./04_Run_Ecalc_Binned/ \
    --output_root ./21_Phenix_Autobuild_Inputs \
    --csv_pattern "*_rs_ecalc_binned.csv" \
    --methods "iREDO,K2,K2_atten" \
    --max_pdbs 1000 \
    --seed 123 \
    --write_index_csv \
    --kappa_max 200

Target a specific set:
  python 21_prepare_autobuild_mtz_inputs.py \
    --input_csv_folder ./04_Run_Ecalc_Binned/ \
    --output_root ./21_Phenix_Autobuild_Inputs \
    --csv_pattern "*_rs_ecalc_binned.csv" \
    --methods "K2,K2_atten" \
    --pdb_ids "1a6g,1a6k" \
    --kappa_max 200

Notes
-----
- Only prepares MTZs for methods requested by --methods.
- By default it does NOT process the full dataset; it samples up to --max_pdbs.
- Stratified sampler aims to cover many distinct space groups.
- K2_ATTEN_FOM is computed on-the-fly if missing from the CSV.
- Resume-safe: if MTZ exists AND is valid (required columns) it is skipped;
  if exists but invalid, it is rebuilt.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gemmi
import numpy as np
import pandas as pd
import reciprocalspaceship as rs


# ----------------------------
# Pipeline-consistency knobs
# ----------------------------

# MUST match the --fom_col used in your binning script when generating *_K2 columns.
# If you binned with --fom_col FOM, keep this as "FOM".
BASE_FOM_COL_FOR_BINNING: str = "FOM"


# ----------------------------
# Methods registry
# ----------------------------

@dataclass(frozen=True)
class MethodSpec:
    name: str
    phase_col: str
    fom_col: str
    suffix: str


def build_method_specs(*, base_fom_col: str) -> Dict[str, MethodSpec]:
    """
    Central registry for allowed method labels.
    Extend here if you introduce alternative phase sources.

    Kept:
      - K2_atten : PHIC_ALL_K2 + K2_ATTEN_FOM (constant 2/pi)
    """
    return {
        "iREDO":    MethodSpec(name="iREDO",    phase_col="PHIC_ALL",    fom_col="FOM",                 suffix="iREDO"),
        "K2":       MethodSpec(name="K2",       phase_col="PHIC_ALL_K2", fom_col=f"{base_fom_col}_K2", suffix="K2"),
        "K3":       MethodSpec(name="K3",       phase_col="PHIC_ALL_K3", fom_col=f"{base_fom_col}_K3", suffix="K3"),
        "K4":       MethodSpec(name="K4",       phase_col="PHIC_ALL_K4", fom_col=f"{base_fom_col}_K4", suffix="K4"),
        "K2_atten": MethodSpec(name="K2_atten", phase_col="PHIC_ALL_K2", fom_col="K2_ATTEN_FOM",       suffix="K2_atten"),
    }


# ----------------------------
# Logging
# ----------------------------

def configure_logging(*, output_root: Path, log_prefix: str = "PC32_prepare_autobuild_mtz") -> Path:
    logs_dir = output_root.parent / "LOGS"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{log_prefix}_{timestamp}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(filename=str(log_path), mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logging.info("Log file: %s", str(log_path))
    return log_path


# ----------------------------
# Helpers
# ----------------------------

def pdb_id_from_path(*, path: Path) -> str:
    return path.name.split("_")[0].lower()


def map_minus180_to_plus180_deg(*, arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    return np.where(np.isclose(x, -180.0, atol=1e-9), 180.0, x)


def kappa_from_R_array(*, R: np.ndarray) -> np.ndarray:
    """
    Vectorized approximation of kappa from resultant length R (0 <= R < 1),
    using standard piecewise approximations for the von Mises distribution.
    """
    R = np.asarray(R, dtype=float)
    Rc = np.clip(R, 0.0, 0.999999)
    kappa = np.zeros_like(Rc)

    mask0 = Rc < 1e-8

    m1 = (~mask0) & (Rc < 0.53)
    kappa[m1] = 2.0 * Rc[m1] + Rc[m1] ** 3 + (5.0 * Rc[m1] ** 5) / 6.0

    m2 = (Rc >= 0.53) & (Rc < 0.85)
    kappa[m2] = -0.4 + 1.39 * Rc[m2] + 0.43 / (1.0 - Rc[m2])

    m3 = Rc >= 0.85
    kappa[m3] = 1.0 / (Rc[m3] ** 3 - 4.0 * Rc[m3] ** 2 + 3.0 * Rc[m3] + 1e-12)

    return kappa


def k2_mean_cos_attenuation() -> float:
    """
    Mean cosine attenuation for uniform rounding error in [-Δ/2, Δ/2] with Δ=π (K=2):
      m2_prior = 2*sin(Δ/2)/Δ = 2/pi.
    """
    return float(2.0 / np.pi)


def mtz_is_valid_for_autobuild(*, mtz_path: Path) -> Tuple[bool, str]:
    if not mtz_path.is_file():
        return (False, "missing")
    if mtz_path.stat().st_size <= 0:
        return (False, "empty_file")

    required = {"FP", "SIGFP", "PHIB", "FOM", "HLA", "HLB", "HLC", "HLD", "FreeR_flag"}

    try:
        mtz = gemmi.read_mtz_file(str(mtz_path))
    except Exception as exc:
        return (False, f"read_failed: {exc}")

    labels = {c.label for c in mtz.columns}
    missing = sorted(list(required - labels))
    if missing:
        return (False, f"missing_columns: {missing}")

    n_refl = int(getattr(mtz, "nreflections", 0))
    if n_refl <= 0:
        return (False, "no_reflections")

    return (True, f"ok (n_reflections={n_refl})")


def build_mtz_from_csv(
    *,
    df: pd.DataFrame,
    phase_col: str,
    fom_col: str,
    out_path: Path,
    kappa_max: float,
) -> None:
    required_cols = [
        "H", "K", "L",
        "FP", "SIGFP",
        phase_col,
        "FreeR_flag",
        "SG",
        "LENGTH_A", "LENGTH_B", "LENGTH_C",
        "ANGLE_ALPHA", "ANGLE_BETA", "ANGLE_GAMMA",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    synthetic_cols = {"K2_ATTEN_FOM"}
    if (fom_col not in synthetic_cols) and (fom_col not in df.columns):
        raise ValueError(f"Missing required column in CSV: {fom_col}")

    df_use = df.dropna(
        subset=["H", "K", "L", "FP", "SIGFP", phase_col, "FreeR_flag"]
    ).copy()
    if df_use.empty:
        raise ValueError("No valid reflections after dropping NaNs.")

    if (fom_col == "K2_ATTEN_FOM") and ("K2_ATTEN_FOM" not in df_use.columns):
        df_use["K2_ATTEN_FOM"] = float(k2_mean_cos_attenuation())

    df_use = df_use.dropna(subset=[fom_col]).copy()
    if df_use.empty:
        raise ValueError("No valid reflections after dropping NaNs (including FOM).")

    H = df_use["H"].to_numpy(dtype=int)
    K = df_use["K"].to_numpy(dtype=int)
    L = df_use["L"].to_numpy(dtype=int)
    FP = df_use["FP"].to_numpy(dtype=float)
    SIGFP = df_use["SIGFP"].to_numpy(dtype=float)

    PHIB = map_minus180_to_plus180_deg(arr=df_use[phase_col].to_numpy(dtype=float))

    FOM = df_use[fom_col].to_numpy(dtype=float)
    FreeR = df_use["FreeR_flag"].to_numpy(dtype=int)

    FOM = np.where(np.isfinite(FOM), FOM, 0.0)
    FOM = np.clip(FOM, 0.0, 0.9999)

    kappa = np.minimum(kappa_from_R_array(R=FOM), float(kappa_max))

    phi_rad = np.deg2rad(PHIB)
    HLA = kappa * np.cos(phi_rad)
    HLB = kappa * np.sin(phi_rad)
    HLC = np.zeros_like(HLA)
    HLD = np.zeros_like(HLA)

    ds = rs.DataSet()
    ds["H"] = rs.DataSeries(H, dtype="H")
    ds["K"] = rs.DataSeries(K, dtype="H")
    ds["L"] = rs.DataSeries(L, dtype="H")
    ds["FP"] = rs.DataSeries(FP, dtype="F")
    ds["SIGFP"] = rs.DataSeries(SIGFP, dtype="Q")
    ds["PHIB"] = rs.DataSeries(PHIB, dtype="P")
    ds["FOM"] = rs.DataSeries(FOM, dtype="W")
    ds["FreeR_flag"] = rs.DataSeries(FreeR, dtype="I")
    ds["HLA"] = rs.DataSeries(HLA, dtype="A")
    ds["HLB"] = rs.DataSeries(HLB, dtype="A")
    ds["HLC"] = rs.DataSeries(HLC, dtype="A")
    ds["HLD"] = rs.DataSeries(HLD, dtype="A")

    ds.set_index(["H", "K", "L"], inplace=True)

    sg_str = str(df_use["SG"].iloc[0])
    a = float(df_use["LENGTH_A"].iloc[0])
    b = float(df_use["LENGTH_B"].iloc[0])
    c = float(df_use["LENGTH_C"].iloc[0])
    alpha = float(df_use["ANGLE_ALPHA"].iloc[0])
    beta = float(df_use["ANGLE_BETA"].iloc[0])
    gamma = float(df_use["ANGLE_GAMMA"].iloc[0])

    ds.spacegroup = gemmi.SpaceGroup(sg_str)
    ds.cell = gemmi.UnitCell(a, b, c, alpha, beta, gamma)

    mtz_obj = ds.to_gemmi()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mtz_obj.write_to_file(str(out_path))


def read_sg_from_csv_header(*, csv_path: Path) -> str:
    """
    Fast: read only SG from first row.
    """
    try:
        df0 = pd.read_csv(csv_path, nrows=1, usecols=["SG"])
        return str(df0["SG"].iloc[0]).strip()
    except Exception:
        return "unknown"


def stratified_sample_by_spacegroup(
    *,
    csv_files: List[Path],
    max_pdbs: int,
    seed: int,
    sg_min_count: int,
) -> Tuple[List[Path], pd.DataFrame]:
    """
    Choose up to max_pdbs files trying to cover as many distinct SGs as possible.
    Strategy:
      - build table (pdb_id, path, SG)
      - group by SG
      - allocate ~equal quota per SG, with at least 1 per SG where possible
      - if total < max_pdbs, fill remaining from SGs with remaining inventory (weighted by remaining size)
    """
    rng = np.random.default_rng(int(seed))

    records: List[Dict[str, str]] = []
    for p in csv_files:
        pid = pdb_id_from_path(path=p)
        sg = read_sg_from_csv_header(csv_path=p)
        records.append({"pdb_id": pid, "spacegroup": sg, "csv_path": str(p)})

    idx = pd.DataFrame(records)
    idx = idx.drop_duplicates(subset=["pdb_id"], keep="first").reset_index(drop=True)

    sg_counts = idx["spacegroup"].value_counts(dropna=False).to_dict()
    idx["sg_count"] = idx["spacegroup"].map(sg_counts).astype(int)

    major = idx[idx["sg_count"] >= int(max(1, sg_min_count))].copy()
    minor = idx[idx["sg_count"] < int(max(1, sg_min_count))].copy()

    groups = {sg: g.copy() for sg, g in major.groupby("spacegroup", sort=False)}

    n_groups = len(groups)
    if n_groups == 0:
        take = min(int(max_pdbs), int(idx.shape[0]))
        chosen = idx.sample(n=take, replace=False, random_state=int(seed))
        files = [Path(s) for s in chosen["csv_path"].tolist()]
        return files, chosen[["pdb_id", "spacegroup", "csv_path"]].reset_index(drop=True)

    quota = max(1, int(max_pdbs) // int(n_groups))

    chosen_rows: List[pd.DataFrame] = []
    for sg, g in groups.items():
        n_avail = int(g.shape[0])
        n_take = min(int(quota), n_avail)
        if n_take <= 0:
            continue
        chosen_rows.append(g.sample(n=n_take, replace=False, random_state=int(rng.integers(0, 2**31 - 1))))

    chosen = pd.concat(chosen_rows, axis=0, ignore_index=True) if chosen_rows else pd.DataFrame(columns=idx.columns)

    remaining_pool = idx.merge(
        chosen[["pdb_id"]],
        on="pdb_id",
        how="left",
        indicator=True,
    )
    remaining_pool = remaining_pool[remaining_pool["_merge"] == "left_only"].drop(columns=["_merge"])

    # Ensure at least 1 from each minor SG if budget allows
    if (not minor.empty) and (int(chosen.shape[0]) < int(max_pdbs)):
        for sg, g in minor.groupby("spacegroup", sort=False):
            if int(chosen.shape[0]) >= int(max_pdbs):
                break
            pick = g.sample(n=1, replace=False, random_state=int(rng.integers(0, 2**31 - 1)))
            chosen = pd.concat([chosen, pick], axis=0, ignore_index=True)

        remaining_pool = idx.merge(
            chosen[["pdb_id"]],
            on="pdb_id",
            how="left",
            indicator=True,
        )
        remaining_pool = remaining_pool[remaining_pool["_merge"] == "left_only"].drop(columns=["_merge"])

    n_need = int(max_pdbs) - int(chosen.shape[0])
    if n_need > 0 and not remaining_pool.empty:
        weights = remaining_pool["sg_count"].to_numpy(dtype=float)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
        weights = weights / float(np.sum(weights))

        n_need = min(n_need, int(remaining_pool.shape[0]))
        idx_pick = rng.choice(a=np.arange(int(remaining_pool.shape[0])), size=n_need, replace=False, p=weights)
        fill = remaining_pool.iloc[idx_pick].copy()
        chosen = pd.concat([chosen, fill], axis=0, ignore_index=True)

    chosen = chosen.drop_duplicates(subset=["pdb_id"], keep="first").reset_index(drop=True)

    files = [Path(s) for s in chosen["csv_path"].tolist()]
    return files, chosen[["pdb_id", "spacegroup", "csv_path"]].reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PC32: Prepare minimal MTZs for Phenix AutoBuild from CSVs, using stratified sampling across SGs."
    )
    parser.add_argument("--input_csv_folder", type=str, required=True, help="Folder containing CSV files (one per dataset).")
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root folder for output MTZs. Default: <base>/21_Phenix_Autobuild_Inputs",
    )
    parser.add_argument("--csv_pattern", type=str, default="*_rs_ecalc_binned.csv", help="Glob pattern for input CSVs.")
    parser.add_argument(
        "--methods",
        type=str,
        required=True,
        help='Comma-separated method names. Allowed: iREDO,K2,K3,K4,K2_atten.',
    )
    parser.add_argument("--kappa_max", type=float, default=200.0, help="Cap for kappa used to compute HL (default 200).")
    parser.add_argument("--dry_run", action="store_true", help="Only log planned MTZs; do not write any files.")

    # selection controls
    parser.add_argument("--max_pdbs", type=int, default=1000, help="Max PDB IDs to process (default 1000).")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for sampling (default 123).")
    parser.add_argument("--sg_min_count", type=int, default=3,
                        help="Treat SGs with < this count as 'minor' (default 3). Minor SGs get at most 1 pick initially.")
    parser.add_argument("--pdb_ids", type=str, default=None,
                        help='Explicit PDB IDs to process (comma-separated). Overrides random sampling.')
    parser.add_argument("--pdb_ids_file", type=str, default=None,
                        help="Text file listing PDB IDs (one per line or comma-separated). Overrides random sampling.")
    parser.add_argument("--write_index_csv", action="store_true",
                        help="Write the selection index CSV (pdb_id,spacegroup,csv_path) under output_root.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_csv_folder = Path(args.input_csv_folder).expanduser().resolve()
    if not input_csv_folder.is_dir():
        raise SystemExit(f"[ERROR] Input CSV folder does not exist: {input_csv_folder}")

    if args.output_root is None:
        output_root = input_csv_folder.parent / "21_Phenix_Autobuild_Inputs"
    else:
        output_root = Path(args.output_root).expanduser().resolve()

    output_root.mkdir(parents=True, exist_ok=True)
    log_path = configure_logging(output_root=output_root, log_prefix="21_prepare_autobuild_mtz_inputs")
    logging.info("Run prefix: %s", "21_prepare_autobuild_mtz_inputs")
    logging.info("Command line: %s", " ".join(shlex.quote(arg) for arg in sys.argv))

    registry = build_method_specs(base_fom_col=str(BASE_FOM_COL_FOR_BINNING))
    requested = [m.strip() for m in str(args.methods).split(",") if m.strip()]

    normalized: List[str] = []
    for m in requested:
        mm = m.strip()
        mml = mm.lower()

        if mml == "iredo":
            normalized.append("iREDO")
        elif mml == "k2_atten":
            normalized.append("K2_atten")
        else:
            normalized.append(mm.upper())

    unknown = [m for m in normalized if m not in registry]
    if unknown:
        raise SystemExit(f"[ERROR] Unknown methods in --methods: {unknown}. Allowed: {sorted(list(registry.keys()))}")

    methods = [registry[m] for m in normalized]

    logging.info("Parameters:")
    logging.info("  input_csv_folder          = %s", str(input_csv_folder))
    logging.info("  output_root               = %s", str(output_root))
    logging.info("  csv_pattern               = %s", str(args.csv_pattern))
    logging.info("  kappa_max                 = %.2f", float(args.kappa_max))
    logging.info("  dry_run                   = %s", bool(args.dry_run))
    logging.info("  BASE_FOM_COL_FOR_BINNING  = %s", BASE_FOM_COL_FOR_BINNING)
    logging.info("  methods_selected          = %s", [m.name for m in methods])
    logging.info("  K2_atten prior            = %.6f", float(k2_mean_cos_attenuation()))

    csv_files = sorted(list(input_csv_folder.glob(pattern=str(args.csv_pattern))))
    csv_files = [p for p in csv_files if p.is_file() and p.stat().st_size > 0]
    if not csv_files:
        raise SystemExit(f"[ERROR] No CSV files found under {input_csv_folder} with pattern '{args.csv_pattern}'")

    # Selection: explicit IDs override sampling
    explicit_ids: List[str] = []
    if args.pdb_ids:
        explicit_ids.extend([x.strip().lower() for x in str(args.pdb_ids).split(",") if x.strip()])
    if args.pdb_ids_file:
        p = Path(args.pdb_ids_file).expanduser().resolve()
        if p.is_file():
            txt = p.read_text(encoding="utf-8", errors="ignore")
            for line in txt.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                for tok in line.split(","):
                    t = tok.strip().lower()
                    if t:
                        explicit_ids.append(t)
    explicit_ids = sorted(set(explicit_ids))

    if explicit_ids:
        prefix_map = {pdb_id_from_path(path=p): p for p in csv_files}
        selected_files: List[Path] = []
        missing: List[str] = []
        for pid in explicit_ids:
            pp = prefix_map.get(pid)
            if pp is None:
                missing.append(pid)
            else:
                selected_files.append(pp)
        if missing:
            logging.warning("Explicit PDB IDs not found (missing %d): %s", len(missing), missing[:50])
        index_df = pd.DataFrame(
            [{"pdb_id": pdb_id_from_path(path=p), "spacegroup": read_sg_from_csv_header(csv_path=p), "csv_path": str(p)} for p in selected_files]
        )
    else:
        selected_files, index_df = stratified_sample_by_spacegroup(
            csv_files=csv_files,
            max_pdbs=int(args.max_pdbs),
            seed=int(args.seed),
            sg_min_count=int(args.sg_min_count),
        )

    logging.info("Discovered CSV files: %d", len(csv_files))
    logging.info("Selected datasets   : %d", len(selected_files))
    logging.info("Distinct SGs (sel)  : %d", int(index_df["spacegroup"].nunique()) if not index_df.empty else 0)
    if not index_df.empty:
        logging.info("Selection preview (first 20): %s", index_df.head(20).to_dict(orient="records"))

    if bool(args.write_index_csv):
        idx_path = output_root / f"PC32_selected_datasets_seed{int(args.seed)}_n{int(len(selected_files))}.csv"
        index_df.to_csv(idx_path, index=False)
        logging.info("Wrote selection index CSV: %s", str(idx_path))

    total_tasks = len(selected_files) * len(methods)
    logging.info("Planned tasks: %d (datasets=%d × methods=%d)", total_tasks, len(selected_files), len(methods))

    successes = 0
    skipped = 0
    failures = 0
    records: List[Dict[str, Any]] = []

    for csv_path in selected_files:
        pdb_id = pdb_id_from_path(path=csv_path)

        try:
            df = pd.read_csv(filepath_or_buffer=csv_path)
        except Exception as exc:
            logging.warning("Skipping %s (read error: %s)", csv_path.name, exc)
            continue

        for spec in methods:
            phase_col = spec.phase_col
            fom_col = spec.fom_col
            suffix = spec.suffix

            if phase_col not in df.columns:
                records.append(
                    {
                        "pdb_id": pdb_id,
                        "csv": str(csv_path),
                        "method": spec.name,
                        "phase_col": phase_col,
                        "fom_col": fom_col,
                        "suffix": suffix,
                        "status": "SKIP(no_phase_col)",
                        "mtz_path": "",
                        "note": "",
                    }
                )
                skipped += 1
                continue

            # If the method needs a CSV FOM column (not a synthetic one), it must exist.
            if (fom_col != "K2_ATTEN_FOM") and (fom_col not in df.columns):
                records.append(
                    {
                        "pdb_id": pdb_id,
                        "csv": str(csv_path),
                        "method": spec.name,
                        "phase_col": phase_col,
                        "fom_col": fom_col,
                        "suffix": suffix,
                        "status": "SKIP(no_fom_col)",
                        "mtz_path": "",
                        "note": "",
                    }
                )
                skipped += 1
                continue

            out_path = output_root / suffix / pdb_id / f"{pdb_id}_{suffix}_PHIB_input.mtz"

            if out_path.is_file() and (not args.dry_run):
                ok, why = mtz_is_valid_for_autobuild(mtz_path=out_path)
                if ok:
                    logging.info("SKIP %s | %s -> exists+valid (%s)", pdb_id, suffix, why)
                    records.append(
                        {
                            "pdb_id": pdb_id,
                            "csv": str(csv_path),
                            "method": spec.name,
                            "phase_col": phase_col,
                            "fom_col": fom_col,
                            "suffix": suffix,
                            "status": "SKIP(valid_exists)",
                            "mtz_path": str(out_path),
                            "note": why,
                        }
                    )
                    skipped += 1
                    continue
                logging.info("REBUILD %s | %s -> exists but invalid (%s)", pdb_id, suffix, why)

            if args.dry_run:
                logging.info("PLAN %s | method=%s | phase=%s | fom=%s -> %s", pdb_id, spec.name, phase_col, fom_col, str(out_path))
                records.append(
                    {
                        "pdb_id": pdb_id,
                        "csv": str(csv_path),
                        "method": spec.name,
                        "phase_col": phase_col,
                        "fom_col": fom_col,
                        "suffix": suffix,
                        "status": "PLAN",
                        "mtz_path": str(out_path),
                        "note": "",
                    }
                )
                continue

            try:
                build_mtz_from_csv(
                    df=df,
                    phase_col=phase_col,
                    fom_col=fom_col,
                    out_path=out_path,
                    kappa_max=float(args.kappa_max),
                )

                ok2, why2 = mtz_is_valid_for_autobuild(mtz_path=out_path)
                if not ok2:
                    raise RuntimeError(f"MTZ written but failed integrity check: {why2}")

                successes += 1
                status = "OK"
                note = why2
                logging.info(
                    "OK %s | %s -> %s (%s) | phase_col=%s | fom_col=%s",
                    pdb_id, suffix, out_path.name, why2, phase_col, fom_col
                )

            except Exception as exc:
                failures += 1
                status = "FAIL"
                note = str(exc)
                logging.error("FAIL %s | method=%s | suffix=%s | csv=%s | err=%s",
                              pdb_id, spec.name, suffix, csv_path.name, exc)

            records.append(
                {
                    "pdb_id": pdb_id,
                    "csv": str(csv_path),
                    "method": spec.name,
                    "phase_col": phase_col,
                    "fom_col": fom_col,
                    "suffix": suffix,
                    "status": status,
                    "mtz_path": str(out_path),
                    "note": note,
                }
            )

    manifest_name = f"21_autobuild_mtz_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    manifest_path = output_root / manifest_name
    pd.DataFrame(records).to_csv(path_or_buf=manifest_path, index=False)

    logging.info("Summary:")
    logging.info("  OK      = %d", successes)
    logging.info("  FAILED  = %d", failures)
    logging.info("  SKIPPED = %d", skipped)
    logging.info("  TOTAL   = %d", successes + failures + skipped)
    logging.info("Manifest: %s", str(manifest_path))
    logging.info("Log file: %s", str(log_path))
    logging.info("[DONE] AutoBuild MTZ input preparation complete.")


if __name__ == "__main__":
    main()