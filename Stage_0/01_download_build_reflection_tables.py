#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_download_build_reflection_tables.py

Repository-ready stage-01 pipeline for cohort source retrieval and reflection-table generation.

This script performs:
1. Download of PDB-REDO coordinate and MTZ files
2. Download of RCSB FASTA files
3. Validation of downloaded files
4. Extraction of solvent fraction from the RCSB Data API
5. Generation of standardized reciprocalspaceship CSV reflection tables
6. Optional ECALC-like normalization

Outputs (relative to --base_dir):
  01_Download_Files  → <BASE>/01_Download_Files/<PDB>_iREDO.{pdb,mtz}
                    → <BASE>/01_Download_Files/<PDB>_rcsb.fasta
  02_Run_RS          → <BASE>/02_Run_RS/<PDB>_rs.csv
  03_Run_Ecalc       → <BASE>/03_Run_Ecalc/<PDB>_rs_ecalc.csv

Key implementation notes:
- Operates on PHIC_ALL, not PHIC-aligned values
- Forces PHIC_ALL to phase dtype when possible
- Canonicalizes phases and maps -180° to +180°
- Writes PHIC_ALL as integer degrees in the RS CSV
- Optionally computes ECALC-like E and SIGE
- Supports three ECALC modes:
    global_df   : cohort-global shells (RAM heavy)
    per_dataset : per-dataset shells (RAM light)
    none        : skip ECALC

Example:
    python 01_download_build_reflection_tables.py \
        --base_dir /path/to/project \
        --num_processes 25 \
        --min_resol 20.0 \
        --max_resol 2.2 \
        --ecalc_n_shells 20 \
        --config_json sampled_cohort_15000.json \
        --logs_dir ./LOGS \
        --ecalc_mode per_dataset
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import gemmi
import numpy as np
import pandas as pd
import reciprocalspaceship as rs
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Plotting style
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


# =========================
# Defaults
# =========================

BASE_DIR = Path("/home/alba/DIRP-20Dec2025")
NUM_PROCESSES = 1
MIN_RESOL = 20.0
MAX_RESOL = 2.5

MIN_FILE_SIZE_BYTES = 10 * 1024
MIN_FASTA_SIZE_BYTES = 20

DEFAULT_HTTP_TIMEOUT_SECONDS = 30
DEFAULT_RCSB_TIMEOUT_SECONDS = 20
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 0.75


# =========================
# Logging
# =========================

def now_local() -> datetime:
    """Return current local time."""
    return datetime.now()


def configure_logging_stage01(
    *,
    logs_dir: Path,
    script_tag: str,
    n_ids: int,
    num_processes: int,
    min_resol: float,
    max_resol: float,
    ecalc_n_shells: int,
) -> Path:
    """
    Configure root logging to stderr and a timestamped log file.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = now_local().strftime("%Y%m%d_%H%M%S")
    log_filename = (
        f"{script_tag}__{timestamp}"
        f"__n{int(n_ids):04d}"
        f"__p{int(num_processes):04d}"
        f"__d{float(min_resol):.1f}-{float(max_resol):.1f}"
        f"__sh{int(ecalc_n_shells):03d}.log"
    )
    log_path = logs_dir / log_filename

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(filename=log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt=formatter)
    logger.addHandler(hdlr=file_handler)

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(fmt=formatter)
    logger.addHandler(hdlr=stream_handler)

    logging.info("Log file: %s", log_path)
    return log_path


# =========================
# Small utilities
# =========================

def coerce_id_list(*, value: Any) -> list[str]:
    """
    Accept list[str] or comma/whitespace-separated str and return lower-case PDB IDs.
    """
    if value is None:
        return []

    if isinstance(value, list):
        items = value
    else:
        parts: list[str] = []
        for chunk in str(value).replace("\n", ",").split(","):
            parts.extend(chunk.strip().split())
        items = [part for part in parts if part]

    return [item.lower() for item in items]


def force_weight_mtztype(*, ds: rs.DataSet, cols: Iterable[str] = ("FOM",)) -> rs.DataSet:
    """
    Force selected columns to MTZ Weight dtype where possible.
    """
    for col in cols:
        if col not in ds.columns:
            continue

        series = ds[col]
        for dtype in ("Weight", "W"):
            try:
                ds[col] = series.astype(dtype)
                break
            except (TypeError, ValueError):
                continue
        else:
            try:
                ds[col] = rs.DataSeries(
                    data=series.to_numpy(dtype=float, copy=False),
                    name=col,
                    dtype="Weight",
                )
            except Exception:
                pass
    return ds


def looks_like_fasta(*, text: str) -> bool:
    """Return True if text looks like FASTA content."""
    return text.lstrip().startswith(">")


def map_minus180_to_plus180_deg(*, arr: np.ndarray) -> np.ndarray:
    """
    Map -180 degrees to +180 degrees while leaving other values unchanged.
    """
    x = np.asarray(arr, dtype=float)
    return np.where(np.isclose(x, -180.0, atol=1e-9), 180.0, x)


def wrap_deg_m180_180(*, arr: np.ndarray) -> np.ndarray:
    """Wrap degrees to [-180, 180)."""
    x = np.asarray(arr, dtype=float)
    return (x + 180.0) % 360.0 - 180.0


def canonical_phase_deg(*, arr: np.ndarray) -> np.ndarray:
    """
    Canonicalize phases into (-180, 180] with -180° ≡ +180°.
    """
    x = wrap_deg_m180_180(arr=arr)
    x = map_minus180_to_plus180_deg(arr=x)
    return x


def log_nan_counts_df(
    *,
    df: pd.DataFrame,
    label: str,
    cols: list[str],
    logger: logging.Logger | None = None,
) -> None:
    """
    Log NaN counts for selected columns in a DataFrame.
    """
    total_rows = int(df.shape[0])
    parts: list[str] = [f"[NaN-COUNT] {label} | total={total_rows}"]
    for col in cols:
        if col in df.columns:
            n_nan = int(pd.isna(df[col]).sum())
            parts.append(f"{col}={n_nan}")
        else:
            parts.append(f"{col}=<MISSING>")
    message = " | ".join(parts)

    if logger is not None:
        logger.info("%s", message)
    else:
        print(message)


def force_centric_to_int01(*, df: pd.DataFrame, col: str = "CENTRIC") -> pd.DataFrame:
    """
    Ensure CENTRIC is encoded as integers in {0, 1}.
    """
    if col not in df.columns:
        return df

    if pd.api.types.is_bool_dtype(df[col]):
        df[col] = df[col].astype(np.int8)
        return df

    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = (df[col].astype(int) != 0).astype(np.int8)
        return df

    series = df[col].astype(str).str.strip().str.lower()
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0}
    if not series.isin(mapping.keys()).all():
        bad_examples = df.loc[~series.isin(mapping.keys()), col].head(5).tolist()
        raise ValueError(f"Unrecognized CENTRIC values. Examples: {bad_examples}")

    df[col] = series.map(mapping).astype(np.int8)
    return df


def extract_wavelength_from_mtz(*, mtz_path: Path) -> tuple[float, str]:
    """
    Attempt to extract the wavelength from an MTZ file using gemmi.

    Returns
    -------
    tuple
        (wavelength, source_description)
    """
    try:
        mtz = gemmi.read_mtz_file(mtz_path=str(mtz_path))
    except Exception:
        return (float("nan"), "gemmi.read_mtz_file failed")

    try:
        if hasattr(mtz, "datasets"):
            for dataset in mtz.datasets:
                wl = float(getattr(dataset, "wavelength", float("nan")))
                if np.isfinite(wl) and wl > 0.0:
                    name = str(getattr(dataset, "name", "")).strip()
                    return (wl, f"mtz.datasets[{name or 'unnamed'}].wavelength")
    except Exception:
        pass

    try:
        if hasattr(mtz, "crystals"):
            for crystal in mtz.crystals:
                dataset_list = getattr(crystal, "datasets", [])
                for dataset in dataset_list:
                    wl = float(getattr(dataset, "wavelength", float("nan")))
                    if np.isfinite(wl) and wl > 0.0:
                        crystal_name = str(getattr(crystal, "name", "")).strip()
                        dataset_name = str(getattr(dataset, "name", "")).strip()
                        return (
                            wl,
                            f"mtz.crystals[{crystal_name or 'unnamed'}].datasets[{dataset_name or 'unnamed'}].wavelength",
                        )
    except Exception:
        pass

    return (float("nan"), "not found in MTZ header")


def safe_float(*, value: Any) -> float:
    """Safely cast to float, returning NaN on failure."""
    try:
        return float(value)
    except Exception:
        return float("nan")


def create_retry_session() -> requests.Session:
    """
    Create a requests Session with retry behavior suitable for API and file downloads.
    """
    session = requests.Session()
    retry = Retry(
        total=DEFAULT_MAX_RETRIES,
        connect=DEFAULT_MAX_RETRIES,
        read=DEFAULT_MAX_RETRIES,
        backoff_factor=DEFAULT_BACKOFF_SECONDS,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount(prefix="http://", adapter=adapter)
    session.mount(prefix="https://", adapter=adapter)
    return session


def fetch_solvent_fraction_from_rcsb(*, pdb_id: str) -> float:
    """
    Return solvent fraction (0..1) from the RCSB Data API using
    exptl_crystal[].density_percent_sol / 100.

    Returns NaN if unavailable.
    """
    pdb_id_upper = str(pdb_id).strip().upper()
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id_upper}"

    try:
        with create_retry_session() as session:
            response = session.get(url=url, timeout=DEFAULT_RCSB_TIMEOUT_SECONDS)
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logging.warning("[RCSB] %s: solvent fraction fetch failed: %s", pdb_id_upper, exc)
        return float("nan")

    exptl_crystal = data.get("exptl_crystal", None)

    crystals: list[dict[str, Any]] = []
    if isinstance(exptl_crystal, list):
        crystals = [item for item in exptl_crystal if isinstance(item, dict)]
    elif isinstance(exptl_crystal, dict):
        crystals = [exptl_crystal]

    for crystal in crystals:
        value = crystal.get("density_percent_sol", None)
        if value is None:
            continue
        percent_sol = safe_float(value=value)
        if np.isfinite(percent_sol):
            return float(percent_sol) / 100.0

    return float("nan")


# =========================
# ECALC-like helpers
# =========================

def add_epsilon_with_gemmi_per_dataset(
    *,
    df: pd.DataFrame,
    mtz_id_col: str = "mtz_id",
    sg_col: str = "SG",
    h_col: str = "H",
    k_col: str = "K",
    l_col: str = "L",
) -> pd.DataFrame:
    """
    Compute EPSILON for each reflection using gemmi.SpaceGroup(SG) per dataset.
    """
    df_out = df.copy()
    df_out["EPSILON"] = np.nan

    for mtz_id, group in df_out.groupby(by=mtz_id_col, sort=False):
        if group.empty:
            continue

        sg_str = str(group[sg_col].iloc[0])
        try:
            space_group = gemmi.SpaceGroup(sg_str)
            operations = space_group.operations()
        except Exception as exc:
            logging.error("[ECALC] mtz_id=%s: invalid SG='%s': %s", mtz_id, sg_str, exc)
            continue

        hkls = group[[h_col, k_col, l_col]].to_numpy(dtype=int)
        unique_hkls, inverse = np.unique(hkls, axis=0, return_inverse=True)

        eps_unique = np.empty(shape=len(unique_hkls), dtype=np.int16)
        for index, (h, k, l) in enumerate(unique_hkls):
            eps_unique[index] = operations.epsilon_factor_without_centering([int(h), int(k), int(l)])

        eps_all = eps_unique[inverse]
        df_out.loc[group.index, "EPSILON"] = eps_all

    df_out["EPSILON"] = df_out["EPSILON"].astype(np.int16)
    return df_out


def assign_resolution_shells(
    *,
    df: pd.DataFrame,
    d_col: str = "dHKL",
    n_shells: int = 10,
) -> pd.DataFrame:
    """
    Assign resolution shells based on d*^3 quantiles.
    """
    df_out = df.copy()
    df_out["dstar3"] = (1.0 / df_out[d_col]) ** 3

    quantiles = np.linspace(0.0, 1.0, num=n_shells + 1)
    edges = np.quantile(df_out["dstar3"].to_numpy(dtype=float), quantiles)

    shell_idx = np.searchsorted(edges, df_out["dstar3"].to_numpy(dtype=float), side="right") - 1
    shell_idx = np.clip(shell_idx, 0, n_shells - 1)
    df_out["shell"] = shell_idx.astype(int)
    return df_out


def smooth_shell_means_per_mtz(*, shell_means: pd.Series, n_shells: int) -> pd.Series:
    """
    Apply simple 3-point smoothing over shell means.
    """
    shell_vals = np.full(shape=n_shells, fill_value=np.nan, dtype=float)
    shell_vals[shell_means.index.to_numpy(dtype=int)] = shell_means.to_numpy(dtype=float)

    smoothed = np.full_like(shell_vals, fill_value=np.nan, dtype=float)
    for i in range(n_shells):
        neighbors: list[float] = []
        for j in (i - 1, i, i + 1):
            if 0 <= j < n_shells and not np.isnan(shell_vals[j]):
                neighbors.append(float(shell_vals[j]))
        if neighbors:
            smoothed[i] = float(np.mean(neighbors))

    return pd.Series(
        data=smoothed,
        index=np.arange(n_shells, dtype=int),
        name="F_eff_sq_mean_smoothed",
    )


def compute_ecalc_like_single_mtz(
    *,
    df_mtz: pd.DataFrame,
    f_col: str,
    sigf_col: str,
    eps_col: str,
    n_shells: int,
    smooth: bool,
) -> pd.DataFrame:
    """
    Compute ECALC-like normalization for a single dataset.
    """
    df_out = df_mtz.copy()

    df_out["F_eff"] = df_out[f_col] / np.sqrt(df_out[eps_col])
    df_out["F_eff_sq"] = df_out["F_eff"] ** 2

    shell_mean_f2 = (
        df_out
        .groupby(by="shell", sort=True)["F_eff_sq"]
        .mean()
        .rename("F_eff_sq_mean")
    )

    if smooth:
        shell_mean_f2_smoothed = smooth_shell_means_per_mtz(
            shell_means=shell_mean_f2,
            n_shells=n_shells,
        )
        df_out = df_out.join(shell_mean_f2_smoothed, on="shell")
        df_out["rms_Feff_shell"] = np.sqrt(df_out["F_eff_sq_mean_smoothed"])
    else:
        df_out = df_out.join(shell_mean_f2, on="shell")
        df_out["rms_Feff_shell"] = np.sqrt(df_out["F_eff_sq_mean"])

    df_out["rms_Feff_shell"] = df_out["rms_Feff_shell"].replace(to_replace=0.0, value=np.nan)
    df_out["E_ecalc_like"] = df_out["F_eff"] / df_out["rms_Feff_shell"]

    if sigf_col in df_out.columns:
        df_out["SIGF_eff"] = df_out[sigf_col] / np.sqrt(df_out[eps_col])
        df_out["SIGE_ecalc_like"] = df_out["SIGF_eff"] / df_out["rms_Feff_shell"]
    else:
        df_out["SIGE_ecalc_like"] = np.nan

    return df_out


def compute_ecalc_like_all(
    *,
    df: pd.DataFrame,
    mtz_id_col: str = "mtz_id",
    f_col: str = "FP",
    sigf_col: str = "SIGFP",
    eps_col: str = "EPSILON",
    n_shells: int = 10,
    smooth: bool = True,
) -> pd.DataFrame:
    """
    Compute ECALC-like normalization for all datasets, grouped by mtz_id.
    """
    df_out = df.copy()

    if eps_col not in df_out.columns:
        raise KeyError(f"Column '{eps_col}' not found; run epsilon computation first.")
    if "shell" not in df_out.columns:
        raise KeyError("Column 'shell' not found; run shell assignment first.")

    if mtz_id_col not in df_out.columns:
        logging.warning("[ECALC] Column '%s' not found; treating as single dataset.", mtz_id_col)
        return compute_ecalc_like_single_mtz(
            df_mtz=df_out,
            f_col=f_col,
            sigf_col=sigf_col,
            eps_col=eps_col,
            n_shells=n_shells,
            smooth=smooth,
        )

    groups: list[pd.DataFrame] = []
    for _, df_mtz in df_out.groupby(by=mtz_id_col, sort=False):
        if df_mtz.empty:
            continue
        df_norm = compute_ecalc_like_single_mtz(
            df_mtz=df_mtz,
            f_col=f_col,
            sigf_col=sigf_col,
            eps_col=eps_col,
            n_shells=n_shells,
            smooth=smooth,
        )
        groups.append(df_norm)

    df_all = pd.concat(objs=groups, axis=0, ignore_index=False)
    return df_all.reindex(index=df_out.index)


# =========================
# Pipeline
# =========================

class ReflectionTableBuilder:
    """
    Stage-01 pipeline:
    downloads source files and builds standardized reflection tables.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.base_dir = args.base_dir
        self.num_parallel_processes = int(args.num_processes)
        self.min_resol = float(args.min_resol)
        self.max_resol = float(args.max_resol)
        self.ecalc_n_shells = int(args.ecalc_n_shells)
        self.ecalc_smooth = bool(not args.ecalc_no_smooth)
        self.ecalc_mode = str(args.ecalc_mode)

        self.rcsb_query: str | list[str] | None = None
        self.to_remove: str | list[str] = ""
        self.search_meta: dict[str, Any] | None = None

        self.stage_01_dirname = "01_Download_Files"
        self.stage_02_dirname = "02_Run_RS"
        self.stage_03_dirname = "03_Run_Ecalc"

        self.download_dir = self.base_dir / self.stage_01_dirname
        self.rs_dir = self.base_dir / self.stage_02_dirname
        self.ecalc_dir = self.base_dir / self.stage_03_dirname
        self.lock_dir = self.base_dir / ".locks"

        for path in (self.download_dir, self.rs_dir, self.ecalc_dir, self.lock_dir):
            path.mkdir(parents=True, exist_ok=True)

        self.load_config_json(path=args.config_json)

        rcsb_list = coerce_id_list(value=self.rcsb_query)
        to_remove_set = set(coerce_id_list(value=self.to_remove))
        self.pdb_id_list = [pdb_id for pdb_id in self.unique_ordered(seq=rcsb_list) if pdb_id not in to_remove_set]

        script_tag = "01_download_build_reflection_tables"
        log_path = configure_logging_stage01(
            logs_dir=args.logs_dir,
            script_tag=script_tag,
            n_ids=len(self.pdb_id_list),
            num_processes=self.num_parallel_processes,
            min_resol=self.min_resol,
            max_resol=self.max_resol,
            ecalc_n_shells=self.ecalc_n_shells,
        )
        self.logger = logging.getLogger(script_tag)

        self.logger.info("[INIT] stage-01 source download and reflection-table generation")
        self.logger.info("[INIT] command_line=%s", " ".join(shlex.quote(arg) for arg in sys.argv))
        self.logger.info("[INIT] base_dir=%s", self.base_dir)
        self.logger.info("[INIT] config_json=%s", args.config_json)
        self.logger.info("[INIT] log_path=%s", log_path)
        self.logger.info("[CFG] ecalc_mode=%s", self.ecalc_mode)
        self.logger.info(
            "[CFG] min_resol=%.2fÅ max_resol=%.2fÅ ecalc_n_shells=%d ecalc_smooth=%s num_processes=%d",
            self.min_resol,
            self.max_resol,
            self.ecalc_n_shells,
            self.ecalc_smooth,
            self.num_parallel_processes,
        )
        self.logger.info("[CFG] PDB IDs after to_remove: %d", len(self.pdb_id_list))
        self.logger.info("[CFG] First 20 PDB IDs: %s", ", ".join(self.pdb_id_list[:20]))

        if self.search_meta:
            self.logger.info("[SEARCH] JSON annotations (search_meta):")
            for key, value in self.search_meta.items():
                self.logger.info("[SEARCH] %s: %s", key, value)

        self.valid_pdb_list: list[str] = []
        self.solvent_fraction_by_pdb: dict[str, float] = {}

    @staticmethod
    def unique_ordered(*, seq: Sequence[str]) -> list[str]:
        """Return unique items preserving the original order."""
        return list(dict.fromkeys(seq))

    def load_config_json(self, *, path: Path) -> None:
        """
        Load the JSON file containing at least rcsb_query, and optionally to_remove/search_meta.
        """
        try:
            with path.open(mode="r", encoding="utf-8") as file_handle:
                cfg = json.load(fp=file_handle)
        except Exception as exc:
            raise RuntimeError(f"[CFG] Failed to read JSON config '{path}': {exc}") from exc

        cfg_lower = {str(key).lower(): value for key, value in cfg.items()}

        rcsb_value = cfg_lower.get("rcsb_query")
        if not rcsb_value:
            raise RuntimeError("[CFG] JSON must define 'rcsb_query'")
        self.rcsb_query = rcsb_value

        self.to_remove = cfg_lower.get("to_remove", "") or ""

        search_meta_value = cfg_lower.get("search_meta")
        if isinstance(search_meta_value, dict):
            self.search_meta = search_meta_value

    @staticmethod
    def download_binary(*, url: str, out_path: Path, timeout_s: int = DEFAULT_HTTP_TIMEOUT_SECONDS) -> tuple[bool, str]:
        """
        Download a binary file to out_path.
        """
        try:
            with create_retry_session() as session:
                response = session.get(url=url, timeout=timeout_s)
                response.raise_for_status()
                with out_path.open(mode="wb") as file_handle:
                    file_handle.write(response.content)
            return True, ""
        except requests.RequestException as exc:
            return False, str(exc)

    @staticmethod
    def download_fasta_text(*, url: str, out_path: Path, timeout_s: int = DEFAULT_HTTP_TIMEOUT_SECONDS) -> tuple[bool, str]:
        """
        Download a FASTA file from RCSB.
        """
        headers = {"User-Agent": "stage01-rcsb-fasta/1.0"}
        try:
            with create_retry_session() as session:
                response = session.get(url=url, headers=headers, timeout=timeout_s)
                response.raise_for_status()
                text = response.text

            if len(text.encode("utf-8", errors="ignore")) < MIN_FASTA_SIZE_BYTES:
                return False, "FASTA payload too small"

            if not looks_like_fasta(text=text):
                return False, "Response does not look like FASTA"

            with out_path.open(mode="w", encoding="utf-8") as file_handle:
                file_handle.write(text)

            return True, ""
        except requests.RequestException as exc:
            return False, str(exc)

    def download_source_files(self) -> None:
        """
        Download PDB-REDO PDB/MTZ files and RCSB FASTA files.
        """
        pdbredo_base_url = "https://pdb-redo.eu/db/"
        rcsb_fasta_base_url = "https://www.rcsb.org/fasta/entry/"

        downloaded = 0
        failed = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_parallel_processes) as executor:
            futures: list[concurrent.futures.Future] = []

            for pdb_id in self.pdb_id_list:
                pdb_id_lower = pdb_id.lower()
                pdb_id_upper = pdb_id.upper()

                out_pdb = self.download_dir / f"{pdb_id_lower}_iREDO.pdb"
                out_mtz = self.download_dir / f"{pdb_id_lower}_iREDO.mtz"
                out_fasta = self.download_dir / f"{pdb_id_lower}_rcsb.fasta"

                url_pdb = f"{pdbredo_base_url}{pdb_id_lower}/{pdb_id_lower}_final.pdb"
                url_mtz = f"{pdbredo_base_url}{pdb_id_lower}/{pdb_id_lower}_final.mtz"
                url_fasta = f"{rcsb_fasta_base_url}{pdb_id_upper}"

                if (
                    out_pdb.exists() and out_pdb.stat().st_size > 0
                    and out_mtz.exists() and out_mtz.stat().st_size > 0
                    and out_fasta.exists() and out_fasta.stat().st_size > 0
                ):
                    self.logger.info("[DL] Skip %s: PDB/MTZ/FASTA already present", pdb_id_lower)
                    continue

                if not (out_pdb.exists() and out_pdb.stat().st_size > 0):
                    futures.append(executor.submit(self.download_binary, url=url_pdb, out_path=out_pdb))
                if not (out_mtz.exists() and out_mtz.stat().st_size > 0):
                    futures.append(executor.submit(self.download_binary, url=url_mtz, out_path=out_mtz))
                if not (out_fasta.exists() and out_fasta.stat().st_size > 0):
                    futures.append(executor.submit(self.download_fasta_text, url=url_fasta, out_path=out_fasta))

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Downloading source files",
            ):
                ok, err = future.result()
                if ok:
                    downloaded += 1
                else:
                    failed += 1
                    self.logger.error("[DL] Download failed: %s", err)

        self.logger.info("[DL] Done. Files written=%d failures=%d", downloaded, failed)

    def filter_valid_downloads(self) -> list[str]:
        """
        Filter PDB IDs that have valid downloaded PDB and MTZ files.
        """
        valid: list[str] = []
        excluded: list[str] = []

        for pdb_id in self.pdb_id_list:
            pdb_id_lower = pdb_id.lower()

            pdb_file = self.download_dir / f"{pdb_id_lower}_iREDO.pdb"
            mtz_file = self.download_dir / f"{pdb_id_lower}_iREDO.mtz"
            fasta_file = self.download_dir / f"{pdb_id_lower}_rcsb.fasta"

            reasons: list[str] = []

            if not pdb_file.exists():
                reasons.append("PDB missing")
            else:
                size_pdb = pdb_file.stat().st_size
                if size_pdb < MIN_FILE_SIZE_BYTES:
                    reasons.append(f"PDB too small ({size_pdb} bytes)")

            if not mtz_file.exists():
                reasons.append("MTZ missing")
            else:
                size_mtz = mtz_file.stat().st_size
                if size_mtz < MIN_FILE_SIZE_BYTES:
                    reasons.append(f"MTZ too small ({size_mtz} bytes)")

            if reasons:
                self.logger.warning("[CHK] Excluding %s: %s", pdb_id_lower, "; ".join(reasons))
                excluded.append(pdb_id_lower)
                continue

            if not fasta_file.exists() or fasta_file.stat().st_size < MIN_FASTA_SIZE_BYTES:
                self.logger.warning("[CHK] %s: FASTA missing or too small (non-fatal).", pdb_id_lower)

            valid.append(pdb_id_lower)

        self.valid_pdb_list = valid
        self.logger.info("[CHK] Valid PDB IDs after validation: %d", len(valid))
        if excluded:
            self.logger.info("[CHK] Excluded PDB IDs (%d): %s", len(excluded), ", ".join(excluded))
        return valid

    def fetch_solvent_fractions_for_valid_ids(self) -> None:
        """
        Populate solvent fractions (0..1) for valid PDB IDs.
        """
        if not self.valid_pdb_list:
            self.solvent_fraction_by_pdb = {}
            return

        self.logger.info("[RCSB] Fetching solvent fractions for %d PDB IDs", len(self.valid_pdb_list))
        results: dict[str, float] = {}

        def worker(pid: str) -> tuple[str, float]:
            fraction = fetch_solvent_fraction_from_rcsb(pdb_id=pid)
            return pid, fraction

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_parallel_processes) as executor:
            futures = [executor.submit(worker, pid) for pid in self.valid_pdb_list]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="RCSB solvent fraction",
            ):
                pid, fraction = future.result()
                results[str(pid).lower()] = float(fraction)

        self.solvent_fraction_by_pdb = results
        values = np.array(list(results.values()), dtype=float) if results else np.array([], dtype=float)
        n_ok = int(np.isfinite(values).sum())
        self.logger.info("[RCSB] Solvent fractions fetched. finite=%d / %d", n_ok, len(results))

    def build_rs_csv_for_single_id(self, pdb_id: str) -> None:
        """
        Generate the stage-02 reflection table CSV for one PDB ID.
        """
        mtz_in = self.download_dir / f"{pdb_id}_iREDO.mtz"
        out_csv = self.rs_dir / f"{pdb_id}_rs.csv"

        if not mtz_in.exists() or mtz_in.stat().st_size == 0:
            self.logger.warning("[RS] Missing iREDO MTZ for %s: %s", pdb_id, mtz_in)
            return

        if out_csv.exists() and out_csv.stat().st_size > 0:
            self.logger.info("[RS] Skip %s (CSV exists)", pdb_id)
            return

        wavelength, wavelength_source = extract_wavelength_from_mtz(mtz_path=mtz_in)
        if np.isfinite(wavelength):
            self.logger.info("[RS] %s: WAVELENGTH=%.6f Å (%s)", pdb_id, wavelength, wavelength_source)
        else:
            self.logger.info("[RS] %s: WAVELENGTH not found (%s)", pdb_id, wavelength_source)

        try:
            ds = rs.read_mtz(str(mtz_in))
        except Exception as exc:
            self.logger.error("[RS] Failed to read %s: %s", mtz_in, exc)
            return

        try:
            ds.compute_dHKL(inplace=True)
        except Exception as exc:
            self.logger.error("[RS] %s: compute_dHKL() failed; skipping. %s", pdb_id, exc)
            return

        try:
            dmin = float(self.max_resol)
            dmax = float(self.min_resol)
            n_before = int(ds.shape[0])
            mask = (ds["dHKL"] >= dmin) & (ds["dHKL"] <= dmax)
            ds = ds[mask]
            n_after = int(ds.shape[0])
            self.logger.info(
                "[RS] %s: resolution filter kept %d/%d reflections in [%.2f, %.2f] Å",
                pdb_id,
                n_after,
                n_before,
                dmin,
                dmax,
            )
        except Exception as exc:
            self.logger.warning("[RS] Resolution filter failed for %s: %s", pdb_id, exc)

        if "PHIC_ALL" in ds.columns:
            try:
                ds["PHIC_ALL"] = ds["PHIC_ALL"].astype("P")
            except Exception as exc:
                self.logger.warning("[RS] %s: could not cast PHIC_ALL to dtype 'P': %s", pdb_id, exc)
        else:
            self.logger.warning("[RS] %s: PHIC_ALL not found in MTZ columns. Columns=%s", pdb_id, list(ds.columns))

        try:
            ds = ds.label_centrics()
            ds = ds.canonicalize_phases()

            if "PHIC_ALL" in ds.columns:
                try:
                    ph = ds["PHIC_ALL"].to_numpy(dtype=float, copy=False)
                    ph = canonical_phase_deg(arr=ph)
                    ds["PHIC_ALL"] = ph
                    ds["PHIC_ALL"] = ds["PHIC_ALL"].astype("P")
                except Exception as exc:
                    self.logger.warning("[RS] %s: failed to canonicalize PHIC_ALL: %s", pdb_id, exc)

            ds = rs.utils.add_rfree(
                ds,
                fraction=0.05,
                ccp4_convention=True,
                seed=42,
                inplace=False,
            )
        except Exception as exc:
            self.logger.warning("[RS] %s: centric/phase canonicalization/add_rfree issues: %s", pdb_id, exc)

        try:
            ds = ds.infer_mtz_dtypes()
        except Exception as exc:
            self.logger.warning("[RS] %s: infer_mtz_dtypes() failed; proceeding anyway. %s", pdb_id, exc)

        ds = force_weight_mtztype(ds=ds, cols=("FOM",))

        dataset = ds.reset_index()

        try:
            dataset = force_centric_to_int01(df=dataset, col="CENTRIC")
        except Exception as exc:
            self.logger.warning("[RS] %s: failed to coerce CENTRIC to int01: %s", pdb_id, exc)

        try:
            sg_str = str(ds.spacegroup).split('"')[1]
        except Exception:
            sg_str = str(ds.spacegroup)

        cell = ds.cell
        dataset["SG"] = sg_str
        dataset["LENGTH_A"] = float(cell.a)
        dataset["LENGTH_B"] = float(cell.b)
        dataset["LENGTH_C"] = float(cell.c)
        dataset["ANGLE_ALPHA"] = float(cell.alpha)
        dataset["ANGLE_BETA"] = float(cell.beta)
        dataset["ANGLE_GAMMA"] = float(cell.gamma)

        dataset["WAVELENGTH"] = float(wavelength) if np.isfinite(wavelength) else np.nan
        solvent_fraction = float(self.solvent_fraction_by_pdb.get(str(pdb_id).lower(), float("nan")))
        dataset["SOLVENT_FRACTION"] = solvent_fraction

        required_cols = ["H", "K", "L", "FP", "SIGFP", "CENTRIC", "dHKL", "PHIC_ALL", "FOM"]
        if "FreeR_flag" in dataset.columns:
            required_cols.append("FreeR_flag")

        log_nan_counts_df(
            df=dataset,
            label=f"RS/{pdb_id} (pre-drop)",
            cols=["PHIC_ALL", "FOM", "FP", "SIGFP", "dHKL", "FreeR_flag", "CENTRIC", "SOLVENT_FRACTION"],
            logger=self.logger,
        )

        n_before_drop = int(dataset.shape[0])
        existing_required = [col for col in required_cols if col in dataset.columns]
        dataset = dataset.dropna(subset=existing_required)
        n_after_drop = int(dataset.shape[0])
        n_dropped = n_before_drop - n_after_drop
        if n_dropped > 0:
            self.logger.info(
                "[RS] %s: dropped %d row(s) due to NaNs in required columns (%s). Remaining=%d",
                pdb_id,
                n_dropped,
                ",".join(existing_required),
                n_after_drop,
            )

        dataset = dataset.reset_index(drop=True)

        if "PHIC_ALL" in dataset.columns:
            try:
                ph = dataset["PHIC_ALL"].to_numpy(dtype=float, copy=False)
                ph = canonical_phase_deg(arr=ph)
                ph_int = np.rint(ph).astype(np.int16)
                ph_int[ph_int == -180] = 180
                dataset["PHIC_ALL"] = ph_int
            except Exception as exc:
                self.logger.warning("[RS] %s: failed to convert PHIC_ALL to integer degrees: %s", pdb_id, exc)

        log_nan_counts_df(
            df=dataset,
            label=f"RS/{pdb_id} (post-drop)",
            cols=["PHIC_ALL", "FOM", "FP", "SIGFP", "dHKL", "FreeR_flag", "CENTRIC", "SOLVENT_FRACTION"],
            logger=self.logger,
        )

        if dataset.empty:
            self.logger.warning("[RS] %s: no reflections left after filtering/NaN cleanup; not writing RS CSV.", pdb_id)
            return

        dataset["mtz_id"] = pdb_id

        output_cols = [
            "H", "K", "L", "FP", "SIGFP", "CENTRIC", "dHKL",
            "PHIC_ALL", "FOM", "FreeR_flag",
            "LENGTH_A", "LENGTH_B", "LENGTH_C",
            "ANGLE_ALPHA", "ANGLE_BETA", "ANGLE_GAMMA", "SG",
            "WAVELENGTH",
            "SOLVENT_FRACTION",
            "mtz_id",
        ]
        output_cols_effective = [col for col in output_cols if col in dataset.columns]
        missing = [col for col in output_cols if col not in dataset.columns]
        if missing:
            self.logger.warning("[RS] %s: missing expected columns %s; they will be skipped.", pdb_id, missing)

        dataset.to_csv(path_or_buf=out_csv, index=False, columns=output_cols_effective)
        self.logger.info("[RS] OK → %s", out_csv)

    def run_ecalc_per_dataset_individual(self) -> None:
        """
        Memory-light ECALC mode: process one RS CSV at a time.
        """
        self.logger.info("[ECALC-PER] Starting per-dataset ECALC-like normalization")

        wrote = 0
        skipped = 0

        for pdb_id in self.valid_pdb_list:
            csv_in = self.rs_dir / f"{pdb_id}_rs.csv"
            out_csv = self.ecalc_dir / f"{pdb_id}_rs_ecalc.csv"

            if not csv_in.exists() or csv_in.stat().st_size == 0:
                self.logger.warning("[ECALC-PER] Missing RS CSV for %s: %s", pdb_id, csv_in)
                skipped += 1
                continue

            if out_csv.exists() and out_csv.stat().st_size > 0:
                self.logger.info("[ECALC-PER] Skip %s (ECALC CSV exists)", pdb_id)
                continue

            df = pd.read_csv(filepath_or_buffer=csv_in)
            if df.empty:
                self.logger.warning("[ECALC-PER] Empty RS CSV for %s; skipping.", pdb_id)
                skipped += 1
                continue

            if "mtz_id" not in df.columns:
                df["mtz_id"] = pdb_id

            required = ["mtz_id", "SG", "H", "K", "L", "FP", "SIGFP", "dHKL"]
            missing = [col for col in required if col not in df.columns]
            if missing:
                self.logger.error("[ECALC-PER] %s: missing required columns: %s", pdb_id, missing)
                skipped += 1
                continue

            try:
                df_eps = add_epsilon_with_gemmi_per_dataset(
                    df=df,
                    mtz_id_col="mtz_id",
                    sg_col="SG",
                    h_col="H",
                    k_col="K",
                    l_col="L",
                )

                df_shells = assign_resolution_shells(
                    df=df_eps,
                    d_col="dHKL",
                    n_shells=self.ecalc_n_shells,
                )

                df_ecalc = compute_ecalc_like_all(
                    df=df_shells,
                    mtz_id_col="mtz_id",
                    f_col="FP",
                    sigf_col="SIGFP",
                    eps_col="EPSILON",
                    n_shells=self.ecalc_n_shells,
                    smooth=self.ecalc_smooth,
                )

                df_out = df.copy()
                df_out["E"] = df_ecalc["E_ecalc_like"]
                df_out["SIGE"] = df_ecalc["SIGE_ecalc_like"]

            except Exception as exc:
                self.logger.exception("[ECALC-PER] %s: ECALC-like computation failed: %s", pdb_id, exc)
                skipped += 1
                continue

            df_out.to_csv(path_or_buf=out_csv, index=False)
            wrote += 1

            if wrote % 100 == 0:
                self.logger.info("[ECALC-PER] Progress: wrote %d files...", wrote)

        self.logger.info("[ECALC-PER] Done. Wrote=%d skipped=%d", wrote, skipped)

    def run_ecalc_global_dataframe(self) -> None:
        """
        Cohort-global ECALC mode: concatenate all RS CSVs and define global shells.
        """
        self.logger.info("[ECALC-DF] Starting cohort-global ECALC-like normalization")

        frames: list[pd.DataFrame] = []
        for pdb_id in self.valid_pdb_list:
            csv_in = self.rs_dir / f"{pdb_id}_rs.csv"
            if not csv_in.exists() or csv_in.stat().st_size == 0:
                self.logger.warning("[ECALC-DF] Missing RS CSV for %s: %s", pdb_id, csv_in)
                continue

            df = pd.read_csv(filepath_or_buffer=csv_in)

            log_nan_counts_df(
                df=df,
                label=f"ECALC-IN/{pdb_id}",
                cols=["PHIC_ALL", "FOM", "FP", "SIGFP", "dHKL", "FreeR_flag", "CENTRIC", "SOLVENT_FRACTION"],
                logger=self.logger,
            )

            if "mtz_id" not in df.columns:
                df["mtz_id"] = pdb_id
            frames.append(df)

        if not frames:
            self.logger.error("[ECALC-DF] No RS CSVs found; aborting ECALC-like step.")
            return

        df_all = pd.concat(objs=frames, axis=0, ignore_index=True)
        self.logger.info("[ECALC-DF] Aggregated RS DataFrame shape: %s", df_all.shape)

        required_cols = ["mtz_id", "SG", "H", "K", "L", "FP", "SIGFP", "dHKL"]
        missing_required = [col for col in required_cols if col not in df_all.columns]
        if missing_required:
            self.logger.error("[ECALC-DF] Missing required columns: %s", missing_required)
            return

        df_eps = add_epsilon_with_gemmi_per_dataset(
            df=df_all,
            mtz_id_col="mtz_id",
            sg_col="SG",
            h_col="H",
            k_col="K",
            l_col="L",
        )
        df_shells = assign_resolution_shells(
            df=df_eps,
            d_col="dHKL",
            n_shells=self.ecalc_n_shells,
        )
        df_ecalc = compute_ecalc_like_all(
            df=df_shells,
            mtz_id_col="mtz_id",
            f_col="FP",
            sigf_col="SIGFP",
            eps_col="EPSILON",
            n_shells=self.ecalc_n_shells,
            smooth=self.ecalc_smooth,
        )

        df_ecalc["E"] = df_ecalc["E_ecalc_like"]
        df_ecalc["SIGE"] = df_ecalc["SIGE_ecalc_like"]

        df_final = df_all.copy()
        df_final["E"] = df_ecalc["E"]
        df_final["SIGE"] = df_ecalc["SIGE"]

        for pdb_id in self.valid_pdb_list:
            sub = df_final[df_final["mtz_id"] == pdb_id].copy()
            if sub.empty:
                self.logger.warning("[ECALC-DF] No reflections for %s in ECALC DataFrame; skipping.", pdb_id)
                continue

            out_csv = self.ecalc_dir / f"{pdb_id}_rs_ecalc.csv"
            sub.to_csv(path_or_buf=out_csv, index=False)
            self.logger.info("[ECALC-DF] Wrote ECALC CSV → %s", out_csv)

        self.logger.info("[ECALC-DF] Cohort-global ECALC step complete.")

    def run_all(self) -> None:
        """
        Execute the full stage-01 pipeline.
        """
        self.download_source_files()
        self.filter_valid_downloads()
        self.fetch_solvent_fractions_for_valid_ids()

        self.logger.info("[RUN] Valid IDs: %d", len(self.valid_pdb_list))
        if not self.valid_pdb_list:
            self.logger.warning("[RUN] No valid inputs after validation; stopping.")
            return

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_parallel_processes) as executor:
            list(
                tqdm(
                    executor.map(self.build_rs_csv_for_single_id, self.valid_pdb_list),
                    total=len(self.valid_pdb_list),
                    desc="Writing 02_Run_RS CSVs",
                    unit="mtz",
                )
            )

        if self.ecalc_mode == "none":
            self.logger.info("[RUN] ecalc_mode=none: skipping ECALC step.")
        elif self.ecalc_mode == "per_dataset":
            self.run_ecalc_per_dataset_individual()
        else:
            self.run_ecalc_global_dataframe()

        n_rs = len(list(self.rs_dir.glob("*_rs.csv")))
        n_ecalc = len(list(self.ecalc_dir.glob("*_rs_ecalc.csv")))
        n_fasta = len(list(self.download_dir.glob("*_rcsb.fasta")))
        self.logger.info("[DONE] FASTA: %d | 02_Run_RS CSV: %d | 03_Run_Ecalc CSV: %d", n_fasta, n_rs, n_ecalc)


# =========================
# CLI
# =========================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-01 pipeline: download source files and build standardized reflection tables "
            "from PDB-REDO MTZ files, operating on PHIC_ALL."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base_dir", type=Path, default=BASE_DIR, help="Base directory for the project.")
    parser.add_argument("--num_processes", type=int, default=NUM_PROCESSES, help="Number of parallel workers.")
    parser.add_argument("--min_resol", type=float, default=MIN_RESOL, help="Minimum dHKL limit in Å.")
    parser.add_argument("--max_resol", type=float, default=MAX_RESOL, help="Maximum dHKL limit in Å.")
    parser.add_argument("--ecalc_n_shells", type=int, default=20, help="Number of ECALC resolution shells.")
    parser.add_argument("--ecalc_no_smooth", action="store_true", help="Disable 3-point smoothing across shells.")
    parser.add_argument(
        "--ecalc_mode",
        type=str,
        default="global_df",
        choices=["global_df", "per_dataset", "none"],
        help=(
            "ECALC mode: "
            "'global_df' builds a combined dataframe (RAM heavy); "
            "'per_dataset' computes E per RS CSV without concatenating (RAM light); "
            "'none' skips ECALC entirely."
        ),
    )
    parser.add_argument(
        "--config_json",
        type=Path,
        required=True,
        help='Path to the cohort JSON file containing at least {"rcsb_query": ...}.',
    )
    parser.add_argument(
        "--logs_dir",
        type=Path,
        default=Path("LOGS"),
        help="Directory where the timestamped log file will be written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_arguments()
    ReflectionTableBuilder(args=cli_args).run_all()
