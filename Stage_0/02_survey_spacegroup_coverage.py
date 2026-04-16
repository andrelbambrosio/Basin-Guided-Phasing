#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_survey_spacegroup_coverage.py

Repository-ready stage-02 cohort audit script.

This script surveys space-group coverage across per-dataset CSV files
(for example, *_rs.csv or *_rs_ecalc.csv) and reports:

- per-space-group counts and percentages
- per-crystal-system counts and percentages
- number of unique space groups per crystal system
- compact lists of SGs per crystal system
- centrosymmetric space groups and datasets flagged for exclusion

Output policy:
- The main .log file is written directly into --logs_dir
- All tabular outputs are written into a dedicated run subfolder:
      <logs_dir>/<run_prefix>__<timestamp>/

Example:
    python 02_survey_spacegroup_coverage.py \
        --source_csv_dir ./03_Run_Ecalc \
        --source_csv_pattern "*_rs_ecalc.csv" \
        --sg_col SG \
        --logs_dir ./LOGS
"""

from __future__ import annotations

import argparse
import glob
import logging
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import gemmi  # type: ignore
except Exception as exc:
    raise SystemExit(f"[FATAL] gemmi is required for this script: {exc}")


RUN_PREFIX = "02_survey_spacegroup_coverage"


def now_local() -> datetime:
    """Return current local time."""
    return datetime.now()


def ensure_dir(*, path: Path) -> None:
    """Create directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def setup_logging(*, logs_dir: Path, prefix: str = RUN_PREFIX) -> tuple[str, Path, Path]:
    """
    Configure logging and create the run output directory.

    Returns
    -------
    tuple
        (timestamp, log_path, run_output_dir)
    """
    ensure_dir(path=logs_dir)
    timestamp = now_local().strftime("%Y%m%d_%H%M%S")

    log_path = logs_dir / f"{prefix}__{timestamp}.log"
    run_output_dir = logs_dir / f"{prefix}__{timestamp}"
    ensure_dir(path=run_output_dir)

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
    logging.info("Run output folder (non-log artifacts): %s", run_output_dir)
    return timestamp, log_path, run_output_dir


def parse_patterns(*, value: str) -> list[str]:
    """
    Parse comma-separated glob patterns.
    """
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("--source_csv_pattern must include at least one pattern.")
    return parts


def discover_files(*, folder: Path, patterns: list[str]) -> list[Path]:
    """
    Discover files matching the provided glob patterns.
    """
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(path) for path in glob.glob(str(folder / pattern), recursive=("**" in pattern)))
    return sorted(set(files))


def dataset_id_from_filename(*, path: Path) -> str:
    """
    Infer dataset ID from the filename prefix before the first underscore.
    """
    head = path.name.split("_")[0]
    return head if head else path.stem


def normalize_sg_display(*, sg_raw: str) -> str:
    """
    Normalize SG string for reporting.
    """
    value = str(sg_raw).strip().strip('"').strip("'")
    value = re.sub(pattern=r"\s+", repl=" ", string=value)
    return value if value else "<MISSING>"


def sg_compact(*, sg_display: str) -> str:
    """
    Return compact SG representation without spaces.
    """
    if sg_display == "<MISSING>":
        return sg_display
    return sg_display.replace(" ", "")


def crystal_system_str(*, sg_obj: Any) -> str:
    """
    Extract crystal system string from a gemmi.SpaceGroup object.
    """
    try:
        crystal_system = sg_obj.crystal_system()
        crystal_system_str_lower = str(crystal_system).lower()
        for key in (
            "triclinic",
            "monoclinic",
            "orthorhombic",
            "tetragonal",
            "trigonal",
            "hexagonal",
            "cubic",
        ):
            if key in crystal_system_str_lower:
                return key
        return crystal_system_str_lower.replace("crystalsystem.", "")
    except Exception:
        return "unknown"


def is_centrosymmetric(*, sg_obj: Any) -> bool:
    """
    Determine whether a space group is centrosymmetric.
    """
    for attr in ("is_centrosymmetric", "isCentrosymmetric"):
        if hasattr(sg_obj, attr):
            try:
                value = getattr(sg_obj, attr)
                return bool(value()) if callable(value) else bool(value)
            except Exception:
                pass

    try:
        for operation in list(sg_obj.operations()):
            try:
                a = operation.apply_to_hkl([1, 0, 0])
                b = operation.apply_to_hkl([0, 1, 0])
                c = operation.apply_to_hkl([0, 0, 1])
                if tuple(a) == (-1, 0, 0) and tuple(b) == (0, -1, 0) and tuple(c) == (0, 0, -1):
                    return True
            except Exception:
                continue
    except Exception:
        pass

    return False


def percent(*, n: int, denom: int) -> float:
    """
    Compute percentage safely.
    """
    return 100.0 * float(n) / float(denom) if denom > 0 else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Survey space-group coverage across per-dataset CSV files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source_csv_dir",
        type=Path,
        required=True,
        help="Folder containing per-dataset CSV files.",
    )
    parser.add_argument(
        "--source_csv_pattern",
        type=str,
        required=True,
        help='Comma-separated glob patterns, for example "*_rs_ecalc.csv".',
    )
    parser.add_argument(
        "--sg_col",
        type=str,
        default="SG",
        help="Name of the space-group column.",
    )
    parser.add_argument(
        "--logs_dir",
        type=Path,
        default=Path("./LOGS"),
        help="Directory for log files and run output folders.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Top-N space groups to summarize in the log.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_dir = args.source_csv_dir.resolve()
    patterns = parse_patterns(value=args.source_csv_pattern)

    _, log_path, run_output_dir = setup_logging(logs_dir=args.logs_dir)

    logging.info("Run prefix: %s", RUN_PREFIX)
    logging.info("Command line: %s", " ".join(shlex.quote(arg) for arg in sys.argv))
    logging.info("[ARGS] source_csv_dir=%s", source_dir)
    logging.info("[ARGS] source_csv_pattern=%s", patterns)
    logging.info("[ARGS] sg_col=%s", args.sg_col)
    logging.info("[ARGS] logs_dir=%s", args.logs_dir.resolve())
    logging.info("[ARGS] top_n=%d", int(args.top_n))

    csv_paths = discover_files(folder=source_dir, patterns=patterns)
    if not csv_paths:
        raise SystemExit(f"[ERROR] No CSV files found in {source_dir} for patterns {patterns}")
    logging.info("[DISCOVER] %d CSV file(s) found.", len(csv_paths))

    rows: list[dict[str, object]] = []
    missing_sg_files: list[str] = []

    for path in csv_paths:
        dataset_id = dataset_id_from_filename(path=path)

        try:
            df_head = pd.read_csv(filepath_or_buffer=path, nrows=1)
        except Exception as exc:
            logging.warning("[READ] skip %s (%s)", path.name, exc)
            continue

        if args.sg_col not in df_head.columns:
            missing_sg_files.append(path.name)
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "file": path.name,
                    "SG": "<MISSING>",
                    "SG_compact": "<MISSING>",
                    "crystal_system": "unknown",
                    "centrosymmetric": np.nan,
                }
            )
            continue

        sg_raw = df_head[args.sg_col].iloc[0]
        sg_display = normalize_sg_display(sg_raw=str(sg_raw))
        sg_compact_value = sg_compact(sg_display=sg_display)

        crystal_system = "unknown"
        centrosymmetric = np.nan

        if sg_display != "<MISSING>":
            try:
                sg_obj = gemmi.SpaceGroup(sg_display)
                crystal_system = crystal_system_str(sg_obj=sg_obj)
                centrosymmetric = bool(is_centrosymmetric(sg_obj=sg_obj))
            except Exception as exc:
                logging.warning("[SG] %s: gemmi.SpaceGroup failed (%s)", sg_display, exc)
                crystal_system = "unknown"
                centrosymmetric = np.nan

        rows.append(
            {
                "dataset_id": dataset_id,
                "file": path.name,
                "SG": sg_display,
                "SG_compact": sg_compact_value,
                "crystal_system": crystal_system,
                "centrosymmetric": centrosymmetric,
            }
        )

    df = pd.DataFrame(rows)
    df_ok = df[df["SG"] != "<MISSING>"].copy()
    n_datasets = int(df_ok.shape[0])

    out_datasets_csv = run_output_dir / f"{RUN_PREFIX}__datasets.csv"
    df.to_csv(path_or_buf=out_datasets_csv, index=False)

    sg_counts = (
        df_ok
        .groupby(by="SG", dropna=False, sort=False)
        .size()
        .rename("n_datasets")
        .reset_index()
    )
    sg_counts["pct"] = sg_counts["n_datasets"].apply(lambda x: percent(n=int(x), denom=n_datasets))
    sg_counts["SG_compact"] = sg_counts["SG"].apply(lambda s: sg_compact(sg_display=str(s)))
    sg_counts = sg_counts.sort_values(by="n_datasets", ascending=False, kind="mergesort")

    sg_centro = (
        df_ok
        .groupby(by="SG", dropna=False, sort=False)["centrosymmetric"]
        .agg(lambda s: bool(np.nanmax(np.asarray(s, dtype=float))) if np.any(pd.notna(s)) else np.nan)
        .reset_index()
        .rename(columns={"centrosymmetric": "centrosymmetric_any"})
    )
    sg_counts = sg_counts.merge(sg_centro, on="SG", how="left", validate="one_to_one")

    out_sg_csv = run_output_dir / f"{RUN_PREFIX}__spacegroups.csv"
    sg_counts.to_csv(path_or_buf=out_sg_csv, index=False)

    cs_summary = (
        df_ok
        .groupby(by="crystal_system", dropna=False, sort=False)
        .agg(
            n_datasets=pd.NamedAgg(column="dataset_id", aggfunc="count"),
            n_spacegroups=pd.NamedAgg(column="SG", aggfunc=lambda s: int(pd.Series(s).nunique())),
        )
        .reset_index()
    )
    cs_summary["pct_datasets"] = cs_summary["n_datasets"].apply(lambda x: percent(n=int(x), denom=n_datasets))
    cs_summary = cs_summary.sort_values(by="n_datasets", ascending=False, kind="mergesort")

    cs_to_sgs: dict[str, str] = {}
    for crystal_system_name, sub in df_ok.groupby(by="crystal_system", sort=False):
        sgs = sorted(pd.Series(sub["SG"]).unique().tolist())
        cs_to_sgs[str(crystal_system_name)] = ", ".join([sg_compact(sg_display=sg) for sg in sgs])
    cs_summary["spacegroups_compact_list"] = cs_summary["crystal_system"].apply(
        lambda c: cs_to_sgs.get(str(c), "")
    )

    out_cs_csv = run_output_dir / f"{RUN_PREFIX}__crystal_systems.csv"
    cs_summary.to_csv(path_or_buf=out_cs_csv, index=False)

    logging.info("[WRITE] datasets table -> %s", out_datasets_csv)
    logging.info("[WRITE] spacegroups table -> %s", out_sg_csv)
    logging.info("[WRITE] crystal systems table -> %s", out_cs_csv)

    logging.info("")
    logging.info("=== Space-group coverage summary ===")
    logging.info("Datasets with SG parsed: %d", n_datasets)
    logging.info("Distinct SGs: %d", int(sg_counts.shape[0]))

    top_n = int(max(1, args.top_n))
    top = sg_counts.head(top_n)
    top_str = "; ".join(
        [
            f"{row['SG_compact']} ({int(row['n_datasets'])}; {row['pct']:.1f}%)"
            for _, row in top.iterrows()
        ]
    )
    logging.info("Top %d SGs: %s", top_n, top_str)

    tail = sg_counts[sg_counts["pct"] <= 3.0]
    if not tail.empty:
        logging.info("Long tail: %d SGs contribute <= ~3%% each.", int(tail.shape[0]))

    centro_sgs = sg_counts[sg_counts["centrosymmetric_any"] == True]  # noqa: E712
    if not centro_sgs.empty:
        centro_list = ", ".join(centro_sgs["SG_compact"].tolist())
        logging.warning("Centrosymmetric SGs detected (exclude): %s", centro_list)
        df_centro = df_ok[df_ok["centrosymmetric"] == True].copy()  # noqa: E712
        logging.warning(
            "Centrosymmetric datasets (n=%d): %s",
            int(df_centro.shape[0]),
            ", ".join(df_centro["dataset_id"].tolist()),
        )
    else:
        logging.info("No centrosymmetric SGs detected.")

    logging.info("")
    logging.info("=== Crystal system summary ===")
    for _, row in cs_summary.iterrows():
        crystal_system_name = str(row["crystal_system"])
        logging.info(
            "%s: %d datasets spanning %d SGs (%.1f%%) | SGs: %s",
            crystal_system_name,
            int(row["n_datasets"]),
            int(row["n_spacegroups"]),
            float(row["pct_datasets"]),
            str(row["spacegroups_compact_list"]),
        )

    if missing_sg_files:
        logging.warning(
            "Files missing SG column (n=%d). First 20: %s",
            len(missing_sg_files),
            ", ".join(missing_sg_files[:20]),
        )

    logging.info("[DONE] Space-group survey complete.")
    logging.info("Log: %s", log_path)
    logging.info("Outputs folder: %s", run_output_dir)
    logging.info("  - %s", out_datasets_csv.name)
    logging.info("  - %s", out_sg_csv.name)
    logging.info("  - %s", out_cs_csv.name)
    print("\n[DONE] Space-group survey complete.")
    print(f"Log: {log_path}")
    print(f"Outputs folder: {run_output_dir}")
    print(f"  - {out_datasets_csv.name}")
    print(f"  - {out_sg_csv.name}")
    print(f"  - {out_cs_csv.name}")


if __name__ == "__main__":
    main()
