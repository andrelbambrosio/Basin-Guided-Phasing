#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
41_make_antiphase_degradation_mtzs.py

Create anti-phase counterparts of stage-40 degradation Autobuild-ready MTZs.

This utility recursively scans a degradation output tree, finds PHIB-input MTZs
inside folders containing frac_<value>, filters by minimum degradation fraction,
and writes anti-phase MTZs by applying:

    PHIB := PHIB + 180 (mod 360)
    HLA  := -HLA
    HLB  := -HLB
    HLC/HLD unchanged

Preferred CLI:
    --degradation_root
    --min_degradation_fraction

Backward-compatible aliases:
    --root_dir
    --min_fraction
"""

from __future__ import annotations

import argparse
import logging
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import gemmi


RUN_PREFIX = "41_make_antiphase_degradation_mtzs"

FRAC_RE = re.compile(r"frac_(\d+\.\d+)")
MTZ_NAME_RE = re.compile(r"_PHIB_input\.mtz$", re.IGNORECASE)


def configure_logging(*, out_dir: Path, run_prefix: str = RUN_PREFIX) -> Path:
    logs_dir = out_dir.parent / "LOGS"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{run_prefix}__{ts}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(filename=str(log_path), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return log_path


def parse_degradation_fraction(*, path: Path) -> Optional[float]:
    for parent in [path.parent] + list(path.parents):
        match = FRAC_RE.search(str(parent.name))
        if match is not None:
            return float(match.group(1))
    return None


def find_target_mtz_files(
    *,
    degradation_root: Path,
    min_degradation_fraction: float,
) -> List[Path]:
    out_paths: List[Path] = []

    for mtz_path in degradation_root.rglob("*_PHIB_input.mtz"):
        if not mtz_path.is_file():
            continue

        degradation_fraction = parse_degradation_fraction(path=mtz_path)
        if degradation_fraction is None:
            continue

        if float(degradation_fraction) >= float(min_degradation_fraction):
            out_paths.append(mtz_path)

    return sorted(out_paths)


def get_column_index(
    *,
    mtz: gemmi.Mtz,
    label: str,
) -> Optional[int]:
    labels = list(mtz.column_labels())
    try:
        return labels.index(label)
    except ValueError:
        return None


def make_antiphase_mtz(
    *,
    input_mtz_path: Path,
    output_mtz_path: Path,
    phase_label: str = "PHIB",
    hla_label: str = "HLA",
    hlb_label: str = "HLB",
    hlc_label: str = "HLC",
    hld_label: str = "HLD",
) -> Tuple[bool, str]:
    try:
        mtz = gemmi.read_mtz_file(path=str(input_mtz_path))
    except Exception as exc:
        return False, f"Failed reading MTZ: {exc}"

    phase_index = get_column_index(mtz=mtz, label=phase_label)
    hla_index = get_column_index(mtz=mtz, label=hla_label)
    hlb_index = get_column_index(mtz=mtz, label=hlb_label)
    hlc_index = get_column_index(mtz=mtz, label=hlc_label)
    hld_index = get_column_index(mtz=mtz, label=hld_label)

    if phase_index is None:
        return False, f"Missing required phase column: {phase_label}"

    data = np.array(object=mtz, copy=False)

    phase_values = np.asarray(data[:, phase_index], dtype=np.float64)
    phase_values = np.mod(phase_values + 180.0, 360.0)
    data[:, phase_index] = phase_values

    if hla_index is not None:
        data[:, hla_index] = -np.asarray(data[:, hla_index], dtype=np.float64)

    if hlb_index is not None:
        data[:, hlb_index] = -np.asarray(data[:, hlb_index], dtype=np.float64)

    _ = hlc_index
    _ = hld_index

    try:
        output_mtz_path.parent.mkdir(parents=True, exist_ok=True)
        mtz.write_to_file(path=str(output_mtz_path))
    except Exception as exc:
        return False, f"Failed writing MTZ: {exc}"

    return True, "OK"


def build_output_path(*, input_mtz_path: Path) -> Path:
    return input_mtz_path.with_name(input_mtz_path.stem + "_anti.mtz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create anti-phase MTZ files for degradation PHIB-input MTZs above a minimum degradation fraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--degradation_root",
        type=Path,
        default=None,
        help="Root directory containing stage-40 degradation folders.",
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=None,
        help="Deprecated alias for --degradation_root.",
    )
    parser.add_argument(
        "--min_degradation_fraction",
        type=float,
        default=None,
        help="Minimum degradation fraction to include.",
    )
    parser.add_argument(
        "--min_fraction",
        type=float,
        default=None,
        help="Deprecated alias for --min_degradation_fraction.",
    )
    parser.add_argument(
        "--phase_label",
        type=str,
        default="PHIB",
        help="Phase column label to anti-phase.",
    )
    parser.add_argument(
        "--hla_label",
        type=str,
        default="HLA",
        help="HLA label.",
    )
    parser.add_argument(
        "--hlb_label",
        type=str,
        default="HLB",
        help="HLB label.",
    )
    parser.add_argument(
        "--hlc_label",
        type=str,
        default="HLC",
        help="HLC label (left unchanged).",
    )
    parser.add_argument(
        "--hld_label",
        type=str,
        default="HLD",
        help="HLD label (left unchanged).",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=None,
        help="Optional explicit path for the output summary CSV. Default: <degradation_root>/41_antiphase_mtz_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    degradation_root = args.degradation_root if args.degradation_root is not None else args.root_dir
    if degradation_root is None:
        raise SystemExit("Please provide --degradation_root (preferred) or --root_dir.")
    degradation_root = degradation_root.expanduser().resolve()

    min_degradation_fraction = (
        float(args.min_degradation_fraction)
        if args.min_degradation_fraction is not None
        else float(args.min_fraction) if args.min_fraction is not None
        else 0.7
    )

    if not degradation_root.is_dir():
        raise SystemExit(f"Degradation root not found: {degradation_root}")

    log_path = configure_logging(out_dir=degradation_root)
    logging.info("Run prefix: %s", RUN_PREFIX)
    logging.info("Command line: %s", " ".join(shlex.quote(x) for x in sys.argv))
    logging.info("degradation_root=%s", str(degradation_root))
    logging.info("min_degradation_fraction=%.4f", float(min_degradation_fraction))
    logging.info("phase_label=%s | hla_label=%s | hlb_label=%s | hlc_label=%s | hld_label=%s",
                 str(args.phase_label), str(args.hla_label), str(args.hlb_label), str(args.hlc_label), str(args.hld_label))

    target_paths = find_target_mtz_files(
        degradation_root=degradation_root,
        min_degradation_fraction=float(min_degradation_fraction),
    )

    logging.info("Found %d PHIB-input MTZ files with degradation fraction >= %.4f",
                 int(len(target_paths)), float(min_degradation_fraction))

    n_ok = 0
    n_fail = 0
    rows = []

    for input_mtz_path in target_paths:
        output_mtz_path = build_output_path(input_mtz_path=input_mtz_path)

        ok, message = make_antiphase_mtz(
            input_mtz_path=input_mtz_path,
            output_mtz_path=output_mtz_path,
            phase_label=str(args.phase_label),
            hla_label=str(args.hla_label),
            hlb_label=str(args.hlb_label),
            hlc_label=str(args.hlc_label),
            hld_label=str(args.hld_label),
        )

        degradation_fraction = parse_degradation_fraction(path=input_mtz_path)

        row = {
            "degradation_fraction": float(degradation_fraction) if degradation_fraction is not None else np.nan,
            "input_mtz": str(input_mtz_path),
            "output_mtz": str(output_mtz_path),
            "status": "OK" if ok else "FAIL",
            "message": str(message),
        }
        rows.append(row)

        if ok:
            n_ok += 1
            logging.info("[OK] frac=%.4f | %s -> %s",
                         float(degradation_fraction), input_mtz_path.name, output_mtz_path.name)
        else:
            n_fail += 1
            logging.error("[FAIL] frac=%.4f | %s | %s",
                          float(degradation_fraction) if degradation_fraction is not None else float("nan"),
                          input_mtz_path.name, str(message))

    summary_csv = (
        args.summary_csv.expanduser().resolve()
        if args.summary_csv is not None
        else degradation_root / "41_antiphase_mtz_summary.csv"
    )
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    logging.info("Summary CSV: %s", str(summary_csv))
    logging.info("Done. Success: %d | Failed: %d", int(n_ok), int(n_fail))

    print("")
    print(f"Done. Success: {n_ok} | Failed: {n_fail}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
