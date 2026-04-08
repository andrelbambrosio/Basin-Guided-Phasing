#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_select_cohort_ids.py

Repository-ready script to reproducibly sample PDB IDs from one or more
RCSB query-output TXT files.

Main behavior:
- Reads all .txt files from --query_dir (optionally recursively)
- Extracts, validates, and deduplicates PDB IDs
- Samples a reproducible subset using a seed-stable permutation
- Writes a JSON file with the selected IDs and provenance metadata
- Writes a timestamped log file
- Optionally ingests a compact query-summary JSON file created by the user

Prefix-stable guarantee:
- For a fixed seed and identical pooled ID order, increasing --n preserves
  the smaller sample as a prefix of the larger sample.

Example:
    python 00_select_cohort_ids.py \
        --query_dir ./00_RCSB_Query \
        --n 15000 \
        --seed 42 \
        --out_path ./sampled_cohort_15000.json \
        --logs_dir ./LOGS \
        --query_summary_file ./query_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PDB_ID_RE = re.compile(pattern=r"^[0-9][A-Za-z0-9]{3}$")

DEFAULT_ADVANCED_QUERY_ATTRIBUTES = (
    'Structure Determination Methodology = "experimental"\n'
    "AND\n"
    'Experimental Method = "X-RAY DIFFRACTION"\n'
    "AND\n"
    "Number of Distinct Protein Entities = 1\n"
    "AND\n"
    "Number of Protein Instances (Chains) per Assembly = 1\n"
    "AND\n"
    "Refinement Resolution <= 1.7"
)


def now_local() -> datetime:
    """Return the current local time as a naive datetime."""
    return datetime.now()


def configure_logging(*, logs_dir: Path, script_tag: str, n: int, seed: int) -> Path:
    """
    Configure root logging to both stderr and a timestamped log file.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = now_local().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{script_tag}__n{n:04d}__seed{seed:04d}__{timestamp}.log"
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


def list_txt_files(*, query_dir: Path, recursive: bool) -> list[Path]:
    """
    Return all TXT files in the query directory, sorted for reproducibility.
    """
    if not query_dir.exists():
        raise FileNotFoundError(f"--query_dir does not exist: {query_dir}")
    if not query_dir.is_dir():
        raise NotADirectoryError(f"--query_dir is not a directory: {query_dir}")

    if recursive:
        txt_files = sorted(path for path in query_dir.rglob(pattern="*.txt") if path.is_file())
    else:
        txt_files = sorted(path for path in query_dir.glob(pattern="*.txt") if path.is_file())

    return txt_files


def iter_tokens_from_txt(*, path: Path) -> Iterable[str]:
    """
    Yield raw tokens from a TXT file.

    Expected input is primarily comma-separated PDB IDs, but the parser is
    tolerant to newlines and extra surrounding whitespace.
    """
    with path.open(mode="r", encoding="utf-8") as file_handle:
        for line in file_handle:
            for token in line.replace("\n", ",").split(","):
                cleaned = token.strip()
                if cleaned:
                    yield cleaned


def pool_ids_from_files(*, files: Sequence[Path]) -> tuple[list[str], dict[str, int]]:
    """
    Pool, validate, and deduplicate PDB IDs from multiple files.

    Returns
    -------
    tuple
        unique_ids_preserve_order, stats
    """
    total_tokens = 0
    valid_tokens = 0
    invalid_tokens = 0

    seen_ids: set[str] = set()
    unique_ids: list[str] = []

    for path in files:
        for raw_token in iter_tokens_from_txt(path=path):
            total_tokens += 1
            pdb_id = raw_token.upper()

            if not PDB_ID_RE.fullmatch(string=pdb_id):
                invalid_tokens += 1
                continue

            valid_tokens += 1
            if pdb_id not in seen_ids:
                seen_ids.add(pdb_id)
                unique_ids.append(pdb_id)

    stats = {
        "total_tokens": total_tokens,
        "valid_tokens": valid_tokens,
        "invalid_tokens": invalid_tokens,
        "unique_valid_ids": len(unique_ids),
    }
    return unique_ids, stats


def sample_ids_prefix_stable(*, unique_ids: Sequence[str], n: int, seed: int) -> list[str]:
    """
    Sample PDB IDs using a single seeded permutation.

    This preserves the prefix-stable property: for the same seed and the same
    pooled ordered ID list, sample(n_small) is a prefix of sample(n_large).
    """
    if n <= 0:
        raise ValueError("--n must be greater than zero.")
    if n > len(unique_ids):
        raise ValueError(
            f"Requested n={n}, but only {len(unique_ids)} unique valid IDs are available."
        )

    rng = np.random.default_rng(seed=seed)
    permutation = rng.permutation(len(unique_ids))
    chosen_indices = permutation[:n]
    sampled_ids = [unique_ids[int(index)] for index in chosen_indices]
    return sampled_ids


def read_query_summary_file(*, query_summary_file: Path | None) -> dict[str, Any]:
    """
    Read a compact user-supplied query summary JSON file.

    Expected format:
    {
      "queried_on": "2025-12-31",
      "advanced_query_attributes": "Line 1\\nAND\\nLine 2",
      "query_label": "single-chain high-resolution X-ray cohort",
      "notes": "Optional extra provenance"
    }

    Only JSON is supported to avoid extra dependencies.
    """
    if query_summary_file is None:
        return {}

    if not query_summary_file.exists():
        raise FileNotFoundError(f"--query_summary_file does not exist: {query_summary_file}")
    if not query_summary_file.is_file():
        raise ValueError(f"--query_summary_file is not a file: {query_summary_file}")

    with query_summary_file.open(mode="r", encoding="utf-8") as file_handle:
        payload = json.load(fp=file_handle)

    if not isinstance(payload, dict):
        raise ValueError("--query_summary_file must contain a JSON object.")

    allowed_keys = {
        "queried_on",
        "advanced_query_attributes",
        "query_label",
        "notes",
        "source",
        "query_url",
        "query_id",
    }
    unknown_keys = sorted(set(payload.keys()) - allowed_keys)
    if unknown_keys:
        raise ValueError(
            "--query_summary_file contains unsupported key(s): "
            + ", ".join(unknown_keys)
        )

    if "queried_on" in payload and not isinstance(payload["queried_on"], str):
        raise ValueError('"queried_on" in query summary must be a string.')
    if "advanced_query_attributes" in payload and not isinstance(payload["advanced_query_attributes"], str):
        raise ValueError('"advanced_query_attributes" in query summary must be a string.')

    return payload


def resolve_query_metadata(
    *,
    queried_on_cli: str,
    advanced_query_attributes_cli: str,
    query_summary_payload: dict[str, Any],
    query_summary_file: Path | None,
) -> dict[str, Any]:
    """
    Resolve query metadata from CLI defaults plus optional user summary file.

    Values from the summary file override the CLI values when present.
    """
    metadata = {
        "queried_on": queried_on_cli,
        "advanced_query_attributes": advanced_query_attributes_cli,
        "query_label": None,
        "notes": None,
        "source": None,
        "query_url": None,
        "query_id": None,
        "query_summary_file": str(query_summary_file) if query_summary_file is not None else None,
    }

    for key, value in query_summary_payload.items():
        metadata[key] = value

    return metadata


def build_payload(
    *,
    sampled_ids: Sequence[str],
    n_random: int,
    pooled_stats: dict[str, int],
    query_metadata: dict[str, Any],
    files_consumed: Sequence[Path],
) -> dict[str, Any]:
    """
    Build the JSON payload describing the selected cohort and provenance.
    """
    run_time = now_local()
    search_meta = {
        "advanced_query_attributes": str(query_metadata["advanced_query_attributes"]),
        "queried_on": str(query_metadata["queried_on"]),
        "num_entries": int(pooled_stats["unique_valid_ids"]),
        "notes": f"JSON created on {run_time.strftime('%d/%m/%Y, %H:%M')}",
        "run_timestamp_iso": run_time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pooled_stats": dict(pooled_stats),
        "files_consumed": [str(path) for path in files_consumed],
    }

    optional_keys = ("query_label", "source", "query_url", "query_id", "query_summary_file")
    for key in optional_keys:
        value = query_metadata.get(key)
        if value is not None:
            search_meta[key] = value

    user_notes = query_metadata.get("notes")
    if user_notes is not None:
        search_meta["user_notes"] = user_notes

    payload = {
        "rcsb_query": list(sampled_ids),
        "to_remove": [],
        "N_random": int(n_random),
        "search_meta": search_meta,
    }
    return payload


def write_json_query(*, out_path: Path, payload: dict[str, Any]) -> None:
    """
    Write the sampled cohort JSON payload to disk.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open(mode="w", encoding="utf-8") as file_handle:
        json.dump(obj=payload, fp=file_handle, indent=2, sort_keys=False)
        file_handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pool PDB IDs from TXT files, sample N IDs reproducibly, and write "
            "a JSON file with provenance metadata."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query_dir",
        type=Path,
        required=True,
        help="Directory containing TXT files with PDB IDs.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also read TXT files from subdirectories of --query_dir.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=250,
        help="Number of PDB IDs to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        default=Path("sampled_cohort.json"),
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--logs_dir",
        type=Path,
        default=Path("LOGS"),
        help="Directory where the timestamped log file will be written.",
    )
    parser.add_argument(
        "--queried_on",
        type=str,
        default=now_local().strftime("%Y-%m-%d"),
        help='Fallback date when the underlying RCSB query was run, in "YYYY-MM-DD" format.',
    )
    parser.add_argument(
        "--advanced_query_attributes",
        type=str,
        default=DEFAULT_ADVANCED_QUERY_ATTRIBUTES,
        help="Fallback multiline description of the advanced query used to generate the TXT file(s).",
    )
    parser.add_argument(
        "--query_summary_file",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing compact query provenance, including queried_on "
            "and advanced_query_attributes. Values in this file override the CLI fallbacks."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    script_tag = "00_select_cohort_ids"
    log_path = configure_logging(
        logs_dir=args.logs_dir,
        script_tag=script_tag,
        n=args.n,
        seed=args.seed,
    )

    logging.info("Run prefix: %s", script_tag)
    logging.info("Command line: %s", " ".join(shlex.quote(arg) for arg in sys.argv))

    query_summary_payload = read_query_summary_file(query_summary_file=args.query_summary_file)
    query_metadata = resolve_query_metadata(
        queried_on_cli=args.queried_on,
        advanced_query_attributes_cli=args.advanced_query_attributes,
        query_summary_payload=query_summary_payload,
        query_summary_file=args.query_summary_file,
    )

    logging.info("Run parameters:")
    logging.info("  query_dir                 = %s", args.query_dir)
    logging.info("  recursive                 = %s", args.recursive)
    logging.info("  n                         = %d", args.n)
    logging.info("  seed                      = %d", args.seed)
    logging.info("  out_path                  = %s", args.out_path)
    logging.info("  logs_dir                  = %s", args.logs_dir)
    logging.info("  query_summary_file        = %s", args.query_summary_file)
    logging.info("Resolved query metadata:")
    logging.info("  queried_on                = %s", query_metadata["queried_on"])
    logging.info("  advanced_query_attributes =\n%s", query_metadata["advanced_query_attributes"])
    if query_metadata.get("query_label") is not None:
        logging.info("  query_label               = %s", query_metadata["query_label"])
    if query_metadata.get("source") is not None:
        logging.info("  source                    = %s", query_metadata["source"])
    if query_metadata.get("query_url") is not None:
        logging.info("  query_url                 = %s", query_metadata["query_url"])
    if query_metadata.get("query_id") is not None:
        logging.info("  query_id                  = %s", query_metadata["query_id"])

    txt_files = list_txt_files(query_dir=args.query_dir, recursive=args.recursive)
    if not txt_files:
        logging.error("No TXT files found in query_dir: %s", args.query_dir)
        return 1

    logging.info("Discovered %d TXT file(s), sorted for reproducibility:", len(txt_files))
    for path in txt_files:
        logging.info("  - %s", path)

    unique_ids, stats = pool_ids_from_files(files=txt_files)

    logging.info("Pooled ID statistics:")
    logging.info("  total tokens read            = %d", stats["total_tokens"])
    logging.info("  valid PDB-ID tokens          = %d", stats["valid_tokens"])
    logging.info("  invalid tokens dropped       = %d", stats["invalid_tokens"])
    logging.info("  unique valid PDB IDs (pool)  = %d", stats["unique_valid_ids"])

    sampled_ids = sample_ids_prefix_stable(unique_ids=unique_ids, n=args.n, seed=args.seed)

    logging.info("Sampling result:")
    logging.info("  selected n                   = %d", len(sampled_ids))
    logging.info("  seed                         = %d", args.seed)
    logging.info(
        "  prefix property              = sample(n_small) is prefix of sample(n_large) "
        "for the same seed and pooled ordered ID list"
    )

    payload = build_payload(
        sampled_ids=sampled_ids,
        n_random=args.n,
        pooled_stats=stats,
        query_metadata=query_metadata,
        files_consumed=txt_files,
    )
    write_json_query(out_path=args.out_path, payload=payload)

    summary_message = (
        f"Wrote JSON: {args.out_path} | "
        f"N_random={len(sampled_ids)} | "
        f"pool_unique={stats['unique_valid_ids']} | "
        f"log={log_path}"
    )
    logging.info("JSON written to: %s", args.out_path)
    logging.info("Log file saved at: %s", log_path)
    logging.info("%s", summary_message)
    print(summary_message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
