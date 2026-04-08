#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
22_run_phenix_autobuild.py

python 22_run_phenix_autobuild.py  \
  --base_folder ./ \
  --input_phenix_folder ./21_Phenix_Autobuild_Inputs/ \
  --methods iREDO,K2 \
  --jobs 10 \
  --nproc 3

Run phenix.autobuild for PHIB-input MTZs produced by PC32, using per-PDB FASTA files.

Expected MTZ layout (PC32):
  <input_phenix_folder>/<suffix>/<pdb_id>/<pdb_id>_<suffix>_PHIB_input.mtz

Defaults:
  --input_phenix_folder defaults to: <base_folder>/21_Phenix_Autobuild_Inputs
  --fasta_dir defaults to: <base_folder>/01_Download_Files

Critical design choice:
  By default we restrict to PDB IDs that have MTZs for ALL requested methods (intersection),
  so method comparisons are paired on the same dataset list.

Resume-safe:
  Job is treated as successful ONLY IF at least one folder:
    <workdir>/AutoBuild_run_*/overall_best.pdb
    <workdir>/AutoBuild_run_*/overall_best_refine_data.mtz
  and corresponding AutoBuild_run_*.log tail contains:
    "Citations for AutoBuild:"

Outputs:
  - AutoBuild_run_* folders will be created by Phenix inside each <workdir>.
  - A manifest CSV is written under: <output_root>/22_autobuild_manifest_<timestamp>.csv
    where output_root defaults to <base_folder>/22_Run_AutoBuild

UPDATED (non-idling scheduler + method-aware logging)
-----------------------------------------------------
- Never sits idle: as soon as any autobuild finishes, we immediately submit the next runnable task,
  keeping up to --jobs concurrent processes until all tasks are done.
- Preference policy:
    1) Prefer submitting remaining methods for PDB IDs that already started (to complete method sets).
    2) If none are pending among active PDBs, start methods for the next new PDB.
- Logging:
    Prints method name for each run:
      [RUN] method=<suffix> | pdb_id=<id> | workdir=<...>
      [SEQ-CHECK] method=<suffix> | expected=<id> | ...
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import shlex
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import gemmi

def configure_logging(*, output_root: Path, run_prefix: str = "22_run_phenix_autobuild") -> Path:
    logs_dir = output_root.parent / "LOGS"
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


import numpy as np


# =============================================================================
# Lightweight progress bar fallback
# =============================================================================

class SimpleBar:
    def __init__(self, total: int, desc: str) -> None:
        self.total = max(int(total), 1)
        self.desc = desc
        self.n = 0
        self._render()

    def update(self, n: int = 1) -> None:
        self.n += int(n)
        self._render()

    def close(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render(self) -> None:
        width: int = 40
        filled: int = int(width * min(self.n, self.total) / self.total)
        bar: str = "#" * filled + "-" * (width - filled)
        pct: float = 100.0 * min(self.n, self.total) / self.total
        sys.stdout.write(f"\r{self.desc} [{bar}] {pct:6.2f}% ({self.n}/{self.total})")
        sys.stdout.flush()


def get_progress(total: int, desc: str):
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(total=total, desc=desc, unit="job", leave=True)
    except Exception:
        return SimpleBar(total=total, desc=desc)


# =============================================================================
# Utilities
# =============================================================================

_PDB_RE = re.compile(pattern=r"([0-9A-Za-z]{4})")


def parse_pdb_id_from_mtz_filename(*, mtz_filename: str) -> Optional[str]:
    token = mtz_filename.split("_")[0].strip()
    if len(token) == 4 and token.isalnum():
        return token.lower()
    m = _PDB_RE.search(mtz_filename)
    return m.group(1).lower() if m else None


def parse_pdb_id_from_fasta_filename(*, fasta_path: Path) -> Optional[str]:
    m = _PDB_RE.search(fasta_path.name)
    return m.group(1).lower() if m else None


def read_fasta_sequence_stats(*, fasta_path: Path) -> Tuple[int, int]:
    n_seq = 0
    total_len = 0
    with open(fasta_path, mode="r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                n_seq += 1
                continue
            seq = re.sub(pattern=r"[^A-Za-z]", repl="", string=s)
            total_len += len(seq)
    return n_seq, total_len


def resolve_fasta_for_pdb(*, pdb_id: str, fasta_dir: Path) -> Optional[Path]:
    pid_l = pdb_id.lower()
    pid_u = pdb_id.upper()

    candidates: List[Path] = [
        fasta_dir / f"{pid_l}_rcsb.fasta",
        fasta_dir / f"{pid_u}_rcsb.fasta",
        fasta_dir / f"{pid_l}.fasta",
        fasta_dir / f"{pid_u}.fasta",
        fasta_dir / f"{pid_l}.fa",
        fasta_dir / f"{pid_u}.fa",
    ]
    for p in candidates:
        if p.is_file() and p.stat().st_size > 0:
            return p

    # fallback: token substring
    token = pid_l
    for p in sorted(fasta_dir.glob("*.fasta")) + sorted(fasta_dir.glob("*.fa")):
        if token in p.name.lower() and p.is_file() and p.stat().st_size > 0:
            return p
    return None


def read_spacegroup_from_mtz(*, mtz_path: Path) -> str:
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


def print_sequence_safety_check(
    *,
    expected_pdb_id: str,
    mtz_filename: str,
    mtz_path: Path,
    fasta_path: Path,
    method: str,
) -> bool:
    expected = expected_pdb_id.lower()
    mtz_pdb = parse_pdb_id_from_mtz_filename(mtz_filename=mtz_filename)
    fasta_pdb = parse_pdb_id_from_fasta_filename(fasta_path=fasta_path)

    n_seq, total_len = read_fasta_sequence_stats(fasta_path=fasta_path)
    sg_str = read_spacegroup_from_mtz(mtz_path=mtz_path)

    mtz_ok = (mtz_pdb == expected)
    fasta_ok = (fasta_pdb == expected)

    logging.info("[SEQ-CHECK] method=%s | expected=%s | mtz_id=%s (match=%s) | fasta_id=%s (match=%s) | SG=%s | n_seq=%d | seq_len=%d | fasta=%s", method, expected, mtz_pdb, mtz_ok, fasta_pdb, fasta_ok, sg_str, n_seq, total_len, fasta_path.name)

    if total_len <= 0:
        print(f"[SEQ-CHECK][FAIL] method={method} | FASTA has zero sequence length: {fasta_path}", file=sys.stderr)
        return False
    if not mtz_ok:
        print(
            f"[SEQ-CHECK][FAIL] method={method} | MTZ filename PDB ID mismatch: expected={expected} got={mtz_pdb} "
            f"(mtz={mtz_filename})",
            file=sys.stderr,
        )
        return False
    if not fasta_ok:
        print(
            f"[SEQ-CHECK][FAIL] method={method} | FASTA filename PDB ID mismatch: expected={expected} got={fasta_pdb} "
            f"(fasta={fasta_path.name})",
            file=sys.stderr,
        )
        return False
    return True


def _log_tail_has_citations(*, log_path: Path, tail_lines: int = 200) -> bool:
    try:
        with open(log_path, mode="rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 65536))
            data = fh.read().decode("utf-8", errors="ignore").splitlines()
        tail = data[-int(tail_lines):]
        return any("Citations for AutoBuild:" in line for line in tail)
    except Exception:
        return False


def job_already_successful(*, workdir: Path) -> bool:
    run_dirs: List[Path] = [p for p in workdir.glob("AutoBuild_run_*") if p.is_dir()]
    if not run_dirs:
        return False

    for run_dir in sorted(run_dirs):
        best_pdb: Path = run_dir / "overall_best.pdb"
        best_mtz: Path = run_dir / "overall_best_refine_data.mtz"
        if not (best_pdb.is_file() and best_mtz.is_file()):
            continue

        candidates: List[Path] = []
        candidates.extend(run_dir.glob("AutoBuild_run_*.log"))
        candidates.extend(run_dir.glob("AutoBuild_run_*.LOG"))
        candidates.extend(workdir.glob("AutoBuild_run_*.log"))
        candidates.extend(workdir.glob("AutoBuild_run_*.LOG"))

        if not candidates:
            continue

        if any(_log_tail_has_citations(log_path=lp, tail_lines=200) for lp in candidates):
            return True

    return False

    for run_dir in sorted(run_dirs):
        best_pdb: Path = run_dir / "overall_best.pdb"
        best_mtz: Path = run_dir / "overall_best_refine_data.mtz"

        if not (best_pdb.is_file() and best_pdb.stat().st_size > 0):
            continue
        if not (best_mtz.is_file() and best_mtz.stat().st_size > 0):
            continue

        if not require_log_success_marker:
            return True

        candidates: List[Path] = []
        candidates.extend(run_dir.glob("AutoBuild_run_*.log"))
        candidates.extend(run_dir.glob("AutoBuild_run_*.LOG"))
        candidates.extend(workdir.glob("AutoBuild_run_*.log"))
        candidates.extend(workdir.glob("AutoBuild_run_*.LOG"))

        if any(_log_tail_has_citations(log_path=lp, tail_lines=200) for lp in candidates):
            return True

    return False


def list_pdb_ids_for_suffix(*, input_phenix_folder: Path, suffix: str) -> List[str]:
    """
    Find all <pdb_id> subfolders under <input_phenix_folder>/<suffix>/ that contain
    <pdb_id>_<suffix>_PHIB_input.mtz.
    """
    base = input_phenix_folder / suffix
    if not base.is_dir():
        return []
    pdb_ids: List[str] = []
    for sub in sorted(p for p in base.iterdir() if p.is_dir()):
        mtz = sub / f"{sub.name}_{suffix}_PHIB_input.mtz"
        if mtz.is_file():
            pdb_ids.append(sub.name.lower())
    return pdb_ids


def read_index_csv_pdb_ids(*, index_csv: Path) -> List[str]:
    """
    PC32 --write_index_csv produces columns: pdb_id, spacegroup, csv_path
    """
    if not index_csv.is_file():
        return []
    import pandas as pd
    df = pd.read_csv(index_csv)
    if "pdb_id" not in df.columns:
        return []
    return sorted(set([str(x).strip().lower() for x in df["pdb_id"].tolist() if str(x).strip()]))


def subsample_ids(*, ids: List[str], max_pdbs: int, seed: int) -> List[str]:
    if max_pdbs <= 0:
        return ids
    if len(ids) <= max_pdbs:
        return ids
    rng = np.random.default_rng(int(seed))
    pick = rng.choice(a=np.array(ids, dtype=object), size=int(max_pdbs), replace=False)
    return sorted([str(x) for x in pick.tolist()])


# =============================================================================
# Core runner
# =============================================================================

def run_autobuild_one(
    *,
    workdir: Path,
    mtz_filename: str,
    pdb_id: str,
    fasta_path: Path,
    nproc: int,
    resolution_build: float,
    refinement_resolution: float,
    extra_args: List[str],
) -> Tuple[str, str, str, int, float]:
    """
    Execute phenix.autobuild in workdir.
    Returns: (status, suffix, pdb_id, returncode, runtime_s)
    """
    suffix = workdir.parent.name  # method label
    mtz_path = workdir / mtz_filename

    logging.info("[RUN] method=%s | pdb_id=%s | workdir=%s", suffix, pdb_id, str(workdir))

    ok_check = print_sequence_safety_check(
        expected_pdb_id=pdb_id,
        mtz_filename=mtz_filename,
        mtz_path=mtz_path,
        fasta_path=fasta_path,
        method=suffix,
    )
    if not ok_check:
        return ("FAIL_SEQ", suffix, pdb_id, 999, 0.0)

    cmd: List[str] = [
        "phenix.autobuild",
        f"data={mtz_filename}",
        f"seq_file={str(fasta_path)}",
        "input_labels=FP SIGFP PHIB FOM HLA HLB HLC HLD FreeR_flag",
        f"nproc={int(nproc)}",
        f"resolution_build={float(resolution_build)}",
        f"refinement_resolution={float(refinement_resolution)}",
    ]
    cmd.extend([str(x) for x in extra_args if str(x).strip()])

    t0 = time.time()
    try:
        proc: subprocess.CompletedProcess = subprocess.run(
            args=cmd,
            cwd=str(workdir),
            text=True,
            capture_output=True,
            check=False,
        )
        dt = float(time.time() - t0)

        if proc.returncode == 0 and job_already_successful(workdir=workdir):
            return ("OK", suffix, pdb_id, int(proc.returncode), dt)

        tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
        if tail:
            sys.stderr.write(
                f"\n[FAIL] phenix.autobuild method={suffix} pdb_id={pdb_id} returned {proc.returncode} or missing success markers.\n"
                f"--- stderr (tail) ---\n{tail}\n----------------------\n"
            )
        else:
            sys.stderr.write(
                f"\n[FAIL] phenix.autobuild method={suffix} pdb_id={pdb_id} returned {proc.returncode} or missing success markers (no stderr).\n"
            )
        return ("FAIL", suffix, pdb_id, int(proc.returncode), dt)

    except Exception as exc:
        dt = float(time.time() - t0)
        sys.stderr.write(f"\n[EXC] phenix.autobuild method={suffix} pdb_id={pdb_id}: {exc}\n")
        return ("EXC", suffix, pdb_id, 998, dt)


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="22: Run phenix.autobuild on MTZs produced by stage-21 MTZ preparation.")

    p.add_argument("--base_folder", type=str, default=".", help="Pipeline base folder (default: cwd).")
    p.add_argument("--input_phenix_folder", type=str, default=None,
                   help="Root folder containing <suffix>/<pdb_id>/<pdb_id>_<suffix>_PHIB_input.mtz. "
                        "Default: <base_folder>/21_Phenix_Autobuild_Inputs")
    p.add_argument("--fasta_dir", type=str, default=None,
                   help="Directory containing per-PDB FASTA files. Default: <base_folder>/01_Download_Files")

    p.add_argument("--methods", type=str, required=True,
                   help='Comma-separated list of suffixes to run (e.g. "iREDO,K2").')

    p.add_argument("--jobs", type=int, default=4,
                   help="Max number of concurrent phenix.autobuild processes (global).")
    p.add_argument("--nproc", type=int, default=4, help="Threads per job passed to phenix.autobuild (nproc=...).")

    p.add_argument("--resolution_build", type=float, default=2.5, help="phenix.autobuild resolution_build=...")
    p.add_argument("--refinement_resolution", type=float, default=2.5, help="phenix.autobuild refinement_resolution=...")

    p.add_argument("--dry_run", action="store_true", help="Plan only; do not execute phenix.autobuild.")

    # selection controls
    p.add_argument("--index_csv", type=str, default=None,
                   help="Optional PC32 selection index CSV (recommended). If provided, restricts to those pdb_id.")
    p.add_argument("--max_pdbs", type=int, default=0,
                   help="If >0, randomly downsample the final PDB list to this size (paired across methods).")
    p.add_argument("--seed", type=int, default=123, help="Random seed for downsampling (default 123).")
    p.add_argument("--require_all_methods", action="store_true",
                   help="Restrict to PDB IDs that have MTZs for ALL requested methods (intersection). Recommended.")

    # extra phenix args
    p.add_argument("--extra_args", type=str, default="",
                   help='Extra arguments passed verbatim to phenix.autobuild (comma-separated), e.g. "quick=True,build_cycles=2"')

    # output/manifest
    p.add_argument("--output_root", type=str, default=None,
                   help="Folder where manifest CSV will be written. Default: <base_folder>/22_Run_AutoBuild")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    base = Path(args.base_folder).expanduser().resolve()

    input_phenix_folder = Path(args.input_phenix_folder).expanduser().resolve() if args.input_phenix_folder else (base / "21_Phenix_Autobuild_Inputs")
    fasta_dir = Path(args.fasta_dir).expanduser().resolve() if args.fasta_dir else (base / "01_Download_Files")

    if not input_phenix_folder.is_dir():
        raise SystemExit(f"[ERROR] input_phenix_folder not found: {input_phenix_folder}")
    if not fasta_dir.is_dir():
        raise SystemExit(f"[ERROR] fasta_dir not found: {fasta_dir}")

    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else (base / "22_Run_AutoBuild")
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = configure_logging(output_root=output_root)
    logging.info("Log file: %s", str(log_path))

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    if not methods:
        raise SystemExit("[ERROR] --methods must contain at least one suffix (e.g. iREDO,K2).")

    extra_args = [x.strip() for x in str(args.extra_args).split(",") if x.strip()]

    # Discover available PDB IDs per method
    per_method_ids: Dict[str, List[str]] = {}
    for m in methods:
        per_method_ids[m] = list_pdb_ids_for_suffix(input_phenix_folder=input_phenix_folder, suffix=m)

    # Start set
    if args.require_all_methods:
        common: Optional[set] = None
        for m in methods:
            s = set(per_method_ids[m])
            common = s if common is None else (common & s)
        pdb_ids = sorted(list(common)) if common is not None else []
    else:
        union: set = set()
        for m in methods:
            union |= set(per_method_ids[m])
        pdb_ids = sorted(list(union))

    # Restrict by PC32 index CSV if provided
    if args.index_csv:
        idx_ids = set(read_index_csv_pdb_ids(index_csv=Path(args.index_csv).expanduser().resolve()))
        pdb_ids = [pid for pid in pdb_ids if pid in idx_ids]

    # Optional downsample (paired because we sample at pdb_id-level)
    pdb_ids = subsample_ids(ids=pdb_ids, max_pdbs=int(args.max_pdbs), seed=int(args.seed))

    logging.info("[INFO] base_folder         : %s", str(base))
    logging.info("[INFO] run_prefix          : %s", "22_run_phenix_autobuild")
    logging.info("[INFO] command_line        : %s", " ".join(shlex.quote(arg) for arg in sys.argv))
    logging.info("[INFO] input_phenix_folder : %s", str(input_phenix_folder))
    logging.info("[INFO] fasta_dir           : %s", str(fasta_dir))
    logging.info("[INFO] methods             : %s", methods)
    logging.info("[INFO] require_all_methods : %s", bool(args.require_all_methods))
    if args.index_csv:
        logging.info("[INFO] index_csv           : %s", str(args.index_csv))
    if int(args.max_pdbs) > 0:
        logging.info("[INFO] max_pdbs            : %d (seed=%d)", int(args.max_pdbs), int(args.seed))
    logging.info("[INFO] final pdb_ids        : %d", len(pdb_ids))

    if not pdb_ids:
        print("[DONE] No PDB IDs found under the requested constraints.")
        return

    # -----------------------------
    # Build per-PDB task queues (resume-filtered)
    # -----------------------------
    Task = Tuple[Path, str, str, Path]  # (workdir, mtz_filename, pdb_id, fasta_path)

    pid_to_queue: Dict[str, Deque[Task]] = {}
    total_tasks_found = 0
    skipped_ok = 0
    skipped_missing = 0

    for pid in pdb_ids:
        fasta_path = resolve_fasta_for_pdb(pdb_id=pid, fasta_dir=fasta_dir)
        if fasta_path is None or (not fasta_path.is_file()):
            continue

        q: Deque[Task] = deque()
        for m in methods:
            workdir = input_phenix_folder / m / pid
            mtz_filename = f"{pid}_{m}_PHIB_input.mtz"
            mtz_path = workdir / mtz_filename

            if not mtz_path.is_file():
                skipped_missing += 1
                continue

            total_tasks_found += 1

            if job_already_successful(
                workdir=workdir,
            ):
                skipped_ok += 1
                continue

            q.append((workdir, mtz_filename, pid, fasta_path))

        if len(q) > 0:
            pid_to_queue[pid] = q

    exec_total = sum(len(q) for q in pid_to_queue.values())

    logging.info("[INFO] total tasks found   : %d", total_tasks_found)
    logging.info("[INFO] to run now          : %d", exec_total)
    logging.info("[INFO] skipped (already OK): %d", skipped_ok)
    logging.info("[INFO] skipped (missing)   : %d", skipped_missing)

    if exec_total <= 0:
        print("[DONE] No runnable tasks found after resume filtering.")
        return

    if args.dry_run:
        shown = 0
        for pid in sorted(pid_to_queue.keys()):
            for (workdir, mtz_filename, _, fasta_path) in list(pid_to_queue[pid]):
                print(f"[PLAN] {workdir.parent.name}/{pid} | data={mtz_filename} | seq={fasta_path.name}")
                shown += 1
                if shown >= 10:
                    break
            if shown >= 10:
                break
        if exec_total > 10:
            print(f"[PLAN] ... and {exec_total - 10} more")
        return

    # Manifest
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = output_root / f"22_autobuild_manifest_{ts}.csv"

    ok = 0
    fail = 0

    jobs_total = max(1, int(args.jobs))

    # Active bookkeeping
    active_pids: set = set()          # PDBs with at least one task submitted
    started_pids: set = set()         # PDBs for which we've printed START banner
    remaining_by_pid: Dict[str, int] = {pid: len(q) for pid, q in pid_to_queue.items()}

    # Convenience lists for selection
    pid_order: List[str] = sorted(pid_to_queue.keys())
    pid_not_started: Deque[str] = deque(pid_order)

    def pop_next_task() -> Optional[Task]:
        """
        Non-idling policy with preference:
          1) pick from active_pids that still have queued tasks (to complete method sets)
          2) otherwise start next not-yet-started pid
        """
        # 1) prefer active pids, and within them prefer those with fewer remaining (finish sooner)
        candidates_active: List[str] = [pid for pid in active_pids if len(pid_to_queue.get(pid, deque())) > 0]
        if candidates_active:
            # smallest remaining first
            candidates_active.sort(key=lambda pid: int(remaining_by_pid.get(pid, 10**9)))
            pid = candidates_active[0]
            return pid_to_queue[pid].popleft()

        # 2) start a new pid
        while pid_not_started:
            pid = pid_not_started[0]
            if pid not in pid_to_queue or len(pid_to_queue[pid]) == 0:
                pid_not_started.popleft()
                continue
            return pid_to_queue[pid].popleft()

        return None

    def on_submit_task(task: Task) -> None:
        workdir, _mtz_filename, pid, _fasta_path = task
        if pid not in active_pids:
            active_pids.add(pid)
        if pid not in started_pids:
            methods_here = ",".join([t[0].parent.name for t in list(pid_to_queue[pid])])
            # Note: methods_here shows what remains after this first pop; include current method too:
            cur_method = workdir.parent.name
            methods_full = ",".join([cur_method] + ([methods_here] if methods_here else []))
            logging.info("[INFO] START PDB %s (methods remaining+current: %s) ...", pid, methods_full)
            started_pids.add(pid)
            if pid_not_started and pid_not_started[0] == pid:
                pid_not_started.popleft()

    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["timestamp", "suffix", "pdb_id", "workdir", "mtz", "fasta", "status", "returncode", "runtime_s"],
        )
        writer.writeheader()

        pbar = get_progress(total=exec_total, desc="phenix.autobuild (non-idling)")

        with ThreadPoolExecutor(max_workers=jobs_total) as ex:
            future_to_task: Dict = {}

            # Prime: fill all slots immediately
            while len(future_to_task) < jobs_total:
                task = pop_next_task()
                if task is None:
                    break
                on_submit_task(task=task)
                workdir, mtz_filename, pid, fasta_path = task
                fut = ex.submit(
                    run_autobuild_one,
                    workdir=workdir,
                    mtz_filename=mtz_filename,
                    pdb_id=pid,
                    fasta_path=fasta_path,
                    nproc=int(args.nproc),
                    resolution_build=float(args.resolution_build),
                    refinement_resolution=float(args.refinement_resolution),
                    extra_args=extra_args,
                )
                future_to_task[fut] = task

            try:
                while future_to_task:
                    done, _ = wait(set(future_to_task.keys()), return_when=FIRST_COMPLETED)

                    for fut in done:
                        workdir, mtz_filename, pid, fasta_path = future_to_task.pop(fut)
                        status, suffix, pid_out, rc, dt = fut.result()

                        if status == "OK":
                            ok += 1
                        else:
                            fail += 1

                        writer.writerow({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "suffix": suffix,
                            "pdb_id": pid_out,
                            "workdir": str(workdir),
                            "mtz": mtz_filename,
                            "fasta": fasta_path.name,
                            "status": status,
                            "returncode": int(rc),
                            "runtime_s": f"{float(dt):.2f}",
                        })
                        fh.flush()

                        logging.info("[%s] %s/%s (rc=%s, %.1fs)", status, suffix, pid_out, rc, dt)
                        pbar.update(1)

                        # decrement remaining count for this pid (includes submitted + queued)
                        remaining_by_pid[pid_out] = int(remaining_by_pid.get(pid_out, 0)) - 1
                        if remaining_by_pid[pid_out] <= 0:
                            if pid_out in active_pids:
                                active_pids.remove(pid_out)
                            logging.info("[INFO] DONE PDB %s (all methods finished)", pid_out)

                    # Refill immediately (non-idling): submit as many new tasks as there are free slots
                    while len(future_to_task) < jobs_total:
                        task = pop_next_task()
                        if task is None:
                            break
                        on_submit_task(task=task)
                        workdir, mtz_filename, pid, fasta_path = task
                        fut = ex.submit(
                            run_autobuild_one,
                            workdir=workdir,
                            mtz_filename=mtz_filename,
                            pdb_id=pid,
                            fasta_path=fasta_path,
                            nproc=int(args.nproc),
                            resolution_build=float(args.resolution_build),
                            refinement_resolution=float(args.refinement_resolution),
                            extra_args=extra_args,
                        )
                        future_to_task[fut] = task

            finally:
                try:
                    pbar.close()
                except Exception:
                    pass

    logging.info("[SUMMARY]")
    logging.info("  OK                     : %d", ok)
    logging.info("  FAIL/EXC/FAIL_SEQ      : %d", fail)
    logging.info("  skipped already OK      : %d", skipped_ok)
    logging.info("  skipped missing inputs  : %d", skipped_missing)
    logging.info("  manifest                : %s", str(manifest_path))
    logging.info("[DONE]")


if __name__ == "__main__":
    main()