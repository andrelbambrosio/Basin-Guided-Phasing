#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
42_run_phenix_autobuild_from_degradation_mtz.py

python 42_run_phenix_autobuild_from_degradation_mtz.py   --base_folder ./   --input_degradation_folder ./13_BasinScore_Degradation/2bkf-new3/   --jobs 10   --nproc 3

Stage-42 pipeline scheduler, adapted from the original PC33 scheduler to run phenix.autobuild on MTZ files
exported by the PC50 degradation pipeline, i.e. the sampled random states
exported alongside CCP4 maps and listed in the PC50 manifest CSVs. The original
PC33 expected the PC32 layout
  <input_phenix_folder>/<suffix>/<pdb_id>/<pdb_id>_<suffix>_PHIB_input.mtz
whereas this version consumes rows from one or more PC50 manifest CSVs and
launches AutoBuild per exported degradation state. If an `_anti.mtz` companion exists next to `mtz_autobuild`, it is launched as an additional paired state. The original scheduler
design and success criteria are preserved. fileciteturn9file0

Expected manifest columns from PC50
-----------------------------------
At minimum:
  - pdb_id
  - mode
  - degrade_fraction
  - round_idx
  - mtz_autobuild

Optional:
  - ccp4_map
  - task_seed

Resume-safe
-----------
A job is treated as successful ONLY IF in the workdir there is at least one:
  <workdir>/AutoBuild_run_*/overall_best.pdb
  <workdir>/AutoBuild_run_*/overall_best_refine_data.mtz
and a corresponding AutoBuild_run_*.log tail contains:
  "Citations for AutoBuild:"

Typical usage
-------------
python 42_run_phenix_autobuild_from_degradation_mtz.py \
  --base_folder ./ \
  --input_degradation_folder ./13_BasinScore_Degradation \
  --jobs 10 \
  --nproc 3

or explicit manifest(s):
python 42_run_phenix_autobuild_from_degradation_mtz.py \
  --manifest_csv ./13_BasinScore_Degradation/2bkf/degradation_master_001337/tables/degradation__uniform_random__ccp4_manifest.csv \
  --fasta_dir ./01_Download_Files \
  --jobs 10 \
  --nproc 3
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
import os

import pandas as pd


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


def configure_logging(*, output_root: Path, run_prefix: str = "42_run_phenix_autobuild_from_degradation_mtz") -> Path:
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

    token = pid_l
    for p in sorted(fasta_dir.glob("*.fasta")) + sorted(fasta_dir.glob("*.fa")):
        if token in p.name.lower() and p.is_file() and p.stat().st_size > 0:
            return p
    return None


def print_sequence_safety_check(
    *,
    expected_pdb_id: str,
    mtz_filename: str,
    mtz_path: Path,
    fasta_path: Path,
    label: str,
) -> bool:
    expected = expected_pdb_id.lower()
    mtz_pdb = parse_pdb_id_from_mtz_filename(mtz_filename=mtz_filename)
    fasta_pdb = parse_pdb_id_from_fasta_filename(fasta_path=fasta_path)

    n_seq, total_len = read_fasta_sequence_stats(fasta_path=fasta_path)

    mtz_ok = (mtz_pdb == expected)
    fasta_ok = (fasta_pdb == expected)

    logging.info("[SEQ-CHECK] label=%s | expected=%s | mtz_id=%s (match=%s) | fasta_id=%s (match=%s) | n_seq=%d | seq_len=%d | fasta=%s", label, expected, mtz_pdb, mtz_ok, fasta_pdb, fasta_ok, n_seq, total_len, fasta_path.name)

    if total_len <= 0:
        logging.error("[SEQ-CHECK][FAIL] label=%s | FASTA has zero sequence length: %s", label, str(fasta_path))
        return False
    if not mtz_ok:
        logging.error("[SEQ-CHECK][FAIL] label=%s | MTZ filename PDB ID mismatch: expected=%s got=%s (mtz=%s)", label, expected, mtz_pdb, mtz_filename)
        return False
    if not fasta_ok:
        logging.error("[SEQ-CHECK][FAIL] label=%s | FASTA filename PDB ID mismatch: expected=%s got=%s (fasta=%s)", label, expected, fasta_pdb, fasta_path.name)
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


def sanitize_token(*, value: str) -> str:
    s = str(value).strip()
    s = re.sub(pattern=r"[^0-9A-Za-z_.-]+", repl="_", string=s)
    s = re.sub(pattern=r"_+", repl="_", string=s)
    s = s.strip("._")
    return s if s else "unknown"


# =============================================================================
# Manifest discovery
# =============================================================================

def discover_manifest_csvs(
    *,
    manifest_csv: Optional[str],
    input_degradation_folder: Optional[Path],
) -> List[Path]:
    if manifest_csv:
        p = Path(manifest_csv).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"[ERROR] manifest_csv not found: {p}")
        return [p]

    if input_degradation_folder is None:
        raise SystemExit("[ERROR] Provide either --manifest_csv or --input_degradation_folder")

    root = input_degradation_folder.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"[ERROR] input_degradation_folder not found: {root}")

    manifests = []
    manifests.extend(sorted(root.glob("**/40_tables/40_degradation__*__ccp4_manifest.csv")))
    manifests.extend(sorted(root.glob("**/tables/degradation__*__ccp4_manifest.csv")))
    manifests = sorted(set(manifests))
    if not manifests:
        raise SystemExit(f"[ERROR] No degradation manifest CSVs found under: {root}")
    return manifests


def read_manifest_tasks(
    *,
    manifest_paths: List[Path],
    fasta_dir: Path,
    output_root: Path,
    modes_filter: Optional[set],
    pdb_filter: Optional[set],
) -> List[dict]:
    """
    Build runnable AutoBuild tasks from PC50 manifest rows.

    New behavior:
      - always include the primary mtz_autobuild state
      - if a sibling MTZ with suffix "_anti.mtz" exists, include it as a second task
      - keep normal and anti runs in separate workdirs
    """
    tasks: List[dict] = []
    seen_keys = set()

    for manifest_path in manifest_paths:
        try:
            df = pd.read_csv(manifest_path)
        except Exception as exc:
            sys.stderr.write(f"[WARN] Could not read manifest {manifest_path}: {exc}\n")
            continue

        required = {"pdb_id", "mode", "degrade_fraction", "round_idx", "mtz_autobuild"}
        if not required.issubset(set(df.columns)):
            sys.stderr.write(f"[WARN] Manifest missing required columns, skipping: {manifest_path}\n")
            continue

        for _, row in df.iterrows():
            pdb_id = str(row.get("pdb_id", "")).strip().lower()
            mode = str(row.get("mode", "")).strip()
            mtz_path = str(row.get("mtz_autobuild", "")).strip()

            if not pdb_id or not mode or not mtz_path:
                continue

            if modes_filter is not None and mode not in modes_filter:
                continue
            if pdb_filter is not None and pdb_id not in pdb_filter:
                continue

            mtz = Path(mtz_path).expanduser().resolve()
            if not mtz.is_file():
                continue

            fasta_path = resolve_fasta_for_pdb(pdb_id=pdb_id, fasta_dir=fasta_dir)
            if fasta_path is None:
                sys.stderr.write(
                    f"[WARN] FASTA not found for pdb_id={pdb_id}; skipping state "
                    f"{mode} frac={row.get('degrade_fraction')} round={row.get('round_idx')}\n"
                )
                continue

            frac = float(row.get("degrade_fraction"))
            rnd = int(row.get("round_idx"))
            task_seed = row.get("task_seed", "")

            state_stub = (
                f"mode_{sanitize_token(value=mode)}__"
                f"frac_{frac:.4f}__"
                f"round_{rnd:06d}"
            )

            mtz_candidates = [("seed", mtz)]
            anti_mtz = mtz.with_name(mtz.stem + "_anti.mtz")
            if anti_mtz.is_file():
                mtz_candidates.append(("anti", anti_mtz))

            for phase_variant, mtz_candidate in mtz_candidates:
                variant_suffix = "" if phase_variant == "seed" else "__anti"
                workdir = output_root / pdb_id / sanitize_token(value=mode) / (state_stub + variant_suffix)
                key = (pdb_id, mode, round(frac, 8), rnd, phase_variant, str(mtz_candidate))

                if key in seen_keys:
                    continue
                seen_keys.add(key)

                tasks.append({
                    "manifest_csv": str(manifest_path),
                    "pdb_id": pdb_id,
                    "mode": mode,
                    "degrade_fraction": frac,
                    "round_idx": rnd,
                    "task_seed": task_seed,
                    "phase_variant": phase_variant,
                    "mtz_path": mtz_candidate,
                    "mtz_filename": mtz_candidate.name,
                    "fasta_path": fasta_path,
                    "workdir": workdir,
                    "ccp4_map": str(row.get("ccp4_map", "")).strip(),
                })

    tasks.sort(
        key=lambda d: (
            d["pdb_id"],
            str(d["mode"]),
            float(d["degrade_fraction"]),
            int(d["round_idx"]),
            0 if str(d.get("phase_variant", "seed")) == "seed" else 1,
        )
    )
    return tasks


# =============================================================================
# Core runner
# =============================================================================

def run_autobuild_one(
    *,
    workdir: Path,
    mtz_path: Path,
    pdb_id: str,
    fasta_path: Path,
    nproc: int,
    resolution_build: float,
    refinement_resolution: float,
    extra_args: List[str],
    label: str,
) -> Tuple[str, str, str, int, float]:
    """
    Execute phenix.autobuild in workdir.
    Returns: (status, label, pdb_id, returncode, runtime_s)
    """
    workdir.mkdir(parents=True, exist_ok=True)
    mtz_filename = mtz_path.name

    logging.info("[RUN] label=%s | pdb_id=%s | workdir=%s", label, pdb_id, str(workdir))

    ok_check = print_sequence_safety_check(
        expected_pdb_id=pdb_id,
        mtz_filename=mtz_filename,
        mtz_path=mtz_path,
        fasta_path=fasta_path,
        label=label,
    )
    if not ok_check:
        return ("FAIL_SEQ", label, pdb_id, 999, 0.0)

    local_mtz = workdir / mtz_filename
    if not local_mtz.is_file():
        try:
            # Prefer hard link; fall back to copy if needed
            os.link(str(mtz_path), str(local_mtz))
        except Exception:
            import shutil
            shutil.copy2(str(mtz_path), str(local_mtz))

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
            return ("OK", label, pdb_id, int(proc.returncode), dt)

        tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
        if tail:
            logging.error("[FAIL] phenix.autobuild label=%s pdb_id=%s returned %s or missing success markers. stderr tail:\n%s", label, pdb_id, proc.returncode, tail)
        else:
            logging.error("[FAIL] phenix.autobuild label=%s pdb_id=%s returned %s or missing success markers (no stderr).", label, pdb_id, proc.returncode)
        return ("FAIL", label, pdb_id, int(proc.returncode), dt)

    except Exception as exc:
        dt = float(time.time() - t0)
        logging.error("[EXC] phenix.autobuild label=%s pdb_id=%s: %s", label, pdb_id, str(exc))
        return ("EXC", label, pdb_id, 998, dt)


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run phenix.autobuild on MTZ states exported by PC50 degradation pipeline.")

    p.add_argument("--base_folder", type=str, default=".", help="Pipeline base folder (default: cwd).")
    p.add_argument("--input_degradation_folder", type=str, default=None,
                   help="Root folder under which PC50 degradation manifests live. Default: <base_folder>/13_BasinScore_Degradation")
    p.add_argument("--manifest_csv", type=str, default=None,
                   help="Explicit single manifest CSV. If provided, overrides recursive discovery under input_degradation_folder.")
    p.add_argument("--fasta_dir", type=str, default=None,
                   help="Directory containing per-PDB FASTA files. Default: <base_folder>/01_Download_Files")

    p.add_argument("--modes", type=str, default="",
                   help='Optional comma-separated list of degradation modes to include (e.g. "uniform_random").')
    p.add_argument("--pdb_ids", type=str, default="",
                   help='Optional comma-separated list of PDB IDs to include.')

    p.add_argument("--jobs", type=int, default=4,
                   help="Max number of concurrent phenix.autobuild processes (global).")
    p.add_argument("--nproc", type=int, default=4, help="Threads per job passed to phenix.autobuild (nproc=...).")

    p.add_argument("--resolution_build", type=float, default=3.5, help="phenix.autobuild resolution_build=...")
    p.add_argument("--refinement_resolution", type=float, default=2.5, help="phenix.autobuild refinement_resolution=...")

    p.add_argument("--dry_run", action="store_true", help="Plan only; do not execute phenix.autobuild.")

    p.add_argument("--extra_args", type=str, default="",
                   help='Extra arguments passed verbatim to phenix.autobuild (comma-separated), e.g. "quick=True,build_cycles=2"')

    p.add_argument("--output_root", type=str, default=None,
                   help="Folder where AutoBuild workdirs and manifest CSV will be written. Default: <base_folder>/42_Run_AutoBuild_from_Degradation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    base = Path(args.base_folder).expanduser().resolve()
    input_degradation_folder = Path(args.input_degradation_folder).expanduser().resolve() if args.input_degradation_folder else (base / "40_BasinScore_Degradation")
    fasta_dir = Path(args.fasta_dir).expanduser().resolve() if args.fasta_dir else (base / "01_Download_Files")

    if not fasta_dir.is_dir():
        raise SystemExit(f"[ERROR] fasta_dir not found: {fasta_dir}")

    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else (base / "42_Run_AutoBuild_from_Degradation")
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = configure_logging(output_root=output_root)
    logging.info("Log file: %s", str(log_path))
    logging.info("Run prefix: %s", "42_run_phenix_autobuild_from_degradation_mtz")
    logging.info("Command line: %s", " ".join(shlex.quote(arg) for arg in sys.argv))

    modes_filter = None
    if str(args.modes).strip():
        modes_filter = set([m.strip() for m in str(args.modes).split(",") if m.strip()])

    pdb_filter = None
    if str(args.pdb_ids).strip():
        pdb_filter = set([p.strip().lower() for p in str(args.pdb_ids).split(",") if p.strip()])

    extra_args = [x.strip() for x in str(args.extra_args).split(",") if x.strip()]

    manifest_paths = discover_manifest_csvs(
        manifest_csv=args.manifest_csv,
        input_degradation_folder=input_degradation_folder if not args.manifest_csv else None,
    )

    tasks = read_manifest_tasks(
        manifest_paths=manifest_paths,
        fasta_dir=fasta_dir,
        output_root=output_root,
        modes_filter=modes_filter,
        pdb_filter=pdb_filter,
    )

    logging.info("[INFO] base_folder             : %s", str(base))
    if args.manifest_csv:
        logging.info("[INFO] manifest_csv            : %s", str(Path(args.manifest_csv).expanduser().resolve()))
    else:
        logging.info("[INFO] input_degradation_folder : %s", str(input_degradation_folder))
        logging.info("[INFO] manifest CSVs found      : %d", len(manifest_paths))
    logging.info("[INFO] fasta_dir               : %s", str(fasta_dir))
    logging.info("[INFO] output_root             : %s", str(output_root))
    if modes_filter is not None:
        logging.info("[INFO] modes filter            : %s", sorted(list(modes_filter)))
    if pdb_filter is not None:
        logging.info("[INFO] pdb_ids filter          : %s", sorted(list(pdb_filter)))
    logging.info("[INFO] total states discovered  : %d", len(tasks))

    if not tasks:
        print("[DONE] No runnable degradation-state MTZ tasks found.")
        return

    # Resume filter
    runnable: List[dict] = []
    skipped_ok = 0
    for task in tasks:
        if job_already_successful(workdir=task["workdir"]):
            skipped_ok += 1
            continue
        runnable.append(task)

    logging.info("[INFO] to run now              : %d", len(runnable))
    logging.info("[INFO] skipped already OK      : %d", skipped_ok)
    logging.info("[INFO] jobs x nproc            : jobs=%d  nproc=%d", int(args.jobs), int(args.nproc))

    if not runnable:
        print("[DONE] No runnable tasks found after resume filtering.")
        return

    if args.dry_run:
        for task in runnable[:10]:
            print(
                f"[PLAN] pdb_id={task['pdb_id']} | mode={task['mode']} | "
                f"frac={task['degrade_fraction']:.4f} | round={int(task['round_idx'])} | "
                f"variant={task.get('phase_variant', 'seed')} | "
                f"data={task['mtz_path'].name} | seq={task['fasta_path'].name}"
            )
        if len(runnable) > 10:
            print(f"[PLAN] ... and {len(runnable) - 10} more")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_out = output_root / f"PC33_autobuild_from_degradation_manifest_{ts}.csv"

    ok = 0
    fail = 0

    jobs_total = max(1, int(args.jobs))
    pbar = get_progress(total=len(runnable), desc="phenix.autobuild from degradation MTZ")

    with open(manifest_out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "timestamp", "pdb_id", "mode", "degrade_fraction", "round_idx", "task_seed",
                "phase_variant", "workdir", "mtz", "fasta", "ccp4_map", "manifest_csv",
                "status", "returncode", "runtime_s"
            ],
        )
        writer.writeheader()

        with ThreadPoolExecutor(max_workers=jobs_total) as ex:
            future_to_task: Dict = {}

            def submit_task(task: dict) -> None:
                label = f"{task['mode']}|frac={task['degrade_fraction']:.4f}|round={int(task['round_idx'])}|variant={task.get('phase_variant', 'seed')}"
                fut = ex.submit(
                    run_autobuild_one,
                    workdir=task["workdir"],
                    mtz_path=task["mtz_path"],
                    pdb_id=task["pdb_id"],
                    fasta_path=task["fasta_path"],
                    nproc=int(args.nproc),
                    resolution_build=float(args.resolution_build),
                    refinement_resolution=float(args.refinement_resolution),
                    extra_args=extra_args,
                    label=label,
                )
                future_to_task[fut] = task

            queue: Deque[dict] = deque(runnable)

            while queue and len(future_to_task) < jobs_total:
                submit_task(queue.popleft())

            try:
                while future_to_task:
                    done, _ = wait(set(future_to_task.keys()), return_when=FIRST_COMPLETED)

                    for fut in done:
                        task = future_to_task.pop(fut)
                        label = f"{task['mode']}|frac={task['degrade_fraction']:.4f}|round={int(task['round_idx'])}|variant={task.get('phase_variant', 'seed')}"
                        status, _label, pdb_id_out, rc, dt = fut.result()

                        if status == "OK":
                            ok += 1
                        else:
                            fail += 1

                        writer.writerow({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "pdb_id": pdb_id_out,
                            "mode": task["mode"],
                            "degrade_fraction": f"{float(task['degrade_fraction']):.4f}",
                            "round_idx": int(task["round_idx"]),
                            "task_seed": task.get("task_seed", ""),
                            "phase_variant": task.get("phase_variant", "seed"),
                            "workdir": str(task["workdir"]),
                            "mtz": str(task["mtz_path"]),
                            "fasta": task["fasta_path"].name,
                            "ccp4_map": task.get("ccp4_map", ""),
                            "manifest_csv": str(task.get("manifest_csv", "")),
                            "status": status,
                            "returncode": int(rc),
                            "runtime_s": f"{float(dt):.2f}",
                        })
                        fh.flush()

                        logging.info("[%s] %s / %s (rc=%s, %.1fs)", status, label, pdb_id_out, rc, dt)
                        pbar.update(1)

                    while queue and len(future_to_task) < jobs_total:
                        submit_task(queue.popleft())

            finally:
                try:
                    pbar.close()
                except Exception:
                    pass

    logging.info("[SUMMARY]")
    logging.info("  OK                   : %d", ok)
    logging.info("  FAIL/EXC/FAIL_SEQ    : %d", fail)
    logging.info("  skipped already OK   : %d", skipped_ok)
    logging.info("  manifest             : %s", str(manifest_out))
    logging.info("[DONE]")


if __name__ == "__main__":
    main()
