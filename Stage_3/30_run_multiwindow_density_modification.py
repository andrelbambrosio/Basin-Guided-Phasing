#!/usr/bin/env phenix.python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

"""
30_run_multiwindow_density_modification.py

phenix.python 30_run_multiwindow_density_modification.py   --csv_dir ../Attenuated_Signed_Amplitudes-19Mar2026/12_DenMod_BasinScore_Calibration/CSV   --out_dir ./denmod_multiwindow_out   --nproc 25 --job_timeout_sec 15   --skip_existing_mode complete

Multi-window standalone density-modification runner with:
- one CSV, a CSV list, or a folder batch
- one or more resolution windows
- isolated output subfolders per window
- safe parallel execution using one OS process per job
- hard per-job timeout with forced termination
- resume/skip support for completed jobs

Default windows:
20-5.0 A, 20-4.5 A, 20-4.0 A, 20-3.5 A, 20-3.0 A, 20-2.5 A
"""

import os
import sys
import glob
import time
import shutil
import signal
import argparse
import traceback
import multiprocessing
import logging
from datetime import datetime
import re

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

from libtbx.utils import Sorry

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import phenix_like_density_modification_from_csv_py27 as denmod

FOM_K2_ATTEN_FALLBACK = 2.0 / np.pi

_DEFAULT_WINDOWS = [
    (20.0, 5.0),
    (20.0, 4.5),
    (20.0, 4.0),
    (20.0, 3.5),
    (20.0, 3.0),
    (20.0, 2.5),
]


_RE_REFLECTIONS = re.compile(
    r"Number\s+of\s+reflections\s+in\s+input\s+refl_db\s*:\s*([0-9]+)",
    re.IGNORECASE,
)
_RE_DM_MEAN_FOM = re.compile(
    r"DM\s+mean\s+FOM\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)
_RE_CC_PROB_MAP = re.compile(
    r"CC\s+of\s+prob\s+map\s+with\s+current\s+map\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)
_RE_R_FACTOR = re.compile(
    r"Overall\s+R-factor\s+for\s+FC\s+vs\s+FP\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+for\s+([0-9]+)\s+reflections",
    re.IGNORECASE,
)
_RE_BIAS_RATIO = re.compile(
    r"BIAS\s+RATIO\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)
_RE_WINDOW_TAG_LINE = re.compile(
    r"window_tag\s*:\s*(dmax[0-9.]+_dmin[0-9.]+)",
    re.IGNORECASE,
)
_RE_DMAX_LINE = re.compile(
    r"resolution_window_dmax\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)
_RE_DMIN_LINE = re.compile(
    r"resolution_window_dmin\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)
_RE_WINDOW_TAG_DIR = re.compile(r"(dmax[0-9.]+_dmin[0-9.]+)")


def _mkdir_p(path):
    if path and (not os.path.isdir(path)):
        os.makedirs(path)


def _configure_logging(logs_dir, run_prefix):
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(logs_dir)), "..", "LOGS")
    logs_dir = os.path.abspath(logs_dir)
    _mkdir_p(logs_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, "%s__%s.log" % (str(run_prefix), str(ts)))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[:] = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return log_path


def _shell_quote(text):
    try:
        import shlex as _shlex
        if hasattr(_shlex, "quote"):
            return _shlex.quote(str(text))
    except Exception:
        pass
    try:
        import pipes as _pipes
        return _pipes.quote(str(text))
    except Exception:
        pass
    s = str(text)
    if re.search(r"[^A-Za-z0-9_./:=+,-]", s):
        return '"' + s.replace('"', '\"') + '"'
    return s

def _infer_pdb_id_from_csv(csv_in):
    base = os.path.basename(csv_in)
    stem = os.path.splitext(base)[0]
    toks = [t for t in stem.split("_") if t]
    if len(toks) > 0:
        return toks[0].strip().lower()
    return stem.strip().lower()


def _window_tag(dmax, dmin):
    return "dmax%.1f_dmin%.1f" % (float(dmax), float(dmin))


def _detect_solvent_content(df, default_solvent_content=None):
    candidates = [
        "SOLVENT_FRACTION", "solvent_content", "Solvent_content", "Solvent_Content",
        "SOLV_CONT", "solv_cont", "SC", "sc"
    ]
    for col in candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                sc = float(vals[0])
                if 0.0 <= sc <= 1.0:
                    return sc, col

    if default_solvent_content is not None:
        sc = float(default_solvent_content)
        if not (0.0 <= sc <= 1.0):
            raise Sorry("default_solvent_content must be in [0,1], got %s" % str(sc))
        return sc, "DEFAULT"

    raise Sorry(
        "Could not detect solvent content in CSV. Please provide --default_solvent_content."
    )


def _ensure_fom_k2_atten(df):
    added = False
    if "FOM_K2_atten" not in df.columns:
        df["FOM_K2_atten"] = float(FOM_K2_ATTEN_FALLBACK)
        added = True
    return df, added


def _filter_by_resolution_window(df, d_col, dmax, dmin):
    if d_col not in df.columns:
        raise Sorry("Resolution column not found in CSV: %s" % str(d_col))
    dvals = pd.to_numeric(df[d_col], errors="coerce")
    mask = np.isfinite(dvals) & (dvals <= float(dmax)) & (dvals >= float(dmin))
    return df.loc[mask].copy()


def _build_output_mtz(cmn, mtz_path, out, phib_in=None, fom_in=None):
    phib_dm = cmn.phib_out_as_miller_array
    fom_dm = cmn.fom_out_as_miller_array
    hl_dm = cmn.hl_out_as_miller_array
    fp_out = cmn.fp_sigfp_out_as_miller_array

    if phib_dm is None or fom_dm is None or hl_dm is None or fp_out is None:
        raise Sorry("Missing DM outputs; cannot write MTZ.")

    mtz_dataset = fp_out.as_mtz_dataset(
        column_root_label="FP_USE",
        column_types="FQ"
    )

    if phib_in is not None:
        mtz_dataset.add_miller_array(
            miller_array=phib_in,
            column_root_label="PHIB_IN",
            column_types="P"
        )

    if fom_in is not None:
        mtz_dataset.add_miller_array(
            miller_array=fom_in,
            column_root_label="FOM_IN",
            column_types="W"
        )

    mtz_dataset.add_miller_array(
        miller_array=phib_dm,
        column_root_label="PHIB_DM",
        column_types="P"
    )

    mtz_dataset.add_miller_array(
        miller_array=fom_dm,
        column_root_label="FOM_DM",
        column_types="W"
    )

    mtz_dataset.add_miller_array(
        miller_array=hl_dm,
        column_root_label="HLADM",
        column_types="AAAA"
    )

    mtz_object = mtz_dataset.mtz_object()
    mtz_object.write(file_name=mtz_path)
    out.write("Wrote output MTZ: %s\n" % mtz_path)


def _remove_tmp_dir(tmp_dir, out=None):
    try:
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
            if out is not None:
                out.write("Removed temporary directory: %s\n" % tmp_dir)
    except Exception as exc:
        if out is not None:
            out.write("WARNING: could not remove temporary directory %s : %s\n" % (tmp_dir, str(exc)))


def _parse_resolution_windows(text_value):
    if text_value is None or str(text_value).strip() == "":
        return list(_DEFAULT_WINDOWS)

    items = [t.strip() for t in str(text_value).split(",") if t.strip()]
    windows = []
    for item in items:
        norm = item.lower().replace("å", "a").replace("angstrom", "a").replace(" ", "")
        norm = norm.rstrip("a")
        if "-" not in norm:
            raise Sorry("Invalid resolution window specification: %s" % item)
        toks = [t for t in norm.split("-") if t]
        if len(toks) != 2:
            raise Sorry("Invalid resolution window specification: %s" % item)
        dmax = float(toks[0])
        dmin = float(toks[1])
        if dmax <= dmin:
            raise Sorry("Expected dmax > dmin in resolution window: %s" % item)
        windows.append((dmax, dmin))

    unique = []
    seen = set()
    for dmax, dmin in windows:
        key = (float(dmax), float(dmin))
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def _collect_csv_inputs(args):
    csv_list = []
    sources_used = 0

    if args.csv_in is not None:
        sources_used += 1
        csv_in = os.path.abspath(os.path.expanduser(args.csv_in))
        if not os.path.isfile(csv_in):
            raise Sorry("csv_in not found: %s" % csv_in)
        csv_list.append(csv_in)

    if args.csv_list is not None:
        sources_used += 1
        list_path = os.path.abspath(os.path.expanduser(args.csv_list))
        if not os.path.isfile(list_path):
            raise Sorry("csv_list file not found: %s" % list_path)
        with open(list_path, "r") as handle:
            for line in handle:
                line = line.strip()
                if (not line) or line.startswith("#"):
                    continue
                csv_path = os.path.abspath(os.path.expanduser(line))
                if not os.path.isfile(csv_path):
                    raise Sorry("CSV path from csv_list not found: %s" % csv_path)
                csv_list.append(csv_path)

    if args.csv_dir is not None:
        sources_used += 1
        csv_dir = os.path.abspath(os.path.expanduser(args.csv_dir))
        if not os.path.isdir(csv_dir):
            raise Sorry("csv_dir not found: %s" % csv_dir)
        csv_list.extend(sorted(glob.glob(os.path.join(csv_dir, "*.csv"))))

    if sources_used != 1:
        raise Sorry("Provide exactly one input mode: --csv_in OR --csv_list OR --csv_dir")
    if len(csv_list) < 1:
        raise Sorry("No CSV inputs found.")

    deduped = []
    seen = set()
    for csv_path in csv_list:
        key = os.path.abspath(csv_path)
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def _status_template(job):
    pdb_id = _infer_pdb_id_from_csv(job["csv_in"])
    dmax = float(job["dmax"])
    dmin = float(job["dmin"])
    window_tag = _window_tag(dmax, dmin)
    out_dir = os.path.abspath(os.path.expanduser(job["window_out_dir"]))
    log_path = os.path.join(out_dir, "%s_denmod.log" % pdb_id)
    mtz_path = os.path.join(out_dir, "%s_denmod.mtz" % pdb_id)
    return {
        "pdb_id": pdb_id,
        "csv_in": os.path.abspath(os.path.expanduser(job["csv_in"])),
        "window_tag": window_tag,
        "dmax": dmax,
        "dmin": dmin,
        "window_out_dir": out_dir,
        "log_path": log_path,
        "mtz_path": mtz_path,
        "status": "FAILED",
        "error": "",
        "elapsed_sec": np.nan,
    }


def _is_completed_job(log_path, mtz_path):
    if (not os.path.isfile(log_path)) or (not os.path.isfile(mtz_path)):
        return False
    try:
        with open(log_path, "r") as handle:
            text = handle.read()
        return ("\nDONE\n" in text) or text.rstrip().endswith("DONE") or ("MTZ saved to:" in text and "Log saved to:" in text)
    except Exception:
        return False


def _append_timeout_note(log_path, message):
    try:
        with open(log_path, "a") as out:
            out.write("\nTIMEOUT\n")
            out.write(message.rstrip() + "\n")
    except Exception:
        pass


def _run_one_job(job):
    csv_in = os.path.abspath(os.path.expanduser(job["csv_in"]))
    out_dir = os.path.abspath(os.path.expanduser(job["window_out_dir"]))
    dmax = float(job["dmax"])
    dmin = float(job["dmin"])
    args = job["args"]

    status = _status_template(job)
    pdb_id = status["pdb_id"]
    window_tag = status["window_tag"]
    log_path = status["log_path"]
    mtz_path = status["mtz_path"]
    temp_dir = os.path.join(out_dir, "%s_denmod_tmp_pid%s" % (pdb_id, str(os.getpid())))

    if not os.path.isfile(csv_in):
        raise Sorry("Input CSV not found: %s" % csv_in)

    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    _mkdir_p(temp_dir)

    started = time.time()
    with open(log_path, "w") as out:
        out.write("Multi-window standalone density modification from CSV\n")
        out.write("Input CSV: %s\n" % csv_in)
        out.write("pdb_id: %s\n" % pdb_id)
        out.write("window_tag: %s\n" % window_tag)
        out.write("resolution_window_dmax: %.3f\n" % dmax)
        out.write("resolution_window_dmin: %.3f\n" % dmin)
        out.write("phi_col: %s\n" % args.phi_col)
        out.write("fom_col: %s\n" % args.fom_col)
        out.write("mask_type: %s\n" % str(args.mask_type))
        out.write("output_mtz: %s\n" % mtz_path)
        out.write("worker_pid: %s\n\n" % str(os.getpid()))
        out.flush()

        try:
            df = pd.read_csv(csv_in)
            df, used_fom_k2_atten_fallback = _ensure_fom_k2_atten(df)
            out.write("Rows read from CSV: %d\n" % int(df.shape[0]))
            out.write("Used FOM_K2_atten fallback (2/pi): %s\n" % str(bool(used_fom_k2_atten_fallback)))
            out.flush()

            dfw = _filter_by_resolution_window(df=df, d_col=str(args.d_col), dmax=dmax, dmin=dmin)
            out.write("Rows kept after resolution window filter (%.3f >= d >= %.3f): %d\n" % (dmax, dmin, int(dfw.shape[0])))
            if int(dfw.shape[0]) < 1:
                raise Sorry("No reflections remain after resolution window filter.")

            if str(args.fom_col) not in dfw.columns:
                raise Sorry("FOM column not found in CSV: %s" % str(args.fom_col))
            if str(args.phi_col) not in dfw.columns:
                raise Sorry("Phase column not found in CSV: %s" % str(args.phi_col))

            sc, sc_source = _detect_solvent_content(df=dfw, default_solvent_content=args.default_solvent_content)
            out.write("Solvent content: %.4f (source: %s)\n" % (float(sc), str(sc_source)))

            hl_cols_arg = args.hl_cols
            if hl_cols_arg is None or str(hl_cols_arg).strip() == "":
                hl_cols = ("HLA", "HLB", "HLC", "HLD")
                force_recompute_hl = True
                out.write("HL columns: default placeholder labels; recomputing HL from phi/FOM\n")
            else:
                toks = [t.strip() for t in str(hl_cols_arg).split(",") if t.strip()]
                if len(toks) != 4:
                    raise Sorry("hl_cols must have 4 comma-separated labels.")
                hl_cols = tuple(toks)
                force_recompute_hl = bool(args.force_recompute_hl)
                out.write("HL columns provided: %s\n" % ",".join(hl_cols))
                out.write("force_recompute_hl: %s\n" % str(force_recompute_hl))
            out.flush()

            cs, fp_sigfp, phib, fom, hl, df_use = denmod.build_cctbx_arrays_from_csv(
                df=dfw,
                phi_col=str(args.phi_col),
                fom_col=str(args.fom_col),
                hl_cols=hl_cols,
                force_recompute_hl=force_recompute_hl,
                kappa_max=float(args.kappa_max),
                out=out,
            )

            out.write("Running density modification...\n")
            out.flush()

            cmn = denmod.run_density_modification(
                fp_sigfp=fp_sigfp,
                phib=phib,
                fom=fom,
                hendrickson_lattman=hl,
                solvent_content=float(sc),
                mask_cycles=int(args.mask_cycles),
                minor_cycles=int(args.minor_cycles),
                temp_dir=temp_dir,
                clean_up=False,
                mask_type=str(args.mask_type),
                rad_wang=args.rad_wang,
                rad_mask=float(args.rad_mask) if args.rad_mask is not None else None,
                verbose=bool(args.verbose),
                out=out,
            )

            phib_dm = cmn.phib_out_as_miller_array
            fom_dm = cmn.fom_out_as_miller_array
            if phib_dm is None or fom_dm is None:
                raise Sorry("Density modification did not return PHIB/FOM outputs.")

            try:
                fom_in_mean = float(np.nanmean(np.array(fom.data(), dtype=float)))
            except Exception:
                fom_in_mean = np.nan
            try:
                fom_dm_mean = float(np.nanmean(np.array(fom_dm.data(), dtype=float)))
            except Exception:
                fom_dm_mean = np.nan
            out.write("Input mean FOM: %s\n" % str(fom_in_mean))
            out.write("DM mean FOM: %s\n" % str(fom_dm_mean))

            try:
                ph_in = phib.common_set(other=phib_dm)
                ph_dm = phib_dm.common_set(other=phib)
                dphi = denmod.abs_circular_diff_deg(
                    a_deg=np.array(ph_in.data(), dtype=float),
                    b_deg=np.array(ph_dm.data(), dtype=float)
                )
                if dphi.size > 0:
                    out.write("Mean abs phase change (deg): %.4f\n" % float(np.mean(dphi)))
                    out.write("RMS phase change (deg): %.4f\n" % float(np.sqrt(np.mean(dphi ** 2))))
            except Exception as exc:
                out.write("Phase-difference summary could not be computed: %s\n" % str(exc))

            _build_output_mtz(cmn=cmn, mtz_path=mtz_path, out=out, phib_in=phib, fom_in=fom)
            out.write("\nDONE\n")
            out.write("Log saved to: %s\n" % log_path)
            out.write("MTZ saved to: %s\n" % mtz_path)
            out.flush()

            status["status"] = "OK"
            status["elapsed_sec"] = time.time() - started
        except Exception as exc:
            status["status"] = "FAILED"
            status["error"] = "%s: %s" % (exc.__class__.__name__, str(exc))
            status["elapsed_sec"] = time.time() - started
            out.write("\nERROR: %s\n" % status["error"])
            out.write(traceback.format_exc())
            out.flush()
        finally:
            _remove_tmp_dir(tmp_dir=temp_dir, out=out)
            out.flush()

    return status


def _worker_entry(job, result_queue):
    try:
        result = _run_one_job(job)
    except Exception as exc:
        result = _status_template(job)
        result["status"] = "FAILED"
        result["error"] = "%s: %s" % (exc.__class__.__name__, str(exc))
    try:
        result_queue.put(result)
    except Exception:
        pass


def _terminate_process_tree(proc):
    if proc is None:
        return
    if not proc.is_alive():
        try:
            proc.join(timeout=0.1)
        except Exception:
            pass
        return

    try:
        os.kill(proc.pid, signal.SIGTERM)
    except Exception:
        pass
    proc.join(timeout=1.0)

    if proc.is_alive():
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except Exception:
            pass
        proc.join(timeout=1.0)




def _last_match_float(pattern, text):
    matches = pattern.findall(text)
    if not matches:
        return None
    last_value = matches[-1]
    if isinstance(last_value, tuple):
        last_value = last_value[0]
    try:
        return float(last_value)
    except Exception:
        return None


def _last_match_int(pattern, text):
    matches = pattern.findall(text)
    if not matches:
        return None
    last_value = matches[-1]
    if isinstance(last_value, tuple):
        last_value = last_value[0]
    try:
        return int(last_value)
    except Exception:
        return None


def _infer_window_from_path(log_path):
    norm_path = os.path.normpath(log_path)
    for piece in reversed(norm_path.split(os.sep)):
        match = _RE_WINDOW_TAG_DIR.search(piece)
        if match:
            window_tag = match.group(1)
            try:
                toks = window_tag.replace("dmax", "").replace("_dmin", ",").split(",")
                return window_tag, float(toks[0]), float(toks[1])
            except Exception:
                return window_tag, None, None
    return None, None, None


def _extract_metrics_from_denmod_log(log_path):
    pdb_id = _infer_pdb_id_from_csv(log_path.replace("_denmod.log", ".csv"))

    try:
        with open(log_path, "r") as handle:
            text = handle.read()
    except Exception as exc:
        return {
            "PDB_ID": pdb_id,
            "window_tag": None,
            "dmax": None,
            "dmin": None,
            "Reflections": None,
            "DM_mean_FOM": None,
            "CC_prob_map_with_current_map": None,
            "R_factor_FC_vs_FP": None,
            "R_factor_reflections": None,
            "BIAS_RATIO": None,
            "log_path": log_path,
            "status": "READ_ERROR: %s" % str(exc),
        }

    reflections = _last_match_int(_RE_REFLECTIONS, text)
    dm_mean_fom = _last_match_float(_RE_DM_MEAN_FOM, text)
    cc_prob_map = _last_match_float(_RE_CC_PROB_MAP, text)
    bias_ratio = _last_match_float(_RE_BIAS_RATIO, text)

    window_tag = None
    dmax = None
    dmin = None

    tag_matches = _RE_WINDOW_TAG_LINE.findall(text)
    if tag_matches:
        window_tag = tag_matches[-1]
    dmax = _last_match_float(_RE_DMAX_LINE, text)
    dmin = _last_match_float(_RE_DMIN_LINE, text)

    if window_tag is None or dmax is None or dmin is None:
        path_window_tag, path_dmax, path_dmin = _infer_window_from_path(log_path)
        if window_tag is None:
            window_tag = path_window_tag
        if dmax is None:
            dmax = path_dmax
        if dmin is None:
            dmin = path_dmin

    r_factor = None
    r_factor_reflections = None
    r_matches = _RE_R_FACTOR.findall(text)
    if r_matches:
        try:
            r_factor = float(r_matches[-1][0])
        except Exception:
            r_factor = None
        try:
            r_factor_reflections = int(r_matches[-1][1])
        except Exception:
            r_factor_reflections = None

    status = "OK"
    if any(value is None for value in [window_tag, dmax, dmin, reflections, dm_mean_fom, cc_prob_map, r_factor, bias_ratio]):
        status = "PARTIAL"

    return {
        "PDB_ID": pdb_id,
        "window_tag": window_tag,
        "dmax": dmax,
        "dmin": dmin,
        "Reflections": reflections,
        "DM_mean_FOM": dm_mean_fom,
        "CC_prob_map_with_current_map": cc_prob_map,
        "R_factor_FC_vs_FP": r_factor,
        "R_factor_reflections": r_factor_reflections,
        "BIAS_RATIO": bias_ratio,
        "log_path": log_path,
        "status": status,
    }


def _build_denmod_log_summary(out_dir):
    log_paths = []
    for root, dirs, files in os.walk(out_dir):
        for fname in files:
            if fname.endswith("_denmod.log"):
                log_paths.append(os.path.join(root, fname))
    log_paths = sorted(log_paths)

    rows = []
    for log_path in log_paths:
        rows.append(_extract_metrics_from_denmod_log(log_path))
    df = pd.DataFrame(rows)
    if df.shape[0] > 0:
        df = df.sort_values(by=["dmax", "dmin", "PDB_ID"], ascending=[True, False, True])
    out_csv = os.path.join(out_dir, "30_denmod_log_summary_multiwindow.csv")
    df.to_csv(out_csv, index=False)
    return out_csv, df

def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Multi-window standalone density-modification runner from one CSV, a CSV list, or a CSV folder."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--csv_in", default=None, help="Single input CSV file.")
    input_group.add_argument("--csv_list", default=None, help="Text file containing one CSV path per line.")
    input_group.add_argument("--csv_dir", default=None, help="Directory containing input CSV files.")

    parser.add_argument("--out_dir", default="30_DENMOD_multiwindow", help="Root output directory. One subfolder per resolution window will be created inside it. Default: 30_DENMOD_multiwindow")
    parser.add_argument("--resolution_windows", default=None, help="Comma-separated resolution windows. Default: 20-5.0,20-4.5,20-4.0,20-3.5,20-3.0,20-2.5")
    parser.add_argument("--nproc", type=int, default=1, help="Maximum number of concurrent jobs. Default: 1")
    parser.add_argument("--n_max", type=int, default=0, help="If >0, process at most this many CSV files (useful for testing).")
    parser.add_argument("--job_timeout_sec", type=float, default=15.0, help="Hard timeout per job in seconds. Default: 15.0")
    parser.add_argument("--poll_interval_sec", type=float, default=0.2, help="Parent polling interval in seconds. Default: 0.2")
    parser.add_argument("--overwrite", action="store_true", help="Rerun jobs even if completed outputs already exist.")
    parser.add_argument(
        "--skip_existing_mode",
        choices=["complete", "mtz"],
        default="mtz",
        help="Skip criterion on restart: complete=requires completed log+mtz; mtz=skip any non-empty MTZ unless log shows ERROR/TIMEOUT."
    )

    parser.add_argument("--phi_col", default="PHIC_ALL_K2", help="Phase column to use. Default: PHIC_ALL_K2")
    parser.add_argument("--fom_col", default="FOM_K2_atten", help="FOM column to use. Default: FOM_K2_atten")
    parser.add_argument("--d_col", default="dHKL", help="Resolution column. Default: dHKL")
    parser.add_argument("--default_solvent_content", type=float, default=None, help="Fallback solvent content in [0,1] if not present in CSV.")
    parser.add_argument("--hl_cols", default=None, help="Optional comma-separated HL columns. If omitted, helper will recompute HL.")
    parser.add_argument("--force_recompute_hl", action="store_true", help="Force recomputation of HL coefficients from phi/FOM.")
    parser.add_argument("--kappa_max", type=float, default=200.0, help="Maximum kappa used in HL reconstruction.")
    parser.add_argument("--mask_cycles", type=int, default=5, help="Number of mask cycles. Default: 5")
    parser.add_argument("--minor_cycles", type=int, default=10, help="Number of minor cycles. Default: 10")
    parser.add_argument("--mask_type", default="histograms", help="Density-modification mask type. Default: histograms")
    parser.add_argument("--rad_mask", type=float, default=4.0, help="Mask radius. Default: 4.0")
    parser.add_argument("--rad_wang", type=float, default=None, help="Optional Wang radius.")
    parser.add_argument("--verbose", action="store_true", help="Verbose DM output.")
    return parser.parse_args(argv)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv=argv)
    if int(args.nproc) < 1:
        raise Sorry("nproc must be >= 1")
    if float(args.job_timeout_sec) <= 0.0:
        raise Sorry("job_timeout_sec must be > 0")

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    _mkdir_p(out_dir)
    log_path = _configure_logging(os.path.join(out_dir, "logs"), "30_run_multiwindow_density_modification")
    logging.info("Run prefix: %s", "30_run_multiwindow_density_modification")
    logging.info("Command line: %s", " ".join([_shell_quote(x) for x in sys.argv]))
    logging.info("out_dir=%s", out_dir)
    logging.info("csv_in=%s | csv_list=%s | csv_dir=%s", str(args.csv_in), str(args.csv_list), str(args.csv_dir))
    logging.info("phi_col=%s | fom_col=%s | d_col=%s", str(args.phi_col), str(args.fom_col), str(args.d_col))
    logging.info("mask_type=%s | mask_cycles=%d | minor_cycles=%d", str(args.mask_type), int(args.mask_cycles), int(args.minor_cycles))
    logging.info("nproc=%d | job_timeout_sec=%.3f | poll_interval_sec=%.3f", int(args.nproc), float(args.job_timeout_sec), float(args.poll_interval_sec))
    logging.info("Helper module required: phenix_like_density_modification_from_csv_py27.py")
    csv_list = _collect_csv_inputs(args=args)
    if int(args.n_max) > 0:
        csv_list = csv_list[:int(args.n_max)]
    windows = _parse_resolution_windows(text_value=args.resolution_windows)
    logging.info("Discovered CSV files: %d", int(len(csv_list)))
    logging.info("n_max=%d", int(args.n_max))
    logging.info("Resolution windows: %s", ", ".join([_window_tag(dmax, dmin) for (dmax, dmin) in windows]))

    all_jobs = []
    skipped_results = []
    for dmax, dmin in windows:
        window_out_dir = os.path.join(out_dir, _window_tag(dmax=dmax, dmin=dmin))
        _mkdir_p(window_out_dir)
        for csv_in in csv_list:
            job = {
                "csv_in": csv_in,
                "window_out_dir": window_out_dir,
                "dmax": dmax,
                "dmin": dmin,
                "args": args,
            }
            probe = _status_template(job)
            should_skip = False
            if not args.overwrite:
                if args.skip_existing_mode == "complete":
                    should_skip = _is_completed_job(probe["log_path"], probe["mtz_path"])
                else:
                    if os.path.isfile(probe["mtz_path"]) and os.path.getsize(probe["mtz_path"]) > 0:
                        should_skip = True
                        if os.path.isfile(probe["log_path"]):
                            try:
                                with open(probe["log_path"], "r") as handle:
                                    existing_txt = handle.read().upper()
                                if ("\nERROR" in existing_txt) or ("\nTIMEOUT" in existing_txt):
                                    should_skip = False
                            except Exception:
                                should_skip = True
            if should_skip:
                probe["status"] = "SKIPPED"
                probe["error"] = "Existing output detected; skipping."
                skipped_results.append(probe)
            else:
                all_jobs.append(job)

    logging.info("Jobs to run: %d", int(len(all_jobs)))
    logging.info("Jobs skipped: %d", int(len(skipped_results)))
    logging.info("skip_existing_mode=%s | overwrite=%s", str(args.skip_existing_mode), str(bool(args.overwrite)))

    results = list(skipped_results)
    n_ok = 0
    n_fail = 0
    n_timeout = 0
    n_skip = len(skipped_results)

    pending = list(all_jobs)
    active = []
    completed_count = 0
    total_count = len(all_jobs) + len(skipped_results)
    while pending or active:
        while pending and len(active) < int(args.nproc):
            job = pending.pop(0)
            job_queue = multiprocessing.Queue()
            proc = multiprocessing.Process(target=_worker_entry, args=(job, job_queue))
            proc.daemon = False
            proc.start()
            active.append({
                "proc": proc,
                "job": job,
                "started": time.time(),
                "queue": job_queue,
            })

        next_active = []
        for item in active:
            proc = item["proc"]
            job = item["job"]
            job_queue = item["queue"]
            elapsed = time.time() - item["started"]
            if proc.is_alive() and elapsed > float(args.job_timeout_sec):
                _terminate_process_tree(proc)
                result = _status_template(job)
                result["status"] = "TIMEOUT"
                result["error"] = "Job exceeded %.3f s and was terminated." % float(args.job_timeout_sec)
                result["elapsed_sec"] = elapsed
                _append_timeout_note(result["log_path"], result["error"])
                results.append(result)
                completed_count += 1
                n_timeout += 1
                sys.stdout.write("[%d/%d] TIMEOUT: %s in %s\n" % (
                    completed_count + n_skip, total_count, result["pdb_id"], result["window_tag"]
                ))
                sys.stdout.flush()
            elif proc.is_alive():
                next_active.append(item)
            else:
                proc.join(timeout=0.1)
                got_result = False
                try:
                    result = job_queue.get_nowait()
                    got_result = True
                    results.append(result)
                    completed_count += 1
                    if result["status"] == "OK":
                        n_ok += 1
                        logging.info("[%d/%d] OK: %s in %s",
                            completed_count + n_skip, total_count, result["pdb_id"], result["window_tag"])
                    else:
                        n_fail += 1
                        logging.error("[%d/%d] FAILED: %s in %s :: %s",
                            completed_count + n_skip, total_count, result["pdb_id"], result["window_tag"], result["error"])
                except Exception:
                    pass
                try:
                    job_queue.close()
                except Exception:
                    pass
                if not got_result:
                    result = _status_template(job)
                    result["status"] = "FAILED"
                    result["error"] = "Worker exited without returning a result."
                    results.append(result)
                    completed_count += 1
                    n_fail += 1
                    logging.error("[%d/%d] FAILED: %s in %s :: %s",
                        completed_count + n_skip, total_count, result["pdb_id"], result["window_tag"], result["error"])

        active = next_active
        if pending or active:
            time.sleep(float(args.poll_interval_sec))

    # de-duplicate possible queue spillover entries
    deduped = []
    seen = set()
    for row in results:
        key = (row["pdb_id"], row["window_tag"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    results = deduped

    summary_csv = os.path.join(out_dir, "denmod_multiwindow_summary.csv")
    summary_df = pd.DataFrame(results)
    if summary_df.shape[0] > 0:
        summary_df = summary_df.sort_values(by=["dmax", "dmin", "pdb_id"], ascending=[True, False, True])
    summary_df.to_csv(summary_csv, index=False)

    log_summary_csv, log_summary_df = _build_denmod_log_summary(out_dir)
    logging.info("Wrote parsed denmod log summary: %s", log_summary_csv)

    sys.stdout.write("\nBatch finished.\n")
    sys.stdout.write("Successful: %d\n" % n_ok)
    sys.stdout.write("Failed: %d\n" % n_fail)
    sys.stdout.write("Timed out: %d\n" % n_timeout)
    sys.stdout.write("Skipped: %d\n" % n_skip)
    sys.stdout.write("Summary CSV: %s\n" % summary_csv)
    sys.stdout.write("Parsed denmod log summary CSV: %s\n" % log_summary_csv)
    sys.stdout.flush()
    logging.info("Log summary rows: %d", int(log_summary_df.shape[0]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Sorry as exc:
        sys.stderr.write("[ERROR] %s\n" % str(exc))
        sys.exit(2)
    except Exception as exc:
        sys.stderr.write("[UNEXPECTED ERROR] %s\n" % str(exc))
        traceback.print_exc(file=sys.stderr)
        sys.exit(2)
