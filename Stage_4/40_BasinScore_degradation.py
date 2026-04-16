#!/usr/bin/env phenix.python
# -*- coding: utf-8 -*-
"""
phenix.python 40_BasinScore_degradation.py   --csv_in ../Attenuated_Signed_Amplitudes-19Mar2026/12_DenMod_BasinScore_Calibration/CSV/2bkf_rs_ecalc_binned.csv   --out_root ./40_BasinScore_Degradation/2bkf-new   --nproc 29   --seed_master 1337   --dm_score_dmax 20.0   --dm_score_dmin 3.5   --skeleton_helper ./40_api_skeleton_proxy.py   --python3_exe /home/alba/miniforge3/bin/python3   --mtz_writer_helper ./40_api_autobuild_mtz_writer.py   --skeleton_threshold_values 0.8,1.0,1.2,1.4,1.7,2.0   --skeleton_target_threshold 1.2   --skeleton_prune_tip_iterations 3   --skeleton_edge_connectivity 26   --degradation_modes uniform_random   --degradation_fractions 0.00,0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00   --n_rounds 1000   --task_timeout_sec 18   --sample_ccp4_maps 10   --maxtasksperchild 1   --round_chunk_size 250


"""

from __future__ import division, print_function

import os
import subprocess
import sys
import time
import argparse
import logging
from datetime import datetime
import traceback
import re
import shutil
import multiprocessing as mp
import csv
import json
import signal

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

RUN_PREFIX = "40_BasinScore_degradation"
DEFAULT_OUT_ROOT = "40_BasinScore_Degradation"
DEFAULT_API50_MTZ_WRITER = "40_api_autobuild_mtz_writer.py"


class WorkerTaskTimeoutError(Exception):
    pass

def _worker_alarm_handler(signum, frame):
    raise WorkerTaskTimeoutError("worker task timeout")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from libtbx.utils import Sorry

try:
    from cctbx import sgtbx
    from cctbx.array_family import flex
    from iotbx import mtz as iotbx_mtz
except Exception:
    sgtbx = None
    flex = None
    iotbx_mtz = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import phenix_like_density_modification_from_csv_py27 as denmod
try:
    from api43_density_modification import run_density_modification_protocol, ensure_fom_k2_atten_in_dataframe
except Exception:
    from pc43_dm_api__v01 import run_density_modification_protocol, ensure_fom_k2_atten_in_dataframe
import zlib


DEFAULTS = dict(
    control_phib_col="PHIC_ALL_K2",
    fom_col="FOM_K2_atten",

    fom_prior=0.64,
    extension_placeholder_fom=0.0,

    balance_tol=0.03,
    centric_round_deg=1.0,
    kappa_max=200.0,

    mask_cycles=5,
    minor_cycles=10,
    rad_mask=4.0,
    rad_wang=None,
    mask_type="histograms",
    clean_up=True,
    verbose=False,
    no_write_files=True,
    keep_temp=False,

    dm_score_dmin=3.5,
    dm_score_dmax=20.0,
    extension_dmin=2.5,

    deployed_score_name="L_20_3p5",
    deployed_window_label="dmax20.0__dmin3.5",

    score_r_good=0.2470,
    score_r_bad=0.2900,
    score_g_mode="logistic",
    score_g_k=2.0,

    w_outer_endpoint=0.80,
    w_outer_pull=0.10,
    w_outer_tail=0.10,
    alpha_penalty=0.15,

    w_cc=0.53,
    w_r=0.47,
    w_fom=0.0,

    dcc_scale=0.10,
    dr_scale=0.08,
    dfom_scale=0.08,
    w_dcc=0.45,
    w_dr=0.45,
    w_dfom=0.10,

    tail_frac=0.40,
    tail_min_cycles=3,
    w_tail_endpoint=0.60,
    w_tail_stability=0.40,

    cc_mad_scale=0.020,
    r_mad_scale=0.020,
    fom_mad_scale=0.030,
    phase_vol_scale_deg=25.0,
    flip_rate_scale=0.15,

    w_cc_stab=0.25,
    w_r_stab=0.25,
    w_fom_stab=0.10,
    w_cc_mon=0.10,
    w_r_mon=0.10,
    w_fom_mon=0.05,
    w_phase_stab=0.10,
    w_flip_stab=0.05,

    w_cc_reversal_pen=0.15,
    w_r_reversal_pen=0.15,
    w_fom_reversal_pen=0.05,
    w_phase_vol_pen=0.30,
    w_flip_rate_pen=0.20,
    w_low_nref_pen=0.15,
    w_large_jump_pen=0.00,
    penalty_phase_vol_scale_deg=40.0,
    penalty_flip_rate_scale=0.25,
    low_nref_soft_min=1000,
    large_jump_scale_deg=90.0,

    fom_std_eps=1e-6,
)

CURRENT_THR_R_FACTOR = 0.34
CURRENT_THR_DM_MEAN_FOM = 0.86
CURRENT_THR_LCF_AUC = 0.38
CURRENT_THR_ENDPOINT_FRAC = 0.08

CURRENT_DM_LOGIT_INTERCEPT = -18.072
CURRENT_DM_LOGIT_R_COEF = -38.171
CURRENT_DM_LOGIT_FOM_COEF = 34.256

CURRENT_SK_LOGIT_INTERCEPT = -1.436
CURRENT_SK_LOGIT_LCF_AUC_COEF = 6.063
CURRENT_SK_LOGIT_ENDPOINT_COEF = -9.757

try:
    unicode  # noqa
except NameError:
    unicode = str


_WORKER_SHARED = None
_LOG_FH = None


def _init_worker(shared):
    global _WORKER_SHARED
    _WORKER_SHARED = shared


_RE_MASK_CYCLE = re.compile(r"\bMask\s*cycle\s*([0-9]+)\b", re.IGNORECASE)
_RE_OVERALL_AVG_CC = re.compile(
    r"Overall average CC:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)
_RE_RFCFP_AND_N = re.compile(
    r"Overall R-factor for FC vs FP:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)\s*for\s*([0-9]+)\s*reflections",
    re.IGNORECASE,
)
_RE_FOM_PATTERNS = [
    re.compile(r"(?:overall\s+)?(?:average|mean)?\s*FOM[^:]*:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)", re.IGNORECASE),
    re.compile(r"(?:overall\s+)?(?:average|mean)?\s*figure\s+of\s+merit[^:]*:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)", re.IGNORECASE),
    re.compile(r"CORRECTED OVERALL FIGURE OF MERIT OF PHASING:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)", re.IGNORECASE),
    re.compile(r"Mean fom of this map was:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)", re.IGNORECASE),
]

_RE_DM_MEAN_FOM = re.compile(
    r"DM\s+mean\s+FOM\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)
_RE_CC_PROB_MAP = re.compile(
    r"CC\s+of\s+prob\s+map\s+with\s+current\s+map\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)
_RE_BIAS_RATIO = re.compile(
    r"BIAS\s+RATIO\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)
_RE_REFLECTIONS_INPUT = re.compile(
    r"Number\s+of\s+reflections\s+in\s+input\s+refl_db\s*:\s*([0-9]+)",
    re.IGNORECASE,
)


def _u_fs(x):
    if x is None:
        return None
    if isinstance(x, unicode):
        return x
    try:
        fsenc = sys.getfilesystemencoding() or "utf-8"
    except Exception:
        fsenc = "utf-8"
    try:
        return unicode(x, fsenc, "replace")  # type: ignore
    except Exception:
        return unicode(str(x))


class _UnicodeBuffer(object):
    def __init__(self):
        self._parts = []

    def write(self, s):
        if s is None:
            return
        self._parts.append(_u_fs(s))

    def getvalue(self):
        return u"".join(self._parts)



def configure_logging(out_root, run_prefix=RUN_PREFIX):
    out_root = os.path.abspath(os.path.expanduser(str(out_root)))
    logs_dir = os.path.abspath(os.path.join(out_root, os.pardir, "LOGS"))
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


def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _set_log_file(log_path):
    return


def log(msg, level=logging.INFO):
    try:
        logging.log(level, "%s", str(msg))
    except Exception:
        try:
            sys.stdout.write("[%s] %s\n" % (_ts(), str(msg)))
            sys.stdout.flush()
        except Exception:
            pass


def _render_progress_line(done, total, t0, width, prefix):
    total = max(int(total), 1)
    done = int(done)
    frac = float(done) / float(total)
    filled = int(round(width * frac))
    bar = (
        "=" * max(filled - 1, 0)
        + (">" if filled > 0 and done < total else "=")
        + " " * max(width - filled, 0)
    )
    dt = max(time.time() - t0, 1e-6)
    rate = done / dt
    eta = (total - done) / rate if rate > 0 else float("inf")
    return "\r%s[%s] %d/%d (%.1f%%) | %.2f done/s | ETA %.1fs" % (
        prefix, bar, done, total, 100.0 * frac, rate, eta
    )


def _print_progress(done, total, t0, prefix="", force=False, state=None):
    if state is None:
        state = {}
    last_done = state.get("last_done", None)
    if (not force) and (last_done == int(done)):
        return
    msg = _render_progress_line(done=done, total=total, t0=t0, width=28, prefix=prefix)
    sys.stderr.write(msg)
    sys.stderr.flush()
    state["last_done"] = int(done)
    if int(done) >= int(total):
        sys.stderr.write("\n")
        sys.stderr.flush()


def _mkdir_p(path):
    if path and (not os.path.isdir(path)):
        os.makedirs(path)


def _rm_tree(path):
    try:
        if path and os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass


def _require_columns(df, cols, label="CSV"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise Sorry("Missing required columns in %s: %s" % (label, missing))


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _normalize_space_group_symbol(sg_str):
    s = str(sg_str).strip()
    aliases = {
        "P 1 21 1": "P 21",
        "P 1 2 1": "P 2",
        "C 1 2 1": "C 2",
        "I 1 21 1": "I 2",
    }
    if s in aliases:
        return aliases[s]
    return s


def _trace_label(h, k, l):
    h = int(h)
    k = int(k)
    l = int(l)

    if h == 0 and k == 0:
        return "00L"
    if h == 0 and l == 0:
        return "0K0"
    if k == 0 and l == 0:
        return "H00"

    if h == 0 and k != 0 and l != 0:
        if k == l:
            return "0KK"
        if k == -l:
            return "0K-K"
        return "0KL"

    if k == 0 and h != 0 and l != 0:
        if h == l:
            return "H0H"
        if h == -l:
            return "H0-H"
        return "H0L"

    if l == 0 and h != 0 and k != 0:
        if h == k:
            return "HH0"
        if h == -k:
            return "H-H0"
        return "HK0"

    if h == k == l:
        return "H=K=L"
    if h == k:
        return "H=K"
    if h == -k:
        return "H=-K"
    if h == l:
        return "H=L"
    if h == -l:
        return "H=-L"
    if k == l:
        return "K=L"
    if k == -l:
        return "K=-L"

    return "general"


def _canonical_trace_family(space_group, trace_label):
    sg = _normalize_space_group_symbol(space_group)
    t = str(trace_label)

    if sg == "P 21 3":
        if t in ["0K0", "H00", "00L"]:
            return "<axial>"
        if t in ["0KK", "H0H", "HH0", "0K-K", "H0-H", "H-H0"]:
            return "<two_equal>"
        if t in ["0KL", "H0L", "HK0"]:
            return "<general_one_zero>"
        if t in ["H=K=L"]:
            return "<body_diagonal>"
        if t in ["H=K", "H=-K", "H=L", "H=-L", "K=L", "K=-L"]:
            return "<two_equal_general>"
        return "<general>"

    return str(trace_label)


def _support_family_from_phi0(phi0_deg):
    x180 = _wrap360(phi0_deg) % 180.0
    if abs(x180 - 0.0) < 1e-6:
        return "0"
    if abs(x180 - 90.0) < 1e-6:
        return "90"
    return None


def _signed_support_pair_from_support_family_label(support_family_label):
    s = str(support_family_label).strip()
    if s == "0":
        return 0.0, 180.0
    if s == "90":
        return 90.0, 270.0
    return None, None


def _free_index_value_for_trace(trace_label, h, k, l):
    h = int(h)
    k = int(k)
    l = int(l)
    t = str(trace_label)

    if t == "0KL":
        return l
    if t == "H0L":
        return h
    if t == "HK0":
        return k

    if t == "0K0":
        return k
    if t == "H00":
        return h
    if t == "00L":
        return l

    if t == "0KK":
        return k
    if t == "H0H":
        return h
    if t == "HH0":
        return h

    if t == "0K-K":
        return k
    if t == "H0-H":
        return h
    if t == "H-H0":
        return h

    return None


def _parity_rule_applies(rule_text, trace_label, canonical_family, h, k, l):
    txt = str(rule_text).strip()

    if txt == "" or txt == "no parity":
        return True

    h = int(h)
    k = int(k)
    l = int(l)

    if txt == "if H even":
        return (h % 2) == 0
    if txt == "if H odd":
        return (h % 2) == 1
    if txt == "if K even":
        return (k % 2) == 0
    if txt == "if K odd":
        return (k % 2) == 1
    if txt == "if L even":
        return (l % 2) == 0
    if txt == "if L odd":
        return (l % 2) == 1

    if txt == "if H+K even":
        return ((h + k) % 2) == 0
    if txt == "if H+K odd":
        return ((h + k) % 2) == 1
    if txt == "if H+L even":
        return ((h + l) % 2) == 0
    if txt == "if H+L odd":
        return ((h + l) % 2) == 1
    if txt == "if K+L even":
        return ((k + l) % 2) == 0
    if txt == "if K+L odd":
        return ((k + l) % 2) == 1
    if txt == "if H+K+L even":
        return ((h + k + l) % 2) == 0
    if txt == "if H+K+L odd":
        return ((h + k + l) % 2) == 1

    if txt == "if free index even":
        fv = _free_index_value_for_trace(trace_label=trace_label, h=h, k=k, l=l)
        if fv is None:
            return False
        return (int(fv) % 2) == 0

    if txt == "if free index odd":
        fv = _free_index_value_for_trace(trace_label=trace_label, h=h, k=k, l=l)
        if fv is None:
            return False
        return (int(fv) % 2) == 1

    if txt == "if axis index even":
        fv = _free_index_value_for_trace(trace_label=trace_label, h=h, k=k, l=l)
        if fv is None:
            return False
        return (int(fv) % 2) == 0

    if txt == "if axis index odd":
        fv = _free_index_value_for_trace(trace_label=trace_label, h=h, k=k, l=l)
        if fv is None:
            return False
        return (int(fv) % 2) == 1

    if txt == "if equal index even":
        fv = _free_index_value_for_trace(trace_label=trace_label, h=h, k=k, l=l)
        if fv is None:
            return False
        return (int(fv) % 2) == 0

    if txt == "if equal index odd":
        fv = _free_index_value_for_trace(trace_label=trace_label, h=h, k=k, l=l)
        if fv is None:
            return False
        return (int(fv) % 2) == 1

    return False


def _load_centric_prior_json(json_path):
    if json_path is None:
        return None
    with open(json_path, "r") as fh:
        obj = json.load(fh)
    return obj


def _log_actionable_centric_priors(prior_obj):
    if prior_obj is None:
        log("No centric prior JSON provided. Using original centric seeding.")
        return

    log("Loaded centric prior JSON for SG=%s" % str(prior_obj.get("space_group", "")))
    fams = prior_obj.get("families", [])
    if len(fams) < 1:
        log("Centric prior JSON has no families. Using original centric seeding.")
        return

    log("Actionable centric priors:")
    for fam in fams:
        fam_lab = str(fam.get("family_label", ""))
        reps = fam.get("representative_traces", [])
        rules = fam.get("rules", [])
        rep_txt = ",".join([str(x) for x in reps])
        log("  Family %s | representative_traces=%s" % (fam_lab, rep_txt))
        for rule in rules:
            log("    support=%s | parity=%s | mode=%s | p_base=%.3f | p_shift180=%.3f | tol=%.3f | conf=%s" % (
                str(rule.get("support_family", "")),
                str(rule.get("parity_rule", "")),
                str(rule.get("recommended_seed_mode", "")),
                float(rule.get("p_base", 0.5)),
                float(rule.get("p_shift180", 0.5)),
                float(rule.get("balance_tol_local", 0.05)),
                str(rule.get("prior_confidence", "")),
            ))


def _build_centric_prior_assignment_table(df, idx_score, phi0_all, sg_str, prior_obj):
    if prior_obj is None:
        return pd.DataFrame()

    sg_json = _normalize_space_group_symbol(prior_obj.get("space_group", ""))
    sg_here = _normalize_space_group_symbol(sg_str)
    if sg_json != sg_here:
        log("Centric prior JSON SG=%s does not match CSV SG=%s. Ignoring centric priors." % (sg_json, sg_here))
        return pd.DataFrame()

    fam_rules = []
    for fam in prior_obj.get("families", []):
        fam_lab = str(fam.get("family_label", ""))
        for rule in fam.get("rules", []):
            fam_rules.append(dict(
                family_label=fam_lab,
                support_family=str(rule.get("support_family", "")),
                parity_rule=str(rule.get("parity_rule", "")),
                recommended_seed_mode=str(rule.get("recommended_seed_mode", "balanced_joint")),
                p_base=float(rule.get("p_base", 0.5)),
                p_shift180=float(rule.get("p_shift180", 0.5)),
                balance_tol_local=float(rule.get("balance_tol_local", 0.05)),
                prior_confidence=str(rule.get("prior_confidence", "")),
            ))

    if len(fam_rules) < 1:
        return pd.DataFrame()

    idx_score = np.asarray(idx_score, dtype=int)
    rows = []

    for i in idx_score:
        if int(df.iloc[i]["CENTRIC"]) != 1:
            continue

        h = int(df.iloc[i]["H"])
        k = int(df.iloc[i]["K"])
        l = int(df.iloc[i]["L"])

        trace_label = _trace_label(h=h, k=k, l=l)
        canonical_family = _canonical_trace_family(space_group=sg_str, trace_label=trace_label)
        support_family = _support_family_from_phi0(phi0_deg=phi0_all[i])
        base_deg, shift_deg = _signed_support_pair_from_support_family_label(support_family_label=support_family)

        if support_family is None or base_deg is None:
            continue

        for fr in fam_rules:
            if str(fr["family_label"]) != str(canonical_family):
                continue
            if str(fr["support_family"]) != str(support_family):
                continue
            if not _parity_rule_applies(
                rule_text=fr["parity_rule"],
                trace_label=trace_label,
                canonical_family=canonical_family,
                h=h, k=k, l=l,
            ):
                continue

            rows.append(dict(
                row_index=int(i),
                H=int(h),
                K=int(k),
                L=int(l),
                trace_label=str(trace_label),
                canonical_family=str(canonical_family),
                support_family=str(support_family),
                base_deg=float(base_deg),
                shift_deg=float(shift_deg),
                parity_rule=str(fr["parity_rule"]),
                recommended_seed_mode=str(fr["recommended_seed_mode"]),
                p_base=float(fr["p_base"]),
                p_shift180=float(fr["p_shift180"]),
                balance_tol_local=float(fr["balance_tol_local"]),
                prior_confidence=str(fr["prior_confidence"]),
                prior_class_id="%s||%s||%s" % (str(canonical_family), str(support_family), str(fr["parity_rule"])),
            ))
            break

    out = pd.DataFrame(rows)
    return out


def _log_prior_respect_for_phase_vector(ph_full, prior_assign_df, prefix_label):
    if prior_assign_df is None or prior_assign_df.shape[0] < 1:
        log("%s | no centric priors active" % str(prefix_label))
        return

    ph = np.asarray(ph_full, dtype=float) % 360.0
    log("%s | centric-prior respect check (diagnostic only; small deviations are acceptable):" % str(prefix_label))

    grouped = prior_assign_df.groupby(["prior_class_id"], as_index=False)
    for _, sub in grouped:
        first = sub.iloc[0]
        idx = sub["row_index"].astype(int).values
        vals = ph[idx]
        base_deg = float(first["base_deg"])
        shift_deg = float(first["shift_deg"])
        target_p = float(first["p_base"])
        m = int(len(idx))
        n_base = int(np.sum(np.abs(vals - base_deg) < 1e-6))
        frac_base = float(n_base) / float(max(m, 1))
        log("  class=%s | n=%d | base=%s shift=%s | observed base_frac=%.3f | target base_frac=%.3f | tol=%.3f" % (
            str(first["prior_class_id"]),
            int(m),
            _fmt_phase_label(base_deg),
            _fmt_phase_label(shift_deg),
            float(frac_base),
            float(target_p),
            float(first["balance_tol_local"]),
        ))


def _phase_value_counts(ph_deg, mask_bool, round_decimals=3):
    ph = np.asarray(ph_deg, dtype=float) % 360.0
    mask = np.asarray(mask_bool, dtype=bool)
    vals = ph[mask]
    if vals.size < 1:
        return []
    vals_r = np.round(vals, int(round_decimals))
    uniq, counts = np.unique(vals_r, return_counts=True)
    order = np.argsort(uniq)
    rows = []
    for u, c in zip(uniq[order], counts[order]):
        rows.append((float(u), int(c)))
    return rows


def _fmt_phase_label(phi_deg):
    x = float(phi_deg)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return ("%.3f" % x).rstrip("0").rstrip(".")

def _format_phase_distribution_line(ph_deg, cent_flags, total_n, prefix_label):
    cent_flags = np.asarray(cent_flags, dtype=int)
    total_n = max(int(total_n), 1)

    rows_c = _phase_value_counts(
        ph_deg=ph_deg,
        mask_bool=(cent_flags == 1),
        round_decimals=3,
    )
    rows_a = _phase_value_counts(
        ph_deg=ph_deg,
        mask_bool=(cent_flags == 0),
        round_decimals=3,
    )

    parts = []
    for phi, cnt in rows_c:
        pct = 100.0 * float(cnt) / float(total_n)
        parts.append("C %s: %d (%.1f%%)" % (_fmt_phase_label(phi), int(cnt), float(pct)))
    for phi, cnt in rows_a:
        pct = 100.0 * float(cnt) / float(total_n)
        parts.append("A %s: %d (%.1f%%)" % (_fmt_phase_label(phi), int(cnt), float(pct)))

    if len(parts) < 1:
        return "%s = <no phases>" % str(prefix_label)
    return "%s = %s" % (str(prefix_label), "; ".join(parts))


def _log_phase_distribution(ph_deg, cent_flags, total_n, prefix_label):
    log(_format_phase_distribution_line(
        ph_deg=ph_deg,
        cent_flags=cent_flags,
        total_n=total_n,
        prefix_label=prefix_label,
    ))


def _wrap360(phi_deg):
    return float(phi_deg) % 360.0


def _wrap_pm180(phi_deg):
    x = float(phi_deg)
    x = ((x + 180.0) % 360.0) - 180.0
    if abs(x + 180.0) < 1e-9:
        x = 180.0
    return x

def _approx_i1_over_i0(k):
    k = float(k)
    if k < 1e-6:
        return 0.0
    if k < 3.75:
        k2 = k * k
        return (0.5 * k) - (0.0625 * k * k2) + (0.0104166666667 * k * k2 * k2)
    return 1.0 - (1.0 / (2.0 * k)) + (1.0 / (8.0 * k * k)) - (1.0 / (16.0 * k * k * k))


def _kappa_from_fom(m, kappa_max):
    m = float(m)
    if (not np.isfinite(m)) or (m <= 0.0):
        return 0.0
    if m >= 0.9999:
        return float(kappa_max)
    lo, hi = 0.0, float(kappa_max)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _approx_i1_over_i0(mid) < m:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _hl_from_phib_fom(phib_deg, fom_vals, kappa_max):
    if flex is None:
        raise Sorry("flex not available")
    phi = np.asarray(phib_deg, dtype=float)
    m = np.asarray(fom_vals, dtype=float)
    phir = (np.pi / 180.0) * phi
    c = np.cos(phir)
    s = np.sin(phir)
    kappas = np.zeros(phi.size, dtype=float)
    for i in range(phi.size):
        kappas[i] = _kappa_from_fom(m[i], kappa_max)
    A = flex.double((kappas * c).tolist())
    B = flex.double((kappas * s).tolist())
    C = flex.double((np.zeros(phi.size)).tolist())
    D = flex.double((np.zeros(phi.size)).tolist())
    return flex.hendrickson_lattman(A, B, C, D)


def _hl_coeff_arrays(hl_obj):
    """
    Return numpy arrays (A,B,C,D) from a flex.hendrickson_lattman object
    across older/newer Phenix builds.
    """
    try:
        A = np.asarray(list(hl_obj.a()), dtype=float)
        B = np.asarray(list(hl_obj.b()), dtype=float)
        C = np.asarray(list(hl_obj.c()), dtype=float)
        D = np.asarray(list(hl_obj.d()), dtype=float)
        return A, B, C, D
    except Exception:
        pass

    try:
        data = hl_obj.data()
        A = np.asarray(list(data.a()), dtype=float)
        B = np.asarray(list(data.b()), dtype=float)
        C = np.asarray(list(data.c()), dtype=float)
        D = np.asarray(list(data.d()), dtype=float)
        return A, B, C, D
    except Exception:
        pass

    try:
        n = int(hl_obj.size())
        A = np.zeros(n, dtype=float)
        B = np.zeros(n, dtype=float)
        C = np.zeros(n, dtype=float)
        D = np.zeros(n, dtype=float)
        for i in range(n):
            rec = hl_obj[i]
            A[i] = float(rec[0])
            B[i] = float(rec[1])
            C[i] = float(rec[2])
            D[i] = float(rec[3])
        return A, B, C, D
    except Exception:
        pass

    raise Sorry("Unable to extract HL coefficient arrays from hendrickson_lattman object in this Phenix build.")
    
def _phase_match_mask(ph_ref_deg, ph_test_deg, tol_deg=1e-6):
    ref = np.asarray(ph_ref_deg, dtype=float) % 360.0
    tst = np.asarray(ph_test_deg, dtype=float) % 360.0
    diffs = np.abs(ref - tst)
    diffs = np.minimum(diffs, 360.0 - diffs)
    return diffs <= float(tol_deg)


def _format_checkpoint_match_summary(ph_ref_deg, ph_test_deg, d_hkl,
                                     dmax_score, dmin_score, extension_dmin,
                                     low_split=8.0, med_split=4.0, tol_deg=1e-6):
    ph_ref = np.asarray(ph_ref_deg, dtype=float) % 360.0
    ph_test = np.asarray(ph_test_deg, dtype=float) % 360.0
    d = np.asarray(d_hkl, dtype=float)

    same = _phase_match_mask(
        ph_ref_deg=ph_ref,
        ph_test_deg=ph_test,
        tol_deg=float(tol_deg),
    )

    def _pct(mask):
        n = int(np.sum(mask))
        if n < 1:
            return n, np.nan
        return n, 100.0 * float(np.sum(same[mask])) / float(n)

    mask_low = (d <= float(dmax_score)) & (d >= float(low_split))
    mask_med = (d < float(low_split)) & (d >= float(med_split))
    mask_high = (d < float(med_split)) & (d >= float(dmin_score))
    mask_not = (d < float(dmin_score)) & (d >= float(extension_dmin))

    n_low, p_low = _pct(mask_low)
    n_med, p_med = _pct(mask_med)
    n_high, p_high = _pct(mask_high)
    n_not = int(np.sum(mask_not))

    def _fmt_pct(p):
        if not np.isfinite(_safe_float(p, default=np.nan)):
            return "NA"
        return "%.1f%%" % float(p)

    return (
        "Checkpoint phase-match vs Ctrl | "
        "Low (%.1f-%.1fA; %d): %s; "
        "Medium (%.1f-%.1fA; %d): %s; "
        "High (%.1f-%.1fA; %d): %s; "
        "Not phased set (%.1f-%.1fA): %d reflections"
    ) % (
        float(dmax_score), float(low_split), int(n_low), _fmt_pct(p_low),
        float(low_split), float(med_split), int(n_med), _fmt_pct(p_med),
        float(med_split), float(dmin_score), int(n_high), _fmt_pct(p_high),
        float(dmin_score), float(extension_dmin), int(n_not),
    )


def _log_checkpoint_best_details(best_row, best_ph_full, shared):
    if best_row is None:
        log("Checkpoint detail | no valid seed yet")
        return

    log("Checkpoint best detail | seed_idx=%d | hand=%s | L_20_3p5=%.3f" % (
        int(best_row["seed_idx"]),
        str(best_row["hand"]),
        float(best_row["score"]),
    ))

    if best_ph_full is None:
        log("Checkpoint best phase distribution unavailable (best_ph_full missing)")
        return

    _log_phase_distribution(
        ph_deg=np.asarray(best_ph_full, dtype=float) % 360.0,
        cent_flags=np.asarray(shared["df"]["CENTRIC"].astype(int).values, dtype=int),
        total_n=int(shared["df"].shape[0]),
        prefix_label="Checkpoint BS %.2f Seed %d (%s)" % (
            float(best_row["score"]),
            int(best_row["seed_idx"]),
            str(best_row["hand"]),
        ),
    )

    log(_format_checkpoint_match_summary(
        ph_ref_deg=np.asarray(shared["control_phases_full"], dtype=float) % 360.0,
        ph_test_deg=np.asarray(best_ph_full, dtype=float) % 360.0,
        d_hkl=np.asarray(shared["df"]["dHKL"].astype(float).values, dtype=float),
        dmax_score=float(shared["dm_score_dmax"]),
        dmin_score=float(shared["dm_score_dmin"]),
        extension_dmin=float(shared["extension_dmin"]),
        low_split=8.0,
        med_split=4.0,
        tol_deg=1e-6,
    ))

    _log_prior_respect_for_phase_vector(
        ph_full=best_ph_full,
        prior_assign_df=shared.get("centric_prior_assignments_score", pd.DataFrame()),
        prefix_label="Checkpoint BS %.2f Seed %d (%s)" % (
            float(best_row["score"]),
            int(best_row["seed_idx"]),
            str(best_row["hand"]),
        ),
    )


def _clip01(x):
    return float(np.clip(float(x), 0.0, 1.0))


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


def _build_clean_python3_env(python3_exe):
    env = dict(os.environ)
    for key in ["PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "__PYVENV_LAUNCHER__"]:
        if key in env:
            del env[key]
    py_dir = os.path.dirname(os.path.abspath(os.path.expanduser(str(python3_exe))))
    old_path = env.get("PATH", "")
    if py_dir:
        env["PATH"] = py_dir + os.pathsep + old_path
    for key in list(env.keys()):
        if key.startswith("LIBTBX_") or key.startswith("PHENIX_") or key.startswith("CCTBX_"):
            del env[key]
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    return env


def _run_subprocess_with_timeout(cmd, env, timeout_sec):
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=env,
    )
    if timeout_sec is None or float(timeout_sec) <= 0:
        out, _ = proc.communicate()
        return int(proc.returncode), out, False
    t0 = time.time()
    timed_out = False
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        if (time.time() - t0) > float(timeout_sec):
            timed_out = True
            try:
                proc.kill()
            except Exception:
                pass
            break
        time.sleep(0.2)
    try:
        out, _ = proc.communicate()
    except Exception:
        out = ""
    return int(proc.returncode) if proc.returncode is not None else -9, out, timed_out


def _extract_denmod_summary_metrics_from_text(transcript_text):
    txt = _u_fs(transcript_text)
    r_matches = _RE_RFCFP_AND_N.findall(txt)
    r_factor = None
    r_nref = None
    if r_matches:
        try:
            r_factor = float(r_matches[-1][0])
        except Exception:
            r_factor = None
        try:
            r_nref = int(r_matches[-1][1])
        except Exception:
            r_nref = None
    return dict(
        Reflections=_last_match_int(_RE_REFLECTIONS_INPUT, txt),
        DM_mean_FOM=_last_match_float(_RE_DM_MEAN_FOM, txt),
        CC_prob_map_with_current_map=_last_match_float(_RE_CC_PROB_MAP, txt),
        R_factor_FC_vs_FP=r_factor,
        R_factor_reflections=r_nref,
        BIAS_RATIO=_last_match_float(_RE_BIAS_RATIO, txt),
    )


def _sigmoid(x):
    x = float(np.clip(float(x), -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-x)))


def _score_term_dm_r_factor(r_val):
    v = _safe_float(r_val, default=np.nan)
    if not np.isfinite(v):
        return np.nan
    return float(v <= float(CURRENT_THR_R_FACTOR))


def _score_term_dm_mean_fom(fom_val):
    v = _safe_float(fom_val, default=np.nan)
    if not np.isfinite(v):
        return np.nan
    return float(v >= float(CURRENT_THR_DM_MEAN_FOM))


def _score_term_lcf_auc(lcf_auc_val):
    v = _safe_float(lcf_auc_val, default=np.nan)
    if not np.isfinite(v):
        return np.nan
    return float(v >= float(CURRENT_THR_LCF_AUC))


def _score_term_endpoint_fraction(ep_val):
    v = _safe_float(ep_val, default=np.nan)
    if not np.isfinite(v):
        return np.nan
    return float(v <= float(CURRENT_THR_ENDPOINT_FRAC))



def _normalize_skeleton_metrics_for_seed(skel_res):
    """
    Ensure skeleton metrics are always finite enough for scoring/logging.
    If skeleton failed or fields are missing, use conservative fallback values.
    """
    out = {}
    if isinstance(skel_res, dict):
        out.update(skel_res)

    ok = bool(out.get("ok", False))
    lcf = _safe_float(out.get("best_largest_component_fraction_at_target"), default=np.nan)
    epf = _safe_float(out.get("best_endpoint_fraction_at_target"), default=np.nan)
    ncc = _safe_float(out.get("best_n_connected_components_at_target"), default=np.nan)
    mdg = _safe_float(out.get("best_mean_degree_at_target"), default=np.nan)
    lcf_auc = _safe_float(out.get("best_largest_component_fraction_auc"), default=np.nan)

    if (not ok) or (not np.isfinite(lcf)):
        out["best_largest_component_fraction_at_target"] = 0.0
    if (not ok) or (not np.isfinite(epf)):
        out["best_endpoint_fraction_at_target"] = 0.5
    if (not ok) or (not np.isfinite(ncc)):
        out["best_n_connected_components_at_target"] = 999999.0
    if (not ok) or (not np.isfinite(mdg)):
        out["best_mean_degree_at_target"] = 0.0
    if (not ok) or (not np.isfinite(lcf_auc)):
        out["best_largest_component_fraction_auc"] = 0.0

    out["skeleton_ok"] = ok
    if not ok and ("skeleton_error" not in out):
        out["skeleton_error"] = str(out.get("error", "skeleton_failed"))
    return out

def _compute_current_basin_score(dm_metrics, skel_metrics):
    r_val = _safe_float(dm_metrics.get("R_factor_FC_vs_FP", np.nan), default=np.nan)
    fom_val = _safe_float(dm_metrics.get("DM_mean_FOM", np.nan), default=np.nan)
    lcf_auc_val = _safe_float(skel_metrics.get("best_largest_component_fraction_auc", np.nan), default=np.nan)
    endpoint_val = _safe_float(skel_metrics.get("best_endpoint_fraction_at_target", np.nan), default=np.nan)

    p_dm = np.nan
    if np.isfinite(r_val) and np.isfinite(fom_val):
        p_dm = _sigmoid(
            CURRENT_DM_LOGIT_INTERCEPT
            + CURRENT_DM_LOGIT_R_COEF * float(r_val)
            + CURRENT_DM_LOGIT_FOM_COEF * float(fom_val)
        )

    p_sk = np.nan
    if np.isfinite(lcf_auc_val) and np.isfinite(endpoint_val):
        p_sk = _sigmoid(
            CURRENT_SK_LOGIT_INTERCEPT
            + CURRENT_SK_LOGIT_LCF_AUC_COEF * float(lcf_auc_val)
            + CURRENT_SK_LOGIT_ENDPOINT_COEF * float(endpoint_val)
        )

    basin = np.nan
    if np.isfinite(p_dm) and np.isfinite(p_sk):
        basin = 100.0 * 0.5 * (float(p_dm) + float(p_sk))
    elif np.isfinite(p_dm):
        basin = 100.0 * float(p_dm)
    elif np.isfinite(p_sk):
        basin = 100.0 * float(p_sk)

    return dict(
        BasinScore=basin,
        basin_term_r=_score_term_dm_r_factor(r_val),
        basin_term_fom=_score_term_dm_mean_fom(fom_val),
        basin_term_lcf=_score_term_lcf_auc(lcf_auc_val),
        basin_term_endpoint=_score_term_endpoint_fraction(endpoint_val),
        basin_prob_dm=p_dm,
        basin_prob_sk=p_sk,
    )


def _write_seed_proxy_csv(csv_path, df_base, phase_col_name, fom_col_name, ph_full, fom_full):
    df_seed = df_base.copy()
    df_seed[str(phase_col_name)] = np.asarray(ph_full, dtype=float) % 360.0
    df_seed[str(fom_col_name)] = np.asarray(fom_full, dtype=float)
    df_seed.to_csv(csv_path, index=False)



def _run_skeleton_proxy_for_seed(df_base, ph_full, fom_full, temp_root, score_params):
    phase_col_name = "PC50_SEED_PHI"
    fom_col_name = "PC50_SEED_FOM"
    csv_dir = os.path.join(temp_root, "seed_input")
    out_dir = os.path.join(temp_root, "seed_output")
    logs_dir = os.path.join(temp_root, "logs")
    _mkdir_p(csv_dir)
    _mkdir_p(out_dir)
    _mkdir_p(logs_dir)
    csv_path = os.path.join(csv_dir, "seed_rs_ecalc_binned.csv")
    _write_seed_proxy_csv(csv_path, df_base, phase_col_name, fom_col_name, ph_full, fom_full)

    api44_script = os.path.join(_THIS_DIR, "40_api_skeleton_proxy.py")
    legacy_api44_script = os.path.join(_THIS_DIR, "api44_skeleton_proxy.py")
    legacy_pc44_api = os.path.join(_THIS_DIR, "pc44_proxy_api__v01.py")
    if os.path.isfile(api44_script):
        script_to_use = api44_script
    elif os.path.isfile(legacy_api44_script):
        script_to_use = legacy_api44_script
    elif os.path.isfile(legacy_pc44_api):
        script_to_use = legacy_pc44_api
    else:
        return {
            "ok": False,
            "error": "No callable skeleton proxy API found next to 40_BasinScore_degradation.py (expected 40_api_skeleton_proxy.py, api44_skeleton_proxy.py, or pc44_proxy_api__v01.py)",
        }

    json_summary_path = os.path.join(logs_dir, "pc44_result_summary.json")
    stdout_log_path = os.path.join(logs_dir, "pc44_stdout.log")
    cmd_log_path = os.path.join(logs_dir, "pc44_command.txt")

    cmd = [
        str(score_params["python3_exe"]),
        script_to_use,
        "--pc44_script", str(score_params["pc44_script"]),
        "--csv_in", csv_path,
        "--out_dir", out_dir,
        "--phase_unit", "deg",
        "--sample_rate", str(score_params["skeleton_sample_rate"]),
        "--map_kind_for_skeleton", str(score_params["skeleton_map_kind"]),
        "--threshold_mode", "sigma",
        "--threshold_values", str(score_params["skeleton_threshold_values"]),
        "--target_threshold", str(score_params["skeleton_target_threshold"]),
        "--prune_tip_iterations", str(score_params["skeleton_prune_tip_iterations"]),
        "--edge_connectivity", str(score_params["skeleton_edge_connectivity"]),
        "--phase_fom_pairs", "%s:%s" % (phase_col_name, fom_col_name),
        "--dmin", str(score_params["skeleton_dmin"]),
        "--dmax", str(score_params["skeleton_dmax"]),
        "--disable_summary_plots",
        "--json_out", json_summary_path,
    ]

    try:
        with open(cmd_log_path, "w") as fh:
            fh.write(" ".join(cmd) + "\n")
    except Exception:
        pass

    rc, out, timed_out = _run_subprocess_with_timeout(
        cmd=cmd,
        env=_build_clean_python3_env(score_params["python3_exe"]),
        timeout_sec=score_params.get("task_timeout_sec", 60.0),
    )

    try:
        with open(stdout_log_path, "w") as fh:
            if out is None:
                out = ""
            fh.write(str(out))
            if (not str(out).endswith("\n")):
                fh.write("\n")
    except Exception:
        pass

    if timed_out:
        return {
            "ok": False,
            "error": "pc44 timeout after %.1f s" % float(score_params.get("task_timeout_sec", 60.0)),
            "pc44_stdout_log": stdout_log_path,
            "pc44_command_log": cmd_log_path,
            "pc44_json_summary": json_summary_path,
        }
    if int(rc) != 0:
        return {
            "ok": False,
            "error": "pc44 failed rc=%d" % int(rc),
            "pc44_stdout_log": stdout_log_path,
            "pc44_command_log": cmd_log_path,
            "pc44_json_summary": json_summary_path,
        }

    summary = None
    if os.path.isfile(json_summary_path):
        try:
            with open(json_summary_path, "r") as fh:
                summary = json.load(fh)
        except Exception:
            summary = None

    proxy_csv = None
    if isinstance(summary, dict):
        proxy_csv = summary.get("traceability_proxy_summary_csv")
        if proxy_csv is not None:
            proxy_csv = os.path.abspath(os.path.expanduser(str(proxy_csv)))
            if not os.path.isfile(proxy_csv):
                proxy_csv = None

    if proxy_csv is None:
        candidates = []
        for root, dirs, files in os.walk(out_dir):
            for name in files:
                if str(name).endswith("_traceability_proxy_summary.csv"):
                    candidates.append(os.path.join(root, name))
        candidates = sorted(candidates)
        if len(candidates) >= 1:
            proxy_csv = candidates[0]

    if proxy_csv is None:
        return {
            "ok": False,
            "error": "pc44 proxy csv not found",
            "pc44_stdout_log": stdout_log_path,
            "pc44_command_log": cmd_log_path,
            "pc44_json_summary": json_summary_path,
        }

    try:
        proxy_df = pd.read_csv(proxy_csv)
    except Exception as e:
        return {
            "ok": False,
            "error": "failed reading proxy csv: %s" % str(e),
            "pc44_stdout_log": stdout_log_path,
            "pc44_command_log": cmd_log_path,
            "pc44_json_summary": json_summary_path,
            "pc44_proxy_csv": proxy_csv,
        }

    if proxy_df.shape[0] < 1:
        return {
            "ok": False,
            "error": "empty proxy csv",
            "pc44_stdout_log": stdout_log_path,
            "pc44_command_log": cmd_log_path,
            "pc44_json_summary": json_summary_path,
            "pc44_proxy_csv": proxy_csv,
        }

    row = proxy_df.iloc[0].to_dict()
    return {
        "ok": True,
        "best_largest_component_fraction_at_target": _safe_float(row.get("largest_component_fraction_at_target"), default=np.nan),
        "best_n_connected_components_at_target": _safe_float(row.get("n_connected_components_at_target"), default=np.nan),
        "best_endpoint_fraction_at_target": _safe_float(row.get("endpoint_fraction_at_target"), default=np.nan),
        "best_mean_degree_at_target": _safe_float(row.get("mean_degree_at_target"), default=np.nan),
        "best_largest_component_fraction_auc": _safe_float(row.get("largest_component_fraction_auc"), default=np.nan),
        "pc44_stdout_log": stdout_log_path,
        "pc44_command_log": cmd_log_path,
        "pc44_json_summary": json_summary_path,
        "pc44_proxy_csv": proxy_csv,
    }


def _median_or_nan(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 1:
        return np.nan
    return float(np.median(arr))


def _mad_or_nan(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 1:
        return np.nan
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def _weighted_mean_available(values, weights):
    num = 0.0
    den = 0.0
    for v, w in zip(values, weights):
        vv = _safe_float(v, default=np.nan)
        ww = _safe_float(w, default=np.nan)
        if np.isfinite(vv) and np.isfinite(ww) and float(ww) > 0.0:
            num += float(ww) * float(vv)
            den += float(ww)
    if den <= 0.0:
        return np.nan
    return float(num / den)


def _reversal_fraction(values, mode):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    diffs = np.diff(arr)
    if mode == "down":
        bad = diffs > 0
    else:
        bad = diffs < 0
    return float(np.mean(bad.astype(float)))


def _choose_tail_indices(n_rows, tail_frac, tail_min_cycles):
    if n_rows < 1:
        return np.asarray([], dtype=int)
    tail_n = int(np.ceil(float(n_rows) * float(tail_frac)))
    tail_n = max(int(tail_min_cycles), tail_n)
    tail_n = min(int(n_rows), tail_n)
    return np.arange(int(n_rows - tail_n), int(n_rows), dtype=int)


def _style_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel=str(xlabel), fontsize=14)
    ax.set_ylabel(ylabel=str(ylabel), fontsize=14)
    ax.set_title(label=str(title), fontsize=14)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="--", color="gray")
    ax.grid(which="minor", linestyle=":", color="lightgray")
    ax.tick_params(axis="both", which="major", labelsize=14, size=10)
    ax.tick_params(axis="both", which="minor", size=5)


def _save_jitter_box_plot(values, labels, out_png, seed_master, title, ylabel):
    flat = []
    xpos = []
    tickpos = []
    ticklabs = []
    rng = np.random.RandomState(seed=int(seed_master) + 999)
    for i, (lab, vals) in enumerate(zip(labels, values)):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 1:
            continue
        pos = float(i + 1)
        flat.append(vals)
        xpos.append(pos + rng.uniform(low=-0.10, high=0.10, size=int(vals.size)))
        tickpos.append(pos)
        ticklabs.append(str(lab))

    if len(flat) < 1:
        return

    fig = plt.figure(figsize=(max(6, 1.5 * len(flat)), 6))
    ax = fig.add_subplot(111)
    ax.boxplot(
        x=flat,
        positions=tickpos,
        widths=[0.35 for _ in tickpos],
        vert=True,
        patch_artist=False,
        showfliers=True,
    )
    for xj, vals in zip(xpos, flat):
        ax.scatter(xj, vals, s=18, alpha=0.6)

    ax.set_xticks(tickpos)
    ax.set_xticklabels(ticklabs, fontsize=14, rotation=0)
    _style_axes(ax=ax, xlabel="Group", ylabel=ylabel, title=title)
    fig.tight_layout()
    fig.savefig(fname=str(out_png), dpi=300, bbox_inches="tight")
    plt.close(fig)


class RNG(object):
    def __init__(self, seed):
        seed = int(seed) % (2 ** 32)
        self._rs = np.random.RandomState(seed)

    def random(self, size=None):
        if size is None:
            return float(self._rs.rand())
        if isinstance(size, (tuple, list)):
            return self._rs.random_sample(size=size)
        return self._rs.random_sample(size=int(size))

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            return float(self._rs.uniform(low=low, high=high))
        if isinstance(size, (tuple, list)):
            return self._rs.uniform(low=low, high=high, size=tuple(size))
        return self._rs.uniform(low=low, high=high, size=int(size))

    def permutation(self, x):
        return self._rs.permutation(x)

    def shuffle(self, x):
        self._rs.shuffle(x)


def _seed_from_master(seed_master, seed_idx, hand_tag):
    hand_offset = 0 if str(hand_tag) == "normal" else 1
    try:
        seed_idx_int = int(seed_idx)
    except Exception:
        seed_label = _u_fs(seed_idx)
        if isinstance(seed_label, unicode):
            seed_label = seed_label.encode("utf-8")
        elif not isinstance(seed_label, str):
            seed_label = str(seed_label)
        seed_idx_int = int(zlib.crc32(seed_label) & 0xffffffff)
    return int((int(seed_master) + 1000003 * int(seed_idx_int) + 37 * int(hand_offset)) % (2 ** 32))


def _to_flex_double(x):
    if flex is None:
        raise Sorry("flex not available in this phenix.python.")
    return flex.double([float(v) for v in np.asarray(x, dtype=float).ravel()])


def parse_cycle_table_from_text(transcript_text):
    if transcript_text is None:
        return pd.DataFrame(columns=["cycle", "CC", "R", "Nref", "FOM"])

    try:
        lines = _u_fs(transcript_text).splitlines()
    except Exception:
        try:
            lines = unicode(str(transcript_text)).splitlines()
        except Exception:
            lines = []

    cycles = {}
    current_cycle = None

    def _ensure_cycle(cyc):
        if cyc not in cycles:
            cycles[cyc] = {"CC": np.nan, "R": np.nan, "Nref": np.nan, "FOM": np.nan}

    for line in lines:
        mcyc = _RE_MASK_CYCLE.search(line)
        if mcyc:
            try:
                current_cycle = int(mcyc.group(1))
                _ensure_cycle(cyc=current_cycle)
            except Exception:
                current_cycle = None

        mcc = _RE_OVERALL_AVG_CC.search(line)
        if mcc:
            try:
                if current_cycle is None:
                    current_cycle = 9999
                    _ensure_cycle(cyc=current_cycle)
                cycles[current_cycle]["CC"] = float(mcc.group(1))
            except Exception:
                pass

        mr = _RE_RFCFP_AND_N.search(line)
        if mr:
            try:
                if current_cycle is None:
                    current_cycle = 9999
                    _ensure_cycle(cyc=current_cycle)
                cycles[current_cycle]["R"] = float(mr.group(1))
                cycles[current_cycle]["Nref"] = float(int(mr.group(2)))
            except Exception:
                pass

        for regex in _RE_FOM_PATTERNS:
            mf = regex.search(line)
            if mf:
                try:
                    if current_cycle is None:
                        current_cycle = 9999
                        _ensure_cycle(cyc=current_cycle)
                    cycles[current_cycle]["FOM"] = float(mf.group(1))
                except Exception:
                    pass
                break

    rows = []
    keys = sorted([k for k in cycles.keys() if isinstance(k, int)])
    for cyc in keys:
        rows.append({
            "cycle": int(cyc),
            "CC": _safe_float(cycles[cyc].get("CC"), default=np.nan),
            "R": _safe_float(cycles[cyc].get("R"), default=np.nan),
            "Nref": _safe_float(cycles[cyc].get("Nref"), default=np.nan),
            "FOM": _safe_float(cycles[cyc].get("FOM"), default=np.nan),
        })
    return pd.DataFrame(rows, columns=["cycle", "CC", "R", "Nref", "FOM"])


def _extract_float_sequence(pattern, text):
    vals = []
    for m in pattern.finditer(text):
        try:
            vals.append(float(m.group(1)))
        except Exception:
            pass
    return vals


def _parse_global_dm_sequences(transcript):
    try:
        txt = _u_fs(transcript)
    except Exception:
        try:
            txt = unicode(str(transcript))
        except Exception:
            txt = u""

    cc_vals = _extract_float_sequence(pattern=_RE_OVERALL_AVG_CC, text=txt)
    r_vals = []
    nref_vals = []
    for m in _RE_RFCFP_AND_N.finditer(txt):
        try:
            r_vals.append(float(m.group(1)))
        except Exception:
            pass
        try:
            nref_vals.append(float(int(m.group(2))))
        except Exception:
            pass

    fom_vals = []
    for rgx in _RE_FOM_PATTERNS:
        tmp = _extract_float_sequence(pattern=rgx, text=txt)
        if len(tmp) > len(fom_vals):
            fom_vals = tmp

    return {
        "CC": np.asarray(cc_vals, dtype=float),
        "R": np.asarray(r_vals, dtype=float),
        "Nref": np.asarray(nref_vals, dtype=float),
        "FOM": np.asarray(fom_vals, dtype=float),
    }


def _compute_cycle_features_from_df(cycle_df, fom_first_mean, fom_in_std, nref_fallback):
    rec = dict()
    rec["FOM_first"] = float(fom_first_mean)
    rec["FOM_in_std"] = float(fom_in_std)
    rec["n_reflections_last"] = _safe_float(nref_fallback, default=np.nan)

    if cycle_df is None or cycle_df.shape[0] < 1:
        rec["CC_first"] = np.nan
        rec["CC_last"] = np.nan
        rec["R_first"] = np.nan
        rec["R_last"] = np.nan
        rec["FOM_last"] = np.nan
        rec["n_cycles_parsed"] = 0
        rec["CC_tail_median"] = np.nan
        rec["R_tail_median"] = np.nan
        rec["FOM_tail_median"] = np.nan
        rec["CC_tail_mad"] = np.nan
        rec["R_tail_mad"] = np.nan
        rec["FOM_tail_mad"] = np.nan
        rec["CC_tail_reversal_frac"] = np.nan
        rec["R_tail_reversal_frac"] = np.nan
        rec["FOM_tail_reversal_frac"] = np.nan
        return rec

    cycle_df = cycle_df.sort_values(by=["cycle"], ascending=[True]).reset_index(drop=True)
    rec["n_cycles_parsed"] = int(cycle_df.shape[0])

    first = cycle_df.iloc[0]
    last = cycle_df.iloc[-1]
    rec["CC_first"] = _safe_float(first["CC"], default=np.nan)
    rec["CC_last"] = _safe_float(last["CC"], default=np.nan)
    rec["R_first"] = _safe_float(first["R"], default=np.nan)
    rec["R_last"] = _safe_float(last["R"], default=np.nan)
    rec["FOM_last"] = _safe_float(last["FOM"], default=np.nan)
    if np.isfinite(_safe_float(last["Nref"], default=np.nan)):
        rec["n_reflections_last"] = _safe_float(last["Nref"], default=np.nan)

    tail_idx = _choose_tail_indices(
        n_rows=int(cycle_df.shape[0]),
        tail_frac=float(DEFAULTS["tail_frac"]),
        tail_min_cycles=int(DEFAULTS["tail_min_cycles"]),
    )
    tail_df = cycle_df.iloc[tail_idx].copy()

    rec["CC_tail_median"] = _median_or_nan(tail_df["CC"].values)
    rec["R_tail_median"] = _median_or_nan(tail_df["R"].values)
    rec["FOM_tail_median"] = _median_or_nan(tail_df["FOM"].values)

    rec["CC_tail_mad"] = _mad_or_nan(tail_df["CC"].values)
    rec["R_tail_mad"] = _mad_or_nan(tail_df["R"].values)
    rec["FOM_tail_mad"] = _mad_or_nan(tail_df["FOM"].values)

    rec["CC_tail_reversal_frac"] = _reversal_fraction(tail_df["CC"].values, mode="up")
    rec["R_tail_reversal_frac"] = _reversal_fraction(tail_df["R"].values, mode="down")
    rec["FOM_tail_reversal_frac"] = _reversal_fraction(tail_df["FOM"].values, mode="up")
    return rec


def _g_from_r_linear(r_val, r_good, r_bad):
    r = _safe_float(r_val, default=np.nan)
    rg = _safe_float(r_good, default=np.nan)
    rb = _safe_float(r_bad, default=np.nan)
    if not (np.isfinite(r) and np.isfinite(rg) and np.isfinite(rb)):
        return np.nan
    if rb <= rg:
        rb = rg + 0.10
    x = (float(r) - float(rg)) / (float(rb) - float(rg))
    return float(np.clip(1.0 - x, 0.0, 1.0))


def _g_from_r_logistic(r_val, r_good, r_bad, k):
    r = _safe_float(r_val, default=np.nan)
    rg = _safe_float(r_good, default=np.nan)
    rb = _safe_float(r_bad, default=np.nan)
    kk = _safe_float(k, default=np.nan)
    if not (np.isfinite(r) and np.isfinite(rg) and np.isfinite(rb) and np.isfinite(kk)):
        return np.nan
    if rb <= rg:
        rb = rg + 0.10
    if kk <= 0:
        kk = 5.0
    mu = 0.5 * (float(rg) + float(rb))
    s = (float(rb) - float(rg)) / float(kk)
    if s <= 1e-12:
        s = 1e-6
    z = (float(r) - float(mu)) / float(s)
    z = float(np.clip(z, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(z)))


def _g_from_r(r_val, r_good, r_bad, g_mode, g_k):
    gm = str(g_mode).strip().lower()
    if gm == "logistic":
        return _g_from_r_logistic(r_val=r_val, r_good=r_good, r_bad=r_bad, k=g_k)
    return _g_from_r_linear(r_val=r_val, r_good=r_good, r_bad=r_bad)


def _endpoint_score(cc_val, r_val, fom_val, fom_in_std, score_params):
    w_cc = float(score_params["w_cc"])
    w_r = float(score_params["w_r"])
    w_fom = float(score_params["w_fom"])
    fom_std_eps = float(score_params["fom_std_eps"])

    if np.isfinite(_safe_float(fom_in_std, default=np.nan)) and (_safe_float(fom_in_std, default=0.0) < fom_std_eps):
        w_fom_eff = 0.0
    else:
        w_fom_eff = w_fom

    g_r = _g_from_r(
        r_val=r_val,
        r_good=float(score_params["r_good"]),
        r_bad=float(score_params["r_bad"]),
        g_mode=str(score_params["g_mode"]),
        g_k=float(score_params["g_k"]),
    )

    return _weighted_mean_available(
        values=[cc_val, g_r, fom_val],
        weights=[w_cc, w_r, w_fom_eff],
    )


def _pull_score(cc_first, cc_tail, r_first, r_tail, fom_first, fom_tail, score_params):
    dcc = (float(cc_tail) - float(cc_first)) if (np.isfinite(_safe_float(cc_tail)) and np.isfinite(_safe_float(cc_first))) else np.nan
    dr = (float(r_first) - float(r_tail)) if (np.isfinite(_safe_float(r_tail)) and np.isfinite(_safe_float(r_first))) else np.nan
    dfom = (float(fom_tail) - float(fom_first)) if (np.isfinite(_safe_float(fom_tail)) and np.isfinite(_safe_float(fom_first))) else np.nan

    s_dcc = _clip01(dcc / float(score_params["dcc_scale"])) if np.isfinite(_safe_float(dcc)) else np.nan
    s_dr = _clip01(dr / float(score_params["dr_scale"])) if np.isfinite(_safe_float(dr)) else np.nan
    s_dfom = _clip01(dfom / float(score_params["dfom_scale"])) if np.isfinite(_safe_float(dfom)) else np.nan

    return _weighted_mean_available(
        values=[s_dcc, s_dr, s_dfom],
        weights=[float(score_params["w_dcc"]), float(score_params["w_dr"]), float(score_params["w_dfom"])],
    )


def _tail_stability_score(rec, score_params):
    cc_stab = 1.0 - _clip01(_safe_float(rec.get("CC_tail_mad"), default=np.nan) / float(score_params["cc_mad_scale"])) \
        if np.isfinite(_safe_float(rec.get("CC_tail_mad"), default=np.nan)) else np.nan
    r_stab = 1.0 - _clip01(_safe_float(rec.get("R_tail_mad"), default=np.nan) / float(score_params["r_mad_scale"])) \
        if np.isfinite(_safe_float(rec.get("R_tail_mad"), default=np.nan)) else np.nan
    fom_stab = 1.0 - _clip01(_safe_float(rec.get("FOM_tail_mad"), default=np.nan) / float(score_params["fom_mad_scale"])) \
        if np.isfinite(_safe_float(rec.get("FOM_tail_mad"), default=np.nan)) else np.nan

    cc_mon = 1.0 - _safe_float(rec.get("CC_tail_reversal_frac"), default=np.nan) \
        if np.isfinite(_safe_float(rec.get("CC_tail_reversal_frac"), default=np.nan)) else np.nan
    r_mon = 1.0 - _safe_float(rec.get("R_tail_reversal_frac"), default=np.nan) \
        if np.isfinite(_safe_float(rec.get("R_tail_reversal_frac"), default=np.nan)) else np.nan
    fom_mon = 1.0 - _safe_float(rec.get("FOM_tail_reversal_frac"), default=np.nan) \
        if np.isfinite(_safe_float(rec.get("FOM_tail_reversal_frac"), default=np.nan)) else np.nan

    phase_stab = np.nan
    flip_stab = np.nan
    phase_vol = _safe_float(rec.get("phase_volatility_tail_deg"), default=np.nan)
    flip_rate = _safe_float(rec.get("flip_rate_tail"), default=np.nan)

    if np.isfinite(phase_vol):
        phase_stab = 1.0 - _clip01(phase_vol / float(score_params["phase_vol_scale_deg"]))
    if np.isfinite(flip_rate):
        flip_stab = 1.0 - _clip01(flip_rate / float(score_params["flip_rate_scale"]))

    return _weighted_mean_available(
        values=[cc_stab, r_stab, fom_stab, cc_mon, r_mon, fom_mon, phase_stab, flip_stab],
        weights=[
            float(score_params["w_cc_stab"]),
            float(score_params["w_r_stab"]),
            float(score_params["w_fom_stab"]),
            float(score_params["w_cc_mon"]),
            float(score_params["w_r_mon"]),
            float(score_params["w_fom_mon"]),
            float(score_params["w_phase_stab"]),
            float(score_params["w_flip_stab"]),
        ],
    )


def _tail_score(rec, score_params):
    tail_endpoint = _endpoint_score(
        cc_val=_safe_float(rec.get("CC_tail_median", rec.get("CC_last", np.nan)), default=np.nan),
        r_val=_safe_float(rec.get("R_tail_median", rec.get("R_last", np.nan)), default=np.nan),
        fom_val=_safe_float(rec.get("FOM_tail_median", rec.get("FOM_last", np.nan)), default=np.nan),
        fom_in_std=_safe_float(rec.get("FOM_in_std", np.nan), default=np.nan),
        score_params=score_params,
    )
    stability = _tail_stability_score(rec=rec, score_params=score_params)
    return _weighted_mean_available(
        values=[tail_endpoint, stability],
        weights=[float(score_params["w_tail_endpoint"]), float(score_params["w_tail_stability"])],
    )


def _penalty_score(rec, score_params):
    cc_rev = _safe_float(rec.get("CC_tail_reversal_frac"), default=np.nan)
    r_rev = _safe_float(rec.get("R_tail_reversal_frac"), default=np.nan)
    fom_rev = _safe_float(rec.get("FOM_tail_reversal_frac"), default=np.nan)

    phase_pen = np.nan
    flip_pen = np.nan
    low_nref_pen = np.nan
    large_jump_pen = np.nan

    phase_vol = _safe_float(rec.get("phase_volatility_tail_deg"), default=np.nan)
    flip_rate = _safe_float(rec.get("flip_rate_tail"), default=np.nan)
    nref_last = _safe_float(rec.get("n_reflections_last"), default=np.nan)
    large_jump = _safe_float(rec.get("mean_abs_dphi_in_vs_dm_deg"), default=np.nan)

    if np.isfinite(phase_vol):
        phase_pen = _clip01(phase_vol / float(score_params["penalty_phase_vol_scale_deg"]))
    if np.isfinite(flip_rate):
        flip_pen = _clip01(flip_rate / float(score_params["penalty_flip_rate_scale"]))
    if np.isfinite(nref_last):
        soft_min = float(score_params["low_nref_soft_min"])
        if soft_min > 0:
            low_nref_pen = float(np.clip((soft_min - nref_last) / soft_min, 0.0, 1.0))
    if np.isfinite(large_jump):
        large_jump_pen = _clip01(large_jump / float(score_params["large_jump_scale_deg"]))

    return _weighted_mean_available(
        values=[cc_rev, r_rev, fom_rev, phase_pen, flip_pen, low_nref_pen, large_jump_pen],
        weights=[
            float(score_params["w_cc_reversal_pen"]),
            float(score_params["w_r_reversal_pen"]),
            float(score_params["w_fom_reversal_pen"]),
            float(score_params["w_phase_vol_pen"]),
            float(score_params["w_flip_rate_pen"]),
            float(score_params["w_low_nref_pen"]),
            float(score_params["w_large_jump_pen"]),
        ],
    )


def _compute_basin_score_hybrid(rec, score_params):
    endpoint = _endpoint_score(
        cc_val=_safe_float(rec.get("CC_last"), default=np.nan),
        r_val=_safe_float(rec.get("R_last"), default=np.nan),
        fom_val=_safe_float(rec.get("FOM_last"), default=np.nan),
        fom_in_std=_safe_float(rec.get("FOM_in_std"), default=np.nan),
        score_params=score_params,
    )
    pull = _pull_score(
        cc_first=_safe_float(rec.get("CC_first"), default=np.nan),
        cc_tail=_safe_float(rec.get("CC_tail_median", rec.get("CC_last", np.nan)), default=np.nan),
        r_first=_safe_float(rec.get("R_first"), default=np.nan),
        r_tail=_safe_float(rec.get("R_tail_median", rec.get("R_last", np.nan)), default=np.nan),
        fom_first=_safe_float(rec.get("FOM_first"), default=np.nan),
        fom_tail=_safe_float(rec.get("FOM_tail_median", rec.get("FOM_last", np.nan)), default=np.nan),
        score_params=score_params,
    )
    tail = _tail_score(rec=rec, score_params=score_params)
    penalty = _penalty_score(rec=rec, score_params=score_params)

    positive = _weighted_mean_available(
        values=[endpoint, pull, tail],
        weights=[
            float(score_params["w_outer_endpoint"]),
            float(score_params["w_outer_pull"]),
            float(score_params["w_outer_tail"]),
        ],
    )

    if np.isfinite(_safe_float(positive)):
        total01 = float(positive)
        if np.isfinite(_safe_float(penalty)):
            total01 = total01 - float(score_params["alpha_penalty"]) * float(penalty)
        total01 = float(np.clip(total01, 0.0, 1.0))
    else:
        total01 = np.nan

    out = dict(
        BasinScore_DM=(100.0 * total01) if np.isfinite(_safe_float(total01)) else np.nan,
        Basin_endpoint=endpoint,
        Basin_pull=pull,
        Basin_tail=tail,
        Basin_penalty=penalty,
        Basin_positive=positive,
    )
    out[str(DEFAULTS["deployed_score_name"])] = out["BasinScore_DM"]
    return out


def _get_solvent_content_from_csv(df, default_solvent_content=None):
    for c in ["SOLVENT_FRACTION", "solvent_content", "Solvent_content", "SOLV_CONT", "SC"]:
        if c in df.columns:
            vals = df[c].dropna().values
            if vals.size:
                sc = float(vals[0])
                if 0.0 <= sc <= 1.0:
                    return sc, c
    if default_solvent_content is not None:
        return float(default_solvent_content), "DEFAULT(%s)" % str(default_solvent_content)
    raise Sorry("No solvent content column found.")


def _get_spacegroup_from_csv(df):
    for c in ["space_group", "SPACE_GROUP", "sg", "SG", "SpaceGroup", "symmetry_space_group"]:
        if c in df.columns:
            vals = df[c].dropna().values
            if vals.size:
                return str(vals[0]).strip()
    return None


def _detect_freer_flag_column(df):
    for c in ["FreeR_flag", "FreeR", "FREER_FLAG", "Rfree_flag", "FREE_R_FLAG"]:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = str(c).lower()
        if "freer" in cl or "free_r" in cl or "rfree_flag" in cl:
            return c
    return None


def _enumerate_ops_sgtbx(space_group_info):
    sg = space_group_info.group()
    if hasattr(sg, "all_ops"):
        return list(sg.all_ops())
    return list(sg)


def _as_double3_tran(tr_vec):
    if hasattr(tr_vec, "as_double"):
        v = list(tr_vec.as_double())
        return (float(v[0]), float(v[1]), float(v[2]))
    try:
        return (float(tr_vec[0]), float(tr_vec[1]), float(tr_vec[2]))
    except Exception:
        return (0.0, 0.0, 0.0)


def _phi0_from_ht(hkl, tvec):
    ht = float(hkl[0]) * tvec[0] + float(hkl[1]) * tvec[1] + float(hkl[2]) * tvec[2]
    frac = ht - np.floor(ht)
    return (180.0 * frac) % 360.0


def _compute_phi0_for_centrics_sgtbx(df, space_group_info):
    cent = df["CENTRIC"].astype(int).values
    hkls = df[["H", "K", "L"]].astype(int).values
    n = int(df.shape[0])
    phi0 = np.zeros(n, dtype=float)

    ops = _enumerate_ops_sgtbx(space_group_info=space_group_info)
    recip_ops = []
    for rt in ops:
        try:
            RinvT = rt.r().inverse().transpose()
            tvec = _as_double3_tran(rt.t())
            recip_ops.append((RinvT, tvec))
        except Exception:
            continue

    n_none = 0
    n_multi = 0

    for i in range(n):
        if int(cent[i]) != 1:
            continue
        hkl = (int(hkls[i, 0]), int(hkls[i, 1]), int(hkls[i, 2]))
        neg = (-hkl[0], -hkl[1], -hkl[2])

        matches = []
        for RinvT, tvec in recip_ops:
            try:
                hp = RinvT * hkl
            except Exception:
                continue
            if (int(hp[0]), int(hp[1]), int(hp[2])) == neg:
                matches.append(tvec)

        if len(matches) == 0:
            n_none += 1
            phi0[i] = 0.0
        else:
            if len(matches) > 1:
                n_multi += 1
            phi0[i] = _phi0_from_ht(hkl=hkl, tvec=matches[0])

    return phi0, int(n_none), int(n_multi)


def _centric_allowed_pair(phi0_deg):
    a = _wrap360(phi0_deg)
    b = _wrap360(phi0_deg + 180.0)
    return (a, b)


def _is_phase_in_centric_support(phi_deg, phi0_deg, tol_deg=1e-6):
    a, b = _centric_allowed_pair(phi0_deg=phi0_deg)
    x = _wrap360(phi_deg)
    return (abs(x - a) <= tol_deg) or (abs(x - b) <= tol_deg)


def _validate_centric_supports_or_raise(df, phi0_all, ph_vec, tol_deg=1e-6, max_report=20):
    cent = df["CENTRIC"].astype(int).values
    idx = np.where(cent == 1)[0]
    bad = []
    for i in idx:
        if not _is_phase_in_centric_support(phi_deg=ph_vec[i], phi0_deg=phi0_all[i], tol_deg=tol_deg):
            bad.append(int(i))
            if len(bad) >= int(max_report):
                break
    if bad:
        msg = []
        msg.append("Centric support violation(s) detected (showing up to %d):" % int(max_report))
        for i in bad:
            a, b = _centric_allowed_pair(phi0_deg=phi0_all[i])
            msg.append(
                "  row=%d HKL=(%d,%d,%d) phi=%.3f allowed={%.3f,%.3f}"
                % (
                    int(i),
                    int(df.loc[i, "H"]),
                    int(df.loc[i, "K"]),
                    int(df.loc[i, "L"]),
                    float(_wrap360(ph_vec[i])),
                    float(a),
                    float(b),
                )
            )
        raise Sorry("\n".join(msg))


def _analyze_centric_supports(df, phi0_all, control_col=None, round_deg=1.0):
    cent = df["CENTRIC"].astype(int).values
    idx = np.where(cent == 1)[0]
    base = (phi0_all[idx] % 360.0)
    base_r = np.round(base / float(round_deg)) * float(round_deg)
    uniq, counts = np.unique(base_r, return_counts=True)
    order = np.argsort(-counts)

    log("Centric support analysis (sgtbx reversing ops):")
    log("  n_centric=%d unique_phi0=%d" % (int(idx.size), int(uniq.size)))
    for k in range(min(20, int(uniq.size))):
        u = float(uniq[order[k]])
        log("    phi0=%.1f deg support={%.1f, %.1f} count=%d wrapped=%+6.1f" % (
            u, u, (u + 180.0) % 360.0, int(counts[order[k]]), _wrap_pm180(u)
        ))

    if control_col is not None and control_col in df.columns:
        vals = (df.loc[idx, control_col].astype(float).values % 360.0)
        u2, c2 = np.unique(np.round(vals, 3), return_counts=True)
        order2 = np.argsort(-c2)
        log("  Control centric palette from %s:" % str(control_col))
        for k in range(min(10, int(u2.size))):
            vv = float(u2[order2[k]])
            log("    %7.3f deg n=%d wrapped=%+7.3f" % (vv, int(c2[order2[k]]), _wrap_pm180(vv)))


def _make_seed_phases(df, phi0_all, balance_tol, rng, centric_round_deg=1.0,
                      restrict_to_idx=None, centric_prior_assignments=None):
    cent = df["CENTRIC"].astype(int).values
    n = int(df.shape[0])
    ph = np.asarray(df[DEFAULTS["control_phib_col"]].astype(float).values % 360.0, dtype=float)

    if restrict_to_idx is None:
        restrict_to_idx = np.arange(n, dtype=int)
    restrict_to_idx = np.asarray(restrict_to_idx, dtype=int)

    idx_c = restrict_to_idx[np.where(cent[restrict_to_idx] == 1)[0]]
    idx_a = restrict_to_idx[np.where(cent[restrict_to_idx] == 0)[0]]

    prior_assigned = set()

    if centric_prior_assignments is not None and centric_prior_assignments.shape[0] > 0:
        grouped = centric_prior_assignments.groupby(["prior_class_id"], as_index=False)
        idx_c_set = set([int(i) for i in idx_c.tolist()])

        for _, sub in grouped:
            first = sub.iloc[0]
            members = sub["row_index"].astype(int).values
            members = np.asarray([i for i in members if int(i) in idx_c_set], dtype=int)
            if members.size < 1:
                continue

            m = int(members.size)
            p_base = float(first["p_base"])
            tol_local = float(first["balance_tol_local"])
            base_deg = float(first["base_deg"])
            shift_deg = float(first["shift_deg"])

            lo = int(np.floor(max(0.0, p_base - tol_local) * m))
            hi = int(np.ceil(min(1.0, p_base + tol_local) * m))

            accepted = False
            for _ in range(200):
                draw = (rng.random(size=m) < p_base).astype(int)
                n_base = int(np.sum(draw))
                if lo <= n_base <= hi:
                    accepted = True
                    break

            if not accepted:
                n_base_target = int(round(p_base * m))
                draw = np.zeros(m, dtype=int)
                draw[:n_base_target] = 1
                rng.shuffle(draw)

            ph[members] = np.where(draw == 1, base_deg, shift_deg).astype(float)
            for ii in members:
                prior_assigned.add(int(ii))

    if idx_c.size:
        remaining = np.asarray([i for i in idx_c if int(i) not in prior_assigned], dtype=int)
        if remaining.size:
            base = (phi0_all[remaining] % 360.0)
            base_r = np.round(base / float(centric_round_deg)) * float(centric_round_deg)
            for u in np.unique(base_r):
                members = remaining[np.where(base_r == u)[0]]
                m = int(members.size)
                lo = int(np.floor(0.5 * (1.0 - float(balance_tol)) * m))
                hi = int(np.ceil(0.5 * (1.0 + float(balance_tol)) * m))
                for _ in range(200):
                    coin = (rng.random(size=m) < 0.5).astype(int)
                    n_base = m - int(np.sum(coin))
                    if lo <= n_base <= hi:
                        ph[members] = (float(u) + 180.0 * coin.astype(float)) % 360.0
                        break
                else:
                    n_base_target = int(round(0.5 * m))
                    coin = np.ones(m, dtype=int)
                    coin[:n_base_target] = 0
                    rng.shuffle(coin)
                    ph[members] = (float(u) + 180.0 * coin.astype(float)) % 360.0

    if idx_a.size:
        m = int(idx_a.size)
        lo = int(np.floor(0.5 * (1.0 - float(balance_tol)) * m))
        hi = int(np.ceil(0.5 * (1.0 + float(balance_tol)) * m))
        for _ in range(500):
            coin = (rng.random(size=m) < 0.5).astype(int)
            n0 = m - int(np.sum(coin))
            if lo <= n0 <= hi:
                ph[idx_a] = (180.0 * coin.astype(float)) % 360.0
                break
        if idx_a.size > 0 and np.all(ph[idx_a] == ph[idx_a[0]]):
            n0_target = int(round(0.5 * m))
            coin = np.ones(m, dtype=int)
            coin[:n0_target] = 0
            rng.shuffle(coin)
            ph[idx_a] = (180.0 * coin.astype(float)) % 360.0

    _validate_centric_supports_or_raise(df=df, phi0_all=phi0_all, ph_vec=ph, tol_deg=1e-6, max_report=20)
    return ph


def _invert_hand_phases(ph_deg):
    ph = np.asarray(ph_deg, dtype=float)
    return (-ph) % 360.0


def _hl_coeff_arrays(hl_obj):
    try:
        A = np.asarray(list(hl_obj.a()), dtype=float)
        B = np.asarray(list(hl_obj.b()), dtype=float)
        C = np.asarray(list(hl_obj.c()), dtype=float)
        D = np.asarray(list(hl_obj.d()), dtype=float)
        return A, B, C, D
    except Exception:
        pass

    try:
        data = hl_obj.data()
        A = np.asarray(list(data.a()), dtype=float)
        B = np.asarray(list(data.b()), dtype=float)
        C = np.asarray(list(data.c()), dtype=float)
        D = np.asarray(list(data.d()), dtype=float)
        return A, B, C, D
    except Exception:
        pass

    try:
        n = int(hl_obj.size())
        A = np.zeros(n, dtype=float)
        B = np.zeros(n, dtype=float)
        C = np.zeros(n, dtype=float)
        D = np.zeros(n, dtype=float)
        for i in range(n):
            rec = hl_obj[i]
            A[i] = float(rec[0])
            B[i] = float(rec[1])
            C[i] = float(rec[2])
            D[i] = float(rec[3])
        return A, B, C, D
    except Exception:
        pass

    raise Sorry("Unable to extract HL coefficient arrays from hendrickson_lattman object in this Phenix build.")


def _try_run_dm(fp_sigfp, phib_ma, fom_ma, hl_ma, solvent_content, dm_params):
    result = run_density_modification_protocol(
        fp_sigfp=fp_sigfp,
        phib_ma=phib_ma,
        fom_ma=fom_ma,
        hl_ma=hl_ma,
        solvent_content=solvent_content,
        dm_params=dm_params,
        temp_dir=dm_params.get("temp_dir", None),
    )
    if not result.get("ok", False):
        return None, result.get("transcript", u""), result.get("error", "")
    return result.get("cmn"), result.get("transcript", u""), ""


def _run_dm_and_score_hybrid(fp_sigfp_score, phib_score, fom_score, hl_score,
                             solvent_content, dm_params,
                             fom_first_mean, fom_in_std,
                             score_params,
                             debug_print_dm_transcript=False,
                             debug_transcript_file=None):
    cmn, transcript, err = _try_run_dm(
        fp_sigfp=fp_sigfp_score,
        phib_ma=phib_score,
        fom_ma=fom_score,
        hl_ma=hl_score,
        solvent_content=solvent_content,
        dm_params=dm_params,
    )
    if cmn is None:
        return {"ok": False, "error": err}

    if debug_print_dm_transcript:
        log("----- DM TRANSCRIPT START -----")
        try:
            sys.stderr.write(_u_fs(transcript) + u"\n")
        except Exception:
            sys.stderr.write(str(transcript) + "\n")
        log("----- DM TRANSCRIPT END -----")
        sys.stderr.flush()

    if debug_transcript_file:
        try:
            with open(debug_transcript_file, "w") as fh:
                try:
                    fh.write(_u_fs(transcript).encode("utf-8"))
                except Exception:
                    fh.write(str(transcript))
        except Exception:
            pass

    cycle_df = parse_cycle_table_from_text(transcript_text=transcript)
    rec = _compute_cycle_features_from_df(
        cycle_df=cycle_df,
        fom_first_mean=fom_first_mean,
        fom_in_std=fom_in_std,
        nref_fallback=int(fp_sigfp_score.size()) if hasattr(fp_sigfp_score, "size") else np.nan,
    )

    global_seq = _parse_global_dm_sequences(transcript=transcript)
    if not np.isfinite(_safe_float(rec.get("CC_first"), default=np.nan)) and global_seq["CC"].size > 0:
        rec["CC_first"] = float(global_seq["CC"][0])
    if not np.isfinite(_safe_float(rec.get("CC_last"), default=np.nan)) and global_seq["CC"].size > 0:
        rec["CC_last"] = float(global_seq["CC"][-1])
    if not np.isfinite(_safe_float(rec.get("R_first"), default=np.nan)) and global_seq["R"].size > 0:
        rec["R_first"] = float(global_seq["R"][0])
    if not np.isfinite(_safe_float(rec.get("R_last"), default=np.nan)) and global_seq["R"].size > 0:
        rec["R_last"] = float(global_seq["R"][-1])
    if global_seq["Nref"].size > 0:
        rec["n_reflections_last"] = float(global_seq["Nref"][-1])
    if not np.isfinite(_safe_float(rec.get("FOM_first"), default=np.nan)) and global_seq["FOM"].size > 0:
        rec["FOM_first"] = float(global_seq["FOM"][0])
    if not np.isfinite(_safe_float(rec.get("FOM_last"), default=np.nan)) and global_seq["FOM"].size > 0:
        rec["FOM_last"] = float(global_seq["FOM"][-1])

    dm_mean_fom_from_array = np.nan
    try:
        fom_dm = cmn.fom_out_as_miller_array
        dm_mean_fom_from_array = float(np.nanmean(np.array(fom_dm.data(), dtype=float)))
        rec["DM_mean_FOM"] = dm_mean_fom_from_array
        if not np.isfinite(_safe_float(rec.get("FOM_last"), default=np.nan)):
            rec["FOM_last"] = dm_mean_fom_from_array
    except Exception:
        rec["DM_mean_FOM"] = np.nan

    txt_metrics = _extract_denmod_summary_metrics_from_text(transcript)
    rec.update(txt_metrics)
    if not np.isfinite(_safe_float(rec.get("DM_mean_FOM"), default=np.nan)):
        rec["DM_mean_FOM"] = _safe_float(dm_mean_fom_from_array, default=np.nan)
    if not np.isfinite(_safe_float(rec.get("DM_mean_FOM"), default=np.nan)):
        rec["DM_mean_FOM"] = _safe_float(txt_metrics.get("DM_mean_FOM"), default=np.nan)

    rec["phase_volatility_tail_deg"] = np.nan
    rec["flip_rate_tail"] = np.nan
    rec["mean_abs_dphi_in_vs_dm_deg"] = np.nan

    if not np.isfinite(_safe_float(rec.get("CC_last"), default=np.nan)):
        return {"ok": False, "error": "CC_last not parsed from DM transcript"}
    if not np.isfinite(_safe_float(rec.get("R_last"), default=np.nan)):
        return {"ok": False, "error": "R_last not parsed from DM transcript"}

    out = dict(rec)
    out["ok"] = True
    out["error"] = ""
    out["transcript"] = transcript
    return out


def _force_mtz_column_types(mtz_obj, ph_label_root="PHIB", fom_label_root="FOM", hl_label_root="HL", freer_label_root="FreeR"):
    try:
        cols = list(mtz_obj.columns())
    except Exception:
        cols = []
    for c in cols:
        try:
            lab = str(c.label())
        except Exception:
            continue
        lab_u = lab.upper()
        try:
            if lab_u.startswith(str(ph_label_root).upper()):
                c.set_type('P')
            elif lab_u.startswith(str(fom_label_root).upper()):
                c.set_type('W')
            elif lab_u.startswith(str(hl_label_root).upper()):
                c.set_type('A')
            elif lab_u.startswith(str(freer_label_root).upper()):
                c.set_type('I')
        except Exception:
            pass


def _write_best_mtz(out_mtz, fp_sigfp, phib_ma, fom_ma, hl_ma, freer_ma=None):
    if iotbx_mtz is None:
        raise Sorry("iotbx.mtz not available.")
    mtz_ds = fp_sigfp.as_mtz_dataset(column_root_label="F")
    mtz_ds.add_miller_array(miller_array=phib_ma, column_root_label="PHIB")
    mtz_ds.add_miller_array(miller_array=fom_ma, column_root_label="FOM")
    mtz_ds.add_miller_array(miller_array=hl_ma, column_root_label="HL")
    if freer_ma is not None:
        mtz_ds.add_miller_array(miller_array=freer_ma, column_root_label="FreeR")
    mtz_obj = mtz_ds.mtz_object()
    _force_mtz_column_types(mtz_obj=mtz_obj)
    mtz_obj.write(file_name=str(out_mtz))


def _map_minus180_to_plus180_deg(arr):
    x = np.asarray(arr, dtype=float)
    x = ((x + 180.0) % 360.0) - 180.0
    x = np.where(np.isclose(x, -180.0, atol=1e-9), 180.0, x)
    return x



def _build_clean_python3_env(python3_exe):
    env = dict(os.environ)

    for key in ["PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "__PYVENV_LAUNCHER__"]:
        if key in env:
            del env[key]

    py_dir = os.path.dirname(os.path.abspath(os.path.expanduser(str(python3_exe))))
    old_path = env.get("PATH", "")
    env["PATH"] = py_dir + os.pathsep + old_path if py_dir else old_path

    for key in list(env.keys()):
        if key.startswith("LIBTBX_") or key.startswith("PHENIX_") or key.startswith("CCTBX_"):
            del env[key]

    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    return env



def _api50_mtz_writer_script_path(shared):
    helper_script = shared.get("mtz_writer_helper", None)
    if helper_script is None or str(helper_script).strip() == "" or str(helper_script).strip().lower() == "none":
        helper_script = shared.get("api50_mtz_writer_script", None)
    if helper_script is None or str(helper_script).strip() == "" or str(helper_script).strip().lower() == "none":
        helper_script = shared.get("mtz_helper_script", None)
    if helper_script is None or str(helper_script).strip() == "" or str(helper_script).strip().lower() == "none":
        helper_script = os.path.join(_THIS_DIR, DEFAULT_API50_MTZ_WRITER)
    return os.path.abspath(os.path.expanduser(str(helper_script)))


def _kappa_from_R_array_api50(R):
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


def _write_best_mtz_autobuild_internal(out_mtz, shared, ph_full, fom_full):
    try:
        import gemmi
        import reciprocalspaceship as rs
    except Exception as exc:
        raise Sorry("Internal Autobuild MTZ export failed in this Phenix build: %s" % str(exc))

    df_full = shared.get("df_full", None)
    if df_full is None:
        raise Sorry("shared['df_full'] not available for Autobuild MTZ export.")

    df_use = df_full.copy()
    required_cols = [
        "H", "K", "L", "FP", "SIGFP",
        "SG", "LENGTH_A", "LENGTH_B", "LENGTH_C",
        "ANGLE_ALPHA", "ANGLE_BETA", "ANGLE_GAMMA",
    ]
    missing = [c for c in required_cols if c not in df_use.columns]
    if missing:
        raise Sorry("Missing required columns for Autobuild MTZ export: %s" % str(missing))

    if "FreeR_flag" not in df_use.columns:
        df_use["FreeR_flag"] = 0

    PHIB = _map_minus180_to_plus180_deg(np.asarray(ph_full, dtype=float))
    FOM = np.clip(np.asarray(fom_full, dtype=float), 0.0, 0.9999)

    if "dHKL" in df_use.columns:
        d_hkl_payload = pd.to_numeric(df_use["dHKL"], errors="coerce").to_numpy(dtype=float)
        high_res_mask = np.isfinite(d_hkl_payload) & (d_hkl_payload < float(DEFAULTS["dm_score_dmin"]))
        PHIB = PHIB.astype(object)
        FOM = FOM.astype(object)
        PHIB[high_res_mask] = np.nan
        FOM[high_res_mask] = np.nan

    PHIB_num = pd.to_numeric(pd.Series(PHIB), errors="coerce").to_numpy(dtype=float)
    FOM_num = pd.to_numeric(pd.Series(FOM), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    kappa = np.minimum(_kappa_from_R_array_api50(FOM_num), float(shared["kappa_max"]))
    phi_rad = np.deg2rad(PHIB_num)
    HLA = kappa * np.cos(phi_rad)
    HLB = kappa * np.sin(phi_rad)
    HLC = np.zeros_like(HLA)
    HLD = np.zeros_like(HLA)

    ds = rs.DataSet()
    ds["H"] = rs.DataSeries(pd.to_numeric(df_use["H"], errors="coerce").fillna(0).astype(int).to_numpy(), dtype="H")
    ds["K"] = rs.DataSeries(pd.to_numeric(df_use["K"], errors="coerce").fillna(0).astype(int).to_numpy(), dtype="H")
    ds["L"] = rs.DataSeries(pd.to_numeric(df_use["L"], errors="coerce").fillna(0).astype(int).to_numpy(), dtype="H")
    ds["FP"] = rs.DataSeries(pd.to_numeric(df_use["FP"], errors="coerce").astype(float).to_numpy(), dtype="F")
    ds["SIGFP"] = rs.DataSeries(pd.to_numeric(df_use["SIGFP"], errors="coerce").astype(float).to_numpy(), dtype="Q")
    ds["PHIB"] = rs.DataSeries(PHIB_num, dtype="P")
    ds["FOM"] = rs.DataSeries(FOM_num, dtype="W")
    ds["FreeR_flag"] = rs.DataSeries(pd.to_numeric(df_use["FreeR_flag"], errors="coerce").fillna(0).astype(int).to_numpy(), dtype="I")
    ds["HLA"] = rs.DataSeries(HLA, dtype="A")
    ds["HLB"] = rs.DataSeries(HLB, dtype="A")
    ds["HLC"] = rs.DataSeries(HLC, dtype="A")
    ds["HLD"] = rs.DataSeries(HLD, dtype="A")
    ds.set_index(["H", "K", "L"], inplace=True)

    ds.spacegroup = gemmi.SpaceGroup(str(df_use["SG"].iloc[0]))
    ds.cell = gemmi.UnitCell(
        float(df_use["LENGTH_A"].iloc[0]),
        float(df_use["LENGTH_B"].iloc[0]),
        float(df_use["LENGTH_C"].iloc[0]),
        float(df_use["ANGLE_ALPHA"].iloc[0]),
        float(df_use["ANGLE_BETA"].iloc[0]),
        float(df_use["ANGLE_GAMMA"].iloc[0]),
    )

    out_mtz_abs = os.path.abspath(out_mtz)
    out_dir = os.path.dirname(out_mtz_abs)
    _mkdir_p(out_dir)
    mtz = ds.to_gemmi()
    mtz.write_to_file(str(out_mtz_abs))
    if not os.path.isfile(out_mtz_abs):
        raise Sorry("Internal Autobuild MTZ export did not produce output file: %s" % str(out_mtz_abs))
    return out_mtz_abs


def _write_best_mtz_autobuild(out_mtz, shared, ph_full, fom_full):
    """
    Faithful PC50 behavior: always write Autobuild-ready MTZs through the
    external Python 3 helper in a cleaned environment, rather than attempting
    internal MTZ export under phenix.python first.
    """
    return _write_best_mtz_autobuild_external(out_mtz=out_mtz, shared=shared, ph_full=ph_full, fom_full=fom_full)


def _write_best_mtz_autobuild_external(out_mtz, shared, ph_full, fom_full):
    df_full = shared.get("df_full", None)
    if df_full is None:
        raise Sorry("shared['df_full'] not available for Autobuild MTZ export.")

    helper_script = _api50_mtz_writer_script_path(shared)
    python3_exe = shared.get("python3_exe", None)
    if not python3_exe:
        raise Sorry("No python3_exe configured for Autobuild MTZ export fallback.")

    df_use = df_full.copy()

    required_cols = [
        "H", "K", "L", "FP", "SIGFP",
        "SG", "LENGTH_A", "LENGTH_B", "LENGTH_C",
        "ANGLE_ALPHA", "ANGLE_BETA", "ANGLE_GAMMA",
    ]
    missing = [c for c in required_cols if c not in df_use.columns]
    if missing:
        raise Sorry("Missing required columns for Autobuild MTZ export: %s" % str(missing))

    if "FreeR_flag" not in df_use.columns:
        df_use["FreeR_flag"] = 0

    out_mtz_abs = os.path.abspath(out_mtz)
    out_dir = os.path.dirname(out_mtz_abs)
    _mkdir_p(out_dir)

    payload_csv = os.path.join(
        out_dir,
        os.path.basename(out_mtz_abs).replace(".mtz", "__payload.csv")
    )
    helper_log = os.path.join(
        out_dir,
        os.path.basename(out_mtz_abs).replace(".mtz", "__mtz_helper.log")
    )

    phib_payload = _map_minus180_to_plus180_deg(np.asarray(ph_full, dtype=float))
    fom_payload = np.clip(np.asarray(fom_full, dtype=float), 0.0, 0.9999)
    d_hkl_payload = pd.to_numeric(df_use["dHKL"], errors="coerce").to_numpy(dtype=float) if "dHKL" in df_use.columns else np.full(shape=len(df_use), fill_value=np.nan, dtype=float)
    high_res_mask = np.isfinite(d_hkl_payload) & (d_hkl_payload < float(DEFAULTS["dm_score_dmin"]))
    phib_payload = phib_payload.astype(object)
    fom_payload = fom_payload.astype(object)
    phib_payload[high_res_mask] = ""
    fom_payload[high_res_mask] = ""

    df_payload = pd.DataFrame({
        "H": pd.to_numeric(df_use["H"], errors="coerce").fillna(0).astype(int),
        "K": pd.to_numeric(df_use["K"], errors="coerce").fillna(0).astype(int),
        "L": pd.to_numeric(df_use["L"], errors="coerce").fillna(0).astype(int),
        "FP": pd.to_numeric(df_use["FP"], errors="coerce").astype(float),
        "SIGFP": pd.to_numeric(df_use["SIGFP"], errors="coerce").astype(float),
        "PHIB": phib_payload,
        "FOM": fom_payload,
        "FreeR_flag": pd.to_numeric(df_use["FreeR_flag"], errors="coerce").fillna(0).astype(int),
        "SG": df_use["SG"].astype(str),
        "LENGTH_A": pd.to_numeric(df_use["LENGTH_A"], errors="coerce").astype(float),
        "LENGTH_B": pd.to_numeric(df_use["LENGTH_B"], errors="coerce").astype(float),
        "LENGTH_C": pd.to_numeric(df_use["LENGTH_C"], errors="coerce").astype(float),
        "ANGLE_ALPHA": pd.to_numeric(df_use["ANGLE_ALPHA"], errors="coerce").astype(float),
        "ANGLE_BETA": pd.to_numeric(df_use["ANGLE_BETA"], errors="coerce").astype(float),
        "ANGLE_GAMMA": pd.to_numeric(df_use["ANGLE_GAMMA"], errors="coerce").astype(float),
    })
    df_payload.to_csv(payload_csv, index=False)

    cmd = [
        str(python3_exe),
        str(helper_script),
        "--csv_in", str(payload_csv),
        "--mtz_out", str(out_mtz_abs),
        "--kappa_max", str(float(shared["kappa_max"])),
    ]
    log("API50 MTZ helper command: %s" % " ".join([str(x) for x in cmd]))
    clean_env = _build_clean_python3_env(python3_exe=python3_exe)
    proc = subprocess.Popen(
        args=cmd,
        env=clean_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    out, _ = proc.communicate()
    with open(helper_log, "w") as fh:
        fh.write(out if out is not None else "")

    if proc.returncode != 0:
        raise Sorry("MTZ helper failed rc=%d; see %s" % (int(proc.returncode), str(helper_log)))
    if not os.path.isfile(out_mtz_abs):
        raise Sorry("MTZ helper completed but output file not found: %s" % str(out_mtz_abs))


def _fraction_needs_mtz_backfill(manifest_csv, mode, frac):
    frac_key = round(float(frac), 12)
    if not os.path.isfile(manifest_csv):
        return True
    try:
        dfm = pd.read_csv(manifest_csv)
    except Exception:
        return True
    if dfm.shape[0] == 0:
        return True
    if "mode" not in dfm.columns or "degrade_fraction" not in dfm.columns:
        return True
    sub = dfm.copy()
    sub["degrade_fraction"] = pd.to_numeric(sub["degrade_fraction"], errors="coerce")
    sub = sub.loc[
        (sub["mode"].astype(str) == str(mode)) &
        (np.round(sub["degrade_fraction"].astype(float), 12) == frac_key)
    ].copy()
    if sub.shape[0] == 0:
        return True
    if "mtz_autobuild" not in sub.columns:
        return True
    mtz_nonempty = sub["mtz_autobuild"].fillna("").astype(str).str.strip()
    return bool((mtz_nonempty == "").any())



def _build_full_state_mtz_arrays(shared, ph_full, fom_full):
    ph_full = np.asarray(ph_full, dtype=float) % 360.0
    fom_full = np.asarray(fom_full, dtype=float)

    phib_full_ma = shared["phib0_full"].customized_copy(
        data=_to_flex_double(ph_full)
    )
    fom_full_ma = shared["fom_ma_obs_full"].customized_copy(
        data=_to_flex_double(fom_full)
    )
    hl_full_ma = shared["hl0_full"].customized_copy(
        data=_hl_from_phib_fom(
            phib_deg=ph_full,
            fom_vals=fom_full,
            kappa_max=float(shared["kappa_max"]),
        )
    )
    freer_ma = shared.get("freer_flag_ma_full", None)
    return phib_full_ma, fom_full_ma, hl_full_ma, freer_ma


def _seed_failure_record(seed_idx, hand, error_msg, n_reflections_score_window, seed_wall_sec=np.nan):
    return dict(
        seed_idx=int(seed_idx),
        hand=str(hand),
        ok=False,
        score=0.0,
        BasinScore=0.0,
        L_20_3p5=0.0,
        BasinScore_DM=np.nan,
        DM_mean_FOM=np.nan,
        R_factor_FC_vs_FP=np.nan,
        CC_prob_map_with_current_map=np.nan,
        BIAS_RATIO=np.nan,
        best_largest_component_fraction_at_target=np.nan,
        best_n_connected_components_at_target=np.nan,
        best_endpoint_fraction_at_target=np.nan,
        best_mean_degree_at_target=np.nan,
        best_largest_component_fraction_auc=np.nan,
        basin_term_r=np.nan,
        basin_term_fom=np.nan,
        basin_term_lcf=np.nan,
        basin_term_endpoint=np.nan,
        cc_last=np.nan,
        r_last=np.nan,
        fom_last=np.nan,
        endpoint=np.nan,
        pull=np.nan,
        tail=np.nan,
        n_reflections_score_window=int(n_reflections_score_window),
        error=str(error_msg),
        seed_wall_sec=_safe_float(seed_wall_sec, default=np.nan),
    )


def _seed_rejection_reasons(row):
    reasons = []

    skeleton_ok = bool(row.get("skeleton_ok", False))
    lcf_auc = _safe_float(row.get("best_largest_component_fraction_auc", np.nan), default=np.nan)
    endpoint_target = _safe_float(row.get("best_endpoint_fraction_at_target", np.nan), default=np.nan)
    r_factor = _safe_float(row.get("R_factor_FC_vs_FP", np.nan), default=np.nan)
    dm_mean_fom = _safe_float(row.get("DM_mean_FOM", np.nan), default=np.nan)

    if not skeleton_ok:
        reasons.append("skeleton_ok == False")
    if np.isfinite(r_factor) and (float(r_factor) > float(CURRENT_THR_R_FACTOR)):
        reasons.append("R_factor > %.6f" % float(CURRENT_THR_R_FACTOR))
    if np.isfinite(dm_mean_fom) and (float(dm_mean_fom) < float(CURRENT_THR_DM_MEAN_FOM)):
        reasons.append("DM_mean_FOM < %.6f" % float(CURRENT_THR_DM_MEAN_FOM))
    if np.isfinite(lcf_auc) and (float(lcf_auc) < float(CURRENT_THR_LCF_AUC)):
        reasons.append("LCF_AUC < %.6f" % float(CURRENT_THR_LCF_AUC))
    if np.isfinite(endpoint_target) and (float(endpoint_target) > float(CURRENT_THR_ENDPOINT_FRAC)):
        reasons.append("Endpoint@target > %.6f" % float(CURRENT_THR_ENDPOINT_FRAC))

    return reasons


def _normalize_seed_result(rec, n_reflections_score_window):
    if rec is None:
        return _seed_failure_record(-1, "normal", "Missing worker result", n_reflections_score_window)

    out = dict(rec)
    out["seed_idx"] = int(out.get("seed_idx", -1))
    out["hand"] = str(out.get("hand", "normal"))
    out["n_reflections_score_window"] = int(out.get("n_reflections_score_window", n_reflections_score_window))
    out["seed_wall_sec"] = _safe_float(out.get("seed_wall_sec", np.nan), default=np.nan)

    for key in [
        "score",
        "BasinScore",
        "L_20_3p5",
        "BasinScore_DM",
        "DM_mean_FOM",
        "R_factor_FC_vs_FP",
        "CC_prob_map_with_current_map",
        "BIAS_RATIO",
        "best_largest_component_fraction_at_target",
        "best_n_connected_components_at_target",
        "best_endpoint_fraction_at_target",
        "best_mean_degree_at_target",
        "best_largest_component_fraction_auc",
        "basin_term_r",
        "basin_term_fom",
        "basin_term_lcf",
        "basin_term_endpoint",
        "cc_last",
        "r_last",
        "fom_last",
        "endpoint",
        "pull",
        "tail",
    ]:
        out[key] = _safe_float(out.get(key, np.nan), default=np.nan)

    if np.isfinite(out["L_20_3p5"]):
        out["score"] = float(out["L_20_3p5"])
    elif np.isfinite(out["score"]):
        out["L_20_3p5"] = float(out["score"])
    elif np.isfinite(out["BasinScore"]):
        out["score"] = float(out["BasinScore"])
        out["L_20_3p5"] = float(out["BasinScore"])

    if not np.isfinite(out["BasinScore"]) and np.isfinite(out["score"]):
        out["BasinScore"] = float(out["score"])
    if not np.isfinite(out["BasinScore_DM"]) and np.isfinite(out["score"]):
        out["BasinScore_DM"] = float(out["score"])

    ok = bool(out.get("ok", False)) and np.isfinite(_safe_float(out.get("L_20_3p5", np.nan)))
    out["ok"] = bool(ok)
    out["error"] = str(out.get("error", ""))
    return out


def _log_best_row(best_row, state=None):
    if best_row is None:
        return
    if state is None:
        state = {}
    key = (int(best_row["seed_idx"]), str(best_row["hand"]), float(best_row["score"]))
    if state.get("last_best_key", None) == key:
        return
    state["last_best_key"] = key
    log("Seeds best-so-far | seed_idx=%d | hand=%s | BasinScore=%.3f | R_factor=%.4f (<= %.4f) | DM_mean_FOM=%.4f (>= %.4f) | LCF_AUC=%.4f (>= %.4f) | Endpoint@target=%.4f (<= %.4f) | n_comp=%s | mean_degree=%.4f | skeleton_ok=%s" % (
        int(best_row["seed_idx"]), str(best_row["hand"]), float(best_row["score"]),
        float(best_row.get("R_factor_FC_vs_FP", np.nan)), float(CURRENT_THR_R_FACTOR),
        float(best_row.get("DM_mean_FOM", np.nan)), float(CURRENT_THR_DM_MEAN_FOM),
        float(best_row.get("best_largest_component_fraction_auc", np.nan)), float(CURRENT_THR_LCF_AUC),
        float(best_row.get("best_endpoint_fraction_at_target", np.nan)), float(CURRENT_THR_ENDPOINT_FRAC),
        str(best_row.get("best_n_connected_components_at_target", "NA")),
        float(best_row.get("best_mean_degree_at_target", np.nan)),
        str(bool(best_row.get("skeleton_ok", False))),
    ))


def _score_state_top(ph_full, temp_tag):
    global _WORKER_SHARED
    S = _WORKER_SHARED

    idx_score = S["idx_score"]
    ph_s = ph_full[idx_score]

    phib_s = S["phib0_score"].customized_copy(data=_to_flex_double(ph_s))
    hl_s = S["hl0_score"].customized_copy(
        data=_hl_from_phib_fom(ph_s, S["fom_vals_prior_score"], float(S["kappa_max"]))
    )

    dm_params = dict(S["dm_params_base"])
    dm_params["temp_dir"] = os.path.join(S["seed_temp_root"], str(temp_tag), "dm")
    _mkdir_p(dm_params["temp_dir"])

    dbg_file = None
    if S["debug_transcript_file"]:
        base, ext = os.path.splitext(S["debug_transcript_file"])
        dbg_file = "%s.%s%s" % (base, str(temp_tag), ext if ext else ".log")

    res = _run_dm_and_score_hybrid(
        fp_sigfp_score=S["fp_sigfp_score"],
        phib_score=phib_s,
        fom_score=S["fom_ma_seed_score"],
        hl_score=hl_s,
        solvent_content=float(S["sol"]),
        dm_params=dm_params,
        fom_first_mean=float(S["fom_first_mean_score"]),
        fom_in_std=float(S["fom_in_std_score"]),
        score_params=S["score_params"],
        debug_print_dm_transcript=bool(S["debug_print_dm_transcript"]),
        debug_transcript_file=dbg_file,
    )

    if not bool(S["keep_temp"]):
        _rm_tree(dm_params["temp_dir"])

    if not bool(res.get("ok", False)):
        return res

    skel_temp_root = os.path.join(S["seed_temp_root"], str(temp_tag), "skel")
    _mkdir_p(skel_temp_root)
    skel = _run_skeleton_proxy_for_seed(
        df_base=S["df"],
        ph_full=ph_full,
        fom_full=S["fom_seed_full_template"],
        temp_root=skel_temp_root,
        score_params=S,
    )
    if not bool(S["keep_temp"]):
        _rm_tree(skel_temp_root)

    skel = _normalize_skeleton_metrics_for_seed(skel)
    if bool(skel.get("skeleton_ok", False)) and ("skeleton_error" not in skel):
        skel["skeleton_error"] = ""

    basin_bits = _compute_current_basin_score(dm_metrics=res, skel_metrics=skel)

    out = dict(res)
    out.update(skel)
    out.update(basin_bits)
    out["score"] = float(out.get("BasinScore", 0.0))
    out["L_20_3p5"] = float(out.get("BasinScore", 0.0))
    out["ok"] = bool(np.isfinite(_safe_float(out.get("BasinScore", np.nan))))
    return out


def _seed_worker_top(seed_idx):
    global _WORKER_SHARED
    S = _WORKER_SHARED

    out_rows = []
    best_local = None

    try:
        rng = RNG(_seed_from_master(S["seed_master"], seed_idx, "normal"))
        ph_full_normal = _make_seed_phases(
            df=S["df"],
            phi0_all=S["phi0_all"],
            balance_tol=float(S["balance_tol"]),
            rng=rng,
            centric_round_deg=float(S["centric_round_deg"]),
            restrict_to_idx=S["idx_score"],
            centric_prior_assignments=S.get("centric_prior_assignments_score", pd.DataFrame()),
        )

        t0 = time.time()
        res_normal = _score_state_top(ph_full=ph_full_normal, temp_tag="seed_%06d__normal" % int(seed_idx))
        normal_rejection_reasons = _seed_rejection_reasons(res_normal)
        normal_ok = bool(res_normal.get("ok", False)) and np.isfinite(_safe_float(res_normal.get("BasinScore", np.nan))) and (len(normal_rejection_reasons) == 0)
        normal_error_bits = []
        if not bool(res_normal.get("ok", False)):
            normal_error_bits.append(str(res_normal.get("error", "")))
        if len(normal_rejection_reasons) > 0:
            normal_error_bits.append("REJECTED: " + "; ".join(normal_rejection_reasons))
        if str(res_normal.get("skeleton_error", "")):
            normal_error_bits.append(str(res_normal.get("skeleton_error", "")))

        row_normal = dict(
            seed_idx=int(seed_idx),
            hand="normal",
            ok=normal_ok,
            score=float(res_normal.get("BasinScore", np.nan)),
            L_20_3p5=float(res_normal.get("BasinScore", np.nan)),
            BasinScore=float(res_normal.get("BasinScore", np.nan)),
            basin_term_r=float(res_normal.get("basin_term_r", np.nan)),
            basin_term_fom=float(res_normal.get("basin_term_fom", np.nan)),
            basin_term_lcf=float(res_normal.get("basin_term_lcf", np.nan)),
            basin_term_endpoint=float(res_normal.get("basin_term_endpoint", np.nan)),
            Reflections=float(res_normal.get("Reflections", np.nan)),
            DM_mean_FOM=float(res_normal.get("DM_mean_FOM", np.nan)),
            CC_prob_map_with_current_map=float(res_normal.get("CC_prob_map_with_current_map", np.nan)),
            R_factor_FC_vs_FP=float(res_normal.get("R_factor_FC_vs_FP", np.nan)),
            BIAS_RATIO=float(res_normal.get("BIAS_RATIO", np.nan)),
            best_largest_component_fraction_at_target=float(res_normal.get("best_largest_component_fraction_at_target", np.nan)),
            best_n_connected_components_at_target=float(res_normal.get("best_n_connected_components_at_target", np.nan)),
            best_endpoint_fraction_at_target=float(res_normal.get("best_endpoint_fraction_at_target", np.nan)),
            best_mean_degree_at_target=float(res_normal.get("best_mean_degree_at_target", np.nan)),
            best_largest_component_fraction_auc=float(res_normal.get("best_largest_component_fraction_auc", np.nan)),
            skeleton_ok=bool(res_normal.get("skeleton_ok", False)),
            skeleton_error=str(res_normal.get("skeleton_error", "")),
            n_reflections_score_window=int(S["n_reflections_score_window"]),
            error=" | ".join([x for x in normal_error_bits if str(x).strip() != ""]),
            seed_wall_sec=float(time.time() - t0),
        )
        row_normal = _normalize_seed_result(row_normal, int(S["n_reflections_score_window"]))
        out_rows.append(row_normal)

        if row_normal.get("ok", False):
            best_local = dict(row_normal)
            best_local["ph_full"] = ph_full_normal

        if bool(S["evaluate_inverted_hand"]):
            ph_full_inv = _invert_hand_phases(ph_full_normal)
            t1 = time.time()
            res_inv = _score_state_top(ph_full=ph_full_inv, temp_tag="seed_%06d__inverted" % int(seed_idx))
            inv_rejection_reasons = _seed_rejection_reasons(res_inv)
            inv_ok = bool(res_inv.get("ok", False)) and np.isfinite(_safe_float(res_inv.get("BasinScore", np.nan))) and (len(inv_rejection_reasons) == 0)
            inv_error_bits = []
            if not bool(res_inv.get("ok", False)):
                inv_error_bits.append(str(res_inv.get("error", "")))
            if len(inv_rejection_reasons) > 0:
                inv_error_bits.append("REJECTED: " + "; ".join(inv_rejection_reasons))
            if str(res_inv.get("skeleton_error", "")):
                inv_error_bits.append(str(res_inv.get("skeleton_error", "")))

            row_inv = dict(
                seed_idx=int(seed_idx),
                hand="inverted",
                ok=inv_ok,
                score=float(res_inv.get("BasinScore", np.nan)),
                L_20_3p5=float(res_inv.get("BasinScore", np.nan)),
                BasinScore=float(res_inv.get("BasinScore", np.nan)),
                basin_term_r=float(res_inv.get("basin_term_r", np.nan)),
                basin_term_fom=float(res_inv.get("basin_term_fom", np.nan)),
                basin_term_lcf=float(res_inv.get("basin_term_lcf", np.nan)),
                basin_term_endpoint=float(res_inv.get("basin_term_endpoint", np.nan)),
                Reflections=float(res_inv.get("Reflections", np.nan)),
                DM_mean_FOM=float(res_inv.get("DM_mean_FOM", np.nan)),
                CC_prob_map_with_current_map=float(res_inv.get("CC_prob_map_with_current_map", np.nan)),
                R_factor_FC_vs_FP=float(res_inv.get("R_factor_FC_vs_FP", np.nan)),
                BIAS_RATIO=float(res_inv.get("BIAS_RATIO", np.nan)),
                best_largest_component_fraction_at_target=float(res_inv.get("best_largest_component_fraction_at_target", np.nan)),
                best_n_connected_components_at_target=float(res_inv.get("best_n_connected_components_at_target", np.nan)),
                best_endpoint_fraction_at_target=float(res_inv.get("best_endpoint_fraction_at_target", np.nan)),
                best_mean_degree_at_target=float(res_inv.get("best_mean_degree_at_target", np.nan)),
                best_largest_component_fraction_auc=float(res_inv.get("best_largest_component_fraction_auc", np.nan)),
                skeleton_ok=bool(res_inv.get("skeleton_ok", False)),
                skeleton_error=str(res_inv.get("skeleton_error", "")),
                n_reflections_score_window=int(S["n_reflections_score_window"]),
                error=" | ".join([x for x in inv_error_bits if str(x).strip() != ""]),
                seed_wall_sec=float(time.time() - t1),
            )
            row_inv = _normalize_seed_result(row_inv, int(S["n_reflections_score_window"]))
            out_rows.append(row_inv)

            if row_inv.get("ok", False):
                if (best_local is None) or (float(row_inv["score"]) > float(best_local["score"])):
                    best_local = dict(row_inv)
                    best_local["ph_full"] = ph_full_inv

        return dict(
            seed_idx=int(seed_idx),
            rows=out_rows,
            best_local=best_local,
            ok=True,
            error="",
        )

    except Exception as e:
        return dict(
            seed_idx=int(seed_idx),
            rows=[_seed_failure_record(
                seed_idx=seed_idx,
                hand="normal",
                error_msg="%s: %s" % (e.__class__.__name__, str(e)),
                n_reflections_score_window=int(S["n_reflections_score_window"]),
                seed_wall_sec=np.nan,
            )],
            best_local=None,
            ok=False,
            error="%s: %s" % (e.__class__.__name__, str(e)),
        )


def _seed_entrypoint(seed_idx, shared, queue_out):
    try:
        _init_worker(shared)
        rec = _seed_worker_top(seed_idx)
        queue_out.put(rec)
    except Exception as e:
        try:
            queue_out.put(dict(
                seed_idx=int(seed_idx),
                rows=[_seed_failure_record(
                    seed_idx=seed_idx,
                    hand="normal",
                    error_msg="ENTRYPOINT %s: %s" % (e.__class__.__name__, str(e)),
                    n_reflections_score_window=int(shared["n_reflections_score_window"]),
                    seed_wall_sec=np.nan,
                )],
                best_local=None,
                ok=False,
                error="ENTRYPOINT %s: %s" % (e.__class__.__name__, str(e)),
            ))
        except Exception:
            pass


def _drain_result_queue(queue_out):
    out = []
    while True:
        try:
            out.append(queue_out.get_nowait())
        except Exception:
            break
    return out


def _open_stream_csv(csv_path):
    exists = os.path.isfile(csv_path)
    fh = open(csv_path, "ab")
    writer = csv.writer(fh)
    if not exists:
        writer.writerow([
            "seed_idx",
            "hand",
            "ok",
            "score",
            "BasinScore",
            "L_20_3p5",
            "BasinScore_DM",
            "DM_mean_FOM",
            "R_factor_FC_vs_FP",
            "CC_prob_map_with_current_map",
            "BIAS_RATIO",
            "best_largest_component_fraction_at_target",
            "best_n_connected_components_at_target",
            "best_endpoint_fraction_at_target",
            "best_mean_degree_at_target",
            "best_largest_component_fraction_auc",
            "basin_term_r",
            "basin_term_fom",
            "basin_term_lcf",
            "basin_term_endpoint",
            "cc_last",
            "r_last",
            "fom_last",
            "endpoint",
            "pull",
            "tail",
            "n_reflections_score_window",
            "error",
            "seed_wall_sec",
        ])
        fh.flush()
    return fh, writer


def _append_stream_row(writer, fh, row):
    writer.writerow([
        int(row.get("seed_idx", -1)),
        str(row.get("hand", "normal")),
        bool(row.get("ok", False)),
        _safe_float(row.get("score", 0.0), 0.0),
        _safe_float(row.get("BasinScore", 0.0), 0.0),
        _safe_float(row.get("L_20_3p5", 0.0), 0.0),
        _safe_float(row.get("BasinScore_DM", np.nan), np.nan),
        _safe_float(row.get("DM_mean_FOM", np.nan), np.nan),
        _safe_float(row.get("R_factor_FC_vs_FP", np.nan), np.nan),
        _safe_float(row.get("CC_prob_map_with_current_map", np.nan), np.nan),
        _safe_float(row.get("BIAS_RATIO", np.nan), np.nan),
        _safe_float(row.get("best_largest_component_fraction_at_target", np.nan), np.nan),
        _safe_float(row.get("best_n_connected_components_at_target", np.nan), np.nan),
        _safe_float(row.get("best_endpoint_fraction_at_target", np.nan), np.nan),
        _safe_float(row.get("best_mean_degree_at_target", np.nan), np.nan),
        _safe_float(row.get("best_largest_component_fraction_auc", np.nan), np.nan),
        _safe_float(row.get("basin_term_r", np.nan), np.nan),
        _safe_float(row.get("basin_term_fom", np.nan), np.nan),
        _safe_float(row.get("basin_term_lcf", np.nan), np.nan),
        _safe_float(row.get("basin_term_endpoint", np.nan), np.nan),
        _safe_float(row.get("cc_last", np.nan), np.nan),
        _safe_float(row.get("r_last", np.nan), np.nan),
        _safe_float(row.get("fom_last", np.nan), np.nan),
        _safe_float(row.get("endpoint", np.nan), np.nan),
        _safe_float(row.get("pull", np.nan), np.nan),
        _safe_float(row.get("tail", np.nan), np.nan),
        int(row.get("n_reflections_score_window", 0)),
        str(row.get("error", "")),
        _safe_float(row.get("seed_wall_sec", np.nan), np.nan),
    ])
    fh.flush()


def _run_seed_process_supervisor_streaming(seed_indices, nproc, timeout_sec, shared,
                                           all_csv_path, normal_csv_path, inverted_csv_path,
                                           progress_prefix="Seeds ", checkpoint_every=1000):
    total_seeds = int(len(seed_indices))
    if total_seeds < 1:
        return None, 0, 0, 0

    nproc = max(1, int(nproc))
    timeout_sec = float(timeout_sec)
    queue_out = mp.Queue()

    pending = list(seed_indices)
    running = {}
    t0 = time.time()
    progress_state = {}
    best_state = {}

    done_seeds = 0
    valid_eval_count = 0
    done_eval_count = 0
    fail_eval_count = 0
    next_checkpoint = int(max(1, checkpoint_every))

    best_row = None
    best_ph_full = None

    all_fh, all_writer = _open_stream_csv(all_csv_path)
    norm_fh, norm_writer = _open_stream_csv(normal_csv_path)
    inv_fh = None
    inv_writer = None
    if inverted_csv_path is not None:
        inv_fh, inv_writer = _open_stream_csv(inverted_csv_path)

    def _launch_one(seed_idx):
        proc = mp.Process(target=_seed_entrypoint, args=(int(seed_idx), shared, queue_out))
        proc.daemon = True
        proc.start()
        running[int(seed_idx)] = dict(
            process=proc,
            start_time=time.time(),
        )

    while done_seeds < total_seeds:
        while (len(running) < nproc) and pending:
            _launch_one(pending.pop(0))

        for rec in _drain_result_queue(queue_out):
            sid = int(rec.get("seed_idx", -1))
            if sid not in running:
                continue
            proc = running[sid]["process"]
            try:
                proc.join(timeout=0.0)
            except Exception:
                pass
            running.pop(sid, None)

            rows = rec.get("rows", [])
            if rows is None:
                rows = []

            for row in rows:
                row = _normalize_seed_result(row, int(shared["n_reflections_score_window"]))
                _append_stream_row(all_writer, all_fh, row)
                if str(row.get("hand", "normal")) == "normal":
                    _append_stream_row(norm_writer, norm_fh, row)
                elif (str(row.get("hand", "")) == "inverted") and (inv_writer is not None):
                    _append_stream_row(inv_writer, inv_fh, row)

                done_eval_count += 1
                if bool(row.get("ok", False)):
                    valid_eval_count += 1
                    if (best_row is None) or (float(row["score"]) > float(best_row["score"])):
                        best_row = dict(row)
                        best_ph_full = None
                        if rec.get("best_local", None) is not None:
                            bl = rec["best_local"]
                            if (int(bl.get("seed_idx", -1)) == int(row["seed_idx"])) and (str(bl.get("hand", "")) == str(row["hand"])):
                                best_ph_full = np.asarray(bl["ph_full"], dtype=float)
                        _log_best_row(best_row=best_row, state=best_state)
                else:
                    fail_eval_count += 1

            if rec.get("best_local", None) is not None:
                bl = rec["best_local"]
                if best_row is not None:
                    if (int(bl.get("seed_idx", -1)) == int(best_row["seed_idx"])) and (str(bl.get("hand", "")) == str(best_row["hand"])):
                        best_ph_full = np.asarray(bl["ph_full"], dtype=float)

            done_seeds += 1
            _print_progress(done=done_seeds, total=total_seeds, t0=t0, prefix=progress_prefix, force=False, state=progress_state)

            if done_seeds >= next_checkpoint:
                if best_row is not None:
                    log("Checkpoint | done=%d | best_seed_idx=%d | hand=%s | L_20_3p5=%.3f" % (
                        int(done_seeds),
                        int(best_row["seed_idx"]),
                        str(best_row["hand"]),
                        float(best_row["score"]),
                    ))
                    _log_checkpoint_best_details(
                        best_row=best_row,
                        best_ph_full=best_ph_full,
                        shared=shared,
                    )
                else:
                    log("Checkpoint | done=%d | no valid seed yet" % int(done_seeds))
                next_checkpoint += int(max(1, checkpoint_every))

        now = time.time()
        timed_out = []
        for sid, meta in running.items():
            proc = meta["process"]
            if not proc.is_alive():
                try:
                    proc.join(timeout=0.0)
                except Exception:
                    pass
                timed_out.append((sid, False))
            elif (now - float(meta["start_time"])) > timeout_sec:
                timed_out.append((sid, True))

        for sid, is_timeout in timed_out:
            meta = running.get(sid, None)
            if meta is None:
                continue
            proc = meta["process"]
            wall = float(now - float(meta["start_time"]))
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass
            try:
                proc.join(timeout=1.0)
            except Exception:
                pass
            running.pop(sid, None)

            row = _seed_failure_record(
                seed_idx=sid,
                hand="normal",
                error_msg=("TIMEOUT after %.1f s" % timeout_sec) if is_timeout else "Worker exited without returning a result",
                n_reflections_score_window=int(shared["n_reflections_score_window"]),
                seed_wall_sec=wall,
            )
            row = _normalize_seed_result(row, int(shared["n_reflections_score_window"]))

            _append_stream_row(all_writer, all_fh, row)
            _append_stream_row(norm_writer, norm_fh, row)

            done_eval_count += 1
            fail_eval_count += 1
            done_seeds += 1

            _print_progress(done=done_seeds, total=total_seeds, t0=t0, prefix=progress_prefix, force=False, state=progress_state)

            if done_seeds >= next_checkpoint:
                if best_row is not None:
                    log("Checkpoint | done=%d | best_seed_idx=%d | hand=%s | L_20_3p5=%.3f" % (
                        int(done_seeds),
                        int(best_row["seed_idx"]),
                        str(best_row["hand"]),
                        float(best_row["score"]),
                    ))
                    _log_checkpoint_best_details(
                        best_row=best_row,
                        best_ph_full=best_ph_full,
                        shared=shared,
                    )
                else:
                    log("Checkpoint | done=%d | no valid seed yet" % int(done_seeds))
                next_checkpoint += int(max(1, checkpoint_every))

        if done_seeds < total_seeds:
            time.sleep(0.05)

    for sid, meta in list(running.items()):
        proc = meta["process"]
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass
        try:
            proc.join(timeout=1.0)
        except Exception:
            pass
        running.pop(sid, None)

    try:
        queue_out.close()
    except Exception:
        pass

    try:
        all_fh.close()
    except Exception:
        pass
    try:
        norm_fh.close()
    except Exception:
        pass
    try:
        if inv_fh is not None:
            inv_fh.close()
    except Exception:
        pass

    _print_progress(done=total_seeds, total=total_seeds, t0=t0, prefix=progress_prefix, force=True, state=progress_state)
    _log_best_row(best_row=best_row, state=best_state)

    return best_row, best_ph_full, valid_eval_count, fail_eval_count




def _detect_amplitude_column(df):
    candidates = ["EOBS", "FOBS", "FP", "F", "|E|", "E"]
    for c in candidates:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            if np.any(np.isfinite(vals)):
                return c
    return None


def _normalize_mode_name(mode):
    m = str(mode).strip().lower()
    aliases = {
        "uniform": "uniform_random",
        "uniform_random": "uniform_random",
        "random": "uniform_random",
        "resolution": "resolution_low_to_high",
        "shell": "resolution_low_to_high",
        "resolution_low_to_high": "resolution_low_to_high",
        "low_to_high": "resolution_low_to_high",
        "amplitude": "amplitude_descending",
        "amplitude_descending": "amplitude_descending",
        "high_amp": "amplitude_descending",
    }
    if m not in aliases:
        raise Sorry("Unknown degradation mode: %s" % str(mode))
    return aliases[m]


def _parse_float_list_csv(text):
    out = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if len(out) == 0:
        raise Sorry("Expected a comma-separated list of numeric values.")
    return out


def _parse_mode_list_csv(text):
    out = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(_normalize_mode_name(tok))
    if len(out) == 0:
        raise Sorry("Expected at least one degradation mode.")
    seen = []
    for m in out:
        if m not in seen:
            seen.append(m)
    return seen


def _mode_pretty_label(mode):
    return {
        "uniform_random": "uniform-random flips",
        "resolution_low_to_high": "low-to-high resolution flips",
        "amplitude_descending": "high-amplitude-first flips",
    }.get(str(mode), str(mode))


def _flip_fraction_to_count(n, frac):
    n = int(n)
    frac = float(frac)
    if n <= 0:
        return 0
    if frac <= 0.0:
        return 0
    if frac >= 1.0:
        return int(n)
    return int(round(frac * float(n)))


def _rank_indices_for_mode(indices, mode, df, amplitude_col, rng):
    idx = np.asarray(indices, dtype=int)
    if idx.size == 0:
        return idx
    mode = _normalize_mode_name(mode)
    if mode == "uniform_random":
        order = rng.permutation(idx.size)
        return idx[order]

    jitter = rng.uniform(low=0.0, high=1e-9, size=idx.size)

    if mode == "resolution_low_to_high":
        dvals = pd.to_numeric(df.iloc[idx]["dHKL"], errors="coerce").to_numpy(dtype=float)
        dvals = np.where(np.isfinite(dvals), dvals, -np.inf)
        order = np.argsort(-(dvals + jitter))
        return idx[order]

    if mode == "amplitude_descending":
        if amplitude_col is None:
            order = rng.permutation(idx.size)
            return idx[order]
        avals = np.abs(pd.to_numeric(df.iloc[idx][amplitude_col], errors="coerce").to_numpy(dtype=float))
        avals = np.where(np.isfinite(avals), avals, -np.inf)
        order = np.argsort(-(avals + jitter))
        return idx[order]

    raise Sorry("Unhandled degradation mode: %s" % str(mode))


def _apply_controlled_degradation(task, shared):
    mode = _normalize_mode_name(task["mode"])
    frac = float(task["degrade_fraction"])
    rng = RNG(int(task["task_seed"]))

    ph_full = np.asarray(shared["control_phases_full"], dtype=float).copy()

    idx_acentric = np.asarray(shared["idx_score_acentric"], dtype=int)
    idx_centric = np.asarray(shared["idx_score_centric"], dtype=int)

    n_flip_acentric = _flip_fraction_to_count(idx_acentric.size, frac)
    n_flip_centric = _flip_fraction_to_count(idx_centric.size, frac)

    rank_acentric = _rank_indices_for_mode(
        indices=idx_acentric,
        mode=mode,
        df=shared["df"],
        amplitude_col=shared.get("amplitude_col", None),
        rng=rng,
    )
    rank_centric = _rank_indices_for_mode(
        indices=idx_centric,
        mode=mode,
        df=shared["df"],
        amplitude_col=shared.get("amplitude_col", None),
        rng=rng,
    )

    flip_acentric = rank_acentric[:n_flip_acentric]
    flip_centric = rank_centric[:n_flip_centric]
    if flip_acentric.size == 0 and flip_centric.size == 0:
        flip_idx = np.asarray([], dtype=int)
    else:
        flip_idx = np.unique(np.concatenate([flip_acentric, flip_centric]).astype(int))

    if flip_idx.size > 0:
        ph_full[flip_idx] = (ph_full[flip_idx] + 180.0) % 360.0

    return dict(
        ph_full=ph_full,
        flip_idx=flip_idx,
        flip_idx_acentric=np.asarray(flip_acentric, dtype=int),
        flip_idx_centric=np.asarray(flip_centric, dtype=int),
        n_flip_total=int(flip_idx.size),
        n_flip_acentric=int(flip_acentric.size),
        n_flip_centric=int(flip_centric.size),
    )


def _compute_realspace_cc_against_control(ph_score_deg, shared):
    try:
        phib_ma = shared["phib0_score"].customized_copy(data=_to_flex_double(np.asarray(ph_score_deg, dtype=float) % 360.0))
        map_coeffs = shared["fp_sigfp_score"].phase_transfer(phase_source=phib_ma, deg=True)
        fft_map = map_coeffs.fft_map(resolution_factor=float(shared.get("map_resolution_factor", 0.25)))
        try:
            fft_map.apply_sigma_scaling()
        except Exception:
            pass
        test_map = fft_map.real_map_unpadded()
        test_flat = np.asarray(list(test_map.as_1d()), dtype=float)
        ref_flat = np.asarray(shared["control_map_flat_score"], dtype=float)
        if test_flat.size != ref_flat.size or test_flat.size < 2:
            return np.nan, np.nan, "realspace_cc_size_mismatch"
        if (not np.any(np.isfinite(test_flat))) or (not np.any(np.isfinite(ref_flat))):
            return np.nan, np.nan, "realspace_cc_nonfinite_map"
        cc = np.corrcoef(ref_flat, test_flat)[0, 1]
        cc_neg = np.corrcoef(-ref_flat, test_flat)[0, 1]
        return float(cc), float(cc_neg), ""
    except Exception as e:
        return np.nan, np.nan, "%s: %s" % (e.__class__.__name__, str(e))


def _build_control_map_flat_score(fp_sigfp_score, phib0_score, ph_control_score, map_resolution_factor):
    phib_ma = phib0_score.customized_copy(data=_to_flex_double(np.asarray(ph_control_score, dtype=float) % 360.0))
    map_coeffs = fp_sigfp_score.phase_transfer(phase_source=phib_ma, deg=True)
    fft_map = map_coeffs.fft_map(resolution_factor=float(map_resolution_factor))
    try:
        fft_map.apply_sigma_scaling()
    except Exception:
        pass
    control_map = fft_map.real_map_unpadded()
    return np.asarray(list(control_map.as_1d()), dtype=float)


def _detect_pdb_id_from_df_or_path(df, csv_path):
    for col in ["PDB_ID", "pdb_id", "PDB", "pdb", "entry_id", "ENTRY_ID"]:
        if col in df.columns:
            vals = df[col].dropna().astype(str).values
            if len(vals) > 0:
                v = str(vals[0]).strip()
                if v:
                    return v
    base = os.path.splitext(os.path.basename(str(csv_path)))[0]
    for suffix in ["_rs_ecalc_binned", "_rs_ecalc", "_binned", "_atten"]:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base



def _sanitize_filename_token(token):
    s = str(token)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    return s if s else "unknown"


def _choose_evenly_spaced_tasks(task_rows, n_keep):
    rows = list(task_rows)
    if len(rows) <= int(n_keep):
        return rows
    order = sorted(rows, key=lambda r: (int(r.get("round_idx", -1)), int(r.get("task_seed", -1))))
    pick_pos = np.linspace(0, len(order) - 1, num=int(n_keep))
    pick_idx = []
    for p in pick_pos:
        idx = int(round(float(p)))
        if idx not in pick_idx:
            pick_idx.append(idx)
    if len(pick_idx) < int(n_keep):
        for idx in range(len(order)):
            if idx not in pick_idx:
                pick_idx.append(idx)
            if len(pick_idx) >= int(n_keep):
                break
    pick_idx = sorted(pick_idx[: int(n_keep)])
    return [order[idx] for idx in pick_idx]


def _write_ccp4_map_from_phases(out_map, fp_sigfp_ma, phib_template_ma, ph_deg, map_resolution_factor, labels=None):
    phib_ma = phib_template_ma.customized_copy(data=_to_flex_double(np.asarray(ph_deg, dtype=float) % 360.0))
    map_coeffs = fp_sigfp_ma.phase_transfer(phase_source=phib_ma, deg=True)
    fft_map = map_coeffs.fft_map(resolution_factor=float(map_resolution_factor))
    try:
        fft_map.apply_sigma_scaling()
    except Exception:
        pass
    lbls = labels
    try:
        if lbls is not None and flex is not None:
            lbls = flex.std_string([str(x) for x in lbls])
    except Exception:
        lbls = None
    try:
        if lbls is None:
            fft_map.as_ccp4_map(file_name=str(out_map))
        else:
            fft_map.as_ccp4_map(file_name=str(out_map), labels=lbls)
    except TypeError:
        fft_map.as_ccp4_map(file_name=str(out_map))
    return str(out_map)


def _choose_export_fraction_indices(fractions, n_keep):
    fr = [float(x) for x in fractions]
    if len(fr) == 0:
        return []
    n_keep = int(max(1, n_keep))
    always = []
    for target in [0.0, 1.0]:
        best = min(range(len(fr)), key=lambda i: abs(fr[i] - target))
        if best not in always:
            always.append(best)
    if n_keep <= len(always):
        return sorted(always[:n_keep], key=lambda i: fr[i])
    extra_needed = n_keep - len(always)
    remaining = [i for i in range(len(fr)) if i not in always]
    if extra_needed >= len(remaining):
        picks = always + remaining
        return sorted(picks, key=lambda i: fr[i])
    grid = np.linspace(0, len(fr) - 1, num=n_keep)
    picks = set(always)
    for g in grid:
        idx = int(round(g))
        picks.add(idx)
    if len(picks) < n_keep:
        for i in remaining:
            picks.add(i)
            if len(picks) >= n_keep:
                break
    picks = sorted(list(picks), key=lambda i: fr[i])
    return picks[:n_keep]


def _export_sample_ccp4_maps_for_mode(mode, tasks, shared, out_dir, maps_per_fraction):
    _mkdir_p(out_dir)
    if len(tasks) == 0 or int(maps_per_fraction) <= 0:
        return []

    tasks_by_fraction = {}
    for task in tasks:
        frac = float(task["degrade_fraction"])
        if frac not in tasks_by_fraction:
            tasks_by_fraction[frac] = []
        tasks_by_fraction[frac].append(task)

    pdb_token = _sanitize_filename_token(shared.get("pdb_id", "unknown"))
    mode_token = _sanitize_filename_token(mode)

    out_rows = []
    for frac in sorted(tasks_by_fraction.keys()):
        chosen_tasks = _choose_evenly_spaced_tasks(
            task_rows=tasks_by_fraction[frac],
            n_keep=int(maps_per_fraction),
        )
        for task in chosen_tasks:
            deg = _apply_controlled_degradation(task=task, shared=shared)
            ph_full = np.asarray(deg["ph_full"], dtype=float)
            fom_full = np.asarray(shared["fom_control_full_vals"], dtype=float)

            labels = [
                "PC50 degradation mode=%s" % str(mode),
                "pdb_id=%s frac=%.4f round=%d" % (
                    str(shared.get("pdb_id", "unknown")),
                    float(task["degrade_fraction"]),
                    int(task["round_idx"]),
                ),
            ]

            base_stub = "pdb_%s__degradation__%s__frac_%0.4f__round_%03d" % (
                str(pdb_token),
                str(mode_token),
                float(task["degrade_fraction"]),
                int(task["round_idx"]),
            )
            out_map = os.path.join(out_dir, base_stub + ".ccp4")
            out_mtz = os.path.join(out_dir, base_stub + "_PHIB_input.mtz")

            map_ok = False
            mtz_ok = False

            try:
                _write_ccp4_map_from_phases(
                    out_map=out_map,
                    fp_sigfp_ma=shared["fp_sigfp_full"],
                    phib_template_ma=shared["phib0_full"],
                    ph_deg=ph_full,
                    map_resolution_factor=float(shared.get("map_resolution_factor", 0.25)),
                    labels=labels,
                )
                map_ok = True
            except Exception as e:
                log("WARNING: failed CCP4 map export for mode %s fraction %.4f round %d: %s" % (
                    str(mode), float(task["degrade_fraction"]), int(task["round_idx"]), str(e)
                ))

            try:
                _write_best_mtz_autobuild(
                    out_mtz=out_mtz,
                    shared=shared,
                    ph_full=ph_full,
                    fom_full=fom_full,
                )
                mtz_ok = True
            except Exception as e:
                log("WARNING: failed MTZ export for mode %s fraction %.4f round %d: %s" % (
                    str(mode), float(task["degrade_fraction"]), int(task["round_idx"]), str(e)
                ))

            if map_ok or mtz_ok:
                out_rows.append(dict(
                    pdb_id=str(shared.get("pdb_id", "unknown")),
                    mode=str(mode),
                    degrade_fraction=float(task["degrade_fraction"]),
                    round_idx=int(task["round_idx"]),
                    task_seed=int(task["task_seed"]),
                    ccp4_map=(str(out_map) if map_ok else ""),
                    mtz_autobuild=(str(out_mtz) if mtz_ok else ""),
                ))

    return out_rows


def _degradation_row_failure(task, shared, error_msg, wall_sec=np.nan):
    return dict(
        mode=str(task.get("mode", "")),
        mode_label=_mode_pretty_label(task.get("mode", "")),
        degrade_fraction=float(task.get("degrade_fraction", np.nan)),
        round_idx=int(task.get("round_idx", -1)),
        task_seed=int(task.get("task_seed", -1)),
        ok=False,
        rejected_by_gate=False,
        rejection_reasons="",
        n_reflections_score_window=int(shared.get("n_reflections_score_window", 0)),
        n_acentric_score_window=int(shared.get("n_acentric_score_window", 0)),
        n_centric_score_window=int(shared.get("n_centric_score_window", 0)),
        n_flip_total=np.nan,
        n_flip_acentric=np.nan,
        n_flip_centric=np.nan,
        frac_flip_total=np.nan,
        frac_flip_acentric=np.nan,
        frac_flip_centric=np.nan,
        realspace_cc_control=np.nan,
        realspace_cc_neg_control=np.nan,
        realspace_cc_error="",
        BasinScore=np.nan,
        basin_term_r=np.nan,
        basin_term_fom=np.nan,
        basin_term_lcf=np.nan,
        basin_term_endpoint=np.nan,
        Reflections=np.nan,
        DM_mean_FOM=np.nan,
        CC_prob_map_with_current_map=np.nan,
        R_factor_FC_vs_FP=np.nan,
        BIAS_RATIO=np.nan,
        best_largest_component_fraction_at_target=np.nan,
        best_n_connected_components_at_target=np.nan,
        best_endpoint_fraction_at_target=np.nan,
        best_mean_degree_at_target=np.nan,
        best_largest_component_fraction_auc=np.nan,
        skeleton_ok=False,
        skeleton_error="",
        wall_sec=float(wall_sec),
        error=str(error_msg),
    )


def _format_degradation_terminal_line(row):
    return (
        "MODE=%s | frac=%.3f | round=%d | ok=%s | gate=%s | flips(all/ac/ce)=%s/%s/%s | "
        "RSCC=%.4f | RSCC(-ctrl)=%.4f | BasinScore=%.3f | R=%.4f | FOM=%.4f | LCF_AUC=%.4f | Endpoint=%.4f | n_comp=%s | mean_degree=%.4f"
    ) % (
        str(row.get("mode", "")),
        float(_safe_float(row.get("degrade_fraction", np.nan), np.nan)),
        int(row.get("round_idx", -1)),
        str(bool(row.get("ok", False))),
        ("reject" if bool(row.get("rejected_by_gate", False)) else "pass"),
        str(row.get("n_flip_total", "NA")),
        str(row.get("n_flip_acentric", "NA")),
        str(row.get("n_flip_centric", "NA")),
        float(_safe_float(row.get("realspace_cc_control", np.nan), np.nan)),
        float(_safe_float(row.get("realspace_cc_neg_control", np.nan), np.nan)),
        float(_safe_float(row.get("BasinScore", np.nan), np.nan)),
        float(_safe_float(row.get("R_factor_FC_vs_FP", np.nan), np.nan)),
        float(_safe_float(row.get("DM_mean_FOM", np.nan), np.nan)),
        float(_safe_float(row.get("best_largest_component_fraction_auc", np.nan), np.nan)),
        float(_safe_float(row.get("best_endpoint_fraction_at_target", np.nan), np.nan)),
        str(row.get("best_n_connected_components_at_target", "NA")),
        float(_safe_float(row.get("best_mean_degree_at_target", np.nan), np.nan)),
    )


def _degradation_csv_fieldnames():
    return [
        "mode",
        "mode_label",
        "degrade_fraction",
        "round_idx",
        "task_seed",
        "ok",
        "rejected_by_gate",
        "rejection_reasons",
        "n_reflections_score_window",
        "n_acentric_score_window",
        "n_centric_score_window",
        "n_flip_total",
        "n_flip_acentric",
        "n_flip_centric",
        "frac_flip_total",
        "frac_flip_acentric",
        "frac_flip_centric",
        "realspace_cc_control",
        "realspace_cc_neg_control",
        "realspace_cc_error",
        "BasinScore",
        "basin_term_r",
        "basin_term_fom",
        "basin_term_lcf",
        "basin_term_endpoint",
        "Reflections",
        "DM_mean_FOM",
        "CC_prob_map_with_current_map",
        "R_factor_FC_vs_FP",
        "BIAS_RATIO",
        "best_largest_component_fraction_at_target",
        "best_n_connected_components_at_target",
        "best_endpoint_fraction_at_target",
        "best_mean_degree_at_target",
        "best_largest_component_fraction_auc",
        "skeleton_ok",
        "skeleton_error",
        "wall_sec",
        "error",
    ]


def _write_degradation_csv_rows(csv_path, rows):
    fieldnames = _degradation_csv_fieldnames()
    with open(csv_path, "w") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict((k, row.get(k, "")) for k in fieldnames)
            writer.writerow(out)


def _read_degradation_csv_rows(csv_path):
    if (csv_path is None) or (not os.path.isfile(csv_path)):
        return []
    out_rows = []
    try:
        with open(csv_path, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                out_rows.append(dict(row))
    except Exception:
        return []
    return out_rows


def _task_row_key(task_or_row):
    try:
        return (
            str(task_or_row.get("mode", "")),
            round(float(task_or_row.get("degrade_fraction", np.nan)), 12),
            int(task_or_row.get("round_idx", -1)),
        )
    except Exception:
        return (
            str(task_or_row.get("mode", "")),
            np.nan,
            int(task_or_row.get("round_idx", -1)),
        )


def _existing_task_key_set_from_csv(csv_path):
    rows = _read_degradation_csv_rows(csv_path)
    keys = set()
    for row in rows:
        keys.add(_task_row_key(row))
    return keys


def _append_degradation_csv_rows(csv_path, rows):
    fieldnames = _degradation_csv_fieldnames()
    existing_keys = _existing_task_key_set_from_csv(csv_path)
    rows_to_append = []
    for row in rows:
        key = _task_row_key(row)
        if key in existing_keys:
            continue
        existing_keys.add(key)
        rows_to_append.append(row)

    if len(rows_to_append) == 0:
        return 0

    write_header = (not os.path.isfile(csv_path)) or (os.path.getsize(csv_path) == 0)
    with open(csv_path, "a") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows_to_append:
            out = dict((k, row.get(k, "")) for k in fieldnames)
            writer.writerow(out)
    return int(len(rows_to_append))


def _rows_for_mode_from_csv(csv_path, mode):
    mode = str(mode)
    rows = _read_degradation_csv_rows(csv_path)
    out = []
    for row in rows:
        if str(row.get("mode", "")) == mode:
            out.append(row)
    return out


def _rows_for_fraction(rows, frac):
    out = []
    target = round(float(frac), 12)
    for row in rows:
        try:
            row_frac = round(float(row.get("degrade_fraction", np.nan)), 12)
        except Exception:
            row_frac = np.nan
        if row_frac == target:
            out.append(row)
    return out


def _existing_round_idx_set(rows, mode, frac):
    out = set()
    mode = str(mode)
    target = round(float(frac), 12)
    for row in rows:
        try:
            row_mode = str(row.get("mode", ""))
            row_frac = round(float(row.get("degrade_fraction", np.nan)), 12)
            row_round = int(row.get("round_idx", -1))
        except Exception:
            continue
        if (row_mode == mode) and (row_frac == target) and (row_round >= 0):
            out.add(int(row_round))
    return out


def _write_fraction_checkpoint_json(out_json, payload):
    tmp = str(out_json) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.rename(tmp, out_json)


def _refresh_mode_artifacts_from_csv(mode, csv_out, tables_root, pdb_id, disable_summary_plots):
    rows = _rows_for_mode_from_csv(csv_out, mode)
    rows = sorted(rows, key=lambda r: (float(_safe_float(r.get("degrade_fraction", np.nan), np.nan)), int(r.get("round_idx", -1)), str(r.get("mode", ""))))
    _log_mode_summary(mode=mode, rows=rows)

    if not bool(disable_summary_plots):
        try:
            panel_out = os.path.join(tables_root, "40_degradation__%s__panel.png" % str(mode))
            _plot_mode_summary(
                mode=mode,
                rows=rows,
                out_png=panel_out,
                pdb_id=pdb_id,
            )
            log("Wrote mode panel: %s" % str(panel_out))
        except Exception as e:
            log("WARNING: failed to write mode panel for %s: %s" % (str(mode), str(e)))

        try:
            signed_out = os.path.join(tables_root, "40_degradation__%s__signed_control_corr.png" % str(mode))
            _plot_signed_control_cc(
                mode=mode,
                rows=rows,
                out_png=signed_out,
                pdb_id=pdb_id,
            )
            log("Wrote signed-control correlation plot: %s" % str(signed_out))
        except Exception as e:
            log("WARNING: failed to write signed-control correlation plot for %s: %s" % (str(mode), str(e)))

    return rows


def _refresh_ccp4_manifest(mode, manifest_csv, exported_rows):
    if len(exported_rows) == 0:
        return
    old_rows = []
    if os.path.isfile(manifest_csv):
        try:
            old_rows = pd.read_csv(manifest_csv).to_dict(orient="records")
        except Exception:
            old_rows = []
    merged_by_key = {}
    for row in list(old_rows) + list(exported_rows):
        key = (
            str(row.get("mode", "")),
            round(float(row.get("degrade_fraction", np.nan)), 12),
            int(row.get("round_idx", -1)),
        )
        merged_by_key[key] = row
    merged = [merged_by_key[k] for k in sorted(merged_by_key.keys())]
    if len(merged) > 0:
        pd.DataFrame(merged).to_csv(manifest_csv, index=False)


def _degradation_worker_top(task):
    global _WORKER_SHARED
    S = _WORKER_SHARED
    t0 = time.time()
    timeout_sec = float(S.get("task_timeout_sec", 0.0))
    old_handler = None
    alarm_set = False
    try:
        if timeout_sec > 0.0:
            try:
                old_handler = signal.signal(signal.SIGALRM, _worker_alarm_handler)
                signal.alarm(int(max(1, np.ceil(timeout_sec))))
                alarm_set = True
            except Exception:
                old_handler = None
                alarm_set = False

        deg = _apply_controlled_degradation(task=task, shared=S)
        ph_full = np.asarray(deg["ph_full"], dtype=float)
        ph_score = ph_full[S["idx_score"]]

        rscc, rscc_neg, rscc_err = _compute_realspace_cc_against_control(ph_score_deg=ph_score, shared=S)

        hl_score = S["hl0_score"].customized_copy(
            data=_hl_from_phib_fom(ph_score, S["fom_control_score_vals"], float(S["kappa_max"]))
        )

        dm_params = dict(S["dm_params_base"])
        dm_params["temp_dir"] = os.path.join(
            S["degradation_temp_root"],
            "%s__f%0.4f__r%03d" % (str(task["mode"]), float(task["degrade_fraction"]), int(task["round_idx"])),
            "dm",
        )
        _mkdir_p(dm_params["temp_dir"])

        res_dm = _run_dm_and_score_hybrid(
            fp_sigfp_score=S["fp_sigfp_score"],
            phib_score=S["phib0_score"].customized_copy(data=_to_flex_double(ph_score)),
            fom_score=S["fom_control_ma_score"],
            hl_score=hl_score,
            solvent_content=float(S["sol"]),
            dm_params=dm_params,
            fom_first_mean=float(S["fom_control_mean_score"]),
            fom_in_std=float(S["fom_control_std_score"]),
            score_params=S["score_params"],
            debug_print_dm_transcript=bool(S.get("debug_print_dm_transcript", False)),
            debug_transcript_file=None,
        )
        if not bool(S.get("keep_temp", False)):
            _rm_tree(dm_params["temp_dir"])

        if not bool(res_dm.get("ok", False)):
            row = _degradation_row_failure(task=task, shared=S, error_msg=str(res_dm.get("error", "DM_failed")), wall_sec=time.time() - t0)
            row["realspace_cc_control"] = float(rscc) if np.isfinite(_safe_float(rscc)) else np.nan
            row["realspace_cc_neg_control"] = float(rscc_neg) if np.isfinite(_safe_float(rscc_neg)) else np.nan
            row["realspace_cc_error"] = str(rscc_err)
            return row

        skel_temp = os.path.join(
            S["degradation_temp_root"],
            "%s__f%0.4f__r%03d" % (str(task["mode"]), float(task["degrade_fraction"]), int(task["round_idx"])),
            "skel",
        )
        _mkdir_p(skel_temp)
        skel = _run_skeleton_proxy_for_seed(
            df_base=S["df"],
            ph_full=ph_full,
            fom_full=S["fom_control_full_vals"],
            temp_root=skel_temp,
            score_params=S,
        )
        if not bool(S.get("keep_temp", False)):
            _rm_tree(skel_temp)
        skel = _normalize_skeleton_metrics_for_seed(skel)
        if bool(skel.get("skeleton_ok", False)) and ("skeleton_error" not in skel):
            skel["skeleton_error"] = ""

        basin = _compute_current_basin_score(dm_metrics=res_dm, skel_metrics=skel)
        merged_metrics = {}
        merged_metrics.update(res_dm)
        merged_metrics.update(skel)
        merged_metrics.update(basin)
        rejection_reasons = _seed_rejection_reasons(merged_metrics)
        rejected = len(rejection_reasons) > 0

        row = dict(
            mode=str(task["mode"]),
            mode_label=_mode_pretty_label(task["mode"]),
            degrade_fraction=float(task["degrade_fraction"]),
            round_idx=int(task["round_idx"]),
            task_seed=int(task["task_seed"]),
            ok=bool(np.isfinite(_safe_float(basin.get("BasinScore", np.nan)))) and (not rejected),
            rejected_by_gate=bool(rejected),
            rejection_reasons="; ".join(rejection_reasons),
            n_reflections_score_window=int(S["n_reflections_score_window"]),
            n_acentric_score_window=int(S["n_acentric_score_window"]),
            n_centric_score_window=int(S["n_centric_score_window"]),
            n_flip_total=int(deg["n_flip_total"]),
            n_flip_acentric=int(deg["n_flip_acentric"]),
            n_flip_centric=int(deg["n_flip_centric"]),
            frac_flip_total=(float(deg["n_flip_total"]) / float(max(1, S["n_reflections_score_window"]))),
            frac_flip_acentric=(float(deg["n_flip_acentric"]) / float(max(1, S["n_acentric_score_window"]))),
            frac_flip_centric=(float(deg["n_flip_centric"]) / float(max(1, S["n_centric_score_window"]))),
            realspace_cc_control=float(rscc) if np.isfinite(_safe_float(rscc)) else np.nan,
            realspace_cc_neg_control=float(rscc_neg) if np.isfinite(_safe_float(rscc_neg)) else np.nan,
            realspace_cc_error=str(rscc_err),
            BasinScore=float(basin.get("BasinScore", np.nan)),
            basin_term_r=float(basin.get("basin_term_r", np.nan)),
            basin_term_fom=float(basin.get("basin_term_fom", np.nan)),
            basin_term_lcf=float(basin.get("basin_term_lcf", np.nan)),
            basin_term_endpoint=float(basin.get("basin_term_endpoint", np.nan)),
            Reflections=float(res_dm.get("Reflections", np.nan)),
            DM_mean_FOM=float(res_dm.get("DM_mean_FOM", np.nan)),
            CC_prob_map_with_current_map=float(res_dm.get("CC_prob_map_with_current_map", np.nan)),
            R_factor_FC_vs_FP=float(res_dm.get("R_factor_FC_vs_FP", np.nan)),
            BIAS_RATIO=float(res_dm.get("BIAS_RATIO", np.nan)),
            best_largest_component_fraction_at_target=float(skel.get("best_largest_component_fraction_at_target", np.nan)),
            best_n_connected_components_at_target=float(skel.get("best_n_connected_components_at_target", np.nan)),
            best_endpoint_fraction_at_target=float(skel.get("best_endpoint_fraction_at_target", np.nan)),
            best_mean_degree_at_target=float(skel.get("best_mean_degree_at_target", np.nan)),
            best_largest_component_fraction_auc=float(skel.get("best_largest_component_fraction_auc", np.nan)),
            skeleton_ok=bool(skel.get("skeleton_ok", False)),
            skeleton_error=str(skel.get("skeleton_error", "")),
            wall_sec=float(time.time() - t0),
            error="",
        )
        if rejected:
            row["error"] = "REJECTED: %s" % row["rejection_reasons"]
        return row
    except WorkerTaskTimeoutError:
        return _degradation_row_failure(task=task, shared=S, error_msg="worker hard timeout after %.1f s" % float(timeout_sec), wall_sec=time.time() - t0)
    except Exception as e:
        return _degradation_row_failure(task=task, shared=S, error_msg="%s: %s" % (e.__class__.__name__, str(e)), wall_sec=time.time() - t0)
    finally:
        if alarm_set:
            try:
                signal.alarm(0)
            except Exception:
                pass
        if old_handler is not None:
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except Exception:
                pass

def _coerce_mode_rows_dataframe(dfm):
    if dfm is None:
        return pd.DataFrame()
    if not isinstance(dfm, pd.DataFrame):
        dfm = pd.DataFrame(dfm)
    if dfm.shape[0] == 0:
        return dfm

    numeric_cols = [
        "degrade_fraction",
        "round_idx",
        "realspace_cc_control",
        "realspace_cc_neg_control",
        "BasinScore",
        "DM_mean_FOM",
        "R_factor_FC_vs_FP",
        "best_largest_component_fraction_at_target",
        "best_endpoint_fraction_at_target",
        "best_mean_degree_at_target",
        "best_n_connected_components_at_target",
        "best_largest_component_fraction_auc",
        "wall_sec",
    ]
    for col in numeric_cols:
        if col in dfm.columns:
            dfm[col] = pd.to_numeric(dfm[col], errors="coerce")

    bool_like_cols = ["ok", "rejected_by_gate", "skeleton_ok"]
    for col in bool_like_cols:
        if col in dfm.columns:
            vals = dfm[col].astype(str).str.strip().str.lower()
            dfm[col] = vals.isin(["1", "true", "t", "yes", "y"])

    for col in ["error", "rejection_reasons", "mode", "mode_label", "realspace_cc_error", "skeleton_error"]:
        if col in dfm.columns:
            dfm[col] = dfm[col].fillna("").astype(str)

    return dfm


def _build_degradation_tasks(modes, fractions, n_rounds, seed_master):
    tasks_by_mode = {}
    for mode in modes:
        mode = _normalize_mode_name(mode)
        rows = []
        for frac in fractions:
            frac = float(frac)
            if frac < 0.0 or frac > 1.0:
                raise Sorry("Degradation fractions must lie in [0, 1]. Got %s" % str(frac))
            rounds_here = 1 if abs(frac) < 1e-12 else int(max(1, n_rounds))
            for round_idx in range(rounds_here):
                task_seed = int(_seed_from_master(int(seed_master), "%s|%.6f|%d" % (mode, frac, int(round_idx)), "degrade"))
                rows.append(dict(
                    mode=mode,
                    degrade_fraction=float(frac),
                    round_idx=int(round_idx),
                    task_seed=int(task_seed),
                ))
        tasks_by_mode[mode] = rows
    return tasks_by_mode


def _log_mode_summary(mode, rows):
    if len(rows) == 0:
        log("No rows returned for mode %s" % str(mode))
        return
    dfm = _coerce_mode_rows_dataframe(pd.DataFrame(rows))
    if dfm.shape[0] == 0:
        log("No rows returned for mode %s" % str(mode))
        return

    n_total = int(dfm.shape[0])
    n_ok = int(dfm["ok"].sum()) if "ok" in dfm.columns else 0
    n_rejected = int(dfm["rejected_by_gate"].sum()) if "rejected_by_gate" in dfm.columns else 0
    if "error" in dfm.columns:
        err_nonempty = (dfm["error"].fillna("").astype(str).str.strip() != "")
        n_failed = int((((~dfm.get("rejected_by_gate", pd.Series([False] * n_total, index=dfm.index))).astype(bool)) & err_nonempty).sum())
    else:
        n_failed = 0

    log("SUMMARY | mode=%s | rows=%d | accepted=%d | rejected=%d | failed=%d" % (
        str(mode), n_total, n_ok, n_rejected, int(n_failed),
    ))

    agg_cols = [
        "realspace_cc_control",
        "realspace_cc_neg_control",
        "BasinScore",
        "DM_mean_FOM",
        "R_factor_FC_vs_FP",
        "best_largest_component_fraction_at_target",
        "best_endpoint_fraction_at_target",
        "best_mean_degree_at_target",
    ]
    agg_map = {}
    for col in agg_cols:
        if col in dfm.columns:
            agg_map[col] = "mean"
    if "ok" in dfm.columns:
        agg_map["ok"] = "sum"
    if "degrade_fraction" not in dfm.columns or len(agg_map) == 0:
        return

    grp = dfm.groupby("degrade_fraction", as_index=False).agg(agg_map)
    grp = grp.sort_values("degrade_fraction")
    for _, rr in grp.iterrows():
        log("  frac=%.3f | mean_RSCC=%.4f | mean_RSCC_neg=%.4f | mean_BasinScore=%.3f | mean_R=%.4f | mean_FOM=%.4f | mean_LCF_AUC=%.4f | mean_Endpoint=%.4f | mean_degree=%.4f | accepted=%d" % (
            float(_safe_float(rr.get("degrade_fraction", np.nan), np.nan)),
            float(_safe_float(rr.get("realspace_cc_control", np.nan), np.nan)),
            float(_safe_float(rr.get("realspace_cc_neg_control", np.nan), np.nan)),
            float(_safe_float(rr.get("BasinScore", np.nan), np.nan)),
            float(_safe_float(rr.get("R_factor_FC_vs_FP", np.nan), np.nan)),
            float(_safe_float(rr.get("DM_mean_FOM", np.nan), np.nan)),
            float(_safe_float(rr.get("best_largest_component_fraction_auc", np.nan), np.nan)),
            float(_safe_float(rr.get("best_endpoint_fraction_at_target", np.nan), np.nan)),
            float(_safe_float(rr.get("best_mean_degree_at_target", np.nan), np.nan)),
            int(_safe_float(rr.get("ok", 0), 0)),
        ))


def _plot_mode_summary(mode, rows, out_png, pdb_id=None):
    if len(rows) == 0:
        return
    dfm = _coerce_mode_rows_dataframe(pd.DataFrame(rows))
    if dfm.shape[0] == 0:
        return

    metric_specs = [
        ("realspace_cc_control", "RSCC to control"),
        ("BasinScore", "Basin Score"),
        ("R_factor_FC_vs_FP", "R factor"),
        ("DM_mean_FOM", "Mean FOM"),
        ("best_largest_component_fraction_auc", "LCF_AUC"),
        ("best_endpoint_fraction_at_target", "Endpoint@target"),
    ]

    threshold_lines = {
        "BasinScore": [
            (60.0, "in-basin threshold"),
            (40.0, "out-of-basin threshold"),
        ],
        "R_factor_FC_vs_FP": [
            (float(CURRENT_THR_R_FACTOR), "32_ threshold"),
        ],
        "DM_mean_FOM": [
            (float(CURRENT_THR_DM_MEAN_FOM), "32_ threshold"),
        ],
        "best_largest_component_fraction_auc": [
            (float(CURRENT_THR_LCF_AUC), "32_ threshold"),
        ],
        "best_endpoint_fraction_at_target": [
            (float(CURRENT_THR_ENDPOINT_FRAC), "32_ threshold"),
        ],
    }

    agg_map = {}
    for metric_name, _metric_label in metric_specs:
        if metric_name in dfm.columns:
            agg_map[metric_name] = ["mean", "std"]

    if "degrade_fraction" not in dfm.columns or len(agg_map) == 0:
        return

    grp = dfm.groupby("degrade_fraction").agg(agg_map)
    grp.columns = ["%s__%s" % (str(a), str(b)) for (a, b) in grp.columns]
    grp = grp.reset_index().sort_values("degrade_fraction")

    x = pd.to_numeric(grp["degrade_fraction"], errors="coerce").astype(float).values

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(14, 8), sharex=True)
    axes = axes.ravel()

    for ax, (metric_name, metric_label) in zip(axes, metric_specs):
        mean_key = "%s__mean" % str(metric_name)
        std_key = "%s__std" % str(metric_name)
        if mean_key not in grp.columns:
            ax.set_visible(False)
            continue
        y = pd.to_numeric(grp[mean_key], errors="coerce").astype(float).values
        yerr = pd.to_numeric(grp[std_key], errors="coerce").astype(float).values if std_key in grp.columns else np.zeros_like(y)
        finite_err = np.isfinite(yerr)
        yerr = np.where(finite_err, yerr, 0.0) if np.any(finite_err) else np.zeros_like(y)
        ax.plot(x, y, marker="o", linewidth=1.5, label="mean")
        if np.any(np.isfinite(yerr) & (yerr > 0.0)):
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.20, label="+/- 1 SD")
        for thr_value, thr_label in threshold_lines.get(metric_name, []):
            if np.isfinite(_safe_float(thr_value, default=np.nan)):
                ax.axhline(y=float(thr_value), linestyle="--", linewidth=1.2, alpha=0.9, label=str(thr_label))
        ax.set_title(metric_label, fontsize=12)
        ax.set_xlabel("Degradation fraction", fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.minorticks_on()
        ax.grid(which="major", linestyle="--", color="gray", alpha=0.7)
        ax.grid(which="minor", linestyle=":", color="lightgray", alpha=0.7)
        ax.tick_params(axis="both", which="major", labelsize=10, length=8)
        ax.tick_params(axis="both", which="minor", length=4)

    title = "Controlled degradation summary: %s" % str(mode)
    if pdb_id is not None:
        title += " | %s" % str(pdb_id)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_signed_control_cc(mode, rows, out_png, pdb_id=None):
    if len(rows) == 0:
        return
    dfm = _coerce_mode_rows_dataframe(pd.DataFrame(rows))
    if dfm.shape[0] == 0:
        return
    if "degrade_fraction" not in dfm.columns:
        return
    if not (("realspace_cc_control" in dfm.columns) and ("realspace_cc_neg_control" in dfm.columns)):
        return

    grp = dfm.groupby("degrade_fraction", as_index=False)[
        ["realspace_cc_control", "realspace_cc_neg_control"]
    ].agg(["mean", "std"])
    grp.columns = ["degrade_fraction" if c[0] == "degrade_fraction" else "%s__%s" % (str(c[0]), str(c[1])) for c in grp.columns]
    if "degrade_fraction" not in grp.columns:
        # older pandas layout fallback
        grp = grp.reset_index()
        if "degrade_fraction" not in grp.columns and "index" in grp.columns:
            grp = grp.rename(columns={"index": "degrade_fraction"})
    grp = grp.sort_values("degrade_fraction")
    x = pd.to_numeric(grp["degrade_fraction"], errors="coerce").astype(float).values

    def _col(name, stat):
        key = "%s__%s" % (name, stat)
        return pd.to_numeric(grp[key], errors="coerce").astype(float).values if key in grp.columns else np.full_like(x, np.nan, dtype=float)

    y_pos = _col("realspace_cc_control", "mean")
    s_pos = _col("realspace_cc_control", "std")
    y_neg = _col("realspace_cc_neg_control", "mean")
    s_neg = _col("realspace_cc_neg_control", "std")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(x, y_pos, marker="o", linewidth=1.7, label="corr(map, control)")
    if np.any(np.isfinite(s_pos) & (s_pos > 0.0)):
        ax.fill_between(x, y_pos - np.where(np.isfinite(s_pos), s_pos, 0.0), y_pos + np.where(np.isfinite(s_pos), s_pos, 0.0), alpha=0.20)
    ax.plot(x, y_neg, marker="s", linewidth=1.7, label="corr(map, -control)")
    if np.any(np.isfinite(s_neg) & (s_neg > 0.0)):
        ax.fill_between(x, y_neg - np.where(np.isfinite(s_neg), s_neg, 0.0), y_neg + np.where(np.isfinite(s_neg), s_neg, 0.0), alpha=0.20)
    _style_axes(ax=ax, xlabel="Degradation fraction", ylabel="Real-space correlation", title=("Signed-control RSCC: %s%s" % (str(mode), (" | %s" % str(pdb_id)) if pdb_id is not None else "")))
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--csv_in", required=True)
    p.add_argument("--out_root", default=DEFAULT_OUT_ROOT,
                   help="Output directory root. Default: %s" % str(DEFAULT_OUT_ROOT))
    p.add_argument("--nproc", type=int, default=8)
    p.add_argument("--seed_master", type=int, default=1337)
    p.add_argument("--dm_score_dmax", type=float, default=DEFAULTS["dm_score_dmax"])
    p.add_argument("--dm_score_dmin", type=float, default=DEFAULTS["dm_score_dmin"])
    p.add_argument("--debug_print_dm_transcript", action="store_true")
    p.add_argument("--keep_temp", action="store_true")
    p.add_argument("--checkpoint_every", type=int, default=50)
    p.add_argument("--maxtasksperchild", type=int, default=1,
                   help="Recycle worker processes after this many tasks to limit memory growth")
    p.add_argument("--round_chunk_size", type=int, default=50,
                   help="Number of rounds to process per chunk within each degradation fraction")
    p.add_argument("--resume_from_checkpoints", action="store_true", default=True,
                   help="Resume by skipping completed rounds already present in per-mode CSVs")
    p.add_argument("--no_resume_from_checkpoints", dest="resume_from_checkpoints", action="store_false")
    p.add_argument("--log_file", default=None)
    p.add_argument("--skeleton_helper", default=None,
                   help="Path to the stage-40 skeleton proxy helper (recommended: ./40_api_skeleton_proxy.py).")
    p.add_argument("--pc44_script", default=None,
                   help="Deprecated alias for --skeleton_helper.")
    p.add_argument("--python3_exe", default="python3",
                   help="Python 3 executable for running the skeleton helper and MTZ writer helper.")
    p.add_argument("--mtz_writer_helper", default=None,
                   help="Path to the stage-40 Autobuild MTZ writer helper (recommended: ./40_api_autobuild_mtz_writer.py).")
    p.add_argument("--api50_mtz_writer_script", default=None,
                   help="Deprecated alias for --mtz_writer_helper.")
    p.add_argument("--mtz_helper_script", default=None,
                   help="Deprecated alias for --mtz_writer_helper.")
    p.add_argument("--skeleton_target_threshold", type=float, default=1.2)
    p.add_argument("--skeleton_threshold_values", default="1.2")
    p.add_argument("--skeleton_sample_rate", type=float, default=3.0)
    p.add_argument("--skeleton_prune_tip_iterations", type=int, default=3)
    p.add_argument("--skeleton_edge_connectivity", type=int, default=26)
    p.add_argument("--skeleton_map_kind", default="fom_weighted")
    p.add_argument("--degradation_modes", default="uniform_random,resolution_low_to_high,amplitude_descending")
    p.add_argument("--degradation_fractions", default="0.00,0.02,0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00")
    p.add_argument("--n_rounds", type=int, default=5)
    p.add_argument("--map_resolution_factor", type=float, default=0.25,
                   help="Resolution factor used for FFT map generation in real-space CC calculations.")
    p.add_argument("--task_timeout_sec", type=float, default=180.0,
                   help="Per degradation trial wall-clock timeout in seconds.")
    p.add_argument("--disable_summary_plots", action="store_true",
                   help="Skip writing per-mode 2x3 PNG summary panels.")
    p.add_argument("--sample_ccp4_maps", type=int, default=5,
                   help="Maximum number of CCP4 maps to export for each degradation fraction, per mode.")
    return p.parse_args(argv)


def main(argv=None):
    global _LOG_FH

    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    if sgtbx is None:
        raise Sorry("sgtbx not available in phenix.python.")
    if flex is None:
        raise Sorry("flex not available in phenix.python.")

    csv_in = os.path.abspath(os.path.expanduser(args.csv_in))
    out_root_input = os.path.abspath(os.path.expanduser(args.out_root))
    out_root = os.path.join(out_root_input, "40_degradation_master_%06d" % int(args.seed_master))
    _mkdir_p(out_root)

    if args.log_file is None:
        log_file = configure_logging(out_root=out_root, run_prefix=RUN_PREFIX)
    else:
        log_file = os.path.abspath(os.path.expanduser(args.log_file))
    log("Run prefix: %s" % str(RUN_PREFIX))
    log("Command line: %s" % " ".join(argv if argv is not None else sys.argv[1:]))
    log("csv_in=%s" % str(csv_in))
    log("out_root=%s" % str(out_root))
    log("Run log file: %s" % str(log_file))
    log("nproc=%d | seed_master=%d | task_timeout_sec=%.1f" % (int(args.nproc), int(args.seed_master), float(args.task_timeout_sec)))
    skeleton_helper = args.skeleton_helper
    if skeleton_helper is None or str(skeleton_helper).strip() == "":
        skeleton_helper = args.pc44_script
    if skeleton_helper is None or str(skeleton_helper).strip() == "":
        raise Sorry("Please provide --skeleton_helper (preferred) or --pc44_script (deprecated alias).")
    args.skeleton_helper = skeleton_helper

    mtz_writer_helper = args.mtz_writer_helper
    if mtz_writer_helper is None or str(mtz_writer_helper).strip() == "":
        mtz_writer_helper = args.api50_mtz_writer_script
    if mtz_writer_helper is None or str(mtz_writer_helper).strip() == "":
        mtz_writer_helper = args.mtz_helper_script
    args.mtz_writer_helper = mtz_writer_helper

    log("dm_score_window=dmax %.1f dmin %.1f" % (float(args.dm_score_dmax), float(args.dm_score_dmin)))
    log("Skeleton helper: %s" % str(args.skeleton_helper))
    log("MTZ writer helper: %s" % str(args.mtz_writer_helper if args.mtz_writer_helper else DEFAULT_API50_MTZ_WRITER))

    temp_root = os.path.join(out_root, "40_degradation_temp")
    tables_root = os.path.join(out_root, "40_tables")
    _mkdir_p(temp_root)
    _mkdir_p(tables_root)

    modes = _parse_mode_list_csv(args.degradation_modes)
    fractions = _parse_float_list_csv(args.degradation_fractions)
    log("Degradation modes: %s" % ", ".join([str(m) for m in modes]))
    log("Degradation fractions: %s" % ", ".join(["%.4f" % float(x) for x in fractions]))
    log("Rounds per non-zero fraction: %d" % int(args.n_rounds))
    log("Per-trial timeout: %.1f s" % float(args.task_timeout_sec))

    df = pd.read_csv(csv_in)
    df = ensure_fom_k2_atten_in_dataframe(df)
    pdb_id = _detect_pdb_id_from_df_or_path(df=df, csv_path=csv_in)
    log("PDB ID / dataset label: %s" % str(pdb_id))
    _require_columns(df, ["H", "K", "L", "CENTRIC", "dHKL"], "csv_in")
    if DEFAULTS["fom_col"] not in df.columns:
        raise Sorry("Missing required control FOM column in csv_in: %s" % str(DEFAULTS["fom_col"]))
    _require_columns(df, [DEFAULTS["control_phib_col"]], "csv_in")

    sol, solsrc = _get_solvent_content_from_csv(df, default_solvent_content=None)
    log("Solvent content = %.4f (source: %s)" % (float(sol), str(solsrc)))

    sg_str = _get_spacegroup_from_csv(df)
    if not sg_str:
        raise Sorry("Space group not found in CSV.")
    log("Space group: %s" % str(sg_str))
    sgi = sgtbx.space_group_info(symbol=str(sg_str))

    cent = df["CENTRIC"].astype(int).values
    n_centric = int(np.sum(cent == 1))
    n_acentric = int(np.sum(cent == 0))
    log("Reflection classes in full CSV: centric=%d acentric=%d total=%d" % (n_centric, n_acentric, int(df.shape[0])))

    log("Computing centric phi0(h) via sgtbx reversing ops...")
    phi0_all, n_none, n_multi = _compute_phi0_for_centrics_sgtbx(df=df, space_group_info=sgi)
    log("Reversing-op match stats: none=%d multi=%d" % (int(n_none), int(n_multi)))
    _validate_centric_supports_or_raise(
        df=df,
        phi0_all=phi0_all,
        ph_vec=df[DEFAULTS["control_phib_col"]].astype(float).values % 360.0,
        tol_deg=1e-3,
        max_report=20,
    )
    log("Control phases validated against centric supports.")

    dmin_score = float(args.dm_score_dmin)
    dmax_score = float(args.dm_score_dmax)
    mask_score = (df["dHKL"].astype(float).values >= float(dmin_score)) & (df["dHKL"].astype(float).values <= float(dmax_score))
    idx_score = np.where(mask_score)[0].astype(int)
    if idx_score.size < 200:
        raise Sorry("Too few reflections in DM scoring window [%g, %g] A: n=%d" % (dmax_score, dmin_score, int(idx_score.size)))
    df_score = df.iloc[idx_score].copy()
    idx_score_acentric = idx_score[df.iloc[idx_score]["CENTRIC"].astype(int).values == 0]
    idx_score_centric = idx_score[df.iloc[idx_score]["CENTRIC"].astype(int).values == 1]
    log("Scoring/degradation window: d in [%.2f, %.2f] A -> total=%d | acentric=%d | centric=%d" % (
        float(dmax_score), float(dmin_score), int(idx_score.size), int(idx_score_acentric.size), int(idx_score_centric.size)
    ))

    amplitude_col = _detect_amplitude_column(df_score)
    if amplitude_col is None:
        log("No amplitude-like column detected for amplitude-descending mode. That mode will fall back to randomized order.")
    else:
        log("Amplitude column for amplitude-descending mode: %s" % str(amplitude_col))

    cs_score, fp_sigfp_score, phib0_score, fom_ma_obs_score, hl0_score, df_use_score = denmod.build_cctbx_arrays_from_csv(
        df=df_score,
        phi_col=str(DEFAULTS["control_phib_col"]),
        fom_col=str(DEFAULTS["fom_col"]),
        hl_cols=("HLA", "HLB", "HLC", "HLD"),
        force_recompute_hl=True,
        kappa_max=float(DEFAULTS["kappa_max"]),
        out=sys.stderr,
    )

    cs_full, fp_sigfp_full, phib0_full, fom_ma_obs_full, hl0_full, df_use_full = denmod.build_cctbx_arrays_from_csv(
        df=df,
        phi_col=str(DEFAULTS["control_phib_col"]),
        fom_col=str(DEFAULTS["fom_col"]),
        hl_cols=("HLA", "HLB", "HLC", "HLD"),
        force_recompute_hl=True,
        kappa_max=float(DEFAULTS["kappa_max"]),
        out=sys.stderr,
    )

    control_phases_full = np.asarray(df[str(DEFAULTS["control_phib_col"])].astype(float).values % 360.0, dtype=float)
    control_phases_score = control_phases_full[idx_score]
    fom_control_full_vals = pd.to_numeric(df[str(DEFAULTS["fom_col"])], errors="coerce").to_numpy(dtype=float)
    fom_control_score_vals = pd.to_numeric(df_score[str(DEFAULTS["fom_col"])], errors="coerce").to_numpy(dtype=float)
    fom_control_ma_score = fom_ma_obs_score.customized_copy(data=_to_flex_double(fom_control_score_vals))

    score_params = dict(
        r_good=float(DEFAULTS["score_r_good"]),
        r_bad=float(DEFAULTS["score_r_bad"]),
        g_mode=str(DEFAULTS["score_g_mode"]),
        g_k=float(DEFAULTS["score_g_k"]),
        w_outer_endpoint=float(DEFAULTS["w_outer_endpoint"]),
        w_outer_pull=float(DEFAULTS["w_outer_pull"]),
        w_outer_tail=float(DEFAULTS["w_outer_tail"]),
        alpha_penalty=float(DEFAULTS["alpha_penalty"]),
        w_cc=float(DEFAULTS["w_cc"]),
        w_r=float(DEFAULTS["w_r"]),
        w_fom=float(DEFAULTS["w_fom"]),
        dcc_scale=float(DEFAULTS["dcc_scale"]),
        dr_scale=float(DEFAULTS["dr_scale"]),
        dfom_scale=float(DEFAULTS["dfom_scale"]),
        w_dcc=float(DEFAULTS["w_dcc"]),
        w_dr=float(DEFAULTS["w_dr"]),
        w_dfom=float(DEFAULTS["w_dfom"]),
        tail_frac=float(DEFAULTS["tail_frac"]),
        tail_min_cycles=int(DEFAULTS["tail_min_cycles"]),
        w_tail_endpoint=float(DEFAULTS["w_tail_endpoint"]),
        w_tail_stability=float(DEFAULTS["w_tail_stability"]),
        cc_mad_scale=float(DEFAULTS["cc_mad_scale"]),
        r_mad_scale=float(DEFAULTS["r_mad_scale"]),
        fom_mad_scale=float(DEFAULTS["fom_mad_scale"]),
        phase_vol_scale_deg=float(DEFAULTS["phase_vol_scale_deg"]),
        flip_rate_scale=float(DEFAULTS["flip_rate_scale"]),
        w_cc_stab=float(DEFAULTS["w_cc_stab"]),
        w_r_stab=float(DEFAULTS["w_r_stab"]),
        w_fom_stab=float(DEFAULTS["w_fom_stab"]),
        w_cc_mon=float(DEFAULTS["w_cc_mon"]),
        w_r_mon=float(DEFAULTS["w_r_mon"]),
        w_fom_mon=float(DEFAULTS["w_fom_mon"]),
        w_phase_stab=float(DEFAULTS["w_phase_stab"]),
        w_flip_stab=float(DEFAULTS["w_flip_stab"]),
        w_cc_reversal_pen=float(DEFAULTS["w_cc_reversal_pen"]),
        w_r_reversal_pen=float(DEFAULTS["w_r_reversal_pen"]),
        w_fom_reversal_pen=float(DEFAULTS["w_fom_reversal_pen"]),
        w_phase_vol_pen=float(DEFAULTS["w_phase_vol_pen"]),
        w_flip_rate_pen=float(DEFAULTS["w_flip_rate_pen"]),
        w_low_nref_pen=float(DEFAULTS["w_low_nref_pen"]),
        w_large_jump_pen=float(DEFAULTS["w_large_jump_pen"]),
        penalty_phase_vol_scale_deg=float(DEFAULTS["penalty_phase_vol_scale_deg"]),
        penalty_flip_rate_scale=float(DEFAULTS["penalty_flip_rate_scale"]),
        low_nref_soft_min=float(DEFAULTS["low_nref_soft_min"]),
        large_jump_scale_deg=float(DEFAULTS["large_jump_scale_deg"]),
        fom_std_eps=float(DEFAULTS["fom_std_eps"]),
    )

    dm_params_base = dict(
        mask_cycles=int(DEFAULTS["mask_cycles"]),
        minor_cycles=int(DEFAULTS["minor_cycles"]),
        rad_mask=float(DEFAULTS["rad_mask"]),
        rad_wang=DEFAULTS["rad_wang"],
        mask_type=DEFAULTS["mask_type"],
        clean_up=bool(DEFAULTS["clean_up"]),
        verbose=bool(DEFAULTS["verbose"]),
        no_write_files=bool(DEFAULTS["no_write_files"]),
        temp_dir=temp_root,
    )

    log("Building control map in the 20-3.5 A calibration window for real-space CC tracking...")
    control_map_flat_score = _build_control_map_flat_score(
        fp_sigfp_score=fp_sigfp_score,
        phib0_score=phib0_score,
        ph_control_score=control_phases_score,
        map_resolution_factor=float(args.map_resolution_factor),
    )
    log("Control map grid size (flattened): %d" % int(control_map_flat_score.size))

    control_dm_temp = os.path.join(temp_root, "control_reference_dm")
    res_control = _run_dm_and_score_hybrid(
        fp_sigfp_score=fp_sigfp_score,
        phib_score=phib0_score.customized_copy(data=_to_flex_double(control_phases_score)),
        fom_score=fom_control_ma_score,
        hl_score=hl0_score.customized_copy(data=_hl_from_phib_fom(control_phases_score, fom_control_score_vals, float(DEFAULTS["kappa_max"]))),
        solvent_content=float(sol),
        dm_params=dict(dm_params_base, temp_dir=control_dm_temp),
        fom_first_mean=float(np.nanmean(fom_control_score_vals)),
        fom_in_std=float(np.nanstd(fom_control_score_vals)),
        score_params=score_params,
        debug_print_dm_transcript=False,
        debug_transcript_file=None,
    )
    if os.path.isdir(control_dm_temp) and (not args.keep_temp):
        _rm_tree(control_dm_temp)

    control_skel_temp = os.path.join(temp_root, "control_reference_skeleton")
    _mkdir_p(control_skel_temp)
    control_skel = _run_skeleton_proxy_for_seed(
        df_base=df,
        ph_full=control_phases_full,
        fom_full=fom_control_full_vals,
        temp_root=control_skel_temp,
        score_params=dict(
            python3_exe=args.python3_exe,
            pc44_script=args.skeleton_helper,
            skeleton_sample_rate=float(args.skeleton_sample_rate),
            skeleton_map_kind=str(args.skeleton_map_kind),
            skeleton_threshold_values=str(args.skeleton_threshold_values),
            skeleton_target_threshold=float(args.skeleton_target_threshold),
            skeleton_prune_tip_iterations=int(args.skeleton_prune_tip_iterations),
            skeleton_edge_connectivity=int(args.skeleton_edge_connectivity),
            skeleton_dmin=float(args.dm_score_dmin),
            skeleton_dmax=float(args.dm_score_dmax),
            task_timeout_sec=3600.0,
        ),
    )
    control_skel = _normalize_skeleton_metrics_for_seed(control_skel)
    if not bool(control_skel.get("skeleton_ok", False)):
        msg = "CONTROL skeleton/proxy step failed"
        err = control_skel.get("skeleton_error", control_skel.get("error", "unknown_error"))
        log("%s: %s" % (msg, str(err)))
        if str(control_skel.get("pc44_command_log", "")):
            log("PC44 command log: %s" % str(control_skel.get("pc44_command_log")))
        if str(control_skel.get("pc44_stdout_log", "")):
            log("PC44 stdout/stderr log: %s" % str(control_skel.get("pc44_stdout_log")))
        if str(control_skel.get("pc44_json_summary", "")):
            log("PC44 JSON summary: %s" % str(control_skel.get("pc44_json_summary")))
        if str(control_skel.get("pc44_proxy_csv", "")):
            log("PC44 proxy CSV: %s" % str(control_skel.get("pc44_proxy_csv")))
        raise Sorry("%s: %s" % (msg, str(err)))
    if os.path.isdir(control_skel_temp) and (not args.keep_temp):
        _rm_tree(control_skel_temp)

    freer_flag_ma_full = None
    freer_col = _detect_freer_flag_column(df)
    if freer_col is not None:
        try:
            freer_vals_full = pd.to_numeric(df[str(freer_col)], errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
            freer_flag_ma_full = fp_sigfp_full.customized_copy(data=flex.int([int(x) for x in freer_vals_full]))
            log("FreeR flag column detected for MTZ export: %s" % str(freer_col))
        except Exception as e:
            freer_flag_ma_full = None
            log("WARNING: failed to build FreeR miller array from column %s: %s" % (str(freer_col), str(e)))
    else:
        log("WARNING: No FreeR flag column detected; exported MTZs will omit FreeR_flag.")

    control_basin = _compute_current_basin_score(dm_metrics=res_control, skel_metrics=control_skel)
    log("CONTROL | BasinScore=%.3f | R=%.4f | FOM=%.4f | LCF_AUC=%.4f | Endpoint=%.4f | n_comp=%s | mean_degree=%.4f | skeleton_ok=%s" % (
        float(control_basin.get("BasinScore", np.nan)),
        float(res_control.get("R_factor_FC_vs_FP", np.nan)),
        float(res_control.get("DM_mean_FOM", np.nan)),
        float(control_skel.get("best_largest_component_fraction_auc", np.nan)),
        float(control_skel.get("best_endpoint_fraction_at_target", np.nan)),
        str(control_skel.get("best_n_connected_components_at_target", "NA")),
        float(control_skel.get("best_mean_degree_at_target", np.nan)),
        str(bool(control_skel.get("skeleton_ok", False))),
    ))

    shared = dict(
        df=df,
        idx_score=idx_score,
        idx_score_acentric=np.asarray(idx_score_acentric, dtype=int),
        idx_score_centric=np.asarray(idx_score_centric, dtype=int),
        n_reflections_score_window=int(idx_score.size),
        n_acentric_score_window=int(idx_score_acentric.size),
        n_centric_score_window=int(idx_score_centric.size),
        control_phases_full=control_phases_full,
        control_phases_score=control_phases_score,
        control_map_flat_score=control_map_flat_score,
        fp_sigfp_score=fp_sigfp_score,
        phib0_score=phib0_score,
        hl0_score=hl0_score,
        fom_control_ma_score=fom_control_ma_score,
        fom_control_score_vals=fom_control_score_vals,
        fom_control_mean_score=float(np.nanmean(fom_control_score_vals)),
        fom_control_std_score=float(np.nanstd(fom_control_score_vals)),
        fom_control_full_vals=fom_control_full_vals,
        score_params=score_params,
        sol=float(sol),
        kappa_max=float(DEFAULTS["kappa_max"]),
        dm_params_base=dm_params_base,
        degradation_temp_root=temp_root,
        debug_print_dm_transcript=bool(args.debug_print_dm_transcript),
        keep_temp=bool(args.keep_temp),
        python3_exe=args.python3_exe,
        pc44_script=args.skeleton_helper,
        skeleton_target_threshold=float(args.skeleton_target_threshold),
        skeleton_threshold_values=str(args.skeleton_threshold_values),
        skeleton_sample_rate=float(args.skeleton_sample_rate),
        skeleton_prune_tip_iterations=int(args.skeleton_prune_tip_iterations),
        skeleton_edge_connectivity=int(args.skeleton_edge_connectivity),
        skeleton_map_kind=str(args.skeleton_map_kind),
        skeleton_dmin=float(args.dm_score_dmin),
        skeleton_dmax=float(args.dm_score_dmax),
        map_resolution_factor=float(args.map_resolution_factor),
        amplitude_col=amplitude_col,
        task_timeout_sec=float(args.task_timeout_sec),
        pdb_id=str(pdb_id),
        fp_sigfp_full=fp_sigfp_full,
        phib0_full=phib0_full,
        fom_ma_obs_full=fom_ma_obs_full,
        hl0_full=hl0_full,
        freer_flag_ma_full=freer_flag_ma_full,
        df_full=df.copy(),
        mtz_writer_helper=str(args.mtz_writer_helper or args.api50_mtz_writer_script or args.mtz_helper_script or ""),
    )

    tasks_by_mode = _build_degradation_tasks(
        modes=modes,
        fractions=fractions,
        n_rounds=int(args.n_rounds),
        seed_master=int(args.seed_master),
    )

    checkpoints_root = os.path.join(out_root, "40_checkpoints")
    _mkdir_p(checkpoints_root)

    for mode in modes:
        mode_tasks_all = tasks_by_mode[mode]
        csv_out = os.path.join(tables_root, "40_degradation__%s.csv" % str(mode))
        manifest_csv = os.path.join(tables_root, "40_degradation__%s__ccp4_manifest.csv" % str(mode))
        mode_checkpoint_json = os.path.join(checkpoints_root, "40_degradation__%s__progress.json" % str(mode))

        log("Starting mode %s (%s) with %d total scheduled tasks..." % (str(mode), _mode_pretty_label(mode), int(len(mode_tasks_all))))
        t0_mode = time.time()

        for frac in fractions:
            frac = float(frac)
            rounds_here = 1 if abs(frac) < 1e-12 else int(max(1, args.n_rounds))
            tasks_this_fraction = [t for t in mode_tasks_all if round(float(t["degrade_fraction"]), 12) == round(frac, 12)]
            existing_rows_mode = _rows_for_mode_from_csv(csv_out, mode)
            existing_rounds = _existing_round_idx_set(existing_rows_mode, mode, frac)

            if bool(args.resume_from_checkpoints):
                tasks_to_run = [t for t in tasks_this_fraction if int(t["round_idx"]) not in existing_rounds]
            else:
                tasks_to_run = list(tasks_this_fraction)

            log("MODE=%s | fraction=%.4f | expected_rounds=%d | existing=%d | pending=%d" % (
                str(mode), float(frac), int(rounds_here), int(len(existing_rounds)), int(len(tasks_to_run))
            ))

            fraction_t0 = time.time()
            rows_new = []
            progress_state = {}

            if len(tasks_to_run) > 0:
                chunk_size = max(1, int(args.round_chunk_size))
                total_done_fraction = 0
                n_chunks = int((len(tasks_to_run) + chunk_size - 1) // chunk_size)
                for chunk_idx, chunk_start in enumerate(range(0, len(tasks_to_run), chunk_size), start=1):
                    chunk_tasks = tasks_to_run[chunk_start: chunk_start + chunk_size]
                    if len(chunk_tasks) == 0:
                        continue
                    log("CHUNK | mode=%s | frac=%.4f | chunk=%d/%d | tasks=%d | round_idx=%d..%d" % (
                        str(mode), float(frac), int(chunk_idx), int(n_chunks), int(len(chunk_tasks)),
                        int(chunk_tasks[0]["round_idx"]), int(chunk_tasks[-1]["round_idx"]),
                    ))
                    rows_chunk = []
                    completed_rounds_chunk = set()
                    processes_this_chunk = min(max(1, int(args.nproc)), len(chunk_tasks))
                    pool = mp.Pool(
                        processes=processes_this_chunk,
                        initializer=_init_worker,
                        initargs=(shared,),
                        maxtasksperchild=max(1, int(args.maxtasksperchild)),
                    )
                    iterator = None
                    heartbeat_sec = max(5.0, min(30.0, float(args.task_timeout_sec) if float(args.task_timeout_sec) > 0 else 10.0))
                    stall_timeout_sec = max(60.0, 3.0 * float(args.task_timeout_sec)) if float(args.task_timeout_sec) > 0 else 120.0
                    last_progress_ts = time.time()
                    chunk_aborted = False
                    try:
                        iterator = pool.imap_unordered(_degradation_worker_top, chunk_tasks, chunksize=1)
                        while len(rows_chunk) < len(chunk_tasks):
                            try:
                                row = iterator.next(timeout=heartbeat_sec)
                                rows_chunk.append(row)
                                try:
                                    completed_rounds_chunk.add(int(row.get("round_idx", -1)))
                                except Exception:
                                    pass
                                total_done_fraction += 1
                                last_progress_ts = time.time()
                                _print_progress(done=total_done_fraction, total=len(tasks_to_run), t0=fraction_t0, prefix=("%s f=%.4f " % (str(mode), float(frac))), force=False, state=progress_state)
                                log(_format_degradation_terminal_line(row))
                                if str(row.get("error", "")).strip() != "":
                                    log("  detail: %s" % str(row.get("error", "")))
                            except mp.TimeoutError:
                                idle_sec = time.time() - last_progress_ts
                                log("HEARTBEAT | mode=%s | frac=%.3f | chunk=%d/%d | completed_in_chunk=%d/%d | idle_for=%.1fs" % (
                                    str(mode), float(frac), int(chunk_idx), int(n_chunks), int(len(rows_chunk)), int(len(chunk_tasks)), float(idle_sec),
                                ))
                                if idle_sec >= stall_timeout_sec:
                                    log("STALL watchdog triggered; terminating pool for mode=%s frac=%.3f chunk=%d after %.1fs without progress." % (
                                        str(mode), float(frac), int(chunk_idx), float(idle_sec),
                                    ))
                                    chunk_aborted = True
                                    break
                        if chunk_aborted:
                            try:
                                pool.terminate()
                            except Exception:
                                pass
                            try:
                                pool.join()
                            except Exception:
                                pass
                            missing_tasks = [tt for tt in chunk_tasks if int(tt.get("round_idx", -1)) not in completed_rounds_chunk]
                            for tt in missing_tasks:
                                row = _degradation_row_failure(task=tt, shared=shared, error_msg=("chunk_stall_timeout_after_%.1fs" % float(stall_timeout_sec)), wall_sec=np.nan)
                                rows_chunk.append(row)
                                total_done_fraction += 1
                                _print_progress(done=total_done_fraction, total=len(tasks_to_run), t0=fraction_t0, prefix=("%s f=%.4f " % (str(mode), float(frac))), force=False, state=progress_state)
                                log(_format_degradation_terminal_line(row))
                                if str(row.get("error", "")).strip() != "":
                                    log("  detail: %s" % str(row.get("error", "")))
                        else:
                            pool.close()
                            pool.join()
                    finally:
                        try:
                            pool.terminate()
                        except Exception:
                            pass

                    rows_chunk = sorted(rows_chunk, key=lambda r: (float(r.get("degrade_fraction", np.nan)), int(r.get("round_idx", -1)), str(r.get("mode", ""))))
                    if len(rows_chunk) > 0:
                        n_appended_chunk = _append_degradation_csv_rows(csv_out, rows_chunk)
                        rows_new.extend(rows_chunk)
                        log("Appended %d new chunk rows to mode CSV: %s" % (int(n_appended_chunk), str(csv_out)))
                    chunk_completed_rounds = _existing_round_idx_set(_rows_for_mode_from_csv(csv_out, mode), mode, frac)
                    _write_fraction_checkpoint_json(
                        out_json=os.path.join(checkpoints_root, "40_degradation__%s__frac_%0.4f__chunk_%04d.json" % (str(mode), float(frac), int(chunk_idx))),
                        payload=dict(
                            mode=str(mode),
                            degrade_fraction=float(frac),
                            chunk_index=int(chunk_idx),
                            n_chunks=int(n_chunks),
                            chunk_size=int(chunk_size),
                            chunk_round_start=int(chunk_tasks[0]["round_idx"]),
                            chunk_round_end=int(chunk_tasks[-1]["round_idx"]),
                            completed_rounds=int(len(chunk_completed_rounds)),
                            expected_rounds=int(rounds_here),
                            csv_path=str(csv_out),
                            wall_sec_fraction=float(time.time() - fraction_t0),
                        ),
                    )
                _print_progress(done=len(tasks_to_run), total=len(tasks_to_run), t0=fraction_t0, prefix=("%s f=%.4f " % (str(mode), float(frac))), force=True, state=progress_state)
            else:
                log("Skipping execution for mode=%s fraction=%.4f because all rounds are already present; no refresh/export needed." % (str(mode), float(frac)))
                continue

            # refresh summaries and plots from on-disk CSV so memory stays bounded
            rows_mode_all = _refresh_mode_artifacts_from_csv(
                mode=mode,
                csv_out=csv_out,
                tables_root=tables_root,
                pdb_id=pdb_id,
                disable_summary_plots=bool(args.disable_summary_plots),
            )

            # export sample maps only for this fraction, then update manifest incrementally
            if int(args.sample_ccp4_maps) > 0:
                try:
                    maps_dir = os.path.join(out_root, "40_ccp4_maps", str(mode), ("frac_%0.4f" % float(frac)))
                    exported_rows = _export_sample_ccp4_maps_for_mode(
                        mode=mode,
                        tasks=tasks_this_fraction,
                        shared=shared,
                        out_dir=maps_dir,
                        maps_per_fraction=int(args.sample_ccp4_maps),
                    )
                    if len(exported_rows) > 0:
                        _refresh_ccp4_manifest(mode=mode, manifest_csv=manifest_csv, exported_rows=exported_rows)
                        log("Updated CCP4/MTZ manifest: %s" % str(manifest_csv))
                        log("Exported %d sampled states (CCP4 and/or MTZ) for mode %s at fraction %.4f" % (int(len(exported_rows)), str(mode), float(frac)))
                    else:
                        log("No CCP4/MTZ states exported for mode %s at fraction %.4f" % (str(mode), float(frac)))
                except Exception as e:
                    log("WARNING: failed CCP4 export for %s fraction %.4f: %s" % (str(mode), float(frac), str(e)))

            completed_rounds_after = _existing_round_idx_set(_rows_for_mode_from_csv(csv_out, mode), mode, frac)
            checkpoint_payload = dict(
                mode=str(mode),
                mode_label=_mode_pretty_label(mode),
                degrade_fraction=float(frac),
                expected_rounds=int(rounds_here),
                completed_rounds=int(len(completed_rounds_after)),
                pending_rounds=int(max(0, int(rounds_here) - int(len(completed_rounds_after)))),
                csv_path=str(csv_out),
                manifest_csv=str(manifest_csv),
                wall_sec_fraction=float(time.time() - fraction_t0),
                resume_from_checkpoints=bool(args.resume_from_checkpoints),
                seed_master=int(args.seed_master),
                n_rounds_requested=int(args.n_rounds),
            )
            _write_fraction_checkpoint_json(
                out_json=os.path.join(checkpoints_root, "40_degradation__%s__frac_%0.4f.json" % (str(mode), float(frac))),
                payload=checkpoint_payload,
            )

            mode_payload = dict(
                mode=str(mode),
                csv_path=str(csv_out),
                manifest_csv=str(manifest_csv),
                fractions=[float(x) for x in fractions],
                seed_master=int(args.seed_master),
                n_rounds_requested=int(args.n_rounds),
                updated_after_fraction=float(frac),
                wall_sec_mode_elapsed=float(time.time() - t0_mode),
                total_rows_in_mode_csv=int(len(rows_mode_all)),
            )
            _write_fraction_checkpoint_json(out_json=mode_checkpoint_json, payload=mode_payload)

            try:
                import gc
                gc.collect()
            except Exception:
                pass

        log("Completed mode %s in %.1f s" % (str(mode), float(time.time() - t0_mode)))


    final_summary_rows = []
    for mode in modes:
        csv_out = os.path.join(tables_root, "40_degradation__%s.csv" % str(mode))
        rows_mode = _rows_for_mode_from_csv(csv_out, mode)
        final_summary_rows.append(dict(
            mode=str(mode),
            mode_label=_mode_pretty_label(mode),
            n_rows=int(len(rows_mode)),
            csv_path=str(csv_out),
        ))
    final_summary_csv = os.path.join(out_root, "40_BasinScore_degradation_summary.csv")
    pd.DataFrame(final_summary_rows).to_csv(final_summary_csv, index=False)
    log("Summary CSV: %s" % str(final_summary_csv))
    log("Completed controlled degradation protocol for %d mode(s)." % int(len(modes)))


if __name__ == "__main__":
    main()