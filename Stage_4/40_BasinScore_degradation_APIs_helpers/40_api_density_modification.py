#!/usr/bin/env phenix.python
# -*- coding: utf-8 -*-
"""
Callable DM API aligned to the PC43 protocol, renamed for integration with stage 40.

Py2-safe version for import under phenix.python.
"""
from __future__ import division, print_function

import os
import sys
import shutil
import traceback

try:
    from io import BytesIO as _BytesBuffer
except Exception:
    try:
        from cStringIO import StringIO as _BytesBuffer
    except Exception:
        from StringIO import StringIO as _BytesBuffer

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import phenix_like_density_modification_from_csv_py27 as denmod

FOM_K2_ATTEN_FALLBACK = 2.0 / np.pi

PC43_DEFAULTS = dict(
    mask_cycles=5,
    minor_cycles=10,
    mask_type="histograms",
    rad_wang=None,
    rad_mask=4.0,
    verbose=False,
    no_write_files=True,
    clean_up=True,
)


def ensure_fom_k2_atten_in_dataframe(df):
    df2 = df.copy()
    if "FOM_K2_atten" not in df2.columns:
        df2["FOM_K2_atten"] = float(FOM_K2_ATTEN_FALLBACK)
    return df2


def _safe_bool(value, default=False):
    try:
        if value is None:
            return bool(default)
        return bool(value)
    except Exception:
        return bool(default)


def _safe_float(value, default=np.nan):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value, default=0):
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _mkdir_p(path):
    if path and (not os.path.isdir(path)):
        os.makedirs(path)


def run_density_modification_protocol(fp_sigfp,
                                      phib_ma,
                                      fom_ma,
                                      hl_ma,
                                      solvent_content,
                                      dm_params=None,
                                      temp_dir=None):
    """
    Execute density modification using PC43-aligned defaults.

    Parameters are array-level so PC50 can keep its current orchestration and output style.
    Returns a dict with keys:
        ok, cmn, transcript, error, used_mask_type
    """
    params = dict(PC43_DEFAULTS)
    if dm_params:
        for key, value in dm_params.items():
            params[key] = value
    if not params.get("mask_type"):
        params["mask_type"] = "histograms"

    local_temp_dir = temp_dir
    made_temp_dir = False
    if local_temp_dir is None:
        local_temp_dir = os.path.abspath(os.path.join(os.getcwd(), "pc43_dm_tmp"))
    if not os.path.isdir(local_temp_dir):
        _mkdir_p(local_temp_dir)
        made_temp_dir = True

    out_buf = _BytesBuffer()
    kwargs = dict(
        fp_sigfp=fp_sigfp,
        phib=phib_ma,
        fom=fom_ma,
        hendrickson_lattman=hl_ma,
        solvent_content=float(solvent_content),
        mask_cycles=_safe_int(params.get("mask_cycles"), default=5),
        minor_cycles=_safe_int(params.get("minor_cycles"), default=10),
        temp_dir=local_temp_dir,
        clean_up=False,
        mask_type=str(params.get("mask_type", "histograms")),
        rad_wang=params.get("rad_wang"),
        rad_mask=_safe_float(params.get("rad_mask"), default=4.0),
        verbose=_safe_bool(params.get("verbose"), default=False),
        out=out_buf,
    )
    if params.get("no_write_files") is not None:
        kwargs["no_write_files"] = _safe_bool(params.get("no_write_files"), default=True)

    try:
        try:
            cmn = denmod.run_density_modification(**kwargs)
        except TypeError:
            if "no_write_files" in kwargs:
                del kwargs["no_write_files"]
            cmn = denmod.run_density_modification(**kwargs)
        transcript = out_buf.getvalue()
        return dict(
            ok=True,
            cmn=cmn,
            transcript=transcript,
            error="",
            used_mask_type=str(kwargs["mask_type"]),
        )
    except Exception as exc:
        return dict(
            ok=False,
            cmn=None,
            transcript=out_buf.getvalue(),
            error="%s: %s\n%s" % (exc.__class__.__name__, str(exc), traceback.format_exc()),
            used_mask_type=str(kwargs["mask_type"]),
        )
    finally:
        if _safe_bool(params.get("clean_up"), default=True) and made_temp_dir:
            try:
                shutil.rmtree(local_temp_dir)
            except Exception:
                pass
