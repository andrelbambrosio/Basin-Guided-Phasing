#!/usr/bin/env phenix.python
# -*- coding: utf-8 -*-
"""
phenix_like_density_modification_from_csv_py27.py

Goal
----
Run Phenix/RESOLVE density modification "as close as possible" to
`phenix.density_modification` (the MTZ-driven wrapper), but using a single
per-dataset CSV as the data source.

Key points
----------
- No gemmi (Phenix py2.7 env typically lacks it)
- Crystal symmetry built via cctbx (NOT iotbx.symmetry callable)
- Uses solve_resolve.resolve_python.density_modify_in_memory (same engine)
- Attempts to mimic the MTZ wrapper behavior:
  * remove_aniso=True (default in phenix.density_modification)
  * workdir temp_denmod (user-selectable)
  * mask_cycles/minor_cycles (user-selectable; defaults match)
  * solvent_content required
  * writes no "sigma-scaled" maps (RAW maps only)
  * can write optional "raw" CCP4 maps for input/ref/DM maps

CSV requirements (minimum)
-------------------------
Reflection columns:
  H,K,L, FP, SIGFP, <PHIB>, <FOM>

Symmetry/unit cell (single-valued per file):
  SG, LENGTH_A, LENGTH_B, LENGTH_C, ANGLE_ALPHA, ANGLE_BETA, ANGLE_GAMMA

Optional:
  FreeR_flag (ignored by RESOLVE in-memory run; safe to omit)
  HL columns (HLA,HLB,HLC,HLD) -- can be supplied, or recomputed from PHIB/FOM

Output CSV
----------
Writes a CSV with density-modified:
  PHIB_DM, FOM_DM, HLA_DM, HLB_DM, HLC_DM, HLD_DM
plus copies of key identifiers/inputs for convenience.

Example
-------
phenix.python phenix_like_density_modification_from_csv_py27.py \
  --csv_in  ./04_Run_Ecalc_Binned/1cu2_rs_ecalc_binned.csv \
  --csv_out ./1cu2_denmod.csv \
  --solvent_content 0.56 \
  --phib_col PHIC_ALL_K2 \
  --fom_col  FOM_K2 \
  --hl_cols  HLA_K2,HLB_K2,HLC_K2,HLD_K2 \
  --mask_cycles 5 \
  --minor_cycles 10 \
  --temp_dir temp_denmod \
  --write_maps \
  --verbose

Notes
-----
- If you want the *full* RESOLVE transcript like the MTZ wrapper prints,
  make sure you run with `--verbose`. This script routes `out` to stdout,
  but RESOLVE itself may still be less chatty depending on build/options.
"""

from __future__ import division, print_function

import os
import sys
import argparse

import numpy as np
import pandas as pd

from libtbx.utils import Sorry
from libtbx.utils import null_out


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def require_cols(df, cols, where):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise Sorry("Missing required columns in %s: %s" % (where, missing))


def map_minus180_to_plus180_deg(x):
    x = np.asarray(x, dtype=float)
    return np.where(np.isclose(x, -180.0, atol=1e-9), 180.0, x)


def kappa_from_R_array(R):
    """
    Approximate von Mises kappa from resultant length R (0<=R<1).
    Standard piecewise approximation.
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


def compute_hl_from_phi_fom(phi_deg, fom, kappa_max):
    """
    Compute HL coefficients (HLA,HLB,HLC,HLD) from PHI (deg) + FOM,
    using a von Mises kappa approximation from R~FOM.

    This mirrors the "quick HL-on-the-fly" approach used in your CSV->MTZ code:
      HLA = kappa*cos(phi)
      HLB = kappa*sin(phi)
      HLC = 0
      HLD = 0
    """
    phi_deg = np.asarray(phi_deg, dtype=float)
    fom = np.asarray(fom, dtype=float)
    fom = np.clip(np.where(np.isfinite(fom), fom, 0.0), 0.0, 0.9999)

    kappa = np.minimum(kappa_from_R_array(R=fom), float(kappa_max))

    phi_rad = np.deg2rad(phi_deg)
    HLA = kappa * np.cos(phi_rad)
    HLB = kappa * np.sin(phi_rad)
    HLC = np.zeros_like(HLA)
    HLD = np.zeros_like(HLA)
    return HLA, HLB, HLC, HLD


def wrap_angle_deg(x):
    """Wrap degrees to (-180, 180]."""
    y = (x + 180.0) % 360.0 - 180.0
    y = np.where(np.isclose(y, -180.0, atol=1e-9), 180.0, y)
    return y


def abs_circular_diff_deg(a_deg, b_deg):
    """Absolute circular difference in degrees."""
    d = wrap_angle_deg(np.asarray(a_deg, dtype=float) - np.asarray(b_deg, dtype=float))
    return np.abs(d)


# ---------------------------------------------------------------------------
# CCP4 map writing (RAW maps only; no sigma scaling)
# ---------------------------------------------------------------------------

def write_ccp4_map_from_map_coeffs(map_coeffs, crystal_symmetry, file_name, out):
    """
    Write a raw CCP4 map from complex map coefficients.

    Notes:
    - No sigma scaling is applied (as requested).
    - Uses cctbx FFT defaults via fft_map().
    """
    from iotbx import mrcfile
    from scitbx.array_family import flex

    if map_coeffs is None or map_coeffs.data() is None or map_coeffs.size() == 0:
        print("No map coefficients to write for %s" % file_name, file=out)
        return

    fft_map = map_coeffs.fft_map(resolution_factor=0.25)
    map_data = fft_map.real_map_unpadded()

    mrcfile.write_ccp4_map(
        file_name=file_name,
        unit_cell=crystal_symmetry.unit_cell(),
        space_group=crystal_symmetry.space_group(),
        map_data=map_data,
        labels=flex.std_string(["RAW map (no sigma scaling)"])
    )
    print("Wrote RAW map: %s" % file_name, file=out)


def map_coeffs_from_fp_phi_fom(fp_sigfp, phib_deg, fom):
    """
    Construct map coefficients ~ (FP*FOM) * exp(i*phi).
    """
    if fp_sigfp is None or phib_deg is None or fom is None:
        return None
    fp = fp_sigfp
    mc = fp.array(data=fp.data() * fom.data()).phase_transfer(phase_source=phib_deg, deg=True)
    return mc


# ---------------------------------------------------------------------------
# Build cctbx arrays from CSV (NO gemmi; robust py2.7)
# ---------------------------------------------------------------------------

def build_cctbx_arrays_from_csv(df,
                               phi_col,
                               fom_col,
                               hl_cols,
                               force_recompute_hl,
                               kappa_max,
                               out):
    base_cols = [
        "H", "K", "L",
        "FP", "SIGFP",
        "SG",
        "LENGTH_A", "LENGTH_B", "LENGTH_C",
        "ANGLE_ALPHA", "ANGLE_BETA", "ANGLE_GAMMA",
    ]
    require_cols(df=df, cols=base_cols, where="CSV")

    if phi_col not in df.columns:
        raise Sorry("Missing phase column '%s' in CSV." % phi_col)
    if fom_col not in df.columns:
        raise Sorry("Missing FOM column '%s' in CSV." % fom_col)

    use = df.dropna(subset=["H", "K", "L", "FP", "SIGFP", phi_col, fom_col]).copy()
    if use.empty:
        raise Sorry("No valid rows after dropping NaNs in required columns.")

    H = use["H"].astype(int).values
    K = use["K"].astype(int).values
    L = use["L"].astype(int).values

    FP = use["FP"].astype(float).values
    SIGFP = use["SIGFP"].astype(float).values

    PHI = map_minus180_to_plus180_deg(use[phi_col].astype(float).values)

    FOM = use[fom_col].astype(float).values
    FOM = np.where(np.isfinite(FOM), FOM, 0.0)
    FOM = np.clip(FOM, 0.0, 0.9999)

    sg_str = str(use["SG"].iloc[0]).strip()
    a = float(use["LENGTH_A"].iloc[0])
    b = float(use["LENGTH_B"].iloc[0])
    c = float(use["LENGTH_C"].iloc[0])
    alpha = float(use["ANGLE_ALPHA"].iloc[0])
    beta = float(use["ANGLE_BETA"].iloc[0])
    gamma = float(use["ANGLE_GAMMA"].iloc[0])

    # ---- robust Phenix py2.7 crystal symmetry construction (NO iotbx.symmetry callable) ----
    from cctbx import crystal
    from cctbx import uctbx
    from cctbx import sgtbx
    from cctbx import miller as cctbx_miller
    from cctbx.array_family import flex

    uc = uctbx.unit_cell((a, b, c, alpha, beta, gamma))
    sgi = sgtbx.space_group_info(symbol=sg_str)
    cs = crystal.symmetry(unit_cell=uc, space_group_info=sgi)

    indices = flex.miller_index(list(zip(H.tolist(), K.tolist(), L.tolist())))
    miller_set = cctbx_miller.set(crystal_symmetry=cs, indices=indices, anomalous_flag=False)

    fp_sigfp = cctbx_miller.array(miller_set=miller_set, data=flex.double(FP.tolist()),
                                  sigmas=flex.double(SIGFP.tolist())).set_observation_type_xray_amplitude()
    phib = cctbx_miller.array(miller_set=miller_set, data=flex.double(PHI.tolist()))
    fom_ma = cctbx_miller.array(miller_set=miller_set, data=flex.double(FOM.tolist()))

    hla_col, hlb_col, hlc_col, hld_col = hl_cols
    have_hl = all([(c in use.columns) for c in [hla_col, hlb_col, hlc_col, hld_col]])

    if have_hl and (not force_recompute_hl):
        print("Using HL from CSV columns: %s" % (",".join([hla_col, hlb_col, hlc_col, hld_col])), file=out)
        HLA = use[hla_col].astype(float).values
        HLB = use[hlb_col].astype(float).values
        HLC = use[hlc_col].astype(float).values
        HLD = use[hld_col].astype(float).values
    else:
        if have_hl and force_recompute_hl:
            print("Forcing HL recompute from %s/%s (ignoring CSV HL columns)." % (phi_col, fom_col), file=out)
        else:
            print("HL columns not found (or incomplete). Computing HL from %s/%s." % (phi_col, fom_col), file=out)
        HLA, HLB, HLC, HLD = compute_hl_from_phi_fom(phi_deg=PHI, fom=FOM, kappa_max=kappa_max)

    # flex.hendrickson_lattman expects 4 flex.double arrays
    hl_data = flex.hendrickson_lattman(
        flex.double(HLA.tolist()),
        flex.double(HLB.tolist()),
        flex.double(HLC.tolist()),
        flex.double(HLD.tolist())
    )
    hendrickson_lattman = cctbx_miller.array(miller_set=miller_set, data=hl_data)

    # mimic wrapper behavior: unique + map to ASU
    fp_sigfp = fp_sigfp.unique_under_symmetry().map_to_asu()
    phib = phib.unique_under_symmetry().map_to_asu(deg=True)
    fom_ma = fom_ma.unique_under_symmetry().map_to_asu()
    hendrickson_lattman = hendrickson_lattman.unique_under_symmetry().map_to_asu()

    return cs, fp_sigfp, phib, fom_ma, hendrickson_lattman, use


# ---------------------------------------------------------------------------
# Density modification runner (calls Phenix/RESOLVE engine)
# ---------------------------------------------------------------------------

def run_density_modification(fp_sigfp,
                             phib,
                             fom,
                             hendrickson_lattman,
                             solvent_content,
                             mask_cycles,
                             minor_cycles,
                             temp_dir,
                             clean_up,
                             mask_type,
                             rad_wang,
                             rad_mask,
                             verbose,
                             out):
    """
    Run solve_resolve.resolve_python.density_modify_in_memory.run()
    in the same spirit as phenix.density_modification.

    We pass `input_text` commands to mimic wrapper defaults:
      - no_write_files
      - workdir <temp_dir>
      - protein_mask_file mask_file.map
      - ncs_mask_file ncs_mask_file.map
    """
    from solve_resolve.resolve_python import density_modify_in_memory as dmim

    # Optional anisotropy correction (matches default wrapper remove_aniso=True)
    # dmim.remove_aniso prints the same block as MTZ wrapper
    fp_sigfp = dmim.remove_aniso(fp_sigfp, params=None, out=out)

    input_text = ""
    input_text += "\nno_write_files\n"
    input_text += "\nno_create_free\nuse_all_for_test\nno_optimize_ncs\n"

    # temp_dir / workdir (matches wrapper)
    if temp_dir:
        input_text += "\nworkdir %s\n" % str(temp_dir)
        if (not os.path.isdir(temp_dir)):
            os.makedirs(temp_dir)

        # The MTZ wrapper always sets these because output_mask_file/output_ncs_mask_file have defaults.
        input_text += "\nprotein_mask_file mask_file.map\n"
        input_text += "\nncs_mask_file ncs_mask_file.map\n"

    # mask selection hints (wrapper uses mask_type phil; RESOLVE may default to Wang if none)
    if mask_type and mask_type.lower() != "none":
        if mask_type.lower() == "histograms":
            input_text += "\nuse_hist_prob\n"
        elif mask_type.lower() == "probability":
            input_text += "\nuse_prob\n"
        elif mask_type.lower() == "wang":
            input_text += "\nuse_wang\n"

    if rad_wang is not None:
        input_text += "\nwang_radius %7.2f\n" % float(rad_wang)

    # Run
    cmn = dmim.run(
        input_text=input_text,
        fp_sigfp=fp_sigfp,
        phib=phib,
        fom=fom,
        hendrickson_lattman=hendrickson_lattman,
        solvent_content=float(solvent_content),
        resolution=None,
        mask_cycles=int(mask_cycles),
        minor_cycles=int(minor_cycles),
        rad_mask=(float(rad_mask) if rad_mask is not None else None),
        temp_dir=str(temp_dir) if temp_dir else None,
        verbose=bool(verbose),
        denmod_with_model=False,
        optimize_rad_wang=False,
        rad_wang=(float(rad_wang) if rad_wang is not None else None),
        mask_type=(mask_type if mask_type is not None else None),
        nohl=False,
        out=out
    )

    if cmn is None or (not hasattr(cmn, "output_refl_db")):
        raise Sorry("Density modification failed: cmn/output_refl_db missing.")

    # Optionally cleanup temp_dir
    if clean_up and temp_dir and os.path.isdir(temp_dir):
        try:
            # phenix wrapper uses phenix.autosol.delete_dir; we keep it simple
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            print("Removed temporary directory %s" % str(temp_dir), file=out)
        except Exception as exc:
            print("Warning: could not remove temp_dir (%s): %s" % (temp_dir, exc), file=out)

    return cmn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Phenix-like density modification from a single CSV (py2.7-safe, no gemmi)."
    )
    p.add_argument("--csv_in", required=True, help="Input CSV path.")
    p.add_argument("--csv_out", required=True, help="Output CSV path.")

    p.add_argument("--solvent_content", required=True, type=float, help="Solvent content (0-1). Required.")
    p.add_argument("--phib_col", required=True, help="Column name for PHIB (degrees).")
    p.add_argument("--fom_col", required=True, help="Column name for FOM (0..1).")

    p.add_argument("--hl_cols", default=None,
                   help="Comma-separated HL columns (HLA,HLB,HLC,HLD). If omitted, HL computed from PHIB/FOM.")
    p.add_argument("--force_recompute_hl", action="store_true",
                   help="If HL columns exist, ignore them and recompute from PHIB/FOM.")

    p.add_argument("--kappa_max", type=float, default=200.0, help="Cap for kappa when computing HL (default 200).")

    p.add_argument("--mask_cycles", type=int, default=5, help="Mask cycles (default 5).")
    p.add_argument("--minor_cycles", type=int, default=10, help="Minor cycles per mask cycle (default 10).")
    p.add_argument("--rad_mask", type=float, default=4.0, help="rad_mask (default 4).")
    p.add_argument("--rad_wang", type=float, default=None, help="Optional wang_radius override (A).")
    p.add_argument("--mask_type", default=None,
                   help="Mask type hint: None|histograms|probability|wang (default None).")

    p.add_argument("--temp_dir", default="temp_denmod", help="Temp directory (default temp_denmod).")
    p.add_argument("--clean_up", action="store_true", help="Remove temp_dir after run (default False).")

    p.add_argument("--write_maps", action="store_true",
                   help="Write RAW CCP4 maps for (ref/input/DM) maps (no sigma scaling).")
    p.add_argument("--ref_phib_col", default=None,
                   help="Optional reference phase column for a 'ref' map (e.g., PHIC_ALL). If omitted, no ref map.")
    p.add_argument("--ref_fom_col", default=None,
                   help="Optional reference FOM column for a 'ref' map. If omitted, uses same FOM as input map.")

    p.add_argument("--verbose", action="store_true", help="Verbose logging (passes through to RESOLVE).")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    csv_in = os.path.abspath(os.path.expanduser(args.csv_in))
    csv_out = os.path.abspath(os.path.expanduser(args.csv_out))

    out = sys.stdout if args.verbose else sys.stdout  # keep stdout; RESOLVE will be quieter if verbose=False

    print("Input CSV : %s" % csv_in, file=out)
    print("Output CSV: %s" % csv_out, file=out)
    print("Using PHIB/FOM columns: %s / %s" % (args.phib_col, args.fom_col), file=out)
    print("Solvent content: %.4f" % float(args.solvent_content), file=out)
    print("mask_cycles=%d  minor_cycles=%d" % (int(args.mask_cycles), int(args.minor_cycles)), file=out)
    print("temp_dir=%s  clean_up=%s" % (str(args.temp_dir), bool(args.clean_up)), file=out)

    # Read CSV
    if not os.path.isfile(csv_in):
        raise Sorry("CSV not found: %s" % csv_in)
    df = pd.read_csv(csv_in)

    # HL columns handling
    if args.hl_cols is None or str(args.hl_cols).strip() == "":
        hl_cols = ("HLA", "HLB", "HLC", "HLD")  # placeholders; will compute
        print("HL columns: (none provided) -> compute HL from %s/%s" % (args.phib_col, args.fom_col), file=out)
    else:
        toks = [t.strip() for t in str(args.hl_cols).split(",") if t.strip()]
        if len(toks) != 4:
            raise Sorry("--hl_cols must have exactly 4 comma-separated labels (HLA,HLB,HLC,HLD). Got: %s" % toks)
        hl_cols = tuple(toks)
        print("Using HL columns: %s" % ",".join(hl_cols), file=out)

    # Build arrays
    cs, fp_sigfp, phib, fom, hl = (None, None, None, None, None)
    cs, fp_sigfp, phib, fom, hl, df_use = build_cctbx_arrays_from_csv(
        df=df,
        phi_col=args.phib_col,
        fom_col=args.fom_col,
        hl_cols=hl_cols,
        force_recompute_hl=bool(args.force_recompute_hl) or (args.hl_cols is None),
        kappa_max=float(args.kappa_max),
        out=out
    )

    # Run DM
    cmn = run_density_modification(
        fp_sigfp=fp_sigfp,
        phib=phib,
        fom=fom,
        hendrickson_lattman=hl,
        solvent_content=float(args.solvent_content),
        mask_cycles=int(args.mask_cycles),
        minor_cycles=int(args.minor_cycles),
        temp_dir=str(args.temp_dir) if args.temp_dir else None,
        clean_up=bool(args.clean_up),
        mask_type=args.mask_type,
        rad_wang=args.rad_wang,
        rad_mask=float(args.rad_mask) if args.rad_mask is not None else None,
        verbose=bool(args.verbose),
        out=out
    )

    # Extract outputs (miller arrays were already copied out by dmim.get_refl_arrays)
    phib_dm = cmn.phib_out_as_miller_array
    fom_dm = cmn.fom_out_as_miller_array
    hl_dm = cmn.hl_out_as_miller_array
    fp_sigfp_out = cmn.fp_sigfp_out_as_miller_array

    if phib_dm is None or fom_dm is None or hl_dm is None or fp_sigfp_out is None:
        raise Sorry("Missing DM outputs (phib/fom/hl/fp_sigfp).")

    # Compare phases (input vs DM) for quick sanity check
    # Need same indexing: use common set via match_indices
    from cctbx import miller
    match = phib.common_set(other=phib_dm).match_indices(phib_dm.common_set(other=phib))
    # safer: just take intersection using common_set
    common = phib.common_set(other=phib_dm)
    ph_in = phib.common_set(other=phib_dm)
    ph_dm = phib_dm.common_set(other=phib)

    dphi = abs_circular_diff_deg(a_deg=np.array(ph_in.data(), dtype=float),
                                 b_deg=np.array(ph_dm.data(), dtype=float))
    rms = np.sqrt(np.mean(dphi ** 2)) if dphi.size else float("nan")
    mean_abs = np.mean(dphi) if dphi.size else float("nan")
    print("RMS |Δφ| (deg): %.2f" % rms, file=out)
    print("Mean |Δφ| (deg): %.2f" % mean_abs, file=out)

    # Optional RAW maps (no sigma scaling)
    if bool(args.write_maps):
        # input map from selected phib/fom
        mc_in = map_coeffs_from_fp_phi_fom(fp_sigfp=fp_sigfp_out, phib_deg=phib, fom=fom)
        # DM map from DM phases/FOM
        mc_dm = map_coeffs_from_fp_phi_fom(fp_sigfp=fp_sigfp_out, phib_deg=phib_dm, fom=fom_dm)

        # Optional reference map
        mc_ref = None
        if args.ref_phib_col is not None and str(args.ref_phib_col).strip() != "":
            ref_phi_col = str(args.ref_phib_col).strip()
            if ref_phi_col not in df.columns:
                print("WARNING: ref_phib_col '%s' not found in CSV; skipping ref map." % ref_phi_col, file=out)
            else:
                ref_fom_col = args.ref_fom_col
                if ref_fom_col is None or str(ref_fom_col).strip() == "":
                    ref_fom_col = args.fom_col
                if ref_fom_col not in df.columns:
                    print("WARNING: ref_fom_col '%s' not found in CSV; skipping ref map." % ref_fom_col, file=out)
                else:
                    # Build temporary miller arrays for ref phases/FOM on the same reflection set used
                    # Reuse build_cctbx_arrays_from_csv but only to get phib/fom
                    cs2, fp2, ph2, fo2, hl2, _ = build_cctbx_arrays_from_csv(
                        df=df,
                        phi_col=ref_phi_col,
                        fom_col=ref_fom_col,
                        hl_cols=hl_cols,
                        force_recompute_hl=True,
                        kappa_max=float(args.kappa_max),
                        out=null_out()
                    )
                    mc_ref = map_coeffs_from_fp_phi_fom(fp_sigfp=fp_sigfp_out, phib_deg=ph2, fom=fo2)

        base = os.path.splitext(os.path.basename(csv_out))[0]
        dir_out = os.path.dirname(csv_out) if os.path.dirname(csv_out) else "."
        f_in = os.path.join(dir_out, "%s__map_in__%s_%s.ccp4" % (base, args.phib_col, args.fom_col))
        f_dm = os.path.join(dir_out, "%s__map_dm__%s_DM_%s_DM.ccp4" % (base, args.phib_col, args.fom_col))
        write_ccp4_map_from_map_coeffs(map_coeffs=mc_in, crystal_symmetry=cs, file_name=f_in, out=out)
        write_ccp4_map_from_map_coeffs(map_coeffs=mc_dm, crystal_symmetry=cs, file_name=f_dm, out=out)

        if mc_ref is not None:
            f_ref = os.path.join(dir_out, "%s__map_ref__%s_%s.ccp4" % (base, args.ref_phib_col, args.ref_fom_col or args.fom_col))
            write_ccp4_map_from_map_coeffs(map_coeffs=mc_ref, crystal_symmetry=cs, file_name=f_ref, out=out)

    # Build output dataframe at reflection level
    # Use HKL + copy some inputs + append DM results
    # Ensure we write DM values in the ASU indexing used by the returned miller arrays.
    from cctbx.array_family import flex

    hkl = np.array(phib_dm.indices(), dtype=int)
    out_df = pd.DataFrame({
        "H": hkl[:, 0],
        "K": hkl[:, 1],
        "L": hkl[:, 2],
        "PHIB_IN": np.array(phib.common_set(other=phib_dm).data(), dtype=float),
        "FOM_IN":  np.array(fom.common_set(other=fom_dm).data(), dtype=float),
        "PHIB_DM": np.array(phib_dm.data(), dtype=float),
        "FOM_DM":  np.array(fom_dm.data(), dtype=float),
    })

    # HL DM (as_abcd returns 4 flex.double arrays)
    abcd = hl_dm.data().as_abcd()
    out_df["HLA_DM"] = np.array(abcd[0], dtype=float)
    out_df["HLB_DM"] = np.array(abcd[1], dtype=float)
    out_df["HLC_DM"] = np.array(abcd[2], dtype=float)
    out_df["HLD_DM"] = np.array(abcd[3], dtype=float)

    # Also store FP/SIGFP after aniso correction (what RESOLVE actually used)
    out_df["FP_USE"] = np.array(fp_sigfp_out.data(), dtype=float)
    out_df["SIGFP_USE"] = np.array(fp_sigfp_out.sigmas(), dtype=float)

    # Write output CSV
    out_dir = os.path.dirname(csv_out)
    if out_dir and (not os.path.isdir(out_dir)):
        os.makedirs(out_dir)

    out_df.to_csv(csv_out, index=False)
    print("Wrote: %s" % csv_out, file=out)


if __name__ == "__main__":
    try:
        main()
    except Sorry as e:
        print("\n[PHENIX-LIKE CSV DENMOD ERROR] %s\n" % str(e), file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print("\n[PHENIX-LIKE CSV DENMOD UNEXPECTED ERROR] %s\n" % str(e), file=sys.stderr)
        raise