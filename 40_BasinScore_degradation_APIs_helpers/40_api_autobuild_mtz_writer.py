#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def _kappa_from_R_array(R):
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

def main():
    p = argparse.ArgumentParser(description="Write Autobuild-ready MTZ from payload CSV")
    p.add_argument("--csv_in", required=True)
    p.add_argument("--mtz_out", required=True)
    p.add_argument("--kappa_max", type=float, default=200.0)
    args = p.parse_args()

    import gemmi
    import reciprocalspaceship as rs

    df = pd.read_csv(args.csv_in)
    required = ["H","K","L","FP","SIGFP","PHIB","FOM","FreeR_flag","SG","LENGTH_A","LENGTH_B","LENGTH_C","ANGLE_ALPHA","ANGLE_BETA","ANGLE_GAMMA"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit("Missing required columns: %s" % missing)

    PHIB = pd.to_numeric(df["PHIB"], errors="coerce").astype(float).to_numpy()
    FOM = pd.to_numeric(df["FOM"], errors="coerce").fillna(0.0).astype(float).to_numpy()
    FOM = np.clip(FOM, 0.0, 0.9999)
    kappa = np.minimum(_kappa_from_R_array(FOM), float(args.kappa_max))
    phi_rad = np.deg2rad(PHIB)
    HLA = kappa * np.cos(phi_rad)
    HLB = kappa * np.sin(phi_rad)
    HLC = np.zeros_like(HLA)
    HLD = np.zeros_like(HLA)

    ds = rs.DataSet()
    ds["H"] = rs.DataSeries(pd.to_numeric(df["H"], errors="coerce").fillna(0).astype(int).to_numpy(), dtype="H")
    ds["K"] = rs.DataSeries(pd.to_numeric(df["K"], errors="coerce").fillna(0).astype(int).to_numpy(), dtype="H")
    ds["L"] = rs.DataSeries(pd.to_numeric(df["L"], errors="coerce").fillna(0).astype(int).to_numpy(), dtype="H")
    ds["FP"] = rs.DataSeries(pd.to_numeric(df["FP"], errors="coerce").astype(float).to_numpy(), dtype="F")
    ds["SIGFP"] = rs.DataSeries(pd.to_numeric(df["SIGFP"], errors="coerce").astype(float).to_numpy(), dtype="Q")
    ds["PHIB"] = rs.DataSeries(PHIB, dtype="P")
    ds["FOM"] = rs.DataSeries(FOM, dtype="W")
    ds["FreeR_flag"] = rs.DataSeries(pd.to_numeric(df["FreeR_flag"], errors="coerce").fillna(0).astype(int).to_numpy(), dtype="I")
    ds["HLA"] = rs.DataSeries(HLA, dtype="A")
    ds["HLB"] = rs.DataSeries(HLB, dtype="A")
    ds["HLC"] = rs.DataSeries(HLC, dtype="A")
    ds["HLD"] = rs.DataSeries(HLD, dtype="A")
    ds.set_index(["H","K","L"], inplace=True)

    sg_str = str(df["SG"].iloc[0])
    a=float(df["LENGTH_A"].iloc[0]); b=float(df["LENGTH_B"].iloc[0]); c=float(df["LENGTH_C"].iloc[0])
    alpha=float(df["ANGLE_ALPHA"].iloc[0]); beta=float(df["ANGLE_BETA"].iloc[0]); gamma=float(df["ANGLE_GAMMA"].iloc[0])
    ds.spacegroup = gemmi.SpaceGroup(sg_str)
    ds.cell = gemmi.UnitCell(a,b,c,alpha,beta,gamma)

    out = Path(args.mtz_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mtz = ds.to_gemmi()
    mtz.write_to_file(str(out))
    print("Wrote MTZ:", out)

if __name__ == "__main__":
    main()
