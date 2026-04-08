#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
32_compute_basin_score_deployed.py

python 32_compute_basin_score_deployed_windowtag.py   --metrics_csv ./23_Autobuild_Analysis/23_AutoBuild_basin_score_targets.csv   --proxy_csv ./31_Skeleton_multiwindow/31_skeleton_traceability_summary_multiwindow.csv   --denmod_csv ./30_DENMOD_multiwindow/30_denmod_log_summary_multiwindow.csv   --resol_window 20.0 3.5

Compute the deployed Basin Score for a user-selected resolution window,
using metrics already summarized in input CSV files that contain a
"window_tag" column with values such as:
    dmax20.0_dmin5.0
    dmax20.0_dmin4.0
    dmax20.0_dmin3.5
    dmax20.0_dmin3.0
    dmax20.0_dmin2.5

The score label is reported from dmin, e.g.:
- --resol_window 20.0 3.5 -> S3p5
- --resol_window 20.0 4.0 -> S4p0
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


RUN_PREFIX = "32_compute_basin_score_deployed"

DM_INTERCEPT = -18.072
DM_COEF_R = -38.171
DM_COEF_FOM = 34.256

SK_INTERCEPT = -1.436
SK_COEF_LCF_AUC = 6.063
SK_COEF_ENDPOINT = -9.757

BASIN_IN_BASIN_THRESHOLD = 60.0
BASIN_OUT_OF_BASIN_THRESHOLD = 40.0

THRESHOLD_R_FACTOR = 0.34
THRESHOLD_DM_MEAN_FOM = 0.86
THRESHOLD_LCF_AUC = 0.38
THRESHOLD_ENDPOINT_FRACTION = 0.08


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


def logistic(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def read_csv_auto(path: Path) -> pd.DataFrame:
    return pd.read_csv(filepath_or_buffer=path, sep=None, engine="python")


def resolve_first_present(df: pd.DataFrame, candidates: List[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError("Could not resolve %s. Tried: %s" % (label, candidates))


def normalize_selection(
    pdb_id: Optional[str],
    pdb_ids: Optional[str],
    pdb_ids_file: Optional[Path],
) -> Optional[Set[str]]:
    selected: Set[str] = set()
    if pdb_id:
        selected.add(str(pdb_id).strip().upper())
    if pdb_ids:
        for item in str(pdb_ids).split(","):
            item = item.strip().upper()
            if item:
                selected.add(item)
    if pdb_ids_file:
        with pdb_ids_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = line.strip().upper()
                if item:
                    selected.add(item)
    return selected if selected else None


def format_float_tag(value: float) -> str:
    return f"{float(value):.1f}"


def build_window_tag(*, dmax: float, dmin: float) -> str:
    return f"dmax{format_float_tag(dmax)}_dmin{format_float_tag(dmin)}"


def score_label_from_dmin(*, dmin: float) -> str:
    return "S" + format_float_tag(dmin).replace(".", "p")


def parse_window_tag(value: object) -> Tuple[float, float]:
    text = str(value).strip()
    m = re.fullmatch(r"dmax([0-9]+(?:\.[0-9]+)?)_dmin([0-9]+(?:\.[0-9]+)?)", text)
    if m is None:
        raise ValueError("Could not parse window_tag: %r" % text)
    return float(m.group(1)), float(m.group(2))


def window_mask(
    *,
    df: pd.DataFrame,
    requested_dmax: float,
    requested_dmin: float,
) -> pd.Series:
    target_tag = build_window_tag(dmax=requested_dmax, dmin=requested_dmin)

    if "window_tag" in df.columns:
        tags = df["window_tag"].astype(str).str.strip()
        return tags.eq(target_tag)

    dmax_candidates = ["dmax", "DM_dmax", "window_dmax"]
    dmin_candidates = ["dmin", "DM_dmin", "window_dmin"]
    dmax_col = resolve_first_present(df, dmax_candidates, "dmax column")
    dmin_col = resolve_first_present(df, dmin_candidates, "dmin column")
    return (
        np.isclose(pd.to_numeric(df[dmax_col], errors="coerce"), requested_dmax)
        & np.isclose(pd.to_numeric(df[dmin_col], errors="coerce"), requested_dmin)
    )


def prepare_proxy_dataframe(
    proxy_df: pd.DataFrame,
    *,
    requested_dmax: float,
    requested_dmin: float,
) -> pd.DataFrame:
    df = proxy_df.copy()
    pdb_col = resolve_first_present(df, ["pdb_id", "PDB_ID"], "proxy PDB column")
    df[pdb_col] = df[pdb_col].astype(str).str.upper()

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.upper().eq("OK")].copy()

    df = df[window_mask(df=df, requested_dmax=requested_dmax, requested_dmin=requested_dmin)].copy()

    lcf_auc_col = resolve_first_present(
        df,
        ["best_largest_component_fraction_auc", "largest_component_fraction_auc"],
        "proxy LCF_AUC column",
    )
    endpoint_col = resolve_first_present(
        df,
        ["best_endpoint_fraction_at_target", "endpoint_fraction_at_target"],
        "proxy endpoint column",
    )
    lcf_target_col = resolve_first_present(
        df,
        ["best_largest_component_fraction_at_target", "largest_component_fraction_at_target"],
        "proxy LCF@target column",
    )
    ncc_col = resolve_first_present(
        df,
        ["best_n_connected_components_at_target", "n_connected_components_at_target"],
        "proxy n_connected_components column",
    )
    mean_degree_col = resolve_first_present(
        df,
        ["best_mean_degree_at_target", "mean_degree_at_target"],
        "proxy mean_degree column",
    )

    maybe_nref = None
    for c in ["n_reflections", "Reflections"]:
        if c in df.columns:
            maybe_nref = c
            break

    cols = {
        "PDB_ID": df[pdb_col],
        "best_largest_component_fraction_auc": pd.to_numeric(df[lcf_auc_col], errors="coerce"),
        "best_endpoint_fraction_at_target": pd.to_numeric(df[endpoint_col], errors="coerce"),
        "best_largest_component_fraction_at_target": pd.to_numeric(df[lcf_target_col], errors="coerce"),
        "best_n_connected_components_at_target": pd.to_numeric(df[ncc_col], errors="coerce"),
        "best_mean_degree_at_target": pd.to_numeric(df[mean_degree_col], errors="coerce"),
    }
    if maybe_nref is not None:
        cols["n_reflections_window"] = pd.to_numeric(df[maybe_nref], errors="coerce")
    if "window_tag" in df.columns:
        cols["window_tag_proxy"] = df["window_tag"].astype(str)
    else:
        cols["window_tag_proxy"] = build_window_tag(dmax=requested_dmax, dmin=requested_dmin)

    return pd.DataFrame(cols)


def prepare_denmod_dataframe(
    denmod_df: pd.DataFrame,
    *,
    requested_dmax: float,
    requested_dmin: float,
) -> pd.DataFrame:
    df = denmod_df.copy()
    pdb_col = resolve_first_present(df, ["PDB_ID", "pdb_id"], "denmod PDB column")
    df[pdb_col] = df[pdb_col].astype(str).str.upper()

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.upper().eq("OK")].copy()

    df = df[window_mask(df=df, requested_dmax=requested_dmax, requested_dmin=requested_dmin)].copy()

    cols = {
        "PDB_ID": df[pdb_col],
        "Reflections": pd.to_numeric(df[resolve_first_present(df, ["Reflections"], "Reflections")], errors="coerce"),
        "DM_mean_FOM": pd.to_numeric(df[resolve_first_present(df, ["DM_mean_FOM"], "DM_mean_FOM")], errors="coerce"),
        "R_factor_FC_vs_FP": pd.to_numeric(df[resolve_first_present(df, ["R_factor_FC_vs_FP"], "R_factor_FC_vs_FP")], errors="coerce"),
    }
    if "CC_prob_map_with_current_map" in df.columns:
        cols["CC_prob_map_with_current_map"] = pd.to_numeric(df["CC_prob_map_with_current_map"], errors="coerce")
    if "BIAS_RATIO" in df.columns:
        cols["BIAS_RATIO"] = pd.to_numeric(df["BIAS_RATIO"], errors="coerce")
    if "window_tag" in df.columns:
        cols["window_tag_denmod"] = df["window_tag"].astype(str)
    else:
        cols["window_tag_denmod"] = build_window_tag(dmax=requested_dmax, dmin=requested_dmin)
    return pd.DataFrame(cols)


def prepare_autobuild_dataframe(metrics_df: pd.DataFrame) -> pd.DataFrame:
    df = metrics_df.copy()
    pdb_col = resolve_first_present(df, ["PDB_ID", "pdb_id"], "metrics PDB column")
    df[pdb_col] = df[pdb_col].astype(str).str.upper()
    if pdb_col != "PDB_ID":
        df = df.rename(columns={pdb_col: "PDB_ID"})
    return df


def compute_basin_columns(df: pd.DataFrame, *, score_label: str) -> pd.DataFrame:
    out = df.copy()
    out["P_DM_linear"] = (
        DM_INTERCEPT
        + DM_COEF_R * out["R_factor_FC_vs_FP"].astype(float)
        + DM_COEF_FOM * out["DM_mean_FOM"].astype(float)
    )
    out["P_SK_linear"] = (
        SK_INTERCEPT
        + SK_COEF_LCF_AUC * out["best_largest_component_fraction_auc"].astype(float)
        + SK_COEF_ENDPOINT * out["best_endpoint_fraction_at_target"].astype(float)
    )
    out["P_DM"] = out["P_DM_linear"].map(logistic)
    out["P_SK"] = out["P_SK_linear"].map(logistic)
    out[score_label] = 100.0 * (out["P_DM"] + out["P_SK"]) / 2.0
    out["BasinScore"] = out[score_label]

    out["pass_R_factor_threshold"] = out["R_factor_FC_vs_FP"].astype(float) < THRESHOLD_R_FACTOR
    out["pass_DM_mean_FOM_threshold"] = out["DM_mean_FOM"].astype(float) > THRESHOLD_DM_MEAN_FOM
    out["pass_LCF_AUC_threshold"] = out["best_largest_component_fraction_auc"].astype(float) > THRESHOLD_LCF_AUC
    out["pass_endpoint_fraction_threshold"] = out["best_endpoint_fraction_at_target"].astype(float) < THRESHOLD_ENDPOINT_FRACTION

    out["n_metric_thresholds_passed"] = out[
        [
            "pass_R_factor_threshold",
            "pass_DM_mean_FOM_threshold",
            "pass_LCF_AUC_threshold",
            "pass_endpoint_fraction_threshold",
        ]
    ].astype(int).sum(axis=1)

    out["passes_all_metric_thresholds"] = (
        out["pass_R_factor_threshold"]
        & out["pass_DM_mean_FOM_threshold"]
        & out["pass_LCF_AUC_threshold"]
        & out["pass_endpoint_fraction_threshold"]
    )

    out["is_in_basin_by_score"] = out[score_label].astype(float) >= BASIN_IN_BASIN_THRESHOLD
    out["is_out_of_basin_by_score"] = out[score_label].astype(float) < BASIN_OUT_OF_BASIN_THRESHOLD
    out["is_ambiguous_by_score"] = (~out["is_in_basin_by_score"]) & (~out["is_out_of_basin_by_score"])

    def assign_band(score: float) -> str:
        if score >= BASIN_IN_BASIN_THRESHOLD:
            return "in_basin_likely_productive"
        if score < BASIN_OUT_OF_BASIN_THRESHOLD:
            return "out_of_basin_likely_unproductive"
        return "ambiguous_transition_region"

    out[f"{score_label}_band"] = out[score_label].astype(float).map(assign_band)
    out["BasinScore_band"] = out[f"{score_label}_band"]

    req = ["K2_atten__rfree_percent", "K2_atten__rel_residues_placed_vs_iREDO"]
    if all(c in out.columns for c in req):
        free_r = pd.to_numeric(out["K2_atten__rfree_percent"], errors="coerce")
        rel_chain = pd.to_numeric(out["K2_atten__rel_residues_placed_vs_iREDO"], errors="coerce")
        out["Autobuild_strong_success"] = (free_r <= 30.0) & (rel_chain >= 80.0)
        out["Autobuild_strong_failure"] = (free_r > 40.0) & (rel_chain < 40.0)
        out["Autobuild_ambiguous"] = ~(out["Autobuild_strong_success"] | out["Autobuild_strong_failure"])
    return out


def build_summary_dataframe(
    metrics_path: Path,
    proxy_path: Path,
    denmod_path: Path,
    selected_pdb_ids: Optional[Set[str]],
    *,
    requested_dmax: float,
    requested_dmin: float,
    score_label: str,
) -> pd.DataFrame:
    metrics_df = prepare_autobuild_dataframe(read_csv_auto(metrics_path))
    proxy_df = prepare_proxy_dataframe(
        read_csv_auto(proxy_path),
        requested_dmax=requested_dmax,
        requested_dmin=requested_dmin,
    )
    denmod_df = prepare_denmod_dataframe(
        read_csv_auto(denmod_path),
        requested_dmax=requested_dmax,
        requested_dmin=requested_dmin,
    )

    merged_df = metrics_df.merge(proxy_df, how="inner", on="PDB_ID", validate="one_to_one").merge(
        denmod_df, how="inner", on="PDB_ID", validate="one_to_one"
    )
    if selected_pdb_ids is not None:
        merged_df = merged_df[merged_df["PDB_ID"].isin(selected_pdb_ids)].copy()
    merged_df = compute_basin_columns(merged_df, score_label=score_label)

    preferred = [
        "PDB_ID", "n_reflections_window", "Reflections",
        "K2_atten__rfree_percent", "K2_atten__rel_residues_placed_vs_iREDO",
        score_label, f"{score_label}_band", "BasinScore", "BasinScore_band", "P_DM", "P_SK",
        "R_factor_FC_vs_FP", "DM_mean_FOM",
        "best_largest_component_fraction_auc", "best_endpoint_fraction_at_target",
        "pass_R_factor_threshold", "pass_DM_mean_FOM_threshold",
        "pass_LCF_AUC_threshold", "pass_endpoint_fraction_threshold",
        "passes_all_metric_thresholds",
        "n_metric_thresholds_passed", "is_in_basin_by_score", "is_ambiguous_by_score",
        "is_out_of_basin_by_score", "Autobuild_strong_success", "Autobuild_strong_failure",
        "Autobuild_ambiguous", "best_largest_component_fraction_at_target",
        "best_n_connected_components_at_target", "best_mean_degree_at_target",
        "CC_prob_map_with_current_map", "BIAS_RATIO", "window_tag_proxy", "window_tag_denmod",
    ]
    present = [c for c in preferred if c in merged_df.columns]
    remaining = [c for c in merged_df.columns if c not in present]
    merged_df = merged_df[present + remaining].copy()
    merged_df = merged_df.sort_values(by=[score_label, "PDB_ID"], ascending=[False, True]).reset_index(drop=True)
    return merged_df


def _make_density_sorted(x: np.ndarray, y: np.ndarray):
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    z = kde(xy)
    order = np.argsort(z)
    return x[order], y[order], z[order]


def plot_basin_score_density(
    *,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    out_png: Path,
) -> None:
    work = df[[x_col, y_col]].copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work = work.dropna(subset=[x_col, y_col]).copy()
    if work.shape[0] < 5:
        logging.info("[PLOT] Skipping %s (too few points).", str(out_png))
        return

    x = work[x_col].to_numpy(dtype=float)
    y = work[y_col].to_numpy(dtype=float)

    try:
        xs, ys, zs = _make_density_sorted(x=x, y=y)
    except Exception as exc:
        logging.warning("[PLOT] KDE failed for %s: %s", str(out_png), str(exc))
        return

    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=160)
    sc = ax.scatter(xs, ys, c=zs, s=28)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.minorticks_on()
    ax.grid(which="major", linestyle="--", color="gray", alpha=0.6)
    ax.grid(which="minor", linestyle=":", color="lightgray", alpha=0.7)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Kernel density")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    logging.info("[PLOT] Saved: %s", str(out_png))


def format_formula_text(*, score_label: str) -> str:
    lines = []
    lines.append("P_DM = logistic(-18.072 - 38.171 * R_factor_FC_vs_FP + 34.256 * DM_mean_FOM)")
    lines.append("P_SK = logistic(-1.436 + 6.063 * best_largest_component_fraction_auc - 9.757 * best_endpoint_fraction_at_target)")
    lines.append(f"{score_label} = 100 * (P_DM + P_SK) / 2")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the deployed Basin Score for a user-selected resolution window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metrics_csv", type=Path, required=True)
    parser.add_argument("--proxy_csv", type=Path, required=True)
    parser.add_argument("--denmod_csv", type=Path, required=True)
    parser.add_argument(
        "--resol_window",
        type=float,
        nargs=2,
        metavar=("DMAX", "DMIN"),
        required=True,
        help="Requested resolution window, for example: --resol_window 20.0 3.5",
    )
    parser.add_argument("--out_csv", type=Path, default=None)
    parser.add_argument("--pdb_id", type=str, default=None)
    parser.add_argument("--pdb_ids", type=str, default=None)
    parser.add_argument("--pdb_ids_file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    requested_dmax = float(args.resol_window[0])
    requested_dmin = float(args.resol_window[1])
    requested_window_tag = build_window_tag(dmax=requested_dmax, dmin=requested_dmin)
    score_label = score_label_from_dmin(dmin=requested_dmin)

    default_out_csv = Path(f"32_BasinScore_{score_label}/32_cohort_BasinScore_{score_label}.csv")
    out_csv = (args.out_csv or default_out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    log_path = configure_logging(out_dir=out_csv.parent)
    logging.info("Run prefix: %s", RUN_PREFIX)
    logging.info("Command line: %s", " ".join(shlex.quote(x) for x in sys.argv))
    logging.info("metrics_csv=%s", str(args.metrics_csv))
    logging.info("proxy_csv=%s", str(args.proxy_csv))
    logging.info("denmod_csv=%s", str(args.denmod_csv))
    logging.info("resol_window=dmax %.1f dmin %.1f", requested_dmax, requested_dmin)
    logging.info("requested_window_tag=%s", requested_window_tag)
    logging.info("score_label=%s", score_label)
    logging.info("out_csv=%s", str(out_csv))

    selected_pdb_ids = normalize_selection(args.pdb_id, args.pdb_ids, args.pdb_ids_file)
    if selected_pdb_ids is not None:
        logging.info("Selected PDB IDs: %d", int(len(selected_pdb_ids)))

    summary_df = build_summary_dataframe(
        metrics_path=args.metrics_csv,
        proxy_path=args.proxy_csv,
        denmod_path=args.denmod_csv,
        selected_pdb_ids=selected_pdb_ids,
        requested_dmax=requested_dmax,
        requested_dmin=requested_dmin,
        score_label=score_label,
    )

    if summary_df.shape[0] < 1:
        raise RuntimeError(
            "No rows remained after filtering to window_tag=%s. Check input CSVs and --resol_window."
            % requested_window_tag
        )

    summary_df.to_csv(out_csv, index=False)

    n_total = int(summary_df.shape[0])
    n_in = int(np.sum(summary_df["is_in_basin_by_score"].to_numpy(dtype=bool))) if "is_in_basin_by_score" in summary_df.columns else 0
    n_amb = int(np.sum(summary_df["is_ambiguous_by_score"].to_numpy(dtype=bool))) if "is_ambiguous_by_score" in summary_df.columns else 0
    n_out = int(np.sum(summary_df["is_out_of_basin_by_score"].to_numpy(dtype=bool))) if "is_out_of_basin_by_score" in summary_df.columns else 0
    n_pass_all = int(np.sum(summary_df["passes_all_metric_thresholds"].to_numpy(dtype=bool))) if "passes_all_metric_thresholds" in summary_df.columns else 0

    logging.info("Rows written: %d", n_total)
    logging.info("In-basin (%s >= %.1f): %d", score_label, float(BASIN_IN_BASIN_THRESHOLD), n_in)
    logging.info("Ambiguous (%.1f <= %s < %.1f): %d", float(BASIN_OUT_OF_BASIN_THRESHOLD), score_label, float(BASIN_IN_BASIN_THRESHOLD), n_amb)
    logging.info("Out-of-basin (%s < %.1f): %d", score_label, float(BASIN_OUT_OF_BASIN_THRESHOLD), n_out)
    logging.info("Pass all metric thresholds: %d", n_pass_all)

    plot_basin_score_density(
        df=summary_df,
        x_col="K2_atten__rfree_percent",
        y_col=score_label,
        x_label="AutoBuild Free R for K2_atten (%)",
        y_label=score_label,
        title=f"{score_label} vs AutoBuild Free R (K2_atten)",
        out_png=out_csv.parent / f"32_{score_label}_vs_K2_atten_rfree_kde.png",
    )
    plot_basin_score_density(
        df=summary_df,
        x_col="K2_atten__rel_residues_placed_vs_iREDO",
        y_col=score_label,
        x_label="Relative chain length vs iREDO (%)",
        y_label=score_label,
        title=f"{score_label} vs Relative Chain Length to iREDO",
        out_png=out_csv.parent / f"32_{score_label}_vs_relchain_kde.png",
    )

    formula_text = format_formula_text(score_label=score_label)

    print("Done.")
    print("Window used: dmax=%.1f Å, dmin=%.1f Å" % (requested_dmax, requested_dmin))
    print("Requested window_tag: %s" % requested_window_tag)
    print("Score label: %s" % score_label)
    print("Rows written: %d" % n_total)
    print("Output CSV: %s" % str(out_csv))
    print("Plot PNG: %s" % str(out_csv.parent / f"32_{score_label}_vs_K2_atten_rfree_kde.png"))
    print("Plot PNG: %s" % str(out_csv.parent / f"32_{score_label}_vs_relchain_kde.png"))
    print("Log file: %s" % str(log_path))

    print("\nDeployed Basin Score formula:")
    print(formula_text)

    print("\nThresholds for transition into likely productive AutoBuild regime:")
    print("  Score threshold:")
    print("    %s >= %.1f  -> likely productive / in basin" % (score_label, float(BASIN_IN_BASIN_THRESHOLD)))
    print("  Companion metric thresholds:")
    print("    R_factor_FC_vs_FP < %.2f" % float(THRESHOLD_R_FACTOR))
    print("    DM_mean_FOM > %.2f" % float(THRESHOLD_DM_MEAN_FOM))
    print("    best_largest_component_fraction_auc > %.2f" % float(THRESHOLD_LCF_AUC))
    print("    best_endpoint_fraction_at_target < %.2f" % float(THRESHOLD_ENDPOINT_FRACTION))
    print("  Lower bound for likely unproductive regime:")
    print("    %s < %.1f  -> likely unproductive / out of basin" % (score_label, float(BASIN_OUT_OF_BASIN_THRESHOLD)))

    print("\nBasin Score bands:")
    print("  [%.1f, 100]  -> in_basin_likely_productive" % float(BASIN_IN_BASIN_THRESHOLD))
    print("  [%.1f, %.1f) -> ambiguous_transition_region" % (float(BASIN_OUT_OF_BASIN_THRESHOLD), float(BASIN_IN_BASIN_THRESHOLD)))
    print("  [0, %.1f)    -> out_of_basin_likely_unproductive" % float(BASIN_OUT_OF_BASIN_THRESHOLD))


if __name__ == "__main__":
    main()
