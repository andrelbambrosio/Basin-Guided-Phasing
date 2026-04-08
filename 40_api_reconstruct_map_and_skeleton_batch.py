#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch map reconstruction + skeletonization from ASU reflection CSV files.

python pc44_reconstruct_map_and_skeleton_batch_folder.py --csv_dir ./12_DenMod_BasinScore_Calibration/CSV   --out_dir ./map_skeleton_batch_all   --phase_unit deg   --sample_rate 3.0   --map_kind_for_skeleton fom_weighted   --threshold_mode sigma   --threshold_values 0.8,1.0,1.2,1.5,2.0   --target_threshold 1.2   --prune_tip_iterations 3   --edge_connectivity 26   --phase_fom_pairs PHIC_ALL_K2:FOM_K2_atten   --write_per_run_tables --dmin 3.5 --dmax 20.0

Capabilities
------------
1. Read all CSV reflection tables in a folder.
2. Reconstruct raw and FOM-weighted maps for one or more phase:FOM pairs.
3. Save CCP4 maps for each pair.
4. Run one or more skeleton thresholds on the selected map kind.
5. Export per-run voxel/edge/metric tables and one aggregate summary CSV per PDB.
6. Compute pair-level threshold-integrated summary metrics.
7. Generate automatic threshold-sweep plots for each pair.
8. Emit compact per-PDB traceability proxy summaries using the four prioritized metrics:
      - largest_component_fraction at target threshold
      - n_connected_components at target threshold
      - endpoint_fraction at target threshold
      - largest_component_fraction_auc across thresholds

Notes
-----
- This is a Coot-like workflow, not a literal reimplementation of Coot/Clipper.
- Skeletonization is performed on a thresholded 3D map using scikit-image.
- The "2mFo-like" map here is the FOM-weighted map:
      F_map = FP * FOM * exp(i * phi)
  from the coefficients present in the CSV.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import gemmi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.morphology import skeletonize


BASE_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "H",
    "K",
    "L",
    "FP",
    "LENGTH_A",
    "LENGTH_B",
    "LENGTH_C",
    "ANGLE_ALPHA",
    "ANGLE_BETA",
    "ANGLE_GAMMA",
    "SG",
)


@dataclass(frozen=True)
class CrystalMetadata:
    length_a: float
    length_b: float
    length_c: float
    angle_alpha: float
    angle_beta: float
    angle_gamma: float
    spacegroup_name: str

    def build_unit_cell(self) -> gemmi.UnitCell:
        return gemmi.UnitCell(
            float(self.length_a),
            float(self.length_b),
            float(self.length_c),
            float(self.angle_alpha),
            float(self.angle_beta),
            float(self.angle_gamma),
        )


@dataclass(frozen=True)
class PairSpec:
    phase_column: str
    fom_column: str
    label: str


class InputError(RuntimeError):
    pass


_OFFSETS_6: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)

_OFFSETS_18: Tuple[Tuple[int, int, int], ...] = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
    if (abs(dx) + abs(dy) + abs(dz)) <= 2
)

_OFFSETS_26: Tuple[Tuple[int, int, int], ...] = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
)

_METRIC_PLOT_SPECS: Tuple[Tuple[str, str], ...] = (
    ("largest_component_fraction", "Largest-component fraction"),
    ("n_connected_components", "Connected components"),
    ("n_nodes", "Skeleton nodes"),
    ("endpoint_fraction", "Endpoint fraction"),
    ("branchpoint_fraction", "Branchpoint fraction"),
    ("cyclomatic_number", "Cyclomatic number"),
    ("mean_degree", "Mean degree"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch reconstruct CCP4 maps and skeleton metrics from ASU reflection CSV files."
    )
    parser.add_argument("--csv_dir", type=str, required=True, help="Input directory containing CSV reflection tables.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--phase_unit",
        type=str,
        choices=("deg", "rad"),
        default="deg",
        help="Unit of all phase columns.",
    )
    parser.add_argument(
        "--amplitude_column",
        type=str,
        default="FP",
        help="Amplitude column used to build maps.",
    )
    parser.add_argument(
        "--phase_column",
        type=str,
        default=None,
        help="Single phase column, used if --phase_fom_pairs is not provided.",
    )
    parser.add_argument(
        "--fom_column",
        type=str,
        default=None,
        help="Single FOM column, used if --phase_fom_pairs is not provided.",
    )
    parser.add_argument(
        "--phase_fom_pairs",
        type=str,
        nargs="*",
        default=None,
        help='List like "PHIC_ALL:FOM" "PHIC_ALL_K2:FOM_K2_atten".',
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=3.0,
        help="Gemmi sample_rate passed to transform_f_phi_to_map().",
    )
    parser.add_argument(
        "--map_kind_for_skeleton",
        type=str,
        choices=("fom_weighted", "raw"),
        default="fom_weighted",
        help="Which reconstructed map is thresholded for skeletonization.",
    )
    parser.add_argument(
        "--threshold_mode",
        type=str,
        choices=("sigma", "absolute", "percentile"),
        default="sigma",
        help="How threshold values are interpreted.",
    )
    parser.add_argument(
        "--threshold_value",
        type=float,
        default=None,
        help="Single threshold value. Ignored if --threshold_values is provided.",
    )
    parser.add_argument(
        "--threshold_values",
        type=str,
        default=None,
        help='Comma-separated thresholds, e.g. "0.8,1.0,1.2,1.5,2.0".',
    )
    parser.add_argument(
        "--target_threshold",
        type=float,
        default=1.2,
        help="Threshold used for the fixed-threshold proxy metrics. Default: 1.2",
    )
    parser.add_argument(
        "--binary_closing_iterations",
        type=int,
        default=0,
        help="Optional binary closing iterations before skeletonization.",
    )
    parser.add_argument(
        "--min_component_size",
        type=int,
        default=0,
        help="Discard connected components smaller than this many voxels before skeletonization.",
    )
    parser.add_argument(
        "--prune_tip_iterations",
        type=int,
        default=0,
        help="Number of recursive endpoint-pruning iterations after skeletonization.",
    )
    parser.add_argument(
        "--edge_connectivity",
        type=int,
        choices=(6, 18, 26),
        default=26,
        help="Neighborhood used to connect skeleton voxels into graph edges.",
    )
    parser.add_argument(
        "--dmin",
        type=float,
        default=None,
        help="Optional high-resolution cutoff in Angstrom. Keep reflections with dHKL >= dmin.",
    )
    parser.add_argument(
        "--dmax",
        type=float,
        default=None,
        help="Optional low-resolution cutoff in Angstrom. Keep reflections with dHKL <= dmax.",
    )
    parser.add_argument(
        "--write_per_run_tables",
        action="store_true",
        help="Write per-run voxel/edge tables for every pair x threshold combination.",
    )
    parser.add_argument(
        "--disable_summary_plots",
        action="store_true",
        help="Disable automatic PNG plots of threshold curves.",
    )
    return parser.parse_args()


def ensure_required_columns(*, dataframe: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing_columns = [name for name in required_columns if name not in dataframe.columns]
    if missing_columns:
        raise InputError(f"Missing required columns: {missing_columns}")


def canonicalize_spacegroup_name(*, raw_spacegroup_name: str) -> str:
    cleaned_name = " ".join(str(raw_spacegroup_name).strip().replace("'", " ").split())
    if not cleaned_name:
        raise InputError("Empty space-group name in CSV.")
    sg = gemmi.find_spacegroup_by_name(cleaned_name)
    if sg is None:
        sg = gemmi.find_spacegroup_by_name(cleaned_name.replace(" ", ""))
    if sg is None:
        raise InputError(f"Could not parse space group: {raw_spacegroup_name!r}")
    return sg.hm


def extract_unique_metadata(*, dataframe: pd.DataFrame) -> CrystalMetadata:
    def unique_scalar(column_name: str) -> float | str:
        unique_values = dataframe[column_name].dropna().unique()
        if len(unique_values) != 1:
            raise InputError(
                f"Column {column_name!r} must contain a single unique value, found {len(unique_values)} values."
            )
        return unique_values[0]

    return CrystalMetadata(
        length_a=float(unique_scalar("LENGTH_A")),
        length_b=float(unique_scalar("LENGTH_B")),
        length_c=float(unique_scalar("LENGTH_C")),
        angle_alpha=float(unique_scalar("ANGLE_ALPHA")),
        angle_beta=float(unique_scalar("ANGLE_BETA")),
        angle_gamma=float(unique_scalar("ANGLE_GAMMA")),
        spacegroup_name=canonicalize_spacegroup_name(raw_spacegroup_name=str(unique_scalar("SG"))),
    )


def _safe_to_numeric(*, series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series)
    except Exception:
        return series


def load_reflection_table(*, csv_path: str, amplitude_column: str) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    ensure_required_columns(dataframe=dataframe, required_columns=BASE_REQUIRED_COLUMNS + (amplitude_column,))
    numeric_columns = [
        "H",
        "K",
        "L",
        amplitude_column,
        "LENGTH_A",
        "LENGTH_B",
        "LENGTH_C",
        "ANGLE_ALPHA",
        "ANGLE_BETA",
        "ANGLE_GAMMA",
    ]
    if "dHKL" in dataframe.columns:
        numeric_columns.append("dHKL")
    for column_name in dataframe.columns:
        if column_name in numeric_columns or column_name.startswith(("PHI", "FOM", "SIG", "FP")):
            dataframe[column_name] = _safe_to_numeric(series=dataframe[column_name])
    for column_name in numeric_columns:
        dataframe[column_name] = pd.to_numeric(dataframe[column_name], errors="coerce")
    dataframe = dataframe.dropna(subset=["H", "K", "L", amplitude_column])
    dataframe["H"] = dataframe["H"].astype(int)
    dataframe["K"] = dataframe["K"].astype(int)
    dataframe["L"] = dataframe["L"].astype(int)
    return dataframe


def apply_resolution_filter(*, dataframe: pd.DataFrame, dmin: float | None, dmax: float | None) -> pd.DataFrame:
    if dmin is None and dmax is None:
        return dataframe.copy()
    if "dHKL" not in dataframe.columns:
        raise InputError("dHKL column is required when using --dmin and/or --dmax.")
    filtered = dataframe.copy()
    if dmin is not None:
        filtered = filtered.loc[filtered["dHKL"] >= float(dmin)]
    if dmax is not None:
        filtered = filtered.loc[filtered["dHKL"] <= float(dmax)]
    if filtered.empty:
        raise InputError("Resolution filtering removed all reflections.")
    return filtered


def convert_phase_array_to_degrees(*, phase_values: np.ndarray, phase_unit: str) -> np.ndarray:
    if phase_unit == "deg":
        return phase_values.astype(np.float32, copy=False)
    if phase_unit == "rad":
        return np.rad2deg(phase_values).astype(np.float32, copy=False)
    raise InputError(f"Unsupported phase_unit: {phase_unit}")


def sanitize_label(*, text: str) -> str:
    cleaned = re.sub(pattern=r"[^A-Za-z0-9._-]+", repl="_", string=str(text).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "unnamed"


def infer_pdb_id_from_csv_path(*, csv_path: str) -> str:
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    toks = [tok for tok in stem.split("_") if tok]
    if len(toks) > 0:
        return toks[0].strip().lower()
    return stem.strip().lower()


def parse_pair_specs(*, args: argparse.Namespace, dataframe: pd.DataFrame) -> List[PairSpec]:
    if args.phase_fom_pairs:
        pair_specs: List[PairSpec] = []
        for raw_pair in args.phase_fom_pairs:
            if ":" not in raw_pair:
                raise InputError(f"Invalid phase:fom pair: {raw_pair!r}")
            phase_column, fom_column = raw_pair.split(":", 1)
            phase_column = phase_column.strip()
            fom_column = fom_column.strip()
            ensure_required_columns(dataframe=dataframe, required_columns=(phase_column, fom_column))
            pair_specs.append(
                PairSpec(
                    phase_column=phase_column,
                    fom_column=fom_column,
                    label=sanitize_label(text=f"{phase_column}__{fom_column}"),
                )
            )
        return pair_specs

    phase_column = args.phase_column or "PHIC_ALL"
    fom_column = args.fom_column or "FOM"
    ensure_required_columns(dataframe=dataframe, required_columns=(phase_column, fom_column))
    return [
        PairSpec(
            phase_column=phase_column,
            fom_column=fom_column,
            label=sanitize_label(text=f"{phase_column}__{fom_column}"),
        )
    ]


def parse_threshold_values(*, args: argparse.Namespace) -> List[float]:
    if args.threshold_values is not None:
        values = [float(item.strip()) for item in str(args.threshold_values).split(",") if item.strip()]
        if not values:
            raise InputError("--threshold_values was provided but no valid values were parsed.")
        return values
    if args.threshold_value is not None:
        return [float(args.threshold_value)]
    return [1.2]


def build_mtz_from_dataframe(
    *,
    dataframe: pd.DataFrame,
    metadata: CrystalMetadata,
    amplitude_column: str,
    phase_column: str,
    fom_column: str,
    phase_unit: str,
) -> gemmi.Mtz:
    working_df = dataframe[["H", "K", "L", amplitude_column, phase_column, fom_column]].copy()
    working_df[amplitude_column] = pd.to_numeric(working_df[amplitude_column], errors="coerce")
    working_df[phase_column] = pd.to_numeric(working_df[phase_column], errors="coerce")
    working_df[fom_column] = pd.to_numeric(working_df[fom_column], errors="coerce")
    working_df = working_df.dropna(subset=[amplitude_column, phase_column, fom_column])
    if working_df.empty:
        raise InputError(f"No valid reflections remain for pair {phase_column}:{fom_column}")

    amplitudes = working_df[amplitude_column].to_numpy(dtype=np.float32, copy=True)
    phases_deg = convert_phase_array_to_degrees(
        phase_values=working_df[phase_column].to_numpy(dtype=np.float32, copy=True),
        phase_unit=phase_unit,
    )
    foms = np.clip(working_df[fom_column].to_numpy(dtype=np.float32, copy=True), 0.0, None)
    weighted_amplitudes = amplitudes * foms

    h_values = working_df["H"].to_numpy(dtype=np.int32, copy=False)
    k_values = working_df["K"].to_numpy(dtype=np.int32, copy=False)
    l_values = working_df["L"].to_numpy(dtype=np.int32, copy=False)

    mtz = gemmi.Mtz(with_base=False)
    mtz.spacegroup = gemmi.find_spacegroup_by_name(metadata.spacegroup_name)
    mtz.set_cell_for_all(metadata.build_unit_cell())
    mtz.add_dataset("csv_reconstruction")
    mtz.add_column("H", "H")
    mtz.add_column("K", "H")
    mtz.add_column("L", "H")
    mtz.add_column("F_RAW", "F")
    mtz.add_column("PHI_RAW", "P")
    mtz.add_column("F_FOM", "F")
    mtz.add_column("PHI_FOM", "P")

    mtz_data = np.column_stack(
        [
            h_values.astype(np.float32, copy=False),
            k_values.astype(np.float32, copy=False),
            l_values.astype(np.float32, copy=False),
            amplitudes.astype(np.float32, copy=False),
            phases_deg.astype(np.float32, copy=False),
            weighted_amplitudes.astype(np.float32, copy=False),
            phases_deg.astype(np.float32, copy=False),
        ]
    )
    mtz.set_data(mtz_data)
    return mtz


def transform_mtz_to_map(*, mtz: gemmi.Mtz, f_label: str, phi_label: str, sample_rate: float) -> gemmi.FloatGrid:
    return mtz.transform_f_phi_to_map(f_label, phi_label, [0, 0, 0], [0, 0, 0], float(sample_rate))


def write_ccp4_map(*, map_grid: gemmi.FloatGrid, output_path: str) -> None:
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = map_grid
    ccp4.update_ccp4_header(2, True)
    ccp4.write_ccp4_map(output_path)


def float_grid_to_numpy(*, map_grid: gemmi.FloatGrid) -> np.ndarray:
    return np.array(map_grid, copy=True, dtype=np.float32)


def sigma_normalize_map(*, map_array: np.ndarray) -> Tuple[np.ndarray, float, float]:
    map_mean = float(np.mean(map_array))
    map_std = float(np.std(map_array, ddof=0))
    if map_std <= 0.0:
        normalized = np.zeros_like(map_array, dtype=np.float32)
    else:
        normalized = ((map_array - map_mean) / map_std).astype(np.float32, copy=False)
    return normalized, map_mean, map_std


def threshold_map(
    *,
    map_array: np.ndarray,
    threshold_mode: str,
    threshold_value: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    normalized_map, map_mean, map_std = sigma_normalize_map(map_array=map_array)
    if threshold_mode == "sigma":
        threshold_map_units = map_mean + float(threshold_value) * map_std
        threshold_sigma_equiv = float(threshold_value)
        binary_mask = normalized_map >= float(threshold_value)
    elif threshold_mode == "absolute":
        threshold_map_units = float(threshold_value)
        threshold_sigma_equiv = (threshold_map_units - map_mean) / map_std if map_std > 0.0 else float("nan")
        binary_mask = map_array >= threshold_map_units
    elif threshold_mode == "percentile":
        threshold_map_units = float(np.percentile(map_array, q=float(threshold_value)))
        threshold_sigma_equiv = (threshold_map_units - map_mean) / map_std if map_std > 0.0 else float("nan")
        binary_mask = map_array >= threshold_map_units
    else:
        raise InputError(f"Unsupported threshold_mode: {threshold_mode}")

    return binary_mask.astype(bool, copy=False), {
        "threshold_map_units": float(threshold_map_units),
        "threshold_sigma_equiv": float(threshold_sigma_equiv),
        "map_mean": float(map_mean),
        "map_std": float(map_std),
    }


def apply_binary_closing(*, binary_mask: np.ndarray, iterations: int) -> np.ndarray:
    if int(iterations) <= 0:
        return binary_mask.astype(bool, copy=False)
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    closed = ndimage.binary_closing(input=binary_mask, structure=structure, iterations=int(iterations))
    return closed.astype(bool, copy=False)


def remove_small_components(*, binary_mask: np.ndarray, min_component_size: int) -> np.ndarray:
    if int(min_component_size) <= 1:
        return binary_mask.astype(bool, copy=False)
    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    labeled, n_labels = ndimage.label(input=binary_mask, structure=structure)
    if n_labels == 0:
        return binary_mask.astype(bool, copy=False)
    component_sizes = np.bincount(labeled.ravel())
    keep_labels = np.where(component_sizes >= int(min_component_size))[0]
    keep_labels = keep_labels[keep_labels != 0]
    filtered = np.isin(labeled, keep_labels)
    return filtered.astype(bool, copy=False)


def neighbor_offsets(*, connectivity: int) -> Tuple[Tuple[int, int, int], ...]:
    if connectivity == 6:
        return _OFFSETS_6
    if connectivity == 18:
        return _OFFSETS_18
    if connectivity == 26:
        return _OFFSETS_26
    raise InputError(f"Unsupported connectivity: {connectivity}")


def prune_skeleton_tips(*, skeleton_mask: np.ndarray, connectivity: int, iterations: int) -> np.ndarray:
    pruned = skeleton_mask.astype(bool, copy=True)
    offsets = neighbor_offsets(connectivity=connectivity)
    for _ in range(int(iterations)):
        coords = np.argwhere(pruned)
        if coords.size == 0:
            break
        node_set = set(tuple(int(v) for v in row) for row in coords)
        remove_list: List[Tuple[int, int, int]] = []
        for node in node_set:
            degree = 0
            x0, y0, z0 = node
            for dx, dy, dz in offsets:
                neighbor = (x0 + dx, y0 + dy, z0 + dz)
                if neighbor in node_set:
                    degree += 1
                    if degree > 1:
                        break
            if degree <= 1:
                remove_list.append(node)
        if not remove_list:
            break
        for x0, y0, z0 in remove_list:
            pruned[x0, y0, z0] = False
    return pruned


def skeleton_mask_to_graph(*, skeleton_mask: np.ndarray, connectivity: int) -> nx.Graph:
    graph = nx.Graph()
    coords = np.argwhere(skeleton_mask)
    node_set = set(tuple(int(v) for v in row) for row in coords)
    for node in node_set:
        graph.add_node(node)
    offsets = neighbor_offsets(connectivity=connectivity)
    for x0, y0, z0 in node_set:
        for dx, dy, dz in offsets:
            neighbor = (x0 + dx, y0 + dy, z0 + dz)
            if neighbor in node_set and neighbor > (x0, y0, z0):
                graph.add_edge(
                    (x0, y0, z0),
                    neighbor,
                    length_voxels=float(math.sqrt(dx * dx + dy * dy + dz * dz)),
                )
    return graph


def voxel_to_fractional(*, voxel_index: Tuple[int, int, int], grid_shape: Sequence[int]) -> Tuple[float, float, float]:
    nx_size, ny_size, nz_size = grid_shape
    return (
        (float(voxel_index[0]) + 0.5) / float(nx_size),
        (float(voxel_index[1]) + 0.5) / float(ny_size),
        (float(voxel_index[2]) + 0.5) / float(nz_size),
    )


def skeleton_graph_to_tables(
    *,
    skeleton_graph: nx.Graph,
    map_grid: gemmi.FloatGrid,
    skeleton_mask: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    voxel_rows: List[Dict[str, float | int]] = []
    grid_shape = skeleton_mask.shape
    for node in skeleton_graph.nodes():
        frac = voxel_to_fractional(voxel_index=node, grid_shape=grid_shape)
        pos = map_grid.unit_cell.orthogonalize(gemmi.Fractional(*frac))
        voxel_rows.append(
            {
                "x_idx": int(node[0]),
                "y_idx": int(node[1]),
                "z_idx": int(node[2]),
                "degree": int(skeleton_graph.degree(node)),
                "frac_x": float(frac[0]),
                "frac_y": float(frac[1]),
                "frac_z": float(frac[2]),
                "orth_x": float(pos.x),
                "orth_y": float(pos.y),
                "orth_z": float(pos.z),
            }
        )
    edge_rows: List[Dict[str, float | int]] = []
    for node_u, node_v, edge_data in skeleton_graph.edges(data=True):
        edge_rows.append(
            {
                "u_x_idx": int(node_u[0]),
                "u_y_idx": int(node_u[1]),
                "u_z_idx": int(node_u[2]),
                "v_x_idx": int(node_v[0]),
                "v_y_idx": int(node_v[1]),
                "v_z_idx": int(node_v[2]),
                "length_voxels": float(edge_data.get("length_voxels", 1.0)),
            }
        )
    return pd.DataFrame(voxel_rows), pd.DataFrame(edge_rows)


def _largest_component_graph(*, skeleton_graph: nx.Graph) -> nx.Graph:
    if skeleton_graph.number_of_nodes() == 0:
        return nx.Graph()
    nodes = max(nx.connected_components(skeleton_graph), key=len)
    return skeleton_graph.subgraph(nodes).copy()


def _graph_endpoint_branch_counts(*, graph: nx.Graph) -> Tuple[int, int, int]:
    if graph.number_of_nodes() == 0:
        return 0, 0, 0
    degrees = dict(graph.degree())
    endpoints = int(sum(1 for value in degrees.values() if value == 1))
    branchpoints = int(sum(1 for value in degrees.values() if value >= 3))
    isolated = int(sum(1 for value in degrees.values() if value == 0))
    return endpoints, branchpoints, isolated


def _approx_component_diameter_edges(*, graph: nx.Graph) -> int:
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return 0
    try:
        first_node = next(iter(graph.nodes()))
        lengths_a = nx.single_source_shortest_path_length(graph, first_node)
        farthest_a = max(lengths_a, key=lengths_a.get)
        lengths_b = nx.single_source_shortest_path_length(graph, farthest_a)
        return int(max(lengths_b.values())) if lengths_b else 0
    except Exception:
        return 0


def graph_metrics(
    *,
    skeleton_graph: nx.Graph,
    pre_prune_skeleton_voxels: int,
    post_prune_skeleton_voxels: int,
    threshold_voxels_before_cleanup: int,
    threshold_voxels_after_cleanup: int,
    threshold_meta: Dict[str, float],
    total_map_voxels: int,
) -> Dict[str, float | int]:
    n_nodes = int(skeleton_graph.number_of_nodes())
    n_edges = int(skeleton_graph.number_of_edges())
    if n_nodes > 0:
        degree_values = np.asarray([deg for _, deg in skeleton_graph.degree()], dtype=np.float32)
        mean_degree = float(np.mean(degree_values))
        median_degree = float(np.median(degree_values))
        endpoint_count = int(np.sum(degree_values == 1))
        branchpoint_count = int(np.sum(degree_values >= 3))
        isolated_count = int(np.sum(degree_values == 0))
    else:
        mean_degree = 0.0
        median_degree = 0.0
        endpoint_count = 0
        branchpoint_count = 0
        isolated_count = 0

    if n_nodes > 0:
        component_node_lists = [list(nodes) for nodes in nx.connected_components(skeleton_graph)]
        component_sizes = np.asarray([len(nodes) for nodes in component_node_lists], dtype=np.int32)
        component_sizes_sorted = np.sort(component_sizes)[::-1]
        n_components = int(len(component_node_lists))
        largest_component_nodes = int(component_sizes_sorted[0])
        second_component_nodes = int(component_sizes_sorted[1]) if len(component_sizes_sorted) > 1 else 0
        largest_component_fraction = float(largest_component_nodes / n_nodes)
    else:
        n_components = 0
        largest_component_nodes = 0
        second_component_nodes = 0
        largest_component_fraction = 0.0

    total_edge_length_voxels = float(
        sum(float(edge_data.get("length_voxels", 1.0)) for _, _, edge_data in skeleton_graph.edges(data=True))
    )
    mean_edge_length_voxels = float(total_edge_length_voxels / n_edges) if n_edges > 0 else 0.0
    cyclomatic_number = int(n_edges - n_nodes + n_components) if n_nodes > 0 else 0

    largest_component_graph = _largest_component_graph(skeleton_graph=skeleton_graph)
    largest_component_edges = int(largest_component_graph.number_of_edges())
    largest_endpoints, largest_branchpoints, largest_isolated = _graph_endpoint_branch_counts(graph=largest_component_graph)
    largest_component_diameter_edges = _approx_component_diameter_edges(graph=largest_component_graph)

    if largest_component_nodes > 0:
        largest_endpoint_fraction = float(largest_endpoints / largest_component_nodes)
        largest_branchpoint_fraction = float(largest_branchpoints / largest_component_nodes)
        largest_isolated_fraction = float(largest_isolated / largest_component_nodes)
        largest_component_edge_density = float(largest_component_edges / largest_component_nodes)
    else:
        largest_endpoint_fraction = 0.0
        largest_branchpoint_fraction = 0.0
        largest_isolated_fraction = 0.0
        largest_component_edge_density = 0.0

    return {
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "mean_degree": float(mean_degree),
        "median_degree": float(median_degree),
        "n_connected_components": int(n_components),
        "largest_component_nodes": int(largest_component_nodes),
        "second_component_nodes": int(second_component_nodes),
        "largest_component_edges": int(largest_component_edges),
        "largest_component_fraction": float(largest_component_fraction),
        "largest_component_edge_density": float(largest_component_edge_density),
        "largest_component_diameter_edges": int(largest_component_diameter_edges),
        "endpoints": int(endpoint_count),
        "branchpoints": int(branchpoint_count),
        "isolated_nodes": int(isolated_count),
        "endpoint_fraction": float(endpoint_count / n_nodes) if n_nodes > 0 else 0.0,
        "branchpoint_fraction": float(branchpoint_count / n_nodes) if n_nodes > 0 else 0.0,
        "isolated_node_fraction": float(isolated_count / n_nodes) if n_nodes > 0 else 0.0,
        "largest_component_endpoints": int(largest_endpoints),
        "largest_component_branchpoints": int(largest_branchpoints),
        "largest_component_isolated_nodes": int(largest_isolated),
        "largest_component_endpoint_fraction": float(largest_endpoint_fraction),
        "largest_component_branchpoint_fraction": float(largest_branchpoint_fraction),
        "largest_component_isolated_node_fraction": float(largest_isolated_fraction),
        "total_edge_length_voxels": float(total_edge_length_voxels),
        "mean_edge_length_voxels": float(mean_edge_length_voxels),
        "cyclomatic_number": int(cyclomatic_number),
        "pre_prune_skeleton_voxels": int(pre_prune_skeleton_voxels),
        "post_prune_skeleton_voxels": int(post_prune_skeleton_voxels),
        "threshold_voxels_before_cleanup": int(threshold_voxels_before_cleanup),
        "threshold_voxels_after_cleanup": int(threshold_voxels_after_cleanup),
        "threshold_voxel_fraction_before_cleanup": float(threshold_voxels_before_cleanup / total_map_voxels),
        "threshold_voxel_fraction_after_cleanup": float(threshold_voxels_after_cleanup / total_map_voxels),
        "threshold_sigma_equiv": float(threshold_meta["threshold_sigma_equiv"]),
        "threshold_map_units": float(threshold_meta["threshold_map_units"]),
        "map_mean": float(threshold_meta["map_mean"]),
        "map_std": float(threshold_meta["map_std"]),
    }


def write_skeleton_mask_ccp4(*, reference_grid: gemmi.FloatGrid, skeleton_mask: np.ndarray, output_path: str) -> None:
    skeleton_grid = gemmi.FloatGrid(int(skeleton_mask.shape[0]), int(skeleton_mask.shape[1]), int(skeleton_mask.shape[2]))
    skeleton_grid.spacegroup = reference_grid.spacegroup
    skeleton_grid.set_unit_cell(reference_grid.unit_cell)
    np.array(skeleton_grid, copy=False)[:, :, :] = np.asarray(skeleton_mask, dtype=np.float32)
    write_ccp4_map(map_grid=skeleton_grid, output_path=output_path)


def threshold_tag(*, threshold_value: float) -> str:
    return sanitize_label(text=f"thr_{threshold_value:.4f}")


def save_metrics_long(*, metrics_dict: Dict[str, float | int | str], output_path: str) -> None:
    pd.DataFrame([{"metric": key, "value": value} for key, value in metrics_dict.items()]).to_csv(output_path, index=False)


def _first_threshold_where(*, x_values: np.ndarray, y_values: np.ndarray, predicate) -> float:
    for x_value, y_value in zip(x_values.tolist(), y_values.tolist()):
        if predicate(y_value):
            return float(x_value)
    return float("nan")


def _best_threshold(*, x_values: np.ndarray, y_values: np.ndarray, maximize: bool) -> float:
    if y_values.size == 0:
        return float("nan")
    idx = int(np.nanargmax(y_values) if maximize else np.nanargmin(y_values))
    return float(x_values[idx])


def compute_pair_level_summaries(*, aggregate_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: List[Dict[str, float | int | str]] = []
    if aggregate_df.empty:
        return pd.DataFrame(summary_rows)

    grouping_columns = ["pair_index", "pair_label", "phase_column", "fom_column", "map_kind_for_skeleton", "threshold_mode"]
    for group_keys, group_df in aggregate_df.groupby(grouping_columns, sort=True):
        group_df = group_df.sort_values(by="threshold_value_input").reset_index(drop=True)
        threshold_values = group_df["threshold_value_input"].to_numpy(dtype=np.float64)
        largest_fraction = group_df["largest_component_fraction"].to_numpy(dtype=np.float64)
        n_components = group_df["n_connected_components"].to_numpy(dtype=np.float64)
        endpoint_fraction = group_df["endpoint_fraction"].to_numpy(dtype=np.float64)
        mean_degree = group_df["mean_degree"].to_numpy(dtype=np.float64)

        largest_fraction_auc = float(np.trapezoid(y=largest_fraction, x=threshold_values))
        n_components_auc = float(np.trapezoid(y=n_components, x=threshold_values))
        endpoint_fraction_auc = float(np.trapezoid(y=endpoint_fraction, x=threshold_values))
        mean_degree_auc = float(np.trapezoid(y=mean_degree, x=threshold_values))

        largest_frac_drop_below_050 = _first_threshold_where(
            x_values=threshold_values,
            y_values=largest_fraction,
            predicate=lambda value: value < 0.50,
        )

        best_largest_idx = int(np.nanargmax(largest_fraction))
        row = {
            "pair_index": int(group_keys[0]),
            "pair_label": str(group_keys[1]),
            "phase_column": str(group_keys[2]),
            "fom_column": str(group_keys[3]),
            "map_kind_for_skeleton": str(group_keys[4]),
            "threshold_mode": str(group_keys[5]),
            "n_thresholds": int(len(group_df)),
            "threshold_min": float(np.min(threshold_values)),
            "threshold_max": float(np.max(threshold_values)),
            "largest_component_fraction_auc": float(largest_fraction_auc),
            "n_connected_components_auc": float(n_components_auc),
            "endpoint_fraction_auc": float(endpoint_fraction_auc),
            "mean_degree_auc": float(mean_degree_auc),
            "largest_fraction_first_below_0p50_threshold": float(largest_frac_drop_below_050),
            "best_largest_fraction": float(largest_fraction[best_largest_idx]),
            "best_largest_fraction_threshold": float(threshold_values[best_largest_idx]),
            "best_largest_fraction_n_components": float(n_components[best_largest_idx]),
            "best_largest_fraction_endpoint_fraction": float(endpoint_fraction[best_largest_idx]),
            "best_largest_fraction_mean_degree": float(mean_degree[best_largest_idx]),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["largest_component_fraction_auc", "best_largest_fraction"],
            ascending=[False, False]
        ).reset_index(drop=True)
        summary_df["pair_rank_by_auc"] = np.arange(1, len(summary_df) + 1)
    return summary_df


def build_fixed_threshold_proxy_summary(
    *,
    aggregate_df: pd.DataFrame,
    pair_level_summary_df: pd.DataFrame,
    target_threshold: float,
) -> pd.DataFrame:
    if aggregate_df.empty:
        return pd.DataFrame([])

    proxy_rows: List[Dict[str, float | int | str]] = []
    grouping_columns = ["pair_index", "pair_label", "phase_column", "fom_column"]

    for group_keys, group_df in aggregate_df.groupby(grouping_columns, sort=True):
        group_df = group_df.copy()
        idx_best = int(np.argmin(np.abs(group_df["threshold_value_input"].to_numpy(dtype=float) - float(target_threshold))))
        row_fixed = group_df.iloc[idx_best]

        pair_summary_match = pair_level_summary_df.loc[
            (pair_level_summary_df["pair_index"] == int(group_keys[0])) &
            (pair_level_summary_df["pair_label"] == str(group_keys[1]))
        ].copy()

        auc_value = np.nan
        if pair_summary_match.shape[0] > 0:
            auc_value = float(pair_summary_match.iloc[0]["largest_component_fraction_auc"])

        proxy_rows.append(
            {
                "pair_index": int(group_keys[0]),
                "pair_label": str(group_keys[1]),
                "phase_column": str(group_keys[2]),
                "fom_column": str(group_keys[3]),
                "target_threshold_requested": float(target_threshold),
                "target_threshold_used": float(row_fixed["threshold_value_input"]),
                "largest_component_fraction_at_target": float(row_fixed["largest_component_fraction"]),
                "n_connected_components_at_target": int(row_fixed["n_connected_components"]),
                "endpoint_fraction_at_target": float(row_fixed["endpoint_fraction"]),
                "mean_degree_at_target": float(row_fixed["mean_degree"]),
                "largest_component_fraction_auc": float(auc_value),
            }
        )

    proxy_df = pd.DataFrame(proxy_rows)
    if not proxy_df.empty:
        proxy_df = proxy_df.sort_values(
            by=[
                "largest_component_fraction_at_target",
                "largest_component_fraction_auc",
                "endpoint_fraction_at_target",
                "n_connected_components_at_target",
            ],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        proxy_df["traceability_proxy_rank"] = np.arange(1, len(proxy_df) + 1)
    return proxy_df


def _apply_plot_style(*, axis: plt.Axes) -> None:
    axis.minorticks_on()
    axis.grid(which="major", linestyle="--", color="gray", alpha=0.6)
    axis.grid(which="minor", linestyle=":", color="lightgray", alpha=0.7)
    axis.tick_params(axis="both", which="major", labelsize=14, size=10)
    axis.tick_params(axis="both", which="minor", size=5)
    axis.xaxis.label.set_size(14)
    axis.yaxis.label.set_size(14)
    axis.title.set_size(14)


def write_pair_threshold_plots(*, aggregate_df: pd.DataFrame, out_dir: str, pdb_id: str) -> None:
    if aggregate_df.empty:
        return
    grouping_columns = ["pair_index", "pair_label", "phase_column", "fom_column"]
    plots_dir = os.path.join(out_dir, f"{pdb_id}_plots")
    os.makedirs(plots_dir, exist_ok=True)

    for group_keys, group_df in aggregate_df.groupby(grouping_columns, sort=True):
        group_df = group_df.sort_values(by="threshold_value_input").reset_index(drop=True)
        pair_index = int(group_keys[0])
        pair_label = str(group_keys[1])
        phase_column = str(group_keys[2])
        fom_column = str(group_keys[3])

        x_values = group_df["threshold_value_input"].to_numpy(dtype=np.float64)

        for metric_name, metric_title in _METRIC_PLOT_SPECS:
            if metric_name not in group_df.columns:
                continue
            y_values = group_df[metric_name].to_numpy(dtype=np.float64)
            fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(7.5, 5.5), dpi=160)
            axis.plot(x_values, y_values, marker="o")
            axis.set_xlabel("Threshold")
            axis.set_ylabel(metric_title)
            axis.set_title(f"{phase_column}:{fom_column}\n{metric_title} vs threshold")
            _apply_plot_style(axis=axis)
            fig.tight_layout()
            fig.savefig(
                os.path.join(plots_dir, f"{pdb_id}_{pair_index:02d}_{pair_label}_{metric_name}_vs_threshold.png"),
                bbox_inches="tight"
            )
            plt.close(fig)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11.0, 8.5), dpi=160)
        combo_metrics = (
            ("largest_component_fraction", "Largest-component fraction"),
            ("n_connected_components", "Connected components"),
            ("endpoint_fraction", "Endpoint fraction"),
            ("mean_degree", "Mean degree"),
        )
        for axis, (metric_name, metric_title) in zip(axes.ravel(), combo_metrics):
            axis.plot(x_values, group_df[metric_name].to_numpy(dtype=np.float64), marker="o")
            axis.set_xlabel("Threshold")
            axis.set_ylabel(metric_title)
            axis.set_title(metric_title)
            _apply_plot_style(axis=axis)
        fig.suptitle(f"{phase_column}:{fom_column}")
        fig.tight_layout()
        fig.savefig(
            os.path.join(plots_dir, f"{pdb_id}_{pair_index:02d}_{pair_label}_summary_panel.png"),
            bbox_inches="tight"
        )
        plt.close(fig)


def write_global_ranking_plot(*, proxy_df: pd.DataFrame, out_dir: str, pdb_id: str) -> None:
    if proxy_df.empty:
        return
    ranking_df = proxy_df.sort_values(by="traceability_proxy_rank", ascending=True).reset_index(drop=True)
    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(10.0, max(4.5, 0.6 * len(ranking_df) + 1.5)), dpi=160)
    y_positions = np.arange(len(ranking_df))
    axis.barh(y_positions, ranking_df["largest_component_fraction_at_target"].to_numpy(dtype=np.float64))
    labels = [f"{row.phase_column}:{row.fom_column}" for _, row in ranking_df.iterrows()]
    axis.set_yticks(y_positions)
    axis.set_yticklabels(labels, fontsize=12)
    axis.invert_yaxis()
    axis.set_xlabel("Largest-component fraction at target threshold")
    axis.set_title(f"{pdb_id}: pair ranking by fixed-threshold traceability proxy")
    _apply_plot_style(axis=axis)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{pdb_id}_pair_traceability_proxy_ranking.png"), bbox_inches="tight")
    plt.close(fig)


def process_one_csv(*, csv_path: str, args: argparse.Namespace) -> Dict[str, object]:
    pdb_id = infer_pdb_id_from_csv_path(csv_path=csv_path)
    pdb_out_dir = os.path.join(args.out_dir, pdb_id)
    os.makedirs(pdb_out_dir, exist_ok=True)

    dataframe = load_reflection_table(csv_path=csv_path, amplitude_column=args.amplitude_column)
    dataframe = apply_resolution_filter(dataframe=dataframe, dmin=args.dmin, dmax=args.dmax)
    metadata = extract_unique_metadata(dataframe=dataframe)
    pair_specs = parse_pair_specs(args=args, dataframe=dataframe)
    threshold_values = parse_threshold_values(args=args)

    aggregate_rows: List[Dict[str, float | int | str]] = []
    pair_summary_rows: List[Dict[str, float | int | str]] = []

    print(f"\n=== {pdb_id} ===")
    print(f"Input CSV: {csv_path}")
    print(f"Reflections kept after filters: {len(dataframe)}")
    print(f"Space group: {metadata.spacegroup_name}")
    print(
        f"Cell: {metadata.length_a:.4f} {metadata.length_b:.4f} {metadata.length_c:.4f}  "
        f"{metadata.angle_alpha:.2f} {metadata.angle_beta:.2f} {metadata.angle_gamma:.2f}"
    )

    for pair_index, pair_spec in enumerate(pair_specs, start=1):
        pair_dir = os.path.join(pdb_out_dir, f"{pdb_id}_{pair_index:02d}_{pair_spec.label}")
        os.makedirs(pair_dir, exist_ok=True)

        mtz = build_mtz_from_dataframe(
            dataframe=dataframe,
            metadata=metadata,
            amplitude_column=args.amplitude_column,
            phase_column=pair_spec.phase_column,
            fom_column=pair_spec.fom_column,
            phase_unit=args.phase_unit,
        )
        mtz_path = os.path.join(pair_dir, f"{pdb_id}_{pair_spec.label}_reconstructed_coefficients.mtz")
        mtz.write_to_file(mtz_path)

        raw_map_grid = transform_mtz_to_map(mtz=mtz, f_label="F_RAW", phi_label="PHI_RAW", sample_rate=args.sample_rate)
        raw_map_path = os.path.join(pair_dir, f"{pdb_id}_{pair_spec.label}_raw_map_from_FP_PHIC.ccp4")
        write_ccp4_map(map_grid=raw_map_grid, output_path=raw_map_path)

        fom_map_grid = transform_mtz_to_map(mtz=mtz, f_label="F_FOM", phi_label="PHI_FOM", sample_rate=args.sample_rate)
        fom_map_path = os.path.join(pair_dir, f"{pdb_id}_{pair_spec.label}_fom_weighted_2mFo_like_map.ccp4")
        write_ccp4_map(map_grid=fom_map_grid, output_path=fom_map_path)

        raw_map_array = float_grid_to_numpy(map_grid=raw_map_grid)
        fom_map_array = float_grid_to_numpy(map_grid=fom_map_grid)

        pair_summary_rows.append(
            {
                "pdb_id": pdb_id,
                "pair_index": int(pair_index),
                "pair_label": pair_spec.label,
                "phase_column": pair_spec.phase_column,
                "fom_column": pair_spec.fom_column,
                "mtz_path": mtz_path,
                "raw_map_path": raw_map_path,
                "fom_weighted_map_path": fom_map_path,
                "raw_map_shape_x": int(raw_map_array.shape[0]),
                "raw_map_shape_y": int(raw_map_array.shape[1]),
                "raw_map_shape_z": int(raw_map_array.shape[2]),
            }
        )

        map_grid_for_skeleton = fom_map_grid if args.map_kind_for_skeleton == "fom_weighted" else raw_map_grid
        map_array_for_skeleton = fom_map_array if args.map_kind_for_skeleton == "fom_weighted" else raw_map_array
        total_map_voxels = int(np.prod(map_array_for_skeleton.shape))

        print(f"[{pair_index}/{len(pair_specs)}] {pair_spec.phase_column}:{pair_spec.fom_column}")

        for threshold_value in threshold_values:
            thr_tag = threshold_tag(threshold_value=threshold_value)
            thr_dir = os.path.join(pair_dir, f"{pdb_id}_{pair_spec.label}_{thr_tag}")
            os.makedirs(thr_dir, exist_ok=True)

            binary_mask, threshold_meta = threshold_map(
                map_array=map_array_for_skeleton,
                threshold_mode=args.threshold_mode,
                threshold_value=float(threshold_value),
            )
            threshold_voxels_before_cleanup = int(np.count_nonzero(binary_mask))

            binary_mask = apply_binary_closing(
                binary_mask=binary_mask,
                iterations=args.binary_closing_iterations,
            )
            binary_mask = remove_small_components(
                binary_mask=binary_mask,
                min_component_size=args.min_component_size,
            )
            threshold_voxels_after_cleanup = int(np.count_nonzero(binary_mask))

            skeleton_mask_pre_prune = skeletonize(binary_mask)
            pre_prune_skeleton_voxels = int(np.count_nonzero(skeleton_mask_pre_prune))

            skeleton_mask = prune_skeleton_tips(
                skeleton_mask=skeleton_mask_pre_prune,
                connectivity=args.edge_connectivity,
                iterations=args.prune_tip_iterations,
            )
            post_prune_skeleton_voxels = int(np.count_nonzero(skeleton_mask))

            skeleton_graph = skeleton_mask_to_graph(
                skeleton_mask=skeleton_mask,
                connectivity=args.edge_connectivity,
            )

            metrics = graph_metrics(
                skeleton_graph=skeleton_graph,
                pre_prune_skeleton_voxels=pre_prune_skeleton_voxels,
                post_prune_skeleton_voxels=post_prune_skeleton_voxels,
                threshold_voxels_before_cleanup=threshold_voxels_before_cleanup,
                threshold_voxels_after_cleanup=threshold_voxels_after_cleanup,
                threshold_meta=threshold_meta,
                total_map_voxels=total_map_voxels,
            )

            row: Dict[str, float | int | str] = {
                "pdb_id": pdb_id,
                "pair_index": int(pair_index),
                "pair_label": pair_spec.label,
                "phase_column": pair_spec.phase_column,
                "fom_column": pair_spec.fom_column,
                "map_kind_for_skeleton": args.map_kind_for_skeleton,
                "threshold_mode": args.threshold_mode,
                "threshold_value_input": float(threshold_value),
                "threshold_dir": thr_dir,
            }
            row.update(metrics)
            aggregate_rows.append(row)

            metrics_wide_path = os.path.join(thr_dir, f"{pdb_id}_{pair_spec.label}_{thr_tag}_skeleton_metrics_wide.csv")
            pd.DataFrame([row]).to_csv(metrics_wide_path, index=False)

            metrics_long_path = os.path.join(thr_dir, f"{pdb_id}_{pair_spec.label}_{thr_tag}_skeleton_metrics_long.csv")
            save_metrics_long(metrics_dict=row, output_path=metrics_long_path)

            skeleton_map_path = os.path.join(thr_dir, f"{pdb_id}_{pair_spec.label}_{thr_tag}_skeleton_mask.ccp4")
            write_skeleton_mask_ccp4(
                reference_grid=map_grid_for_skeleton,
                skeleton_mask=skeleton_mask,
                output_path=skeleton_map_path,
            )

            if args.write_per_run_tables:
                voxels_df, edges_df = skeleton_graph_to_tables(
                    skeleton_graph=skeleton_graph,
                    map_grid=map_grid_for_skeleton,
                    skeleton_mask=skeleton_mask,
                )
                voxels_df.to_csv(os.path.join(thr_dir, f"{pdb_id}_{pair_spec.label}_{thr_tag}_skeleton_voxels.csv"), index=False)
                edges_df.to_csv(os.path.join(thr_dir, f"{pdb_id}_{pair_spec.label}_{thr_tag}_skeleton_edges.csv"), index=False)

            print(
                f"  threshold={threshold_value:g}  "
                f"nodes={metrics['n_nodes']}  edges={metrics['n_edges']}  "
                f"largest_frac={metrics['largest_component_fraction']:.4f}  "
                f"components={metrics['n_connected_components']}  "
                f"endpoint_frac={metrics['endpoint_fraction']:.4f}  "
                f"mean_degree={metrics['mean_degree']:.4f}"
            )

    aggregate_df = pd.DataFrame(aggregate_rows)
    pair_map_summary_df = pd.DataFrame(pair_summary_rows)
    pair_level_summary_df = compute_pair_level_summaries(aggregate_df=aggregate_df)
    proxy_df = build_fixed_threshold_proxy_summary(
        aggregate_df=aggregate_df,
        pair_level_summary_df=pair_level_summary_df,
        target_threshold=float(args.target_threshold),
    )

    aggregate_csv_path = os.path.join(pdb_out_dir, f"{pdb_id}_all_skeleton_metrics.csv")
    aggregate_df.to_csv(aggregate_csv_path, index=False)

    pair_map_summary_csv_path = os.path.join(pdb_out_dir, f"{pdb_id}_pair_map_summary.csv")
    pair_map_summary_df.to_csv(pair_map_summary_csv_path, index=False)

    pair_level_summary_csv_path = os.path.join(pdb_out_dir, f"{pdb_id}_pair_threshold_summary.csv")
    pair_level_summary_df.to_csv(pair_level_summary_csv_path, index=False)

    proxy_csv_path = os.path.join(pdb_out_dir, f"{pdb_id}_traceability_proxy_summary.csv")
    proxy_df.to_csv(proxy_csv_path, index=False)

    if not args.disable_summary_plots:
        write_pair_threshold_plots(aggregate_df=aggregate_df, out_dir=pdb_out_dir, pdb_id=pdb_id)
        write_global_ranking_plot(proxy_df=proxy_df, out_dir=pdb_out_dir, pdb_id=pdb_id)

    summary_lines = [
        f"pdb_id: {pdb_id}",
        f"Input CSV: {csv_path}",
        f"Reflections kept after filters: {len(dataframe)}",
        f"Space group: {metadata.spacegroup_name}",
        (
            f"Cell: {metadata.length_a:.4f} {metadata.length_b:.4f} {metadata.length_c:.4f}  "
            f"{metadata.angle_alpha:.2f} {metadata.angle_beta:.2f} {metadata.angle_gamma:.2f}"
        ),
        f"Amplitude column: {args.amplitude_column}",
        f"Pairs processed: {len(pair_specs)}",
        f"Thresholds processed: {threshold_values}",
        f"Target threshold for proxy metrics: {args.target_threshold}",
        f"Skeleton map kind: {args.map_kind_for_skeleton}",
        f"Aggregate metrics CSV: {aggregate_csv_path}",
        f"Pair map summary CSV: {pair_map_summary_csv_path}",
        f"Pair threshold summary CSV: {pair_level_summary_csv_path}",
        f"Traceability proxy CSV: {proxy_csv_path}",
        f"Summary plots enabled: {not args.disable_summary_plots}",
    ]
    run_summary_path = os.path.join(pdb_out_dir, f"{pdb_id}_run_summary.txt")
    with open(run_summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines) + "\n")

    print("\n".join(summary_lines))

    best_pair_row = proxy_df.iloc[0].to_dict() if proxy_df.shape[0] > 0 else {}
    return {
        "pdb_id": pdb_id,
        "csv_in": csv_path,
        "n_reflections": int(len(dataframe)),
        "spacegroup": metadata.spacegroup_name,
        "n_pairs": int(len(pair_specs)),
        "best_pair_label": str(best_pair_row.get("pair_label", "")),
        "best_phase_column": str(best_pair_row.get("phase_column", "")),
        "best_fom_column": str(best_pair_row.get("fom_column", "")),
        "best_largest_component_fraction_at_target": float(best_pair_row.get("largest_component_fraction_at_target", np.nan)) if best_pair_row else np.nan,
        "best_n_connected_components_at_target": float(best_pair_row.get("n_connected_components_at_target", np.nan)) if best_pair_row else np.nan,
        "best_endpoint_fraction_at_target": float(best_pair_row.get("endpoint_fraction_at_target", np.nan)) if best_pair_row else np.nan,
        "best_mean_degree_at_target": float(best_pair_row.get("mean_degree_at_target", np.nan)) if best_pair_row else np.nan,
        "best_largest_component_fraction_auc": float(best_pair_row.get("largest_component_fraction_auc", np.nan)) if best_pair_row else np.nan,
        "traceability_proxy_summary_csv": proxy_csv_path,
        "pair_threshold_summary_csv": pair_level_summary_csv_path,
        "all_skeleton_metrics_csv": aggregate_csv_path,
        "run_summary_txt": run_summary_path,
    }


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(args.csv_dir, "*.csv")))
    if len(csv_paths) < 1:
        raise InputError(f"No CSV files found in directory: {args.csv_dir}")

    batch_rows: List[Dict[str, object]] = []
    n_ok = 0
    n_fail = 0

    print(f"CSV directory: {args.csv_dir}")
    print(f"Output directory: {args.out_dir}")
    print(f"CSV files found: {len(csv_paths)}")

    for index, csv_path in enumerate(csv_paths, start=1):
        print(f"\n===== [{index}/{len(csv_paths)}] {os.path.basename(csv_path)} =====")
        try:
            batch_rows.append(process_one_csv(csv_path=csv_path, args=args))
            n_ok += 1
        except Exception as exc:
            pdb_id = infer_pdb_id_from_csv_path(csv_path=csv_path)
            n_fail += 1
            print(f"ERROR processing {pdb_id}: {exc}", file=sys.stderr)
            batch_rows.append(
                {
                    "pdb_id": pdb_id,
                    "csv_in": csv_path,
                    "status": "FAILED",
                    "error": str(exc),
                }
            )

    batch_summary_df = pd.DataFrame(batch_rows)
    batch_summary_csv_path = os.path.join(args.out_dir, "batch_traceability_proxy_summary.csv")
    batch_summary_df.to_csv(batch_summary_csv_path, index=False)

    print("\n===== BATCH SUMMARY =====")
    print(f"Successful: {n_ok}")
    print(f"Failed: {n_fail}")
    print(f"Batch summary CSV: {batch_summary_csv_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)