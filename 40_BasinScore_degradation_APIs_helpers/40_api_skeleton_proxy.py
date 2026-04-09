#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
api44_skeleton_proxy.py

Bridge wrapper for PC44 versions whose callable API is:
    process_one_csv(*, csv_path: str, args: argparse.Namespace)

This wrapper:
- loads PC44 from --pc44_script path
- builds the exact argparse.Namespace-style object PC44 expects
- writes a compact JSON summary for PC50 or direct debugging
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback


def _parse_phase_fom_pairs(value):
    if value is None:
        return None
    parts = [x.strip() for x in str(value).split(",") if x.strip()]
    return parts if parts else None


def _load_module_from_path(module_name, file_path):
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise IOError("PC44 script not found: %s" % file_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError("Could not build import spec for %s" % file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        try:
            sys.modules.pop(module_name, None)
        except Exception:
            pass
        raise
    return module


def _extract_proxy_metrics(summary):
    proxy = {}
    if isinstance(summary, dict):
        candidate_keys = [
            "best_largest_component_fraction_at_target",
            "best_endpoint_fraction_at_target",
            "best_n_connected_components_at_target",
            "best_mean_degree_at_target",
            "best_largest_component_fraction_auc",
        ]
        for key in candidate_keys:
            if key in summary:
                proxy[key] = summary.get(key)

        nested_keys = [
            "proxy_summary",
            "best_proxy_summary",
            "fixed_threshold_proxy_summary",
        ]
        for nested_key in nested_keys:
            nested = summary.get(nested_key)
            if isinstance(nested, dict):
                for key in candidate_keys:
                    if key in nested and key not in proxy:
                        proxy[key] = nested.get(key)
    return proxy


def main():
    parser = argparse.ArgumentParser(description="Run the callable skeleton proxy API on one CSV and emit a compact JSON summary.")
    parser.add_argument("--pc44_script", required=True, help="Path to pc44_reconstruct_map_and_skeleton_batch_folder.py")
    parser.add_argument("--csv_in", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--phase_unit", default="deg")
    parser.add_argument("--sample_rate", type=float, default=3.0)
    parser.add_argument("--amplitude_column", default="FP")
    parser.add_argument("--map_kind_for_skeleton", default="fom_weighted")
    parser.add_argument("--threshold_mode", default="sigma")
    parser.add_argument("--threshold_values", default="1.2")
    parser.add_argument("--target_threshold", type=float, default=1.2)
    parser.add_argument("--binary_closing_iterations", type=int, default=0)
    parser.add_argument("--min_component_size", type=int, default=0)
    parser.add_argument("--prune_tip_iterations", type=int, default=3)
    parser.add_argument("--edge_connectivity", type=int, default=26)
    parser.add_argument("--phase_fom_pairs", default=None)
    parser.add_argument("--phase_column", default=None)
    parser.add_argument("--fom_column", default=None)
    parser.add_argument("--dmin", type=float, default=3.5)
    parser.add_argument("--dmax", type=float, default=20.0)
    parser.add_argument("--write_per_run_tables", action="store_true")
    parser.add_argument("--disable_summary_plots", action="store_true")
    parser.add_argument("--json_out", default=None, help="Optional explicit JSON output path")
    args_cli = parser.parse_args()

    os.makedirs(args_cli.out_dir, exist_ok=True)

    json_out = args_cli.json_out
    if not json_out:
        json_out = os.path.join(args_cli.out_dir, "pc44_result_summary.json")

    result = {
        "ok": False,
        "csv_in": os.path.abspath(args_cli.csv_in),
        "out_dir": os.path.abspath(args_cli.out_dir),
        "pc44_script": os.path.abspath(args_cli.pc44_script),
    }

    try:
        pc44 = _load_module_from_path("api44_pc44_runtime_module", args_cli.pc44_script)

        args_for_pc44 = argparse.Namespace(
            csv_dir=os.path.dirname(os.path.abspath(args_cli.csv_in)),
            out_dir=args_cli.out_dir,
            phase_unit=args_cli.phase_unit,
            amplitude_column=args_cli.amplitude_column,
            phase_column=args_cli.phase_column,
            fom_column=args_cli.fom_column,
            phase_fom_pairs=_parse_phase_fom_pairs(args_cli.phase_fom_pairs),
            sample_rate=args_cli.sample_rate,
            map_kind_for_skeleton=args_cli.map_kind_for_skeleton,
            threshold_mode=args_cli.threshold_mode,
            threshold_value=None,
            threshold_values=args_cli.threshold_values,
            target_threshold=args_cli.target_threshold,
            binary_closing_iterations=args_cli.binary_closing_iterations,
            min_component_size=args_cli.min_component_size,
            prune_tip_iterations=args_cli.prune_tip_iterations,
            edge_connectivity=args_cli.edge_connectivity,
            dmin=args_cli.dmin,
            dmax=args_cli.dmax,
            write_per_run_tables=bool(args_cli.write_per_run_tables),
            disable_summary_plots=bool(args_cli.disable_summary_plots),
        )

        summary = pc44.process_one_csv(
            csv_path=args_cli.csv_in,
            args=args_for_pc44,
        )

        if summary is None:
            raise RuntimeError("PC44 returned None from process_one_csv().")

        result["ok"] = True
        result["summary"] = summary
        result["proxy_metrics"] = _extract_proxy_metrics(summary)

    except Exception as exc:
        result["ok"] = False
        result["error"] = "%s: %s" % (exc.__class__.__name__, str(exc))
        result["traceback"] = traceback.format_exc()

    with open(json_out, "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)

    if result["ok"]:
        print(json.dumps({
            "ok": True,
            "json_out": json_out,
            "proxy_metrics": result.get("proxy_metrics", {}),
        }, indent=2, sort_keys=True))
        return 0

    print(json.dumps({
        "ok": False,
        "json_out": json_out,
        "error": result.get("error"),
    }, indent=2, sort_keys=True), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
