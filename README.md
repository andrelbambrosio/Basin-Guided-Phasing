# Basin-Guided-Phasing
<<<<<<< HEAD

> **Citation notice**  
> If you use any part of this codebase, or results derived from it, in academic work, presentations, preprints, or downstream software, please cite the associated manuscript:  
> **[Mateus P Otto, Felipe S Lincoln, Andre LB Ambrosio]. Binary phase initialization enables basin entry in macromolecular structure determination. bioRxiv (2026). DOI: [DOI TO BE ADDED]**  
> This repository will be updated with the final citation details once the preprint is posted and later linked to the peer-reviewed publication.

## Repository description

This repository contains a staged computational framework for investigating how much phase information is sufficient to initiate macromolecular crystal structure solution. The codebase accompanies our work on **attenuated signed amplitudes** as a minimal, symmetry-consistent initializer for macromolecular phasing, and it is organized around a simple idea: successful structure determination does not require globally accurate initial phases, but rather entry into a **basin of attraction** from which density modification and automated model building can converge. In this framework, exact centric phase supports are retained, while the acentric phase stream can be aggressively compressed to a binary set {0,pi}, both in an attenuated representation with a universal constant figure of merit of 2/pi. The repository provides the tools needed to test that hypothesis systematically across large cohorts of crystallographic datasets, calibrate low-cost basin-proximity surrogates, and evaluate how robust those surrogates remain under controlled perturbation.

The pipeline is organized into numbered stages that reflect the major components of the analysis: PDB data acquisition and preparation, empirical centric-burden analysis, acentric phase compression, Basin Score calibration, and Basin Score degradation. These stages support cohort-scale processing of reflection tables, symmetry-aware centric analysis, generation of compressed phase initializers, multiwindow density modification, skeleton-based map traceability analysis at multiple map sigma thresholds, and downstream benchmarking with `phenix.autobuild`. The central practical output is a branch-balanced **Basin Score**, calibrated against rebuilding outcomes and designed to estimate whether a reduced phase set is already compatible with productive density modification and model building. Controlled degradation and anti-phase analysis then extend this framework by probing how basin compatibility erodes, transitions, or reappears under explicit phase corruption.

Taken together, the repository is intended both as a reproducible implementation of the analyses reported in the manuscript and as a starting point for future **basin-guided ab initio phasing** strategies. It combines standard Python 3 utilities, `phenix.python` workflows, MTZ and map processing, density modification and skeleton connectivity analysis, degradation experiments, and AutoBuild validation into a single, structured codebase. The broader goal is to support future development of reduced, physically meaningful phase initializers and cheap basin-scoring functions that can guide combinatorial or learning-assisted searches in macromolecular phasing.

## Code naming structure

Code naming follows this convention:

**X = major analysis stage, Y = execution order within that stage**

So the pattern is:

```text
XY_stepname.py
```

with:

- **0Y** = PDB data acquisition and preparation
- **1Y** = empirical centric burden
- **2Y** = acentric compression
- **3Y** = Basin Score calibration
- **4Y** = Basin Score degradation

This numbering is meant to make the pipeline order immediately visible from the filename.

<img width="1785" height="1027" alt="image" src="https://github.com/user-attachments/assets/258f3d12-9df9-438c-8f9f-5606179bb40b" />

## Repository overview

### Stage 0 — PDB data acquisition and preparation
- `00_select_cohort_ids_with_query_summary.py`
- `01_download_build_reflection_tables.py`
- `02_survey_spacegroup_coverage.py`

### Stage 1 — Empirical centric burden
- `10_measure_empirical_centric_burden.py`
- `11_plot_symmetry_complexity_vs_centric_burden.py`
- `12_analyze_trace_resolved_centric_phase_supports.py`

### Stage 2 — Acentric compression
- `20_bin_acentric_phases.py`
- `21_prepare_autobuild_mtz_inputs.py`
- `22_run_phenix_autobuild.py`
- `23_analyze_autobuild_outcomes.py`

### Stage 3 — Basin Score calibration
- `30_run_multiwindow_density_modification.py`
- `31_reconstruct_maps_and_extract_skeleton_proxies.py`
- `32_compute_basin_score_deployed.py`

### Stage 4 — Basin Score degradation
- `40_BasinScore_degradation.py`
- `40_api_density_modification.py`
- `40_api_skeleton_proxy.py`
- `40_api_reconstruct_map_and_skeleton_batch.py`
- `40_api_autobuild_mtz_writer.py`
- `41_make_antiphase_degradation_mtzs.py`
- `42_run_phenix_autobuild_from_degradation_mtz.py`
- `43_analyze_autobuild_degradation_results.py`

### Supporting helper
- `phenix_like_density_modification_from_csv_py27.py`

## Logging philosophy

Pipeline scripts are designed to use comprehensive logging, typically including:

- run prefix
- reconstructed command line
- key input/output paths
- per-stage progress reporting
- final output summary
- timestamped log files stored under a `LOGS/` folder

## License

This repository is distributed under the **BSD 3-Clause License**. See the `LICENSE` file for details.

## Citation metadata

A `CITATION.cff` file is included so GitHub can expose a “Cite this repository” entry.  
Before public release, please update:

- manuscript DOI
- author list
- version / release tag
- repository URL
=======
Study of macromolecular crystallographic phasing from attenuated signed amplitudes: cohort assembly, centric-burden analysis, acentric compression, Basin Score calibration, controlled degradation, anti-phase MTZ generation, AutoBuild validation, and degradation-result analysis for basin-guided ab initio structure solution.
>>>>>>> 9f34fc7b8e8673160f0678353441bf327a339658
