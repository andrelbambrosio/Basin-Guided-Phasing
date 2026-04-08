# Tutorial

This tutorial gives a practical execution path through the repository, from cohort selection to Basin Score degradation and downstream AutoBuild analysis. It follows the **committed repository script names** and reflects the actual behavior of the current scripts.

Phenix (https://phenix-online.org/) must be installed.

The Basin Score pipeline was tested on Ubuntu 24.0 only.

---

## General conventions

### Script naming
Scripts follow the repository stage convention:

```text
XY_stepname.py
```

where:

- `0Y` = PDB data acquisition and preparation
- `1Y` = empirical centric burden
- `2Y` = acentric compression
- `3Y` = Basin Score calibration
- `4Y` = Basin Score degradation

### Python environments
Some scripts must be run with standard `python`, while others must be run with `phenix.python`.


General rule:
- use `python` for pure Python / plotting / CSV workflows
- use `phenix.python` for density-modification or Phenix/cctbx-dependent workflows

### Logging
Most scripts write timestamped logs into a `LOGS/` directory. Inspect these logs first whenever a stage fails or appears stalled.

---

# Stage 0 — PDB data acquisition and preparation

## 00. Select a reproducible cohort of PDB IDs

**Script:** `00_select_cohort_ids_with_query_summary.py`

Starting with querying the Protein Data Bank for your favorite keywords and export the results as a list in one or more `.txt` file(s). Save the `.txt` file(s) in a folder such as ./00_RCSB_Query.

This script reads one or more `.txt` files from `--query_dir`, extracts valid PDB IDs, deduplicates them, applies a seed-stable random permutation, and writes a JSON cohort file with provenance metadata. It also supports a compact query-summary JSON file for storing query date, label, notes, and advanced query text. The sampling is **prefix-stable**: for a fixed seed and the same pooled ID order, increasing `--n` preserves the smaller sample as a prefix of the larger one.

Typical command:

```bash
python 00_select_cohort_ids_with_query_summary.py \
  --query_dir ./00_RCSB_Query \
  --n 50 \
  --seed 42 \
  --out_path ./sampled_cohort_50.json \
  --logs_dir ./LOGS \
  --query_summary_file ./query_summary_template.json
```

Main output:
- `sampled_cohort_50.json`

---

## 01. Download source files and build reflection tables

**Script:** `01_download_build_reflection_tables.py`

This is the main stage-01 data-generation pipeline. It performs:

1. download of **PDB-REDO** coordinate and MTZ files
2. download of **RCSB FASTA** files
3. validation of downloaded files
4. retrieval of solvent fraction from the **RCSB Data API**
5. generation of standardized reciprocal-space CSV reflection tables
6. optional ECALC-like normalization

Outputs are written relative to `--base_dir`:

- `01_Download_Files/`
  - `<PDB>_iREDO.pdb`
  - `<PDB>_iREDO.mtz`
  - `<PDB>_rcsb.fasta`
- `02_Run_RS/`
  - `<PDB>_rs.csv`
- `03_Run_Ecalc/`
  - `<PDB>_rs_ecalc.csv`

Key implementation details:
- uses `PHIC_ALL`, not PHIC-aligned values
- canonicalizes phases and maps `-180°` to `+180°`
- writes `PHIC_ALL` as integer degrees in the RS CSV
- adds `FreeR_flag`
- adds crystal-cell metadata, wavelength, and solvent fraction
- supports three ECALC modes:
  - `global_df`
  - `per_dataset`
  - `none`

Typical command:

```bash
python 01_download_build_reflection_tables.py \
  --base_dir ./ \
  --num_processes 25 \
  --min_resol 20.0 \
  --max_resol 2.5 \
  --ecalc_n_shells 10 \
  --config_json ./sampled_cohort_50.json \
  --logs_dir ./LOGS \
  --ecalc_mode per_dataset
```

---

## 02. Survey space-group coverage

**Script:** `02_survey_spacegroup_coverage.py`

This script audits space-group coverage across a directory of per-dataset CSV files, for example `*_rs.csv` or `*_rs_ecalc.csv`.

It reports:
- per-space-group counts and percentages
- per-crystal-system counts and percentages
- number of unique SGs per crystal system
- lists of SGs per crystal system
- centrosymmetric SGs and datasets flagged for exclusion

Output policy:
- the main `.log` file is written directly into `--logs_dir`
- all tabular outputs are written into a dedicated run folder:
  - `LOGS/02_survey_spacegroup_coverage__<timestamp>/`

Typical command:

```bash
python 02_survey_spacegroup_coverage.py \
  --source_csv_dir ./03_Run_Ecalc \
  --source_csv_pattern "*_rs_ecalc.csv" \
  --sg_col SG \
  --logs_dir ./LOGS
```

---

# Stage 1 — Empirical centric burden

## 10. Measure empirical centric burden

**Script:** `10_measure_empirical_centric_burden.py`

This stage measures empirical centric burden across a cohort of reflection CSV files. It computes:

- centric and acentric reflection counts
- empirical centric phase populations
- observed unique-HKL centric fractions
- theoretical unique-HKL centric fractions
- overall completeness
- centric completeness
- top-10 SG reports

Outputs are written under:

- `10_Empirical_Centric_Burden/01_Centric_Acentric_Counts_By_SG/`
- `10_Empirical_Centric_Burden/02_Centric_Phase_Counts_By_SG/`
- `10_Empirical_Centric_Burden/03_Top10_SG_Reports/`

Typical command:

```bash
python 10_measure_empirical_centric_burden.py \
  --project_dir ./ \
  --input_dir ./03_Run_Ecalc \
  --pattern "*_rs_ecalc.csv" \
  --num_processes 29
```

Important code behavior:
- validates required columns before processing
- requires canonicalized phases in `(-180, 180]` with `-180` forbidden
- uses `gemmi` to compute theoretical ASU-reduced centric counts in the chosen resolution range
- can run serially or in parallel via `ProcessPoolExecutor`

---

## 11. Plot symmetry complexity vs centric burden

**Script:** `11_plot_symmetry_complexity_vs_centric_burden.py`

This stage relates centric-burden quantities to symmetry-derived complexity measures.

For each space group, it computes:

- `n_rev_families`: number of distinct reversing families in full reciprocal space
- `n_asu_traces`: number of distinct ASU-restricted centric traces
- `point_group`

It then summarizes centric-burden quantities at the SG level using **weighted means**, with weights equal to `centric_completeness`, and generates descriptive plots.

Typical command:

```bash
python 11_plot_symmetry_complexity_vs_centric_burden.py \
  --project_dir ./ \
  --counts_dir ./10_Empirical_Centric_Burden/01_Centric_Acentric_Counts_By_SG \
  --out_dir ./12_Symmetry_vs_Centric_Burden \
  --x_axis_mode both \
  --asu_trace_max_index 9
```

Important note:
- model fitting has been intentionally removed in the repository version; this stage is descriptive only

---

## 12. Analyze trace-resolved centric phase supports

**Script:** `12_analyze_trace_resolved_centric_phase_supports.py`

This is the SG-specific mechanistic follow-up to stage 10.

For a chosen space group, it:

1. enumerates reversing operations `(R|t)` such that `(I + R^{-T})` is singular
2. groups the resulting reciprocal-space centric loci into ASU traces
3. determines, for each centric HKL with a finite observed phase, whether the phase is compatible with one of the translation-consistent two-point supports implied by a reversing operation
4. optionally computes parity-conditioned trace partitions for actionable priors

Typical command:

```bash
python 12_analyze_trace_resolved_centric_phase_supports.py \
  --project_dir ./ \
  --input_dir ./03_Run_Ecalc \
  --pattern "*_rs_ecalc.csv" \
  --spacegroup "P 43 21 2" \
  --trace_signature_index 8 \
  --write_invalid_rows_csv \
  --write_actionable_priors_report
```

Main outputs:
- `traces_index.json`
- `trace_table.csv`
- `dataset_summaries.csv`
- `invalid_centric_phase_dataset_ids.csv`
- optionally:
  - `invalid_rows__all_traces.csv`
  - `actionable_priors__CLEAN.csv`
  - `actionable_priors__CLEAN.md`
  - `actionable_rules__CLEAN_unique.csv`

---

# Stage 2 — Acentric compression and AutoBuild benchmarking

## 20. Bin acentric phases

**Script:** `20_bin_acentric_phases.py`

This stage performs **acentric-only phase compression** on the ECALC CSV files and writes binned outputs into a derived folder, typically:

- `03_Run_Ecalc/` → `04_Run_Ecalc_Binned/`

Important code behavior:
- only **acentric** reflections are binned; centric reflections are preserved unchanged
- for each requested `K`, the script writes:
  - `PHIC_ALL_KK`
  - `FOM_KK`
- it also computes dataset-level and global summaries of:
  - RMS phase error
  - mean absolute phase error
  - FOM attenuation
  - acentric phase uniformity via mean resultant length `R`
- it detects **centrosymmetric space groups** and reports them separately
- logs go directly into `LOGS/`, while all non-log artifacts go into a timestamped run subfolder under `LOGS/`

Typical command:

```bash
python 20_bin_acentric_phases.py \
  --source_csv_dir ./03_Run_Ecalc \
  --source_csv_pattern "*_rs_ecalc.csv" \
  --phase_col PHIC_ALL \
  --fom_col FOM \
  --bins "3,2" \
  --max_workers 30 \
  --logs_dir ./LOGS \
  --skip_write_if_exists
```

Main outputs:
- binned CSV files in `04_Run_Ecalc_Binned/`
- run-artifact folder under `LOGS/`

---

## 21. Prepare AutoBuild MTZ inputs

**Script:** `21_prepare_autobuild_mtz_inputs.py`

This stage converts the binned CSV files into minimal **PHIB-input MTZ files** for Phenix AutoBuild.

Important code behavior:
- supports methods:
  - `iREDO`
  - `K2`
  - `K3`
  - `K4`
  - `K2_atten`
- `K2_atten` uses:
  - `phase_col = PHIC_ALL_K2`
  - `fom_col = K2_ATTEN_FOM`
  - where `K2_ATTEN_FOM = 2/pi` if missing from the CSV
- by default, it samples up to `--max_pdbs`
- sampling is **stratified by space group**
- existing MTZs are skipped only if they already contain the required AutoBuild columns
- if `--output_root` is omitted, the default output folder is:
  - `<input_csv_folder_parent>/21_Phenix_Autobuild_Inputs`

Typical command:

```bash
python 21_prepare_autobuild_mtz_inputs.py \
  --input_csv_folder ./04_Run_Ecalc_Binned \
  --output_root ./21_Phenix_Autobuild_Inputs \
  --csv_pattern "*_rs_ecalc_binned.csv" \
  --methods "iREDO,K2_atten" \
  --max_pdbs 1000 \
  --seed 123 \
  --write_index_csv \
  --kappa_max 200
```

Main outputs:
- `21_Phenix_Autobuild_Inputs/<suffix>/<pdb_id>/<pdb_id>_<suffix>_PHIB_input.mtz`
- manifest CSV under the output root
- optional selection index CSV

---

## 22. Run Phenix AutoBuild

**Script:** `22_run_phenix_autobuild.py`

This stage runs `phenix.autobuild` on the MTZs prepared by stage 21.

Important code behavior:
- expected input layout:
  - `<input_phenix_folder>/<suffix>/<pdb_id>/<pdb_id>_<suffix>_PHIB_input.mtz`
- FASTA files are resolved from:
  - `<base_folder>/01_Download_Files`
- scheduler is **non-idling**
- a run is considered successful only if:
  - `overall_best.pdb` exists
  - `overall_best_refine_data.mtz` exists
  - and an AutoBuild log contains the citation marker
- default output folder:
  - `<base_folder>/22_Run_AutoBuild`

Typical command:

```bash
python 22_run_phenix_autobuild.py \
  --base_folder ./ \
  --input_phenix_folder ./21_Phenix_Autobuild_Inputs \
  --methods iREDO,K2_atten \
  --jobs 6 \
  --nproc 4
```

---

## 23. Analyze AutoBuild outcomes

**Script:** `23_analyze_autobuild_outcomes.py`

This stage analyzes valid AutoBuild runs across the requested methods.

Important code behavior:
- scans:
  - `<phenix_root>/<method>/<pdb_id>/AutoBuild_run_*/`
- extracts:
  - final Free R
  - residues built
  - residues placed
  - best cycle
  - output reflection counts
  - SG from the PHIB-input MTZ
- computes:
  - method-wise Free R summaries
  - `ΔFreeR` vs a reference method
  - relative residues placed vs the reference method
  - a paired **success report**
- writes:
  - `23_AutoBuild_basin_score_targets.csv`
- default output folder:
  - `<base_folder>/23_Autobuild_Analysis`

Typical command:

```bash
python 23_analyze_autobuild_outcomes.py \
  --base_folder ./ \
  --phenix_root ./22_Run_AutoBuild \
  --only-suffixes iREDO,K2_atten \
  --ref-method iREDO
```

---

# Stage 3 — Basin Score calibration

## 30. Run multiwindow density modification

**Script:** `30_run_multiwindow_density_modification.py`

This stage performs density modification across one or more **resolution windows**.

Important code behavior:
- accepts:
  - one CSV
  - a CSV list
  - or a full CSV folder
- default windows are:
  - `20–5.0 Å`
  - `20–4.5 Å`
  - `20–4.0 Å`
  - `20–3.5 Å`
  - `20–3.0 Å`
  - `20–2.5 Å`
- each window gets its own subfolder under the output root
- jobs are run as isolated OS processes with timeouts and restart logic
- if `FOM_K2_atten` is missing, the script adds it internally using `2/pi`
- writes:
  - `30_denmod_log_summary_multiwindow.csv`
- default output root:
  - `30_DENMOD_multiwindow`

Typical command:

```bash
phenix.python 30_run_multiwindow_density_modification.py \
  --csv_dir ./04_Run_Ecalc_Binned \
  --nproc 30 \
  --job_timeout_sec 60
```

---

## 31. Reconstruct maps and extract skeleton proxies

**Script:** `31_reconstruct_maps_and_extract_skeleton_proxies.py`

This stage reconstructs CCP4 maps from CSV reflection tables and extracts skeleton-based connectivity metrics.

Important code behavior:
- supports one or more `phase:FOM` pairs
- reconstructs:
  - raw map
  - FOM-weighted map
- computes threshold-level graph metrics and fixed-threshold proxy summaries
- writes:
  - `31_skeleton_traceability_summary_multiwindow.csv`
- default output root:
  - `31_Skeleton_multiwindow`

Typical command:

```bash
python 31_reconstruct_maps_and_extract_skeleton_proxies.py \
  --csv_dir ./04_Run_Ecalc_Binned \
  --phase_unit deg \
  --sample_rate 3.0 \
  --map_kind_for_skeleton fom_weighted \
  --threshold_mode sigma \
  --threshold_values 0.8,1.0,1.2,1.5,2.0 \
  --target_threshold 1.2 \
  --prune_tip_iterations 3 \
  --edge_connectivity 26 \
  --phase_fom_pairs PHIC_ALL_K2:FOM_K2_atten \
  --write_per_run_tables \
  --nproc 20
```

---

## 32. Compute the deployed Basin Score

**Script:** `32_compute_basin_score_deployed.py`

This stage computes the **deployed Basin Score** for a selected resolution window by merging:

- AutoBuild outcome targets from stage 23
- skeleton summary metrics from stage 31
- density-modification summary metrics from stage 30

Important code behavior:
- window is passed via:
  - `--resol_window DMAX DMIN`
- output score label depends on `dmin`, for example:
  - `20.0 3.5` → `S3p5`
- writes:
  - cohort score CSV
  - two kernel-density scatter plots
- prints:
  - the Basin Score formula
  - score thresholds
  - companion metric thresholds

Typical command:

```bash
python 32_compute_basin_score_deployed.py \
  --metrics_csv ./23_Autobuild_Analysis/23_AutoBuild_basin_score_targets.csv \
  --proxy_csv ./31_Skeleton_multiwindow/31_skeleton_traceability_summary_multiwindow.csv \
  --denmod_csv ./30_DENMOD_multiwindow/30_denmod_log_summary_multiwindow.csv \
  --resol_window 20.0 3.5
```

---

# Stage 4 — Basin Score degradation

## 40. Run controlled Basin Score degradation

**Script:** `40_BasinScore_degradation.py`

This is the main stage-40 orchestration script. It starts from one binned CSV and evaluates how the deployed Basin Score changes under **controlled phase flips**.

Important code behavior:
- runs under `phenix.python`
- supports degradation modes:
  - `uniform_random`
  - `resolution_low_to_high`
  - `amplitude_descending`
- degrades both acentric and centric reflections within the **scoring window**
- scores each degraded state using:
  - density-modification outputs
  - skeleton proxy outputs
  - the deployed Basin Score formula
- tracks real-space correlation both to:
  - the control map
  - the negative control map
- exports sampled degradation states as:
  - CCP4 maps
  - Autobuild-ready MTZ files
- uses helper scripts:
  - `40_api_skeleton_proxy.py`
  - `40_api_autobuild_mtz_writer.py`
  - indirectly, `40_api_reconstruct_map_and_skeleton_batch.py`
  - and `40_api_density_modification.py`

Typical command:

```bash
phenix.python 40_BasinScore_degradation.py \
  --csv_in ./04_Run_Ecalc_Binned/2bkf_rs_ecalc_binned.csv \
  --out_root ./40_BasinScore_Degradation/2bkf_test \
  --nproc 8 \
  --seed_master 1337 \
  --dm_score_dmax 20.0 \
  --dm_score_dmin 3.5 \
  --skeleton_helper ./40_api_skeleton_proxy.py \
  --python3_exe /path/to/python3 \
  --mtz_writer_helper ./40_api_autobuild_mtz_writer.py \
  --skeleton_threshold_values 0.8,1.0,1.2,1.4,1.7,2.0 \
  --skeleton_target_threshold 1.2 \
  --skeleton_prune_tip_iterations 3 \
  --skeleton_edge_connectivity 26 \
  --degradation_modes uniform_random \
  --degradation_fractions 0.00,0.10,0.20,0.30,0.50,0.70,1.00 \
  --n_rounds 20 \
  --task_timeout_sec 18 \
  --sample_ccp4_maps 2 \
  --maxtasksperchild 1 \
  --round_chunk_size 20
```

Main outputs:
- `40_BasinScore_Degradation/<run_name>/40_tables/`
- `40_ccp4_maps/`
- per-mode degradation CSVs
- summary PNG panels
- CCP4/MTZ manifest CSVs for downstream AutoBuild

### 40 helper scripts

#### `40_api_density_modification.py`
This is a callable Phenix-side density-modification API aligned to the stage-40 protocol. It wraps `phenix_like_density_modification_from_csv_py27.py` with PC43-style defaults and returns:
- success flag
- DM common object
- transcript
- error text
- used mask type

#### `40_api_reconstruct_map_and_skeleton_batch.py`
This is the underlying batch reconstruction + skeletonization engine. It reconstructs maps from CSV reflection tables, computes threshold-dependent skeleton metrics, and writes:
- per-pair threshold summaries
- per-PDB proxy summary CSVs
- ranking plots and summary panels

#### `40_api_skeleton_proxy.py`
This is a thin Python-3 bridge wrapper. It loads the batch skeleton script dynamically, builds the exact `argparse.Namespace` it expects, runs one CSV, and emits a compact JSON summary for stage 40.

#### `40_api_autobuild_mtz_writer.py`
This helper writes an Autobuild-ready MTZ from a payload CSV containing:
- `FP`, `SIGFP`
- `PHIB`, `FOM`
- `FreeR_flag`
- SG and cell constants

It computes HL coefficients internally from `PHIB` and `FOM`.

---

## 41. Build anti-phase MTZs for highly degraded states

**Script:** `41_make_antiphase_degradation_mtzs.py`

This stage scans the stage-40 output tree for exported PHIB-input MTZs and creates anti-phase companions.

Important code behavior:
- looks for `*_PHIB_input.mtz`
- selects only those in directories matching `frac_<value>`
- applies:
  - `PHIB := PHIB + 180 mod 360`
  - `HLA := -HLA`
  - `HLB := -HLB`
  - `HLC`, `HLD` unchanged
- default summary CSV:
  - `<degradation_root>/41_antiphase_mtz_summary.csv`

Typical command:

```bash
python 41_make_antiphase_degradation_mtzs.py \
  --degradation_root ./40_BasinScore_Degradation/2bkf_test/40_degradation_master_001337 \
  --min_degradation_fraction 0.7
```

---

## 42. Run AutoBuild on degraded and anti-phase MTZs

**Script:** `42_run_phenix_autobuild_from_degradation_mtz.py`

This stage consumes the manifest CSVs created by stage 40 and launches `phenix.autobuild` on the exported degraded states.

Important code behavior:
- discovers manifests under:
  - `**/40_tables/40_degradation__*__ccp4_manifest.csv`
- for each manifest row, launches:
  - the main `mtz_autobuild` state
  - and, if present, the sibling `_anti.mtz` state
- keeps seed and anti-phase runs in separate workdirs
- output root defaults to:
  - `<base_folder>/42_Run_AutoBuild_from_Degradation`
- success criteria match the stage-22 logic:
  - `overall_best.pdb`
  - `overall_best_refine_data.mtz`
  - log contains the citation marker

Typical command:

```bash
python 42_run_phenix_autobuild_from_degradation_mtz.py \
  --base_folder ./ \
  --input_degradation_folder ./40_BasinScore_Degradation/2bkf_test \
  --fasta_dir ./01_Download_Files \
  --jobs 7 \
  --nproc 4
```

---

## 43. Analyze degradation AutoBuild results

**Script:** `43_analyze_autobuild_degradation_results.py`

This is the final analysis stage for the degradation branch.

Important code behavior:
- scans the stage-42 AutoBuild workdirs
- detects valid runs in folders like:
  - `mode_<mode>__frac_<fraction>__round_<round>/`
  - `mode_<mode>__frac_<fraction>__round_<round>__anti/`
- groups results by:
  - source PDB
  - degradation mode
  - phase variant (`seed` / `anti`)
  - degradation fraction
- computes:
  - pooled Free R summaries
  - pooled residues placed summaries
  - success fraction by degradation fraction
  - paired seed-vs-anti differences
- writes:
  - summary CSVs
  - plots grouped into subfolders by source PDB / mode / phase variant
- prints grouped statistics to the terminal and log

Typical command:

```bash
python 43_analyze_autobuild_degradation_results.py \
  --degradation_root ./42_Run_AutoBuild_from_Degradation \
  --source_pdb 2bkf \
  --modes uniform_random
```

Main outputs:
- `pc34_degradation_valid_jobs.csv`
- `pc34_degradation_summary_by_fraction.csv`
- `pc34_degradation_summary_by_fraction_wide.csv`
- `pc34_degradation_seed_vs_anti_by_fraction.csv`
- organized plot folders under:
  - `43_AutoBuild_Degradation_Analysis/<source_pdb>/<mode>/<phase_variant>/`

---

# Final recommendation

Before large cohort runs, test the pipeline on:
- a small sampled cohort
- one representative space group
- one representative PDB for the degradation branch

That makes it much easier to verify paths, dependencies, and outputs before scaling up.
