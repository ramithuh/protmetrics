# CCTBX / MolProbity Parity Review

- Author: Codex (GPT-5)
- Date: March 2, 2026
- Repository: `protein-geometry-metrics`
- Scope: verify whether this package is a correct PyTorch drop-in for MolProbity / CCTBX backbone validation metrics

## Conclusion

This repository is **not** a full drop-in replacement for MolProbity / CCTBX validation.

The implementation is split into two categories:

- `compute_dihedrals()` and `ramachandran_metrics()` are a reasonable CCTBX-style reimplementation of backbone phi/psi/omega calculation and rama8000 table lookup.
- `bond_length_metrics()` and `bond_angle_metrics()` are **proxy backbone geometry metrics**, not true MolProbity / CCTBX geometry-restraint validation.

In practice, the Ramachandran portion appears directionally correct, but the bond/angle portion is not equivalent to CCTBX's restraint-based validation and should not be presented as a direct drop-in.

## Verification Method

I reviewed the local implementation and test suite, then compared the design against primary CCTBX / Phenix documentation.

Local checks performed:

- Read package code in `src/protmetrics/`
- Read the validation and cross-check scripts in `scripts/`
- Ran tests with `PYTHONPATH=src pytest -q`
- Confirmed test status: `27 passed`

Important limitation:

- A direct runtime comparison against `mmtbx.validation.ramalyze` and related CCTBX APIs was **not possible in this environment** because `iotbx` and `mmtbx` are not installed.

## Issues

### 1. Bond and angle metrics are not true CCTBX / MolProbity geometry validation

Severity: High

Evidence in this repo:

- `src/protmetrics/bonds.py`
- `src/protmetrics/angles.py`
- `src/protmetrics/constants.py`

Observed behavior:

- Bond length violations are defined using fixed absolute cutoffs against a single ideal value:
  - `BOND_LENGTH_VIOLATION_THRESHOLD = 0.1`
- Bond angle violations are also defined using fixed absolute cutoffs:
  - `BOND_ANGLE_VIOLATION_THRESHOLD = 10.0`

Why this differs from CCTBX / MolProbity:

- CCTBX validation for covalent geometry is restraint-based and sigma-based.
- The validation APIs expose bond and angle outliers using a `sigma_cutoff` parameter, commonly `4.0`.
- Phenix documents these as geometry-restraint outliers and also reports RMSD / RMSZ-style summaries for restrained geometry.

Impact:

- The values produced by this repo for bonds and angles are not directly comparable to CCTBX or MolProbity outputs.
- The code can still be useful as a training-time proxy metric, but it should be described as such.

### 2. The implementation assumes tensor adjacency equals bonded topology

Severity: High

Evidence in this repo:

- `src/protmetrics/bonds.py`: peptide C-N is always taken between residue `i` and `i+1`
- `src/protmetrics/angles.py`: inter-residue angles always use adjacent tensor positions
- `src/protmetrics/dihedrals.py`: phi/psi/omega always use adjacent tensor positions

Observed behavior:

- Inter-residue geometry is computed purely from neighboring indices in the input tensor.
- There is no explicit notion of chain identity, residue continuity, insertion codes, or chain breaks.

Why this differs from CCTBX / MolProbity:

- CCTBX validation works from the actual structural topology and geometry-restraint graph, not from raw tensor adjacency.

Impact:

- If the tensor contains a chain break, missing loop, cropped fragment boundary, or concatenated segments, this code will score non-bonded residue pairs as if they were peptide-linked unless the caller masks those boundaries manually.
- That can corrupt:
  - peptide bond lengths
  - inter-residue backbone angles
  - phi / psi / omega
  - Ramachandran classification downstream

### 3. Proline-specific peptide bond geometry is declared but not used

Severity: Medium

Evidence in this repo:

- `src/protmetrics/constants.py` defines:
  - `IDEAL_C_N = 1.329`
  - `IDEAL_C_N_PRO = 1.341`
- `src/protmetrics/bonds.py` always scores peptide bonds against `IDEAL_C_N`
- `src/protmetrics/__init__.py` does not pass `aa_seq` into bond or angle scoring

Observed behavior:

- The code declares separate peptide bond ideals for general residues and proline, but the bond metric implementation does not branch on residue type.

Impact:

- Even within the repo's simplified metric definition, peptide bonds preceding proline are not modeled correctly.
- This weakens parity with Engh-Huber-style geometry expectations and any intended MolProbity / CCTBX approximation.

### 4. Bond and angle outputs are summary proxies, not MolProbity-style validation outputs

Severity: Medium

Evidence in this repo:

- `src/protmetrics/bonds.py` returns mean / std / median / average absolute deviation and one overall violation fraction
- `src/protmetrics/angles.py` returns mean / std / average absolute deviation and one overall violation fraction

Observed behavior:

- The package returns flattened logging metrics suitable for ML training.

Why this differs from CCTBX / MolProbity:

- CCTBX validation exposes per-restraint outliers, sigma deviations, and geometry summary statistics tied to restraint proxies.
- MolProbity / Phenix reports focus on geometry-restraint outliers and RMSD / RMSZ summaries, not only backbone-only absolute-threshold fractions.

Impact:

- Even if the raw geometric quantities are meaningful, the reported metric shape does not match the output semantics users expect from CCTBX validation.

### 5. Ramachandran implementation appears substantially closer to CCTBX, but parity is not fully proven here

Severity: Medium

Evidence in this repo:

- `src/protmetrics/dihedrals.py`
- `src/protmetrics/ramachandran.py`
- `scripts/extract_rama_tables.py`
- `scripts/cross_validate_cctbx.py`
- `scripts/noise_sweep_cctbx.py`

What looks correct:

- Phi, psi, and omega are defined using the standard backbone atom quartets.
- The implementation uses six residue classes:
  - general
  - glycine
  - cis_proline
  - trans_proline
  - pre_proline
  - ile_val
- The package uses a packaged `rama_tables.pt` extracted from CCTBX rama8000 data.
- The thresholds in `RAMA_THRESHOLDS` match the expected table-specific favored / allowed style cutoffs.
- The repository includes scripts intended to compare against CCTBX directly.

Remaining caveat:

- Without `mmtbx` installed in this environment, I did not run a direct parity benchmark against CCTBX on a real PDB set.
- Therefore the Ramachandran portion should be described as "intended to reproduce CCTBX behavior" rather than "verified equivalent" unless that comparison is actually executed and recorded.

### 6. Small-sample standard deviation behavior produces warnings

Severity: Low

Evidence in this repo:

- `src/protmetrics/bonds.py`
- `src/protmetrics/angles.py`

Observed behavior:

- The code uses `torch.std()` with the default unbiased estimator.
- For single-element samples, PyTorch emits warnings and returns `nan`.

Impact:

- This is not a MolProbity parity problem, but it is a correctness / usability issue for short fragments or heavily masked batches.

## What Is Correct vs. What Is Not

Reasonable to claim:

- "Pure PyTorch backbone geometry metrics inspired by MolProbity / CCTBX"
- "PyTorch approximation of Ramachandran validation using CCTBX-derived rama8000 tables"
- "Training-time structural quality proxies for backbone coordinates"

Not reasonable to claim without further work:

- "Drop-in replacement for MolProbity"
- "Equivalent to CCTBX geometry validation"
- "Produces the same bond and angle outlier metrics as CCTBX"

## Recommended Follow-Up

If the goal is true parity, the next work should be:

1. Install CCTBX locally and run a fixed PDB regression suite comparing:
   - per-residue phi / psi
   - favored / allowed / outlier fractions
   - residue-class dispatch
2. Make inter-residue calculations break-aware so noncontiguous residues cannot be treated as bonded.
3. Add residue-specific peptide bond handling for proline.
4. Rename or document bond / angle metrics as proxy metrics unless restraint-based validation is implemented.
5. If full geometry parity is required, redesign bond / angle validation around restraint sigma values rather than fixed absolute thresholds.

## References Used

Primary external references:

1. CCTBX developer docs, `mmtbx.validation`:
   https://cctbx.github.io/mmtbx/mmtbx.validation.html
2. CCTBX developer docs, `mmtbx.validation.molprobity`:
   https://cctbx.github.io/mmtbx/mmtbx.validation.molprobity.html
3. CCTBX developer docs, `cctbx.geometry_restraints.energies`:
   https://cctbx.github.io/cctbx/cctbx.geometry_restraints.energies.html
4. Phenix reference, validation tools:
   https://www.phenix-online.org/documentation/reference/validation.html
5. Phenix validation docs describing geometry-restraint outliers and 4-sigma interpretation:
   https://phenix-online.org/version_docs/2.0-5837/reference/validation.html

Local repository references:

- `src/protmetrics/__init__.py`
- `src/protmetrics/constants.py`
- `src/protmetrics/bonds.py`
- `src/protmetrics/angles.py`
- `src/protmetrics/dihedrals.py`
- `src/protmetrics/ramachandran.py`
- `scripts/cross_validate_cctbx.py`
- `scripts/noise_sweep_cctbx.py`
- `tests/`
