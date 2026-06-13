"""Ideal backbone geometry constants.

Bond lengths and angles from Engh & Huber (1991, 2001), as tabulated in:
  - OpenFold: openfold/utils/residue_constants.py (lines 546-551)
  - CCTBX: cctbx_project/mmtbx/validation/ramalyze.py

Ramachandran thresholds from the rama8000 dataset:
  - Lovell et al. (2003) "Structure validation by Calpha geometry"
  - CCTBX: cctbx_project/mmtbx/validation/ramachandran/
  - Richardson Lab: https://github.com/rlabduke/reference_data
"""

# ---------------------------------------------------------------------------
# Bond lengths (Angstroms) — Engh & Huber Table 5.2.1.1
# ---------------------------------------------------------------------------
IDEAL_N_CA = 1.459
IDEAL_CA_C = 1.525
IDEAL_C_N = 1.329  # non-Pro peptide bond
IDEAL_C_N_PRO = 1.341  # Pro peptide bond

# Standard deviations — same source
STDDEV_N_CA = 0.020
STDDEV_CA_C = 0.026
STDDEV_C_N = 0.014
STDDEV_C_N_PRO = 0.016

# ---------------------------------------------------------------------------
# Bond angles (degrees) — Engh & Huber / OpenFold residue_constants.py
# ---------------------------------------------------------------------------
IDEAL_N_CA_C = 111.0  # intra-residue
IDEAL_CA_C_N = 116.568  # inter-residue (cos = -0.4473, stddev_cos = 0.0311)
IDEAL_C_N_CA = 121.352  # inter-residue (cos = -0.5203, stddev_cos = 0.0353)

# Approximate angular stddevs (converted from cos-space stddevs above)
STDDEV_N_CA_C = 2.8
STDDEV_CA_C_N = 1.8
STDDEV_C_N_CA = 2.0

# ---------------------------------------------------------------------------
# Violation thresholds
# ---------------------------------------------------------------------------
BOND_LENGTH_VIOLATION_THRESHOLD = 0.1  # Angstroms
BOND_ANGLE_VIOLATION_THRESHOLD = 10.0  # degrees

# ---------------------------------------------------------------------------
# Ramachandran classification — rama8000 (Lovell et al. 2003)
# Thresholds from CCTBX: cctbx_project/mmtbx/validation/ramalyze.py
# RAMALYZE_FAVORED / RAMALYZE_ALLOWED cutoffs per residue class
# {table_name: (favored_threshold, allowed_threshold)}
# Outlier is anything below the allowed threshold.
# ---------------------------------------------------------------------------
RAMA_THRESHOLDS = {
    "general": (0.02, 0.0005),
    "glycine": (0.02, 0.0005),
    "cis_proline": (0.02, 0.0020),
    "trans_proline": (0.02, 0.0010),
    "pre_proline": (0.02, 0.0010),
    "ile_val": (0.02, 0.0010),
}

# ---------------------------------------------------------------------------
# Standard amino acid indices — 0-indexed alphabetical order
# Matches RF2AA / atomworks / OpenFold convention.
# If your encoding is 1-indexed (padding at 0), pass aa_index_offset=1
# to ramachandran_metrics.
# ---------------------------------------------------------------------------
AA_GLY = 7
AA_PRO = 14
AA_ILE = 9
AA_VAL = 19

# ---------------------------------------------------------------------------
# Van der Waals radii (Angstroms) — Bondi (1964), J. Phys. Chem. 68, 441
# Used for backbone clash detection.
# ---------------------------------------------------------------------------
VDW_RADIUS_N = 1.55
VDW_RADIUS_C = 1.70  # applies to both CA and C
VDW_RADIUS_O = 1.52

# Clash overlap threshold (Angstroms) — MolProbity convention
# overlap = (r_A + r_B) - dist; clash if overlap >= threshold
CLASH_OVERLAP_THRESHOLD = 0.4

# ---------------------------------------------------------------------------
# Carbonyl oxygen reconstruction geometry — Engh & Huber
# ---------------------------------------------------------------------------
IDEAL_C_O = 1.231          # C=O bond length (Angstroms)
IDEAL_CA_C_O = 120.8       # CA-C=O angle (degrees)
