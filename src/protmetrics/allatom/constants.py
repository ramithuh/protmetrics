"""Atom14 sidechain topology: names, chi-angle atoms, and symmetry.

Conventions match OpenFold / AlphaFold2 `residue_constants` and the protmetrics
0-indexed *alphabetical* 3-letter ordering already used in
``protmetrics.backbone.constants`` (ALA=0 … VAL=19, GLY=7, PRO=14).

The generator emits atom37; callers convert atom37 -> atom14 at the protmetrics
entry point (standard fixed mapping) so this module can rely on the canonical
atom14 slot ordering below.

Nothing here is a training target. These tables feed *evaluation* metrics
(chi angles, rotamer lookup, chi-RMSD, Cbeta deviation). See the package
docstring for the no-loss discipline.
"""

import torch

# ---------------------------------------------------------------------------
# Residue order — alphabetical 3-letter, matches backbone.constants
# ---------------------------------------------------------------------------
RESTYPES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
RESTYPE_TO_IDX = {r: i for i, r in enumerate(RESTYPES)}
NUM_RESTYPES = len(RESTYPES)  # 20

# ---------------------------------------------------------------------------
# Canonical atom14 atom names per residue (AF2 restype_name_to_atom14_names).
# Empty string = unused slot. Slot 0-3 are always backbone N, CA, C, O.
# ---------------------------------------------------------------------------
ATOM14_NAMES = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
}

# ---------------------------------------------------------------------------
# Chi-angle atom quartets per residue (chi1..chi4), by atom name.
# ---------------------------------------------------------------------------
CHI_ANGLES_ATOMS = {
    "ALA": [],
    "ARG": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "NE"], ["CG", "CD", "NE", "CZ"]],
    "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
    "CYS": [["N", "CA", "CB", "SG"]],
    "GLN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "GLU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "OE1"]],
    "GLY": [],
    "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
    "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
    "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "LYS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"], ["CB", "CG", "CD", "CE"], ["CG", "CD", "CE", "NZ"]],
    "MET": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "SD"], ["CB", "CG", "SD", "CE"]],
    "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
    "SER": [["N", "CA", "CB", "OG"]],
    "THR": [["N", "CA", "CB", "OG1"]],
    "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
    "VAL": [["N", "CA", "CB", "CG1"]],
}

# Terminal chi with 180-degree symmetry (symmetric terminal group): the angle
# and angle+180 are physically indistinguishable. AF2 `chi_pi_periodic`:
# ASP chi2, GLU chi3, PHE chi2, TYR chi2.  (ASN/GLN/HIS are NOT symmetric — the
# distal O vs N atoms are distinguishable.)
CHI_PI_PERIODIC_LAST = {"ASP", "GLU", "PHE", "TYR"}

MAX_CHI = 4
CB_OUTLIER_THRESHOLD = 0.25  # Angstroms (MolProbity cbetadev)
ROTAMER_OUTLIER_PROB = 0.003  # 0.3% — rotalyze outlier cutoff


def _build_index_tensors():
    """Precompute chi atom14 indices, validity, and symmetry as tensors.

    Returns:
        chi_atom14_index: [20, 4, 4] long — atom14 slot of each of the 4
            atoms defining chi_j for residue i (0 where undefined).
        chi_mask:         [20, 4] bool — whether chi_j is defined for residue i.
        chi_symmetric:    [20, 4] bool — whether chi_j is 180-deg symmetric.
    """
    idx = torch.zeros(NUM_RESTYPES, MAX_CHI, 4, dtype=torch.long)
    mask = torch.zeros(NUM_RESTYPES, MAX_CHI, dtype=torch.bool)
    sym = torch.zeros(NUM_RESTYPES, MAX_CHI, dtype=torch.bool)
    for r, ri in RESTYPE_TO_IDX.items():
        name_to_slot = {n: s for s, n in enumerate(ATOM14_NAMES[r]) if n}
        chis = CHI_ANGLES_ATOMS[r]
        for ci, quartet in enumerate(chis):
            for ai, atom in enumerate(quartet):
                idx[ri, ci, ai] = name_to_slot[atom]
            mask[ri, ci] = True
        if r in CHI_PI_PERIODIC_LAST and chis:
            sym[ri, len(chis) - 1] = True
    return idx, mask, sym


CHI_ATOM14_INDEX, CHI_MASK, CHI_SYMMETRIC = _build_index_tensors()
