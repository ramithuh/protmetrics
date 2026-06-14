"""protmetrics.allatom — sidechain quality metrics for (partial) pocket sidechains.

Input convention (set at the protmetrics boundary; callers convert atom37->atom14):
    atom14_coords : [B, L, 14, 3]  canonical atom14 order (slots 0-3 = N,CA,C,O)
    atom14_mask   : [B, L, 14]     per-slot occupancy (1 real, 0 phantom/missing)
    aa_seq        : [B, L]         residue types, 0-indexed alphabetical (ALA=0..VAL=19)
    has_sidechain : [B, L]         1 for pocket residues whose sidechain was generated

Metrics are POCKET-ONLY by construction (scored over has_sidechain residues).
`rotamer/outlier_frac` in particular is a valid per-residue rate but not directly
comparable to all-residue literature numbers — label it pocket-only when logging.

NO-LOSS DISCIPLINE: everything here is an evaluation/logging metric. Do not wire
rotamer favorability (or clash) as a training loss — optimizing toward favored
rotamers Goodharts into native-looking but wrong placements. Differentiability is
preserved as a property, not an invitation.

Status: cbeta + chi + rotamer outlier + recovery + sidechain covalent geometry
(bond/angle RMSZ, chirality, planarity) are implemented and verified vs CCTBX;
clash remains a stub (heavy-atom bond-topology table).
"""

from protmetrics.allatom.cbeta import cbeta_deviation_metrics, ideal_cb
from protmetrics.allatom.dihedrals import compute_chi
from protmetrics.allatom.energy import sidechain_geometry_energy
from protmetrics.allatom.geometry import sidechain_geometry_metrics
from protmetrics.allatom.losses import (
    sidechain_angle_loss,
    sidechain_bond_loss,
    sidechain_geometry_loss,
)
from protmetrics.allatom.recovery import chi_rmsd_metrics, rotamer_recovery
from protmetrics.allatom.rotamers import rotamer_metrics

__all__ = [
    "cbeta_deviation_metrics",
    "ideal_cb",
    "compute_chi",
    "rotamer_metrics",
    "sidechain_geometry_metrics",
    "sidechain_geometry_energy",
    "sidechain_bond_loss",
    "sidechain_angle_loss",
    "sidechain_geometry_loss",
    "chi_rmsd_metrics",
    "rotamer_recovery",
    "evaluate_sidechains",
]


def evaluate_sidechains(
    atom14_coords,
    aa_seq,
    atom14_mask=None,
    has_sidechain=None,
    native_atom14_coords=None,
    native_atom14_mask=None,
    per_restype=False,
):
    """All-in-one pocket sidechain metrics. Returns a flat dict for log_dict().

    Backbone N/CA/C/O are read from atom14 slots 0-3. If
    `native_atom14_coords` is provided (conditioned/memorization eval), recovery
    metrics (chi-RMSD, rotamer recovery) are added; otherwise they are skipped
    (unconditioned de novo).

    `per_restype=True` adds per-residue-type covalent-geometry keys
    (sidechain/<RES>/bond_rmsz, .../angle_rmsz) — a per-sidechain learnability map.
    """
    metrics = {}
    n, ca, c = atom14_coords[..., 0, :], atom14_coords[..., 1, :], atom14_coords[..., 2, :]
    cb = atom14_coords[..., 4, :]
    cb_present = atom14_mask[..., 4] if atom14_mask is not None else None

    metrics.update(cbeta_deviation_metrics(
        n, ca, c, cb, aa_seq, residue_mask=has_sidechain, cb_present=cb_present,
    ))

    chi, chi_valid = compute_chi(atom14_coords, aa_seq, atom14_mask, has_sidechain)
    metrics.update(rotamer_metrics(chi, chi_valid, aa_seq))

    if atom14_mask is not None:
        metrics.update(sidechain_geometry_metrics(
            atom14_coords, atom14_mask, aa_seq,
            residue_mask=has_sidechain, per_restype=per_restype,
        ))

    if native_atom14_coords is not None:
        nchi, nvalid = compute_chi(native_atom14_coords, aa_seq, native_atom14_mask, has_sidechain)
        both = chi_valid & nvalid
        metrics.update(chi_rmsd_metrics(chi, nchi, aa_seq, both))
        metrics.update(rotamer_recovery(chi, nchi, aa_seq, both))

    return metrics
