"""C-beta deviation — MolProbity cbetadev (exact), and backbone<->pocket check.

In a model where the backbone (N, CA, C) and the pocket block's own CA/CB are
denoised separately, the ideal CB reconstructed from the *backbone* and the
*pocket block's predicted* CB should coincide. Disagreement means the two
representations place the same residue's CB in two spots — a direct
backbone<->pocket consistency check that also validates the assembly step.

The ideal CB uses MolProbity's exact cbetadev construction (Lovell et al. 2003):
build CB two ways with idealized geometry (one anchored N->C, one C->N), average
them, and rescale to the ideal CA-CB bond length. Verified to reproduce CCTBX
mmtbx.validation.cbetadev to within 1e-5 A per residue, so the 0.25 A outlier
classification matches. Residue-type-specific ideal geometry is used for Ala,
Pro, and the beta-branched Ile/Thr/Val.
"""

import math

import torch
from torch import Tensor

from protmetrics.backbone.constants import AA_GLY
from protmetrics.allatom.constants import CB_OUTLIER_THRESHOLD

# Per-residue idealized CB geometry (L-amino acids), alphabetical 0-indexed.
# Columns: dist, angleCAB, dihedralNCAB, angleNAB, dihedralCNAB
_GEN = (1.530, 110.1, 122.8, 110.5, -122.6)
_IDEAL = {
    0:  (1.536, 110.1, 122.9, 110.6, -122.6),   # ALA
    9:  (1.540, 109.1, 123.4, 111.5, -122.0),   # ILE
    14: (1.530, 112.2, 115.1, 103.0, -120.7),   # PRO
    16: (1.540, 109.1, 123.4, 111.5, -122.0),   # THR
    19: (1.540, 109.1, 123.4, 111.5, -122.0),   # VAL
}
_CB_PARAMS = torch.tensor([_IDEAL.get(i, _GEN) for i in range(20)])  # [20, 5]


def _rotate(p1: Tensor, p2: Tensor, point: Tensor, angle_deg: Tensor) -> Tensor:
    """Rodrigues rotation of `point` about the axis p1->p2 by `angle_deg` [..,1]."""
    axis = p2 - p1
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-12)
    v = point - p1
    th = angle_deg * (math.pi / 180.0)
    cos, sin = torch.cos(th), torch.sin(th)
    dot = (axis * v).sum(-1, keepdim=True)
    return p1 + v * cos + torch.linalg.cross(axis, v) * sin + axis * dot * (1 - cos)


def _construct(N, CA, C, dist, angle, dihedral, method: str) -> Tensor:
    """Place CB from N, CA, C with idealized geometry (cbetadev construct_fourth)."""
    if method == "NCAB":
        r0, r1, r2 = N, C, CA
    else:  # CNAB
        r0, r1, r2 = C, N, CA
    c = torch.linalg.cross(r2 - r1, r0 - r1)
    c = c * dist / (c.norm(dim=-1, keepdim=True) + 1e-12) + r2
    newD = _rotate(r1, r2, c, dihedral - 90.0)
    c = torch.linalg.cross(newD - r2, r1 - r2)
    c = c * dist / (c.norm(dim=-1, keepdim=True) + 1e-12) + r2
    return _rotate(r2, c, newD, 90.0 - angle)


def ideal_cb(n: Tensor, ca: Tensor, c: Tensor, aa_seq: Tensor) -> Tensor:
    """Ideal CB from backbone N, CA, C per MolProbity cbetadev. [B,L,3] -> [B,L,3]."""
    p = _CB_PARAMS.to(n.device)[aa_seq]                      # [B, L, 5]
    dist, aCAB, dNCAB, aNAB, dCNAB = (p[..., i:i + 1] for i in range(5))
    bN = _construct(n, ca, c, dist, aCAB, dNCAB, "NCAB")
    bC = _construct(n, ca, c, dist, aNAB, dCNAB, "CNAB")
    beta = (bN + bC) / 2
    bd = (ca - beta).norm(dim=-1, keepdim=True)
    return torch.where(bd > 1e-9, ca + (beta - ca) * dist / (bd + 1e-12), beta)


def cbeta_deviation_metrics(
    n: Tensor,
    ca: Tensor,
    c: Tensor,
    cb: Tensor,
    aa_seq: Tensor,
    residue_mask: Tensor | None = None,
    cb_present: Tensor | None = None,
) -> dict[str, Tensor]:
    """Deviation between predicted CB and the CB implied by the backbone.

    Args:
        n, ca, c: [B, L, 3] backbone atoms (the separately-denoised backbone).
        cb: [B, L, 3] the pocket block's predicted CB.
        aa_seq: [B, L] residue types (Gly is excluded — no CB).
        residue_mask: [B, L] which residues to score (e.g. has_sidechain / pocket).
        cb_present: [B, L] occupancy of the predicted CB slot.

    Returns:
        cbeta/dev_mean, cbeta/dev_max, cbeta/outlier_frac (dev > 0.25 A),
        over the scored (pocket, non-Gly, CB-present) residues.
    """
    dev = torch.linalg.norm(cb - ideal_cb(n, ca, c, aa_seq), dim=-1)  # [B, L]

    mask = (aa_seq != AA_GLY)
    if residue_mask is not None:
        mask = mask & residue_mask.bool()
    if cb_present is not None:
        mask = mask & cb_present.bool()

    valid = dev[mask]
    if valid.numel() == 0:
        nan = torch.tensor(float("nan"), device=dev.device)
        return {"cbeta/dev_mean": nan, "cbeta/dev_max": nan, "cbeta/outlier_frac": nan}
    return {
        "cbeta/dev_mean": valid.mean(),
        "cbeta/dev_max": valid.max(),
        "cbeta/outlier_frac": (valid > CB_OUTLIER_THRESHOLD).float().mean(),
    }
