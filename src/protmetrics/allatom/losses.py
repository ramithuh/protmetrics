"""Differentiable sidechain covalent-geometry losses (Tier-1, no Goodhart).

These reuse the monomer-library ideals from geom_data/geom_restraints.pt, but —
unlike the eval metrics in ``geometry.py`` — they:

  * return a single differentiable scalar (no thresholding / detach / no_grad),
  * use STATIC ideals only (no CDL grid, no proline/disulfide link passes):
    bit-exact CCTBX faithfulness is an *evaluation* concern; a loss only needs to
    pull geometry toward ideal. CA-CB uses the per-restype CDL mean baked into the
    table, which is plenty for a restraint.
  * default to a FLAT-BOTTOM penalty: cost only when |deviation| > tol * sigma, so
    the natural ~1 sigma spread of real structures is left untouched and only gross
    violations are penalized (this is the AlphaFold2 structural-violation form).

Everything is a vectorized gather keyed by the (known, fixed) residue type — the
type selects which atom pairs to measure and their ideal/sigma; gradients flow
only through the coordinates.

USAGE (apply to the model's predicted CLEAN coords x0-hat, never the noised input;
weighting/scheduling is the caller's job):

    from protmetrics.allatom.losses import sidechain_bond_loss
    loss_geom = sidechain_bond_loss(a14_hat, atom14_mask, aa_seq,
                                    residue_mask=has_sidechain)
    loss = loss_fm + w_geom * loss_geom

This is the loss-safe subset only. Do NOT build a rotamer-favorability loss the
same way (Goodhart) — keep that eval-only.
"""

import torch
from torch import Tensor

from protmetrics.allatom.constants import RESTYPES
from protmetrics.allatom.geometry import _load

_BOND: tuple | None = None
_ANGLE: tuple | None = None


def _bond_tables():
    """[20, Bmax, 2] atom14 idx, [20, Bmax] ideal/sigma/mask from the cif table."""
    global _BOND
    if _BOND is None:
        t = _load()
        bmax = max(len(t[r]["bonds"]) for r in RESTYPES)
        idx = torch.zeros(20, bmax, 2, dtype=torch.long)
        ideal = torch.ones(20, bmax)
        sigma = torch.ones(20, bmax)
        mask = torch.zeros(20, bmax, dtype=torch.bool)
        for ri, r in enumerate(RESTYPES):
            for bi, (i, j, d0, esd) in enumerate(t[r]["bonds"]):
                idx[ri, bi, 0], idx[ri, bi, 1] = i, j
                ideal[ri, bi], sigma[ri, bi], mask[ri, bi] = d0, esd, True
        _BOND = (idx, ideal, sigma, mask)
    return _BOND


def _angle_tables():
    """[20, Amax, 3] atom14 idx, [20, Amax] ideal(deg)/sigma/mask from the cif table."""
    global _ANGLE
    if _ANGLE is None:
        t = _load()
        amax = max(len(t[r]["angles"]) for r in RESTYPES)
        idx = torch.zeros(20, amax, 3, dtype=torch.long)
        ideal = torch.ones(20, amax)
        sigma = torch.ones(20, amax)
        mask = torch.zeros(20, amax, dtype=torch.bool)
        for ri, r in enumerate(RESTYPES):
            for ai, (i, j, k, a0, esd) in enumerate(t[r]["angles"]):
                idx[ri, ai, 0], idx[ri, ai, 1], idx[ri, ai, 2] = i, j, k
                ideal[ri, ai], sigma[ri, ai], mask[ri, ai] = a0, esd, True
        _ANGLE = (idx, ideal, sigma, mask)
    return _ANGLE


def _gather_atoms(atom14_coords: Tensor, slot: Tensor) -> Tensor:
    """atom14_coords [B,L,14,3], slot [B,L,K] -> [B,L,K,3] (differentiable in coords)."""
    return torch.gather(atom14_coords, 2, slot.unsqueeze(-1).expand(*slot.shape, 3))


def _penalty(dev: Tensor, sigma: Tensor, mode: str, tol: float, c: float | None) -> Tensor:
    z = dev / sigma  # deviation in sigma units (dimensionless)
    if mode == "flat_bottom":              # cost only beyond tol sigma; preserves natural spread
        return torch.relu(z.abs() - tol) ** 2
    if mode == "harmonic":                 # sigma-weighted squared deviation (= weighted MSE / chi^2)
        return z ** 2
    if mode in ("mse", "harmonic_unweighted"):  # raw squared deviation (= MSE; Angstrom^2 / deg^2)
        return dev ** 2
    if mode == "berhu":                    # reversed Huber on RAW deviation: |dev| in the
        # middle (linear, robust, dense gradient), quadratic in the tails |dev|>c
        # (escalates on gross violations). C1 except the kink at 0. c is in raw units
        # (Angstrom for bonds, degrees for angles) and is supplied by the caller.
        if c is None:
            raise ValueError("mode='berhu' requires c (raw units: Angstrom for bonds, degrees for angles)")
        a = dev.abs()
        return torch.where(a <= c, a, (dev ** 2 + c ** 2) / (2.0 * c))
    if mode == "berhu_sigma":              # ADAPTIVE reversed-Huber: knee at c*sigma per
        # constraint (Engh-Huber physical tolerance). c is the DIMENSIONLESS multiplier k:
        # gentle L1 within normal wiggle (|dev| <= k*sigma), quadratic on unphysical geometry
        # beyond. Self-annealing: as the model improves past tolerance, more mass falls into
        # the L1 core -> gentle constant-gradient polishing. sigma sets the KNEE per
        # constraint, NOT a normalizer (penalty stays raw |dev|/dev^2 -> no chi^2 blowup like
        # the harmonic/flat_bottom modes). NOTE: this adapts the THRESHOLD, not the coord-
        # gradient scale -- angles in degrees still carry ~57x (deg/rad) the coord-gradient of
        # bonds, so mixing bond+angle still needs the angle weight scaled down (~1/57) or a
        # radian angle reformulation.
        if c is None:
            raise ValueError("mode='berhu_sigma' requires c (dimensionless k: knee at k*sigma)")
        c_eff = c * sigma
        a = dev.abs()
        return torch.where(a <= c_eff, a, (dev ** 2 + c_eff ** 2) / (2.0 * c_eff))
    raise ValueError(f"unknown mode {mode!r}")


def _reduce(pen: Tensor, valid: Tensor, reduction: str) -> Tensor:
    pen = pen * valid
    if reduction == "sum":
        return pen.sum()
    return pen.sum() / valid.sum().clamp(min=1)  # mean over valid restraints


def sidechain_bond_loss(
    atom14_coords: Tensor,
    atom14_mask: Tensor,
    aa_seq: Tensor,
    residue_mask: Tensor | None = None,
    mode: str = "flat_bottom",
    tol: float = 4.0,
    c: float | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Differentiable sidechain bond-length geometry loss (scalar).

    For each residue, measures every sidechain bond distance from the coords and
    penalizes deviation from that residue type's ideal length.

    Args:
        atom14_coords: [B, L, 14, 3] predicted (clean) coords. Carries grad.
        atom14_mask:   [B, L, 14] per-slot occupancy.
        aa_seq:        [B, L] residue types, 0-indexed alphabetical (fixed/known).
        residue_mask:  [B, L] optional gate (e.g. has_sidechain); 0 -> not scored.
        mode:          "flat_bottom" (default), "harmonic", "mse"
                       (alias "harmonic_unweighted"), or "berhu".
        tol:           flat-bottom half-width in sigma units (only |dev|>tol*sigma costs).
        c:             reversed-Huber transition in RAW units (Angstrom); REQUIRED for
                       mode="berhu" (linear for |dev|<=c, quadratic beyond). Ignored otherwise.
        reduction:     "mean" over valid bonds (default) or "sum".

    Returns:
        Scalar loss tensor (0-dim), differentiable w.r.t. atom14_coords.
    """
    idx, ideal, sigma, bmask = (x.to(atom14_coords.device) for x in _bond_tables())
    i = idx[aa_seq, :, 0]           # [B, L, Bmax]
    j = idx[aa_seq, :, 1]
    xi, xj = _gather_atoms(atom14_coords, i), _gather_atoms(atom14_coords, j)
    d = (xi - xj).norm(dim=-1)      # [B, L, Bmax]
    dev = d - ideal[aa_seq]
    valid = (bmask[aa_seq]
             & torch.gather(atom14_mask, 2, i).bool()
             & torch.gather(atom14_mask, 2, j).bool())
    if residue_mask is not None:
        valid = valid & residue_mask.bool().unsqueeze(-1)
    return _reduce(_penalty(dev, sigma[aa_seq], mode, tol, c), valid.float(), reduction)


def sidechain_angle_loss(
    atom14_coords: Tensor,
    atom14_mask: Tensor,
    aa_seq: Tensor,
    residue_mask: Tensor | None = None,
    mode: str = "flat_bottom",
    tol: float = 4.0,
    c: float | None = None,
    reduction: str = "mean",
) -> Tensor:
    """Differentiable sidechain bond-angle geometry loss (scalar, ideals in degrees).

    Same contract as ``sidechain_bond_loss``; measures the angle at each vertex atom
    and penalizes deviation from the residue type's ideal angle. For mode="berhu",
    ``c`` is the transition in RAW units (degrees).
    """
    idx, ideal, sigma, amask = (x.to(atom14_coords.device) for x in _angle_tables())
    i, j, k = idx[aa_seq, :, 0], idx[aa_seq, :, 1], idx[aa_seq, :, 2]
    p0, p1, p2 = (_gather_atoms(atom14_coords, s) for s in (i, j, k))
    u, v = p0 - p1, p2 - p1
    cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1)).clamp_min(1e-8)
    ang = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7)) * 180.0 / torch.pi
    dev = ang - ideal[aa_seq]
    valid = (amask[aa_seq]
             & torch.gather(atom14_mask, 2, i).bool()
             & torch.gather(atom14_mask, 2, j).bool()
             & torch.gather(atom14_mask, 2, k).bool())
    if residue_mask is not None:
        valid = valid & residue_mask.bool().unsqueeze(-1)
    return _reduce(_penalty(dev, sigma[aa_seq], mode, tol, c), valid.float(), reduction)


def sidechain_geometry_loss(
    atom14_coords: Tensor,
    atom14_mask: Tensor,
    aa_seq: Tensor,
    residue_mask: Tensor | None = None,
    mode: str = "flat_bottom",
    bond_tol: float = 4.0,
    angle_tol: float = 6.0,
    bond_c: float | None = None,
    angle_c: float | None = None,
    angle_weight: float = 1.0,
) -> Tensor:
    """Convenience: bond loss + angle_weight * angle loss.

    Bonds and angles take SEPARATE knobs because they live on different scales:
      - flat_bottom: bonds are stiff and want a tight tol (~4 sigma catches real
        violations with no false positives); real proteins carry natural angle strain
        beyond 4 sigma, so angles want a looser tol (~6-8). A shared tol would force
        the wrong value on one. (Irrelevant for mode="harmonic"/"mse", which ignore tol.)
      - berhu: pass bond_c (Angstrom) and angle_c (degrees) transitions; required for that mode.
    """
    lb = sidechain_bond_loss(atom14_coords, atom14_mask, aa_seq, residue_mask, mode, bond_tol, bond_c)
    la = sidechain_angle_loss(atom14_coords, atom14_mask, aa_seq, residue_mask, mode, angle_tol, angle_c)
    return lb + angle_weight * la
