"""Compound sidechain covalent-geometry energy (bonds + angles + chirality + planarity).

A single differentiable scalar that reproduces CCTBX's covalent restraint *target*
(everything except clash/nonbonded and dihedral) for the sidechain. One object,
two uses:

  * VAL METRIC  — ``mode="harmonic", reduction="sum"`` is the CCTBX restraint energy
    E = sum z^2 over bonds+angles+chirality+planarity, z = (value - ideal)/sigma.
    Uses the EXACT ideals (CDL CA-CB grid + proline/disulfide links + monomer-library
    chirality volumes), so it matches CCTBX. Call under no_grad and log it (optionally
    with per-family components).
  * LOSS — same function, pick a training-friendly mode (e.g. "mse" or "berhu") and
    ``reduction="mean"``. Differentiable w.r.t. the coordinates; apply to the model's
    predicted clean coords. Weighting/scheduling is the caller's.

Families and their sigmas (all from the monomer library / CCTBX):
  bonds      sidechain bond lengths (CA-CB uses the per-residue CDL phi/psi ideal)
  angles     sidechain bond angles  (+ proline C(i-1)-N-CD and disulfide CB-SG-SG links)
  chirality  signed chiral volume vs ideal (ILE/THR/VAL Cbeta, LEU Cgamma), sigma=0.2
  planarity  per-atom distance to best-fit plane, sigma=0.02

Clash is intentionally excluded (separate repulsion term, and the all-atom clash
metric is unimplemented). Harmonic mode is sigma-normalised so all four families are
dimensionless and additive; other modes mix units across families, so use per-family
``weights`` then.
"""

import torch
from torch import Tensor

from protmetrics.allatom.constants import RESTYPE_TO_IDX, RESTYPES
from protmetrics.allatom.geometry import (
    PRO_LINK_ANGLE, SS_ANGLE, SS_BOND, SS_MAX_DIST,
    _angle_deg, _load, cdl_ca_cb_ideal_esd,
)
from protmetrics.allatom.losses import _angle_tables, _bond_tables, _gather_atoms, _penalty

PLANAR_SIGMA = 0.02
# Per-atom planarity esd overrides (monomer library). Only non-0.020 case across all
# planar residues: ARG CD (sp3, loosely in the guanidinium plane). (res, atom14_slot) -> esd.
PLANAR_ESD = {("ARG", 6): 0.095}  # ARG CD = slot 6
# Ideal signed chiral volume + esd per residue (one sidechain chiral centre each),
# from CCTBX chirality_proxies (static across structures). Atom order follows the
# monomer-library chir record stored in geom_restraints.pt["chirs"].
CHIR_V0 = {"ILE": 2.645, "THR": 2.552, "VAL": -2.629, "LEU": -2.590}
CHIR_SIGMA = 0.2

_FAMILIES = ("bond", "angle", "chirality", "planarity")
_CACB = None  # [20, Bmax] bool: which bond slot is CA-CB (slots 1,4)


def _is_cacb_table():
    global _CACB
    if _CACB is None:
        t = _load()
        bmax = max(len(t[r]["bonds"]) for r in RESTYPES)
        m = torch.zeros(20, bmax, dtype=torch.bool)
        for ri, r in enumerate(RESTYPES):
            for bi, (i, j, *_2) in enumerate(t[r]["bonds"]):
                if tuple(sorted((i, j))) == (1, 4):
                    m[ri, bi] = True
        _CACB = m
    return _CACB


def _collect(coords, mask, aa, rmask, exact):
    """Return {family: (dev[1d], sigma[1d])} over all scored sidechain restraints."""
    dev, sig = {}, {}
    B, L = aa.shape
    dvc = coords.device
    tables = _load()
    if rmask is None:
        rmask = torch.ones(B, L, dtype=torch.bool, device=dvc)
    rmask = rmask.bool()

    # ---- bonds (vectorised; CA-CB overridden by the CDL phi/psi ideal when exact) ----
    bidx, bideal, bsig, bmsk = (x.to(dvc) for x in _bond_tables())
    i, j = bidx[aa, :, 0], bidx[aa, :, 1]
    d = (_gather_atoms(coords, i) - _gather_atoms(coords, j)).norm(dim=-1)  # [B,L,Bmax]
    ideal_b, sig_b = bideal[aa].clone(), bsig[aa].clone()
    if exact:
        ci, ce = cdl_ca_cb_ideal_esd(coords, mask, aa)            # [B,L]
        cbm = _is_cacb_table().to(dvc)[aa]                        # [B,L,Bmax]
        ci_e, ce_e = ci[..., None].expand_as(d), ce[..., None].expand_as(d)
        use_i = cbm & ~torch.isnan(ci_e)
        ideal_b = torch.where(use_i, ci_e, ideal_b)
        sig_b = torch.where(cbm & ~torch.isnan(ce_e), ce_e, sig_b)
    vb = (bmsk[aa] & torch.gather(mask, 2, i).bool()
          & torch.gather(mask, 2, j).bool() & rmask[..., None])
    dev["bond"], sig["bond"] = [(d - ideal_b)[vb]], [sig_b[vb]]

    # ---- angles (vectorised) ----
    aidx, aideal, asig, amsk = (x.to(dvc) for x in _angle_tables())
    ai, aj, ak = aidx[aa, :, 0], aidx[aa, :, 1], aidx[aa, :, 2]
    ang = _angle_deg(_gather_atoms(coords, ai), _gather_atoms(coords, aj), _gather_atoms(coords, ak))
    va = (amsk[aa] & torch.gather(mask, 2, ai).bool() & torch.gather(mask, 2, aj).bool()
          & torch.gather(mask, 2, ak).bool() & rmask[..., None])
    dev["angle"] = [(ang - aideal[aa])[va]]
    sig["angle"] = [asig[aa][va]]

    # ---- chirality (signed volume vs ideal; few residue types) ----
    dev["chirality"], sig["chirality"] = [], []
    for res, V0 in CHIR_V0.items():
        ri = RESTYPE_TO_IDX[res]
        sel = (aa == ri) & rmask
        if not sel.any():
            continue
        c0, a1, a2, a3, _sgn = tables[res]["chirs"][0]
        xyz = coords[sel]
        present = mask[sel][:, [c0, a1, a2, a3]].bool().all(dim=-1)
        if not present.any():
            continue
        pc = xyz[:, c0][present]
        e1, e2, e3 = xyz[:, a1][present] - pc, xyz[:, a2][present] - pc, xyz[:, a3][present] - pc
        vol = (e1 * torch.linalg.cross(e2, e3, dim=-1)).sum(-1)
        dev["chirality"].append(vol - V0)
        sig["chirality"].append(torch.full_like(vol, CHIR_SIGMA))

    # ---- planarity (per-atom distance to best-fit plane) ----
    dev["planarity"], sig["planarity"] = [], []
    for ri in range(20):
        res = RESTYPES[ri]
        planes = tables[res]["planes"]
        if not planes:
            continue
        sel = (aa == ri) & rmask
        if not sel.any():
            continue
        xyz, m = coords[sel], mask[sel].bool()
        for group in planes:
            pres = m[:, group].all(dim=-1)
            if not pres.any():
                continue
            pts = xyz[:, group][pres]                      # [Mp, K, 3]
            esd = torch.tensor([PLANAR_ESD.get((res, s), PLANAR_SIGMA) for s in group],
                               device=coords.device, dtype=pts.dtype)   # [K] per-atom esd
            w = 1.0 / esd ** 2                             # [K] weights
            # weighted best-fit plane: weighted centroid + smallest-eigenvalue eigenvector
            ctr = (w[None, :, None] * pts).sum(1, keepdim=True) / w.sum()  # [Mp,1,3]
            diff = pts - ctr                              # [Mp, K, 3]
            cov = torch.einsum("k,mki,mkj->mij", w, diff, diff)  # [Mp,3,3] weighted covariance
            normal = torch.linalg.eigh(cov)[1][:, :, 0]   # smallest-eigenvalue vector [Mp,3]
            dist = (diff * normal.unsqueeze(1)).sum(-1)   # [Mp, K] signed distance to plane
            dev["planarity"].append(dist.reshape(-1))
            sig["planarity"].append(esd[None, :].expand(dist.shape[0], -1).reshape(-1))

    # ---- inter-residue link terms (exact only) ----
    if exact:
        _proline_link_terms(coords, mask, aa, rmask, dev, sig)
        _disulfide_terms(coords, mask, aa, rmask, dev, sig)

    out = {}
    for fam in _FAMILIES:
        dl, sl = dev.get(fam, []), sig.get(fam, [])
        if dl:
            out[fam] = (torch.cat(dl), torch.cat(sl))
        else:
            out[fam] = (coords.new_zeros(0), coords.new_zeros(0))
    return out


def _proline_link_terms(coords, mask, aa, rmask, dev, sig):
    # Proline links: the ring-closure angle C(i-1)-N-CD AND the planar amide N
    # (plane of [C(i-1), N, CA, CD], esd 0.05) -- both inter-residue, both in CCTBX.
    pro = (aa == RESTYPE_TO_IDX["PRO"]) & rmask
    if not pro.any():
        return
    N, CA, C, CD = coords[..., 0, :], coords[..., 1, :], coords[..., 2, :], coords[..., 6, :]
    Cprev = torch.full_like(C, float("nan")); Cprev[:, 1:] = C[:, :-1]
    mCp = torch.zeros_like(mask[..., 2].bool()); mCp[:, 1:] = mask[:, :-1, 2].bool()
    present = (pro & mask[..., 0].bool() & mask[..., 1].bool() & mask[..., 6].bool() & mCp
               & ((N - Cprev).norm(dim=-1) < 2.0))
    if not present.any():
        return
    ang = _angle_deg(Cprev[present], N[present], CD[present])
    dev["angle"].append(ang - PRO_LINK_ANGLE[0])
    sig["angle"].append(torch.full_like(ang, PRO_LINK_ANGLE[1]))
    # planar proline N
    pts = torch.stack([Cprev[present], N[present], CA[present], CD[present]], dim=1)  # [Mp,4,3]
    ctr = pts.mean(1, keepdim=True)
    normal = torch.linalg.svd(pts - ctr, full_matrices=False)[2][:, -1, :]
    dist = ((pts - ctr) * normal.unsqueeze(1)).sum(-1)        # [Mp,4]
    dev["planarity"].append(dist.reshape(-1))
    sig["planarity"].append(torch.full_like(dist.reshape(-1), 0.05))


def _disulfide_terms(coords, mask, aa, rmask, dev, sig):
    cys = (aa == RESTYPE_TO_IDX["CYS"]) & rmask & mask[..., 5].bool()
    if cys.sum() < 2:
        return
    SG, CB = coords[..., 5, :], coords[..., 4, :]
    for b in range(aa.shape[0]):
        idx = cys[b].nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        sg = SG[b, idx]
        dmat = torch.cdist(sg, sg)
        ii, jj = torch.triu_indices(idx.numel(), idx.numel(), offset=1, device=coords.device)
        sel = dmat[ii, jj] < SS_MAX_DIST
        if not sel.any():
            continue
        ra, rb = idx[ii[sel]], idx[jj[sel]]
        dist = (SG[b, ra] - SG[b, rb]).norm(dim=-1)
        dev["bond"].append(dist - SS_BOND[0])
        sig["bond"].append(torch.full_like(dist, SS_BOND[1]))
        for ca_r, far_r in ((ra, rb), (rb, ra)):
            ang = _angle_deg(CB[b, ca_r], SG[b, ca_r], SG[b, far_r])
            dev["angle"].append(ang - SS_ANGLE[0])
            sig["angle"].append(torch.full_like(ang, SS_ANGLE[1]))


def sidechain_geometry_energy(
    atom14_coords: Tensor,
    atom14_mask: Tensor,
    aa_seq: Tensor,
    residue_mask: Tensor | None = None,
    mode: str = "harmonic",
    weights: dict | None = None,
    tol: float = 4.0,
    c: float | None = None,
    reduction: str = "sum",
    exact: bool = True,
    return_components: bool = False,
):
    """Compound sidechain covalent-geometry energy (bonds + angles + chirality + planarity).

    Args:
        atom14_coords: [B, L, 14, 3] coords (predicted clean coords for a loss).
        atom14_mask:   [B, L, 14] occupancy.
        aa_seq:        [B, L] residue types, 0-indexed alphabetical.
        residue_mask:  [B, L] optional gate (has_sidechain).
        mode:          per-restraint penalty: "harmonic" (z^2 = CCTBX energy; default),
                       "mse"/"harmonic_unweighted", "flat_bottom", "berhu", "berhu_sigma".
        weights:       optional per-family multipliers {bond,angle,chirality,planarity}.
        tol, c:        passed to the penalty (flat_bottom tol / berhu c).
        reduction:     "sum" (CCTBX-comparable energy) or "mean" (per-family mean; better as a loss).
        exact:         True (default) uses CDL CA-CB + proline/disulfide links (matches CCTBX);
                       False uses static ideals only (faster, for a loss where exactness is moot).
        return_components: also return the per-family reduced energies.

    Returns:
        scalar energy (differentiable), or (scalar, {family: scalar}) if return_components.
        mode="harmonic", reduction="sum", exact=True == CCTBX covalent restraint target
        for the sidechain (minus clash/dihedral).
    """
    w = {f: 1.0 for f in _FAMILIES}
    if weights:
        w.update(weights)
    fam = _collect(atom14_coords, atom14_mask, aa_seq, residue_mask, exact)

    total = atom14_coords.new_zeros(())
    comps = {}
    for f, (dv, sg) in fam.items():
        if dv.numel() == 0:
            comps[f] = atom14_coords.new_zeros(())
            continue
        pen = _penalty(dv, sg, mode, tol, c)
        e = pen.sum() if reduction == "sum" else pen.mean()
        comps[f] = e
        total = total + w[f] * e

    return (total, comps) if return_components else total
