"""Sidechain covalent-geometry metrics — diff-from-ideal, faithful to CCTBX.

Ports the four CCTBX geometry-restraint families to batched PyTorch, scored over
the *sidechain* heavy atoms (any restraint touching an atom beyond backbone
N/CA/C/O). Ideals come from the monomer library (geostd), packaged at
``geom_data/geom_restraints.pt`` by ``scripts/extract_geometry_restraints.py``:

    bond length   -> z = (d   - ideal) / esd     -> RMSZ, >4sigma outlier frac
    bond angle    -> z = (ang - ideal) / esd     -> RMSZ, >4sigma outlier frac
    chirality     -> signed chiral volume        -> sign-flip outlier frac (exact)
    planarity     -> RMS dist to best-fit plane  -> rmsd (A), outlier frac (exact)

This catches the failure chi angles are blind to: correct torsions but distorted
local geometry — stretched bonds, bent angles, inverted ILE/THR Cbeta, buckled
aromatic rings. Faithfulness vs CCTBX (verified on real structures, canonical
atom names): bonds, angles, chirality and planarity are all BIT-EXACT.
  - distal bonds (CG onward) and all intra-residue angles: identical monomer-lib ideals.
  - CA-CB bond: CDL-adjusted by CCTBX (varies with backbone phi/psi). We port the
    CDL CA-CB grid (cdl_cacb.pt) and look it up per residue from phi/psi -> exact for
    trans residues. Termini/chain-breaks/cis-peptides fall back to the CDL mean.
  - inter-residue links: proline ring-closure angle C(i-1)-N-CD (static) and
    disulfides (SG-SG bond + CB-SG-SG angles, detected by SG-SG distance) are added,
    so angles match CCTBX even for proline/disulfide-rich structures.
  - chirality (sign) and planarity (RMSD): sigma-independent and exact.

Two non-exact edge cases, both irrelevant to canonical generated pockets:
(1) symmetric-atom nomenclature — CCTBX canonicalizes ARG NH1/NH2, ASP OD1/OD2,
    GLU OE1/OE2, PHE/TYR ring atoms to IUPAC before scoring; we use input labels, so
    experimental structures with non-standard labeling differ on those angles
    (generated atom14 is canonical by construction, so this never arises);
(2) cis-peptides and the 1-2 chain-terminal residues, where CA-CB falls back to the mean.

``per_restype=True`` additionally emits ``sidechain/<RES>/bond_rmsz`` etc. — a
learnability map showing which sidechains are hardest to place. Per-type numbers
are noisy per-batch (few residues each) but meaningful averaged over a val epoch.

POCKET-ONLY when scored over has_sidechain residues. Pure geometry, so unlike
rotamer favorability these ARE valid as auxiliary training losses (no Goodhart).
"""

import math
from pathlib import Path

import torch
from torch import Tensor

from protmetrics.allatom.constants import RESTYPE_TO_IDX, RESTYPES
from protmetrics.backbone.dihedrals import _dihedral

OUTLIER_SIGMA = 4.0
PLANARITY_OUTLIER_A = 0.08  # 4 * 0.02 A per-atom esd; max in-group atom dev

# Inter-residue "link" restraints CCTBX scores but a per-residue table cannot hold.
# All static (verified constant across structures vs CCTBX). atom14 slots:
# N=0, CA=1, C=2, CB=4; CYS SG=5; PRO CD=6.
PRO_LINK_ANGLE = (125.0, 4.1)   # C(i-1)-N-CD proline ring-closure (ideal deg, esd)
SS_BOND = (2.031, 0.020)        # SG-SG disulfide bond (ideal A, esd)
SS_ANGLE = (104.2, 2.1)         # CB-SG-SG disulfide angle (ideal deg, esd)
SS_MAX_DIST = 2.5               # SG-SG distance cutoff for disulfide detection (A)

_TABLES: dict | None = None
_DATA = Path(__file__).parent / "geom_data" / "geom_restraints.pt"
_CDL: dict | None = None
_CDL_DATA = Path(__file__).parent / "geom_data" / "cdl_cacb.pt"
_GLY, _PRO, _ILE, _VAL = (RESTYPE_TO_IDX[x] for x in ("GLY", "PRO", "ILE", "VAL"))


def _load() -> dict:
    global _TABLES
    if _TABLES is None:
        if not _DATA.exists():
            raise FileNotFoundError(
                f"Geometry restraints not found at {_DATA}. Run "
                "scripts/extract_geometry_restraints.py --geostd <geostd>"
            )
        _TABLES = torch.load(_DATA, weights_only=False)
    return _TABLES


def _load_cdl() -> dict:
    global _CDL
    if _CDL is None:
        _CDL = torch.load(_CDL_DATA, weights_only=False)
    return _CDL


def cdl_ca_cb_ideal_esd(
    atom14_coords: Tensor, atom14_mask: Tensor, aa_seq: Tensor,
) -> tuple[Tensor, Tensor]:
    """Per-residue CA-CB (ideal, esd) [B, L] from the CDL phi/psi grid.

    Reproduces CCTBX's conformation-dependent CA-CB bond restraint EXACTLY
    (nearest 10-deg bin, banker's rounding, no interpolation; key=(phi,psi);
    residue class Gly/IleVal/Pro/NonPGIV x prePro). Needs sequential full
    backbone (slots N=0, CA=1, C=2) to form phi/psi from neighbours; returns NaN
    at chain termini/breaks and where backbone is missing (caller falls back to
    the static CDL-mean). Cis-peptides are not special-cased (rare).
    """
    grid = _load_cdl()["grid"].to(atom14_coords.device)  # [6, 36, 36, 2]
    B, L = aa_seq.shape
    x = atom14_coords.double()
    N, CA, C = x[..., 0, :], x[..., 1, :], x[..., 2, :]
    mN, mCA, mC = atom14_mask[..., 0].bool(), atom14_mask[..., 1].bool(), atom14_mask[..., 2].bool()
    Cprev = torch.full_like(C, float("nan")); Cprev[:, 1:] = C[:, :-1]
    Nnext = torch.full_like(N, float("nan")); Nnext[:, :-1] = N[:, 1:]
    mCp = torch.zeros_like(mC); mCp[:, 1:] = mC[:, :-1]
    mNn = torch.zeros_like(mN); mNn[:, :-1] = mN[:, 1:]
    phi = _dihedral(Cprev, N, CA, C)
    psi = _dihedral(N, CA, C, Nnext)
    valid = (mN & mCA & mC & mCp & mNn
             & ((N - Cprev).norm(dim=-1) < 2.0) & ((Nnext - C).norm(dim=-1) < 2.0)
             & ~torch.isnan(phi) & ~torch.isnan(psi))

    def _bin(a):  # round_to_ten + wrap 180 -> -180, then to grid index 0..35
        b = torch.round(a / 10.0) * 10.0
        b = torch.where(b >= 180.0, b - 360.0, b)
        return ((b + 180.0) / 10.0).long().clamp(0, 35)

    pi, psii = _bin(phi), _bin(psi)
    nxt = torch.cat(
        [aa_seq[:, 1:], torch.full((B, 1), -1, dtype=aa_seq.dtype, device=aa_seq.device)], dim=1)
    xpro = (nxt == _PRO).long()
    base = torch.full_like(aa_seq, 2)  # NonPGIV
    base = torch.where((aa_seq == _ILE) | (aa_seq == _VAL), torch.zeros_like(base), base)
    base = torch.where(aa_seq == _PRO, torch.full_like(base, 4), base)
    gidx = (base + xpro).clamp(0, 5)
    ideal = grid[gidx, pi, psii, 0]
    esd = grid[gidx, pi, psii, 1]
    nan = torch.tensor(float("nan"), device=atom14_coords.device)
    return torch.where(valid, ideal, nan), torch.where(valid, esd, nan)


def _rmsz(z: Tensor) -> Tensor:
    return (z**2).mean().sqrt()


def sidechain_geometry_metrics(
    atom14_coords: Tensor,
    atom14_mask: Tensor,
    aa_seq: Tensor,
    residue_mask: Tensor | None = None,
    per_restype: bool = False,
) -> dict[str, Tensor]:
    """Sidechain bond/angle/chirality/planarity deviations vs monomer-library ideals.

    Args:
        atom14_coords: [B, L, 14, 3] canonical atom14 coordinates.
        atom14_mask:   [B, L, 14] per-slot occupancy (1 real, 0 missing).
        aa_seq:        [B, L] residue types, 0-indexed alphabetical.
        residue_mask:  [B, L] optional gate (e.g. has_sidechain); residues with 0
            are not scored. Defaults to all residues.
        per_restype:   if True, also emit per-residue-type keys.

    Returns:
        sidechain/bond_rmsz, sidechain/bond_outlier_frac, sidechain/bond_dev_mean_A,
        sidechain/angle_rmsz, sidechain/angle_outlier_frac, sidechain/angle_dev_mean_deg,
        sidechain/chirality_outlier_frac, sidechain/planarity_rmsd, sidechain/planarity_outlier_frac
        (+ sidechain/<RES>/{bond_rmsz,angle_rmsz} when per_restype).
        NaN for any family with no scorable restraint in the batch.
    """
    tables = _load()
    device = atom14_coords.device
    nan = torch.tensor(float("nan"), device=device)

    if residue_mask is None:
        residue_mask = torch.ones(aa_seq.shape, dtype=torch.bool, device=device)
    residue_mask = residue_mask.bool()

    # Per-residue CDL CA-CB ideal/esd (exact vs CCTBX); NaN -> static fallback.
    cacb_ideal, cacb_esd = cdl_ca_cb_ideal_esd(atom14_coords, atom14_mask, aa_seq)

    bond_z, bond_dev = [], []          # global pooled
    angle_z, angle_dev = [], []
    chir_flip = []                     # 1.0 if sign-inverted else 0.0
    plane_rms, plane_out = [], []      # per (residue, plane-group)
    per_bz, per_az = {}, {}            # res -> list of z tensors (per_restype only)

    for idx in range(20):
        res = RESTYPES[idx]
        t = tables[res]
        sel = (aa_seq == idx) & residue_mask  # [B, L]
        if not sel.any():
            continue
        xyz = atom14_coords[sel]   # [M, 14, 3]
        m = atom14_mask[sel].bool()  # [M, 14]

        ci_all, ce_all = cacb_ideal[sel], cacb_esd[sel]  # [M] per-residue CA-CB
        rb_z = []
        for i, j, d0, esd in t["bonds"]:
            present = m[:, i] & m[:, j]
            if not present.any():
                continue
            d = torch.linalg.norm(xyz[:, i] - xyz[:, j], dim=-1)  # [M]
            if tuple(sorted((i, j))) == (1, 4):  # CA-CB: per-residue CDL, static fallback
                di = torch.where(torch.isnan(ci_all), torch.full_like(ci_all, d0), ci_all)
                ei = torch.where(torch.isnan(ce_all), torch.full_like(ce_all, esd), ce_all)
            else:
                di = torch.full_like(d, d0)
                ei = torch.full_like(d, esd)
            z = ((d - di) / ei)[present]
            bond_z.append(z); bond_dev.append((d - di).abs()[present]); rb_z.append(z)

        ra_z = []
        for i, j, k, a0, esd in t["angles"]:
            present = m[:, i] & m[:, j] & m[:, k]
            if not present.any():
                continue
            u = (xyz[:, i] - xyz[:, j])[present]
            v = (xyz[:, k] - xyz[:, j])[present]
            cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1)).clamp_min(1e-8)
            ang = torch.acos(cos.clamp(-1.0, 1.0)) * 180.0 / math.pi
            z = (ang - a0) / esd
            angle_z.append(z); angle_dev.append((ang - a0).abs()); ra_z.append(z)

        for c, a1, a2, a3, sign in t["chirs"]:
            present = m[:, c] & m[:, a1] & m[:, a2] & m[:, a3]
            if not present.any():
                continue
            pc = xyz[:, c][present]
            e1, e2, e3 = xyz[:, a1][present] - pc, xyz[:, a2][present] - pc, xyz[:, a3][present] - pc
            vol = (e1 * torch.cross(e2, e3, dim=-1)).sum(-1)
            chir_flip.append((vol * sign < 0).float())

        for group in t["planes"]:
            present = m[:, group].all(dim=-1)
            if not present.any():
                continue
            pts = xyz[:, group][present]  # [Mp, K, 3]
            ctr = pts.mean(dim=1, keepdim=True)
            _, _, vh = torch.linalg.svd(pts - ctr, full_matrices=False)
            normal = vh[:, -1, :]  # [Mp, 3]
            dist = ((pts - ctr) * normal.unsqueeze(1)).sum(-1).abs()  # [Mp, K]
            plane_rms.append((dist**2).mean(dim=1).sqrt())
            plane_out.append((dist.max(dim=1).values > PLANARITY_OUTLIER_A).float())

        if per_restype and rb_z:
            per_bz[res] = list(rb_z)
        if per_restype and ra_z:
            per_az[res] = list(ra_z)

    # --- Inter-residue link restraints (proline ring closure, disulfides) ---
    _proline_link(atom14_coords, atom14_mask, aa_seq, residue_mask,
                  angle_z, angle_dev, per_az if per_restype else None)
    _disulfide_links(atom14_coords, atom14_mask, aa_seq, residue_mask,
                     bond_z, bond_dev, angle_z, angle_dev,
                     per_bz if per_restype else None, per_az if per_restype else None)

    out: dict[str, Tensor] = {}

    if bond_z:
        bz = torch.cat(bond_z)
        out["sidechain/bond_rmsz"] = _rmsz(bz)
        out["sidechain/bond_outlier_frac"] = (bz.abs() > OUTLIER_SIGMA).float().mean()
        out["sidechain/bond_dev_mean_A"] = torch.cat(bond_dev).mean()
    else:
        out.update({k: nan for k in
                    ("sidechain/bond_rmsz", "sidechain/bond_outlier_frac", "sidechain/bond_dev_mean_A")})

    if angle_z:
        az = torch.cat(angle_z)
        out["sidechain/angle_rmsz"] = _rmsz(az)
        out["sidechain/angle_outlier_frac"] = (az.abs() > OUTLIER_SIGMA).float().mean()
        out["sidechain/angle_dev_mean_deg"] = torch.cat(angle_dev).mean()
    else:
        out.update({k: nan for k in
                    ("sidechain/angle_rmsz", "sidechain/angle_outlier_frac", "sidechain/angle_dev_mean_deg")})

    out["sidechain/chirality_outlier_frac"] = (
        torch.cat(chir_flip).mean() if chir_flip else nan)

    if plane_rms:
        out["sidechain/planarity_rmsd"] = torch.cat(plane_rms).mean()
        out["sidechain/planarity_outlier_frac"] = torch.cat(plane_out).mean()
    else:
        out["sidechain/planarity_rmsd"] = nan
        out["sidechain/planarity_outlier_frac"] = nan

    for res, zs in per_bz.items():
        out[f"sidechain/{res}/bond_rmsz"] = _rmsz(torch.cat(zs))
    for res, zs in per_az.items():
        out[f"sidechain/{res}/angle_rmsz"] = _rmsz(torch.cat(zs))

    return out


def _angle_deg(p0: Tensor, p1: Tensor, p2: Tensor) -> Tensor:
    """Angle p0-p1-p2 (vertex p1) in degrees, batched over leading dims."""
    u, v = p0 - p1, p2 - p1
    cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1)).clamp_min(1e-8)
    return torch.acos(cos.clamp(-1.0, 1.0)) * 180.0 / math.pi


def _proline_link(coords, mask, aa_seq, residue_mask, angle_z, angle_dev, per_az):
    """Score the proline ring-closure angle C(i-1)-N-CD (inter-residue, static)."""
    pro = (aa_seq == RESTYPE_TO_IDX["PRO"]) & residue_mask
    if not pro.any():
        return
    N, C, CD = coords[..., 0, :], coords[..., 2, :], coords[..., 6, :]
    Cprev = torch.full_like(C, float("nan")); Cprev[:, 1:] = C[:, :-1]
    mCp = torch.zeros_like(mask[..., 2].bool()); mCp[:, 1:] = mask[:, :-1, 2].bool()
    present = (pro & mask[..., 0].bool() & mask[..., 6].bool() & mCp
               & ((N - Cprev).norm(dim=-1) < 2.0))
    if not present.any():
        return
    ang = _angle_deg(Cprev[present], N[present], CD[present])
    z = (ang - PRO_LINK_ANGLE[0]) / PRO_LINK_ANGLE[1]
    angle_z.append(z); angle_dev.append((ang - PRO_LINK_ANGLE[0]).abs())
    if per_az is not None:
        per_az.setdefault("PRO", []).append(z)


def _disulfide_links(coords, mask, aa_seq, residue_mask,
                     bond_z, bond_dev, angle_z, angle_dev, per_bz, per_az):
    """Detect SG-SG pairs (<2.5 A) among scored CYS; score SS bond + CB-SG-SG angles."""
    cys = (aa_seq == RESTYPE_TO_IDX["CYS"]) & residue_mask & mask[..., 5].bool()
    if cys.sum() < 2:
        return
    SG, CB, mCB = coords[..., 5, :], coords[..., 4, :], mask[..., 4].bool()
    B = aa_seq.shape[0]
    for b in range(B):
        idx = cys[b].nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        sg = SG[b, idx]
        d = torch.cdist(sg, sg)
        ii, jj = torch.triu_indices(idx.numel(), idx.numel(), offset=1, device=coords.device)
        sel = d[ii, jj] < SS_MAX_DIST
        if not sel.any():
            continue
        ra, rb = idx[ii[sel]], idx[jj[sel]]      # paired residue indices in L
        dist = (SG[b, ra] - SG[b, rb]).norm(dim=-1)
        zb = (dist - SS_BOND[0]) / SS_BOND[1]
        bond_z.append(zb); bond_dev.append((dist - SS_BOND[0]).abs())
        if per_bz is not None:
            per_bz.setdefault("CYS", []).append(zb)
        for ca_r, far_r in ((ra, rb), (rb, ra)):  # CB-SG-SG at each endpoint
            m = mCB[b, ca_r]
            if not m.any():
                continue
            ang = _angle_deg(CB[b, ca_r], SG[b, ca_r], SG[b, far_r])[m]
            za = (ang - SS_ANGLE[0]) / SS_ANGLE[1]
            angle_z.append(za); angle_dev.append((ang - SS_ANGLE[0]).abs())
            if per_az is not None:
                per_az.setdefault("CYS", []).append(za)
