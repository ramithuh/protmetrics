"""Sidechain chi-angle computation from atom14 coordinates."""

import torch
from torch import Tensor

from protmetrics.backbone.dihedrals import _dihedral
from protmetrics.allatom.constants import CHI_ATOM14_INDEX, CHI_MASK, MAX_CHI


def compute_chi(
    atom14_coords: Tensor,
    aa_seq: Tensor,
    atom14_mask: Tensor | None = None,
    has_sidechain: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute chi1..chi4 for each residue.

    Args:
        atom14_coords: [B, L, 14, 3] heavy-atom coords in canonical atom14 order.
        aa_seq: [B, L] integer residue types (0-indexed alphabetical; ALA=0..VAL=19).
        atom14_mask: [B, L, 14] occupancy (1 = real atom, 0 = phantom/missing).
            A chi is undefined if any of its 4 atoms is missing.
        has_sidechain: [B, L] residue-level mask (1 = this residue's sidechain was
            generated). Non-pocket residues should be 0 so their chi are NaN.

    Returns:
        chi: [B, L, 4] in degrees, range (-180, 180]. NaN where undefined.
        chi_valid: [B, L, 4] bool — True where chi is defined and well-formed.
    """
    B, L = aa_seq.shape
    device = atom14_coords.device

    idx = CHI_ATOM14_INDEX.to(device)[aa_seq]   # [B, L, 4, 4]
    defined = CHI_MASK.to(device)[aa_seq]        # [B, L, 4]

    # Gather the 4 atoms of each chi: [B, L, 4*4, 3] -> [B, L, 4, 4, 3]
    gather_idx = idx.reshape(B, L, MAX_CHI * 4, 1).expand(B, L, MAX_CHI * 4, 3)
    picked = torch.gather(atom14_coords, 2, gather_idx).reshape(B, L, MAX_CHI, 4, 3)
    p0, p1, p2, p3 = picked.unbind(dim=3)
    chi = _dihedral(p0, p1, p2, p3)              # [B, L, 4]

    valid = defined.clone()
    if atom14_mask is not None:
        atoms_present = torch.gather(
            atom14_mask.bool(), 2, idx.reshape(B, L, MAX_CHI * 4)
        ).reshape(B, L, MAX_CHI, 4).all(dim=-1)
        valid = valid & atoms_present
    if has_sidechain is not None:
        valid = valid & has_sidechain.bool().unsqueeze(-1)

    chi = torch.where(valid, chi, torch.full_like(chi, float("nan")))
    return chi, valid
