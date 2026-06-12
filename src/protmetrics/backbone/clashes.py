"""Backbone steric clash detection.

Computes a MolProbity-style clash score on backbone heavy atoms (N, CA, C,
and optionally O).  Uses Bondi van der Waals radii and a 0.4 A overlap
threshold, excluding bonded pairs up to 1-4 connectivity between sequential
residues.

When ``include_O=True`` and the input contains only N/CA/C coordinates, the
carbonyl oxygen is deterministically reconstructed from the peptide-plane
geometry before clash detection.
"""

import math

import torch
from torch import Tensor

from protmetrics.backbone.constants import (
    CLASH_OVERLAP_THRESHOLD,
    IDEAL_C_O,
    IDEAL_CA_C_O,
    VDW_RADIUS_C,
    VDW_RADIUS_N,
    VDW_RADIUS_O,
)

# Atom ordering within a residue: N=0, CA=1, C=2, (O=3)
# Bonded-pair exclusions between residue i and i+1 (up to 1-4 connectivity).
# Each tuple is (local_atom_i, local_atom_j).
_EXCL_NO_O: list[tuple[int, int]] = [
    (2, 0),  # C_i  – N_{i+1}   1-2
    (1, 0),  # CA_i – N_{i+1}   1-3
    (2, 1),  # C_i  – CA_{i+1}  1-3
    (0, 0),  # N_i  – N_{i+1}   1-4
    (1, 1),  # CA_i – CA_{i+1}  1-4
    (2, 2),  # C_i  – C_{i+1}   1-4
]

_EXCL_WITH_O: list[tuple[int, int]] = _EXCL_NO_O + [
    (3, 0),  # O_i  – N_{i+1}   1-3
    (3, 1),  # O_i  – CA_{i+1}  1-4
]


def reconstruct_O(
    backbone_coords: Tensor,
    backbone_mask: Tensor | None = None,
) -> Tensor:
    """Place carbonyl O deterministically from N, CA, C coordinates.

    O sits in the peptide plane at bond length 1.231 A from C, with
    angle CA-C-O = 120.8 deg, on the opposite side of the C-N_{i+1}
    bond from CA.

    Args:
        backbone_coords: [B, L*3, 3] with repeating N, CA, C.
        backbone_mask: [B, L] residue-level mask (optional, unused but
            accepted for API consistency).

    Returns:
        [B, L, 4, 3] tensor with atom order N, CA, C, O.
    """
    B = backbone_coords.shape[0]
    L = backbone_coords.shape[1] // 3
    atoms = backbone_coords.reshape(B, L, 3, 3)  # [B, L, 3(atom), 3(xyz)]

    N = atoms[:, :, 0]   # [B, L, 3]
    CA = atoms[:, :, 1]
    C = atoms[:, :, 2]

    # Unit vector from C toward CA
    u_CA = CA - C  # [B, L, 3]
    u_CA = u_CA / (u_CA.norm(dim=-1, keepdim=True) + 1e-8)

    # Reference direction: N_{i+1} for residues 0..L-2, N_i for last residue
    ref = torch.zeros_like(C)
    if L > 1:
        ref[:, :-1] = N[:, 1:]    # N of next residue
        ref[:, -1] = N[:, -1]     # fallback: own N for last residue
    else:
        ref[:, 0] = N[:, 0]

    u_ref = ref - C
    u_ref = u_ref / (u_ref.norm(dim=-1, keepdim=True) + 1e-8)

    # Peptide-plane normal
    plane_n = torch.cross(u_CA, u_ref, dim=-1)
    plane_n_norm = plane_n.norm(dim=-1, keepdim=True)

    # Handle degenerate case: collinear atoms (cross product ~0).
    # Pick an arbitrary perpendicular direction.
    degenerate = plane_n_norm.squeeze(-1) < 1e-6  # [B, L]
    if degenerate.any():
        # Pick y-axis as arbitrary; if u_CA is along y, pick z instead
        arb = torch.zeros_like(u_CA)
        arb[..., 1] = 1.0  # y-axis
        # Where u_CA is nearly parallel to y, use z instead
        dot_y = u_CA[..., 1].abs()
        use_z = dot_y > 0.9
        arb[use_z, 1] = 0.0
        arb[use_z, 2] = 1.0
        fallback_n = torch.cross(u_CA, arb, dim=-1)
        fallback_n = fallback_n / (fallback_n.norm(dim=-1, keepdim=True) + 1e-8)
        plane_n[degenerate] = fallback_n[degenerate]
        plane_n_norm = plane_n.norm(dim=-1, keepdim=True)

    plane_n = plane_n / (plane_n_norm + 1e-8)

    # In-plane perpendicular to u_CA, pointing toward ref (N_{i+1}) side
    perp = torch.cross(plane_n, u_CA, dim=-1)

    angle_rad = math.radians(IDEAL_CA_C_O)
    # O direction: 120.8 deg from the CA direction, opposite side from N_{i+1}
    d_O = u_CA * math.cos(angle_rad) - perp * math.sin(angle_rad)

    O = C + IDEAL_C_O * d_O  # [B, L, 3]

    # Stack: [B, L, 4, 3]
    return torch.stack([N, CA, C, O], dim=2)


def _build_exclusion_mask(
    L: int, A: int, include_O: bool, device: torch.device,
) -> Tensor:
    """Boolean mask [L*A, L*A] where True = exclude pair from clash check."""
    N_atoms = L * A

    # Same-residue exclusion (block diagonal)
    atom_residue = torch.arange(N_atoms, device=device) // A
    mask = atom_residue.unsqueeze(0) == atom_residue.unsqueeze(1)

    # Sequential bonded exclusions (1-2 through 1-4)
    excl_pairs = _EXCL_WITH_O if include_O else _EXCL_NO_O
    if L > 1:
        res_i = torch.arange(L - 1, device=device)
        for a_i, a_j in excl_pairs:
            row = res_i * A + a_i
            col = (res_i + 1) * A + a_j
            mask[row, col] = True
            mask[col, row] = True

    return mask


def backbone_clash_score(
    backbone_coords: Tensor,
    backbone_mask: Tensor | None = None,
    include_O: bool = False,
) -> dict[str, Tensor]:
    """Compute MolProbity-style backbone clash score.

    A clash is an inter-residue atom pair whose van der Waals spheres
    overlap by >= 0.4 A (the MolProbity convention).  Bonded pairs up
    to 1-4 connectivity between sequential residues are excluded.

    Args:
        backbone_coords: [B, L*3, 3] (N, CA, C) or [B, L, 4, 3] (N, CA, C, O).
        backbone_mask: [B, L] residue-level mask (1 = valid, 0 = padding).
        include_O: If True and input is [B, L*3, 3], reconstruct O first.
            If True and input is [B, L, 4, 3], use O directly.

    Returns:
        Dict with keys:
            clash/score — clashes per 1000 atoms (MolProbity convention)
            clash/count — mean number of clashing atom pairs per structure
            clash/total_atoms — mean atom count per structure
    """
    # --- Shape handling ---
    ndim = backbone_coords.ndim
    if ndim == 4:
        # Already [B, L, 4, 3] — O is present
        B, L, A, _ = backbone_coords.shape
        coords = backbone_coords  # [B, L, A, 3]
        include_O = True  # override since O is already provided
    elif ndim == 3:
        B = backbone_coords.shape[0]
        L = backbone_coords.shape[1] // 3
        if include_O:
            coords = reconstruct_O(backbone_coords, backbone_mask)  # [B, L, 4, 3]
            A = 4
        else:
            coords = backbone_coords.reshape(B, L, 3, 3)  # [B, L, 3, 3]
            A = 3
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {ndim}D")

    device = coords.device

    # --- VdW radii per atom type ---
    if A == 4:
        radii_per_atom = torch.tensor(
            [VDW_RADIUS_N, VDW_RADIUS_C, VDW_RADIUS_C, VDW_RADIUS_O],
            device=device,
        )
    else:
        radii_per_atom = torch.tensor(
            [VDW_RADIUS_N, VDW_RADIUS_C, VDW_RADIUS_C],
            device=device,
        )

    # Tile for all residues: [L*A]
    radii = radii_per_atom.repeat(L)
    # Pairwise VdW sum: [L*A, L*A]
    vdw_sum = radii.unsqueeze(1) + radii.unsqueeze(0)

    # --- Pairwise distances ---
    pos = coords.reshape(B, L * A, 3)  # [B, L*A, 3]
    dists = torch.cdist(pos, pos)       # [B, L*A, L*A]

    # --- Exclusion mask ---
    excl = _build_exclusion_mask(L, A, include_O, device)  # [L*A, L*A]

    # Upper triangle only (avoid double-counting)
    upper = torch.triu(torch.ones(L * A, L * A, dtype=torch.bool, device=device), diagonal=1)
    consider = upper & ~excl  # [L*A, L*A]

    # --- Padding mask ---
    if backbone_mask is not None:
        # Expand residue mask to atom level: [B, L*A]
        atom_mask = backbone_mask.bool().unsqueeze(-1).expand(B, L, A).reshape(B, L * A)
        valid_pairs = atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2)  # [B, L*A, L*A]
        consider = consider.unsqueeze(0) & valid_pairs  # [B, L*A, L*A]
    else:
        consider = consider.unsqueeze(0).expand(B, -1, -1)

    # --- Clash detection ---
    overlap = vdw_sum.unsqueeze(0) - dists  # [B, L*A, L*A]
    is_clash = (overlap >= CLASH_OVERLAP_THRESHOLD) & consider

    clash_count = is_clash.sum(dim=(-1, -2)).float()  # [B]

    # Total atoms per structure
    if backbone_mask is not None:
        total_atoms = backbone_mask.sum(dim=-1).float() * A  # [B]
    else:
        total_atoms = torch.full((B,), L * A, dtype=torch.float32, device=device)

    # Clashes per 1000 atoms
    clashscore = clash_count / total_atoms.clamp(min=1) * 1000.0

    return {
        "clash/score": clashscore.mean(),
        "clash/count": clash_count.mean(),
        "clash/total_atoms": total_atoms.mean(),
    }
