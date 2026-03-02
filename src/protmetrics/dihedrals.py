"""Phi/psi dihedral angle computation for protein backbones."""

import torch
from torch import Tensor


def _dihedral(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
    """Compute dihedral angle (in degrees) for 4-atom sequences A-B-C-D.

    Standard formulation via cross products:
        b1 = B - A, b2 = C - B, b3 = D - C
        n1 = b1 x b2, n2 = b2 x b3
        angle = atan2(dot(n1 x n2, b2/|b2|), dot(n1, n2))

    Args:
        p0, p1, p2, p3: [..., 3] atom coordinates.

    Returns:
        [...] dihedral angles in degrees, range (-180, 180].
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = torch.linalg.cross(b1, b2)
    n2 = torch.linalg.cross(b2, b3)

    # Unit vector along b2
    b2_norm = b2 / (torch.linalg.norm(b2, dim=-1, keepdim=True) + 1e-8)

    m1 = torch.linalg.cross(n1, b2_norm)

    x = torch.sum(n1 * n2, dim=-1)
    y = torch.sum(m1 * n2, dim=-1)

    # Negate y to match IUPAC sign convention (same as CCTBX/MolProbity)
    return torch.atan2(-y, x) * (180.0 / torch.pi)


def compute_dihedrals(
    backbone_coords: Tensor,
    backbone_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute phi, psi, and omega dihedral angles from backbone N-CA-C coordinates.

    Args:
        backbone_coords: [B, L*3, 3] with repeating N, CA, C order.
        backbone_mask: [B, L] residue-level mask (1 = valid, 0 = padding).

    Returns:
        phi: [B, L] in degrees. NaN for first residue and masked positions.
        psi: [B, L] in degrees. NaN for last residue and masked positions.
        omega: [B, L] in degrees. NaN for first residue and masked positions.
            Peptide bond planarity: ~180° for trans, ~0° for cis.
    """
    B, L3, _ = backbone_coords.shape
    L = L3 // 3
    coords = backbone_coords.reshape(B, L, 3, 3)

    N = coords[:, :, 0]  # [B, L, 3]
    CA = coords[:, :, 1]
    C = coords[:, :, 2]

    phi = torch.full((B, L), float("nan"), device=backbone_coords.device)
    psi = torch.full((B, L), float("nan"), device=backbone_coords.device)
    omega = torch.full((B, L), float("nan"), device=backbone_coords.device)

    # phi_i = dihedral(C_{i-1}, N_i, CA_i, C_i)  — undefined for i=0
    if L > 1:
        phi[:, 1:] = _dihedral(C[:, :-1], N[:, 1:], CA[:, 1:], C[:, 1:])

    # psi_i = dihedral(N_i, CA_i, C_i, N_{i+1})  — undefined for i=L-1
    if L > 1:
        psi[:, :-1] = _dihedral(N[:, :-1], CA[:, :-1], C[:, :-1], N[:, 1:])

    # omega_i = dihedral(CA_{i-1}, C_{i-1}, N_i, CA_i) — undefined for i=0
    if L > 1:
        omega[:, 1:] = _dihedral(CA[:, :-1], C[:, :-1], N[:, 1:], CA[:, 1:])

    # Apply mask: NaN out invalid residues + neighbors that depend on them
    if backbone_mask is not None:
        mask = backbone_mask.bool()
        # phi_i and omega_i need residue i and i-1 both valid
        inter_mask_prev = torch.zeros_like(mask)
        inter_mask_prev[:, 1:] = mask[:, 1:] & mask[:, :-1]
        phi[~inter_mask_prev] = float("nan")
        omega[~inter_mask_prev] = float("nan")

        # psi_i needs residue i and i+1 both valid
        psi_mask = torch.zeros_like(mask)
        psi_mask[:, :-1] = mask[:, :-1] & mask[:, 1:]
        psi[~psi_mask] = float("nan")

    return phi, psi, omega
