"""Bond length metrics for protein backbones."""

import torch
from torch import Tensor

from protmetrics.constants import (
    BOND_LENGTH_VIOLATION_THRESHOLD,
    IDEAL_CA_C,
    IDEAL_C_N,
    IDEAL_N_CA,
)


def bond_length_metrics(
    backbone_coords: Tensor,
    backbone_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Compute bond length statistics for N-CA-C backbone atoms.

    Args:
        backbone_coords: [B, L*3, 3] with repeating N, CA, C order.
        backbone_mask: [B, L] residue-level mask (1 = valid, 0 = padding).

    Returns:
        Flat dict of scalar tensors suitable for self.log_dict().
    """
    B, L3, _ = backbone_coords.shape
    L = L3 // 3
    # [B, L, 3, 3] -> residue, {N=0, CA=1, C=2}, xyz
    coords = backbone_coords.reshape(B, L, 3, 3)

    N = coords[:, :, 0]  # [B, L, 3]
    CA = coords[:, :, 1]
    C = coords[:, :, 2]

    # Intra-residue distances
    d_n_ca = torch.linalg.norm(CA - N, dim=-1)  # [B, L]
    d_ca_c = torch.linalg.norm(C - CA, dim=-1)  # [B, L]

    # Inter-residue peptide bond: C_i -> N_{i+1}
    d_c_n = torch.linalg.norm(N[:, 1:] - C[:, :-1], dim=-1)  # [B, L-1]

    # Build masks
    if backbone_mask is not None:
        intra_mask = backbone_mask.bool()  # [B, L]
        # Peptide bond valid only if both residues are valid
        inter_mask = backbone_mask[:, :-1].bool() & backbone_mask[:, 1:].bool()
    else:
        intra_mask = torch.ones(B, L, dtype=torch.bool, device=coords.device)
        inter_mask = torch.ones(B, L - 1, dtype=torch.bool, device=coords.device)

    def _stats(d: Tensor, mask: Tensor, ideal: float, name: str) -> dict[str, Tensor]:
        valid = d[mask]
        if valid.numel() == 0:
            nan = torch.tensor(float("nan"), device=d.device)
            return {
                f"bond/{name}_mean": nan,
                f"bond/{name}_std": nan,
                f"bond/{name}_median": nan,
            }
        dev = (valid - ideal).abs()
        return {
            f"bond/{name}_mean": valid.mean(),
            f"bond/{name}_std": valid.std(),
            f"bond/{name}_median": valid.median(),
            f"bond/{name}_dev_mean": dev.mean(),
        }

    metrics = {}
    metrics.update(_stats(d_n_ca, intra_mask, IDEAL_N_CA, "N_CA"))
    metrics.update(_stats(d_ca_c, intra_mask, IDEAL_CA_C, "CA_C"))
    metrics.update(_stats(d_c_n, inter_mask, IDEAL_C_N, "C_N"))

    # Overall violation fraction: |d - ideal| > threshold
    all_devs = torch.cat([
        (d_n_ca[intra_mask] - IDEAL_N_CA).abs(),
        (d_ca_c[intra_mask] - IDEAL_CA_C).abs(),
        (d_c_n[inter_mask] - IDEAL_C_N).abs(),
    ])
    if all_devs.numel() > 0:
        metrics["bond/violation_frac"] = (
            all_devs > BOND_LENGTH_VIOLATION_THRESHOLD
        ).float().mean()
    else:
        metrics["bond/violation_frac"] = torch.tensor(float("nan"), device=coords.device)

    return metrics
