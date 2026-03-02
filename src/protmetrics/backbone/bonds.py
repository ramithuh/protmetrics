"""Bond length metrics for protein backbones."""

import torch
from torch import Tensor

from protmetrics.backbone.constants import (
    AA_PRO,
    BOND_LENGTH_VIOLATION_THRESHOLD,
    IDEAL_CA_C,
    IDEAL_C_N,
    IDEAL_C_N_PRO,
    IDEAL_N_CA,
    STDDEV_CA_C,
    STDDEV_C_N,
    STDDEV_C_N_PRO,
    STDDEV_N_CA,
)


def bond_length_metrics(
    backbone_coords: Tensor,
    backbone_mask: Tensor | None = None,
    aa_seq: Tensor | None = None,
    aa_index_offset: int = 0,
) -> dict[str, Tensor]:
    """Compute bond length statistics for N-CA-C backbone atoms.

    Args:
        backbone_coords: [B, L*3, 3] with repeating N, CA, C order.
        backbone_mask: [B, L] residue-level mask (1 = valid, 0 = padding).
        aa_seq: [B, L] integer amino acid indices (optional).
            When provided, Pro residues use IDEAL_C_N_PRO for the C-N
            peptide bond (C_{i-1} -> N_i where i is Pro).
        aa_index_offset: Offset for aa_seq (use 1 if 1-indexed/padding at 0).

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

    # Per-bond ideal and sigma for C-N, accounting for proline
    # The peptide bond C_i -> N_{i+1}: use Pro ideal when residue i+1 is Pro
    ideal_c_n = torch.full_like(d_c_n, IDEAL_C_N)
    sigma_c_n = torch.full_like(d_c_n, STDDEV_C_N)
    if aa_seq is not None and L > 1:
        aa = aa_seq - aa_index_offset
        next_is_pro = aa[:, 1:] == AA_PRO  # [B, L-1]
        ideal_c_n[next_is_pro] = IDEAL_C_N_PRO
        sigma_c_n[next_is_pro] = STDDEV_C_N_PRO

    def _stats(d: Tensor, mask: Tensor, ideal, name: str) -> dict[str, Tensor]:
        """ideal can be a scalar or a tensor broadcastable to d[mask]."""
        valid = d[mask]
        if valid.numel() == 0:
            nan = torch.tensor(float("nan"), device=d.device)
            return {
                f"bond/{name}_mean": nan,
                f"bond/{name}_std": nan,
                f"bond/{name}_median": nan,
            }
        ideal_valid = ideal[mask] if isinstance(ideal, Tensor) else ideal
        dev = (valid - ideal_valid).abs()
        return {
            f"bond/{name}_mean": valid.mean(),
            f"bond/{name}_std": valid.std(correction=min(1, valid.numel() - 1)),
            f"bond/{name}_median": valid.median(),
            f"bond/{name}_dev_mean": dev.mean(),
        }

    metrics = {}
    metrics.update(_stats(d_n_ca, intra_mask, IDEAL_N_CA, "N_CA"))
    metrics.update(_stats(d_ca_c, intra_mask, IDEAL_CA_C, "CA_C"))
    metrics.update(_stats(d_c_n, inter_mask, ideal_c_n, "C_N"))

    # Overall violation fraction: |d - ideal| > threshold
    all_devs = torch.cat([
        (d_n_ca[intra_mask] - IDEAL_N_CA).abs(),
        (d_ca_c[intra_mask] - IDEAL_CA_C).abs(),
        (d_c_n[inter_mask] - ideal_c_n[inter_mask]).abs(),
    ])
    if all_devs.numel() > 0:
        metrics["bond/violation_frac"] = (
            all_devs > BOND_LENGTH_VIOLATION_THRESHOLD
        ).float().mean()
    else:
        metrics["bond/violation_frac"] = torch.tensor(float("nan"), device=coords.device)

    # Z-scores and RMSZ
    z_n_ca = (d_n_ca[intra_mask] - IDEAL_N_CA) / STDDEV_N_CA
    z_ca_c = (d_ca_c[intra_mask] - IDEAL_CA_C) / STDDEV_CA_C
    z_c_n = (d_c_n[inter_mask] - ideal_c_n[inter_mask]) / sigma_c_n[inter_mask]

    all_z = torch.cat([z_n_ca, z_ca_c, z_c_n])
    if all_z.numel() > 0:
        metrics["bond/rmsz"] = (all_z ** 2).mean().sqrt()
        metrics["bond/outlier_frac_4sigma"] = (all_z.abs() > 4.0).float().mean()
    else:
        nan = torch.tensor(float("nan"), device=coords.device)
        metrics["bond/rmsz"] = nan
        metrics["bond/outlier_frac_4sigma"] = nan

    return metrics
