"""Bond angle metrics for protein backbones."""

import torch
from torch import Tensor

from protmetrics.backbone.constants import (
    BOND_ANGLE_VIOLATION_THRESHOLD,
    IDEAL_CA_C_N,
    IDEAL_C_N_CA,
    IDEAL_N_CA_C,
    STDDEV_CA_C_N,
    STDDEV_C_N_CA,
    STDDEV_N_CA_C,
)


def _angle_between(v1: Tensor, v2: Tensor) -> Tensor:
    """Angle in degrees between vectors v1 and v2 along last dim."""
    cos = torch.sum(v1 * v2, dim=-1) / (
        torch.linalg.norm(v1, dim=-1) * torch.linalg.norm(v2, dim=-1) + 1e-8
    )
    cos = cos.clamp(-1.0, 1.0)
    return torch.acos(cos) * (180.0 / torch.pi)


def bond_angle_metrics(
    backbone_coords: Tensor,
    backbone_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Compute bond angle statistics for N-CA-C backbone.

    Args:
        backbone_coords: [B, L*3, 3] with repeating N, CA, C order.
        backbone_mask: [B, L] residue-level mask (1 = valid, 0 = padding).

    Returns:
        Flat dict of scalar tensors.
    """
    B, L3, _ = backbone_coords.shape
    L = L3 // 3
    coords = backbone_coords.reshape(B, L, 3, 3)

    N = coords[:, :, 0]
    CA = coords[:, :, 1]
    C = coords[:, :, 2]

    # Intra-residue: N-CA-C (angle at CA)
    ang_n_ca_c = _angle_between(N - CA, C - CA)  # [B, L]

    # Inter-residue: CA_i-C_i-N_{i+1} (angle at C)
    ang_ca_c_n = _angle_between(CA[:, :-1] - C[:, :-1], N[:, 1:] - C[:, :-1])  # [B, L-1]

    # Inter-residue: C_{i-1}-N_i-CA_i (angle at N)
    ang_c_n_ca = _angle_between(C[:, :-1] - N[:, 1:], CA[:, 1:] - N[:, 1:])  # [B, L-1]

    # Masks
    if backbone_mask is not None:
        intra_mask = backbone_mask.bool()
        inter_mask = backbone_mask[:, :-1].bool() & backbone_mask[:, 1:].bool()
    else:
        intra_mask = torch.ones(B, L, dtype=torch.bool, device=coords.device)
        inter_mask = torch.ones(B, L - 1, dtype=torch.bool, device=coords.device)

    def _stats(ang: Tensor, mask: Tensor, ideal: float, name: str) -> dict[str, Tensor]:
        valid = ang[mask]
        if valid.numel() == 0:
            nan = torch.tensor(float("nan"), device=ang.device)
            return {f"angle/{name}_mean": nan, f"angle/{name}_std": nan}
        return {
            f"angle/{name}_mean": valid.mean(),
            f"angle/{name}_std": valid.std(correction=min(1, valid.numel() - 1)),
            f"angle/{name}_dev_mean": (valid - ideal).abs().mean(),
        }

    metrics = {}
    metrics.update(_stats(ang_n_ca_c, intra_mask, IDEAL_N_CA_C, "N_CA_C"))
    metrics.update(_stats(ang_ca_c_n, inter_mask, IDEAL_CA_C_N, "CA_C_N"))
    metrics.update(_stats(ang_c_n_ca, inter_mask, IDEAL_C_N_CA, "C_N_CA"))

    # Violation fraction
    all_devs = torch.cat([
        (ang_n_ca_c[intra_mask] - IDEAL_N_CA_C).abs(),
        (ang_ca_c_n[inter_mask] - IDEAL_CA_C_N).abs(),
        (ang_c_n_ca[inter_mask] - IDEAL_C_N_CA).abs(),
    ])
    if all_devs.numel() > 0:
        metrics["angle/violation_frac"] = (
            all_devs > BOND_ANGLE_VIOLATION_THRESHOLD
        ).float().mean()
    else:
        metrics["angle/violation_frac"] = torch.tensor(float("nan"), device=coords.device)

    # Z-scores and RMSZ
    z_n_ca_c = (ang_n_ca_c[intra_mask] - IDEAL_N_CA_C) / STDDEV_N_CA_C
    z_ca_c_n = (ang_ca_c_n[inter_mask] - IDEAL_CA_C_N) / STDDEV_CA_C_N
    z_c_n_ca = (ang_c_n_ca[inter_mask] - IDEAL_C_N_CA) / STDDEV_C_N_CA

    all_z = torch.cat([z_n_ca_c, z_ca_c_n, z_c_n_ca])
    if all_z.numel() > 0:
        metrics["angle/rmsz"] = (all_z ** 2).mean().sqrt()
        metrics["angle/outlier_frac_4sigma"] = (all_z.abs() > 4.0).float().mean()
    else:
        nan = torch.tensor(float("nan"), device=coords.device)
        metrics["angle/rmsz"] = nan
        metrics["angle/outlier_frac_4sigma"] = nan

    return metrics
