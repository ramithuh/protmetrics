"""Native-pocket recovery metrics: symmetry-aware chi-RMSD and rotamer recovery.

First-class for the *conditioned / memorization* setting where the native pocket
is available. N/A for unconditioned de novo (no reference) — callers simply skip
these when no native is provided.

Both metrics fold 180-degree-symmetric terminal chi (Asp chi2, Glu chi3,
Phe/Tyr chi2) so that a flipped-but-equivalent rotamer is not penalized.
"""

import torch
from torch import Tensor

from protmetrics.allatom.constants import CHI_SYMMETRIC


def _chi_abs_diff(pred: Tensor, native: Tensor, symmetric: Tensor) -> Tensor:
    """Smallest angular difference in degrees, honoring chi symmetry.

    Non-symmetric chi have period 360; symmetric terminal chi have period 180.
    """
    d = (pred - native).abs() % 360.0
    d = torch.minimum(d, 360.0 - d)            # fold to [0, 180]
    d_sym = d % 180.0
    d_sym = torch.minimum(d_sym, 180.0 - d_sym)  # fold to [0, 90] for symmetric
    return torch.where(symmetric, d_sym, d)


def chi_rmsd_metrics(
    chi_pred: Tensor,
    chi_native: Tensor,
    aa_seq: Tensor,
    valid: Tensor,
) -> dict[str, Tensor]:
    """Symmetry-aware chi-RMSD over residues where both pred and native chi exist.

    Args:
        chi_pred, chi_native: [B, L, 4] degrees (NaN where undefined).
        aa_seq: [B, L] residue types (for per-chi symmetry lookup).
        valid: [B, L, 4] bool — chi defined in *both* pred and native.

    Returns:
        chi/rmsd_deg (pooled over all valid chi), chi/mae_deg.
    """
    sym = CHI_SYMMETRIC.to(aa_seq.device)[aa_seq]      # [B, L, 4]
    both = valid & ~torch.isnan(chi_pred) & ~torch.isnan(chi_native)
    d = _chi_abs_diff(chi_pred, chi_native, sym)[both]
    if d.numel() == 0:
        nan = torch.tensor(float("nan"), device=aa_seq.device)
        return {"chi/rmsd_deg": nan, "chi/mae_deg": nan}
    return {"chi/rmsd_deg": (d ** 2).mean().sqrt(), "chi/mae_deg": d.mean()}


def rotamer_recovery(
    chi_pred: Tensor,
    chi_native: Tensor,
    aa_seq: Tensor,
    valid: Tensor,
    tol_deg: float = 40.0,
) -> dict[str, Tensor]:
    """Fraction of residues whose every defined chi is within `tol_deg` of native.

    Tolerance-based proxy for "same rotamer well" (Top8000-bin matching can
    replace this once the rotamer tables are extracted — see rotamers.py).
    Per-residue rate over pocket residues with >=1 defined chi.
    """
    sym = CHI_SYMMETRIC.to(aa_seq.device)[aa_seq]
    both = valid & ~torch.isnan(chi_pred) & ~torch.isnan(chi_native)  # [B, L, 4]
    d = _chi_abs_diff(chi_pred, chi_native, sym)
    within = (d <= tol_deg) | ~both                    # ignore undefined chi
    has_chi = both.any(dim=-1)                          # [B, L]
    recovered = within.all(dim=-1) & has_chi
    n = has_chi.sum()
    if n == 0:
        return {"rotamer/recovery_frac": torch.tensor(float("nan"), device=aa_seq.device)}
    return {"rotamer/recovery_frac": recovered.sum().float() / n.float()}
