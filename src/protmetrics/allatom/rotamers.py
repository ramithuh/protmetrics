"""Rotamer outlier metric — faithful reimplementation of CCTBX rotalyze.

Each residue type has an n-dimensional Top8000 probability grid (rota8000,
the same data CCTBX uses). The rotamer "score" is the grid value at the
residue's chi angles, obtained by bin-center multilinear interpolation
replicating mmtbx.rotamer.n_dim_table.NDimTable.valueAt exactly (2^n
surrounding bins, per-dimension wrap/clamp). Classification matches
mmtbx.validation.rotalyze:

    value >= 0.02   -> Favored
    value >= 0.003  -> Allowed
    value <  0.003  -> OUTLIER          (reported score = value * 100)

Symmetric terminal chi (Asp chi2, Glu chi3, Phe/Tyr chi2) need no special
handling: their grids are defined on a 0-180 wrapping range, so feeding the
raw measured chi and letting the grid wrap reproduces CCTBX (which also feeds
raw, unfolded chi to valueAt).

POCKET-ONLY: scored over the generated pocket residues with all chi defined.
A valid per-residue rate, but NOT directly comparable to all-residue numbers
(e.g. La-Proteina). Label accordingly when logging.

Evaluation only — do NOT use as a training loss (Goodhart).
"""

from pathlib import Path

import torch
from torch import Tensor

from protmetrics.allatom.constants import RESTYPES
from protmetrics.allatom.dihedrals import compute_chi

OUTLIER_THRESHOLD = 0.003
ALLOWED_THRESHOLD = 0.02

_TABLES: dict | None = None
_DATA = Path(__file__).parent / "rota_data" / "rota_tables.pt"


def _load() -> dict:
    global _TABLES
    if _TABLES is None:
        if not _DATA.exists():
            raise FileNotFoundError(
                f"Rotamer grids not found at {_DATA}. Run "
                "scripts/extract_rotamer_tables.py --src <Top8000_rotamer_pct_contour_grids>"
            )
        _TABLES = torch.load(_DATA, weights_only=False)
    return _TABLES


def _value_at(chi: Tensor, g: dict) -> Tensor:
    """Bin-center multilinear interpolation, matching NDimTable.valueAt.

    Args:
        chi: [M, D] chi angles (degrees) for M residues of one type, D = grid dims.
        g: grid dict with n_dim, minVal, wBin, nBins, doWrap, grid.

    Returns:
        [M] interpolated probability values.
    """
    device = chi.device
    D = g["n_dim"]
    minV = torch.tensor(g["minVal"], device=device)
    wBin = torch.tensor(g["wBin"], device=device)
    nB = torch.tensor(g["nBins"], device=device)
    grid_flat = g["grid"].reshape(-1).to(device)
    # C-order strides
    strides = [1] * D
    for i in range(D - 2, -1, -1):
        strides[i] = strides[i + 1] * g["nBins"][i + 1]

    home = torch.minimum(torch.floor((chi - minV) / wBin).long(), nB - 1)  # [M, D]
    home_ctr = minV + wBin * (home.float() + 0.5)
    neighbor = torch.where(chi < home_ctr, home - 1, home + 1)
    contrib = ((chi - home_ctr) / wBin).abs()  # [M, D] in [0, 0.5]

    value = torch.zeros(chi.shape[0], device=device)
    for corner in range(1 << D):
        coeff = torch.ones(chi.shape[0], device=device)
        flat = torch.zeros(chi.shape[0], dtype=torch.long, device=device)
        for d in range(D):
            if corner & (1 << d):
                b = neighbor[:, d]
                coeff = coeff * contrib[:, d]
            else:
                b = home[:, d]
                coeff = coeff * (1.0 - contrib[:, d])
            if g["doWrap"][d]:
                b = b % g["nBins"][d]
            b = b.clamp(0, g["nBins"][d] - 1)
            flat = flat + b * strides[d]
        value = value + coeff * grid_flat[flat]
    return value


def rotamer_metrics(
    chi: Tensor,
    chi_valid: Tensor,
    aa_seq: Tensor,
) -> dict[str, Tensor]:
    """Rotamer favored/allowed/outlier fractions, faithful to CCTBX rotalyze.

    Args:
        chi: [B, L, 4] chi angles in degrees (from compute_chi).
        chi_valid: [B, L, 4] bool — chi defined & well-formed.
        aa_seq: [B, L] residue types (0-indexed alphabetical).

    Returns:
        rotamer/favored_frac, rotamer/allowed_frac, rotamer/outlier_frac,
        over scored residues (pocket residues with all rotameric chi defined).
        Gly/Ala and incomplete residues are excluded.
    """
    tables = _load()
    aa_to_file = tables["aa_to_file"]
    grids = tables["tables"]
    device = chi.device

    values = torch.full(aa_seq.shape, float("nan"), device=device)  # [B, L]

    for idx in range(20):
        aa3 = RESTYPES[idx].lower()
        if aa3 not in aa_to_file:
            continue
        g = grids[aa_to_file[aa3]]
        D = g["n_dim"]
        # residues of this type with all D rotameric chi defined
        sel = (aa_seq == idx) & chi_valid[..., :D].all(dim=-1)
        if sel.any():
            values[sel] = _value_at(chi[sel][:, :D], g)

    scored = ~torch.isnan(values)
    if scored.sum() == 0:
        nan = torch.tensor(float("nan"), device=device)
        return {
            "rotamer/favored_frac": nan,
            "rotamer/allowed_frac": nan,
            "rotamer/outlier_frac": nan,
        }
    v = values[scored]
    favored = (v >= ALLOWED_THRESHOLD).float()
    allowed = ((v >= OUTLIER_THRESHOLD) & (v < ALLOWED_THRESHOLD)).float()
    outlier = (v < OUTLIER_THRESHOLD).float()
    return {
        "rotamer/favored_frac": favored.mean(),
        "rotamer/allowed_frac": allowed.mean(),
        "rotamer/outlier_frac": outlier.mean(),
    }
