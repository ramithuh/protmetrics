"""Ramachandran classification using rama8000 lookup tables.

Tables are pre-extracted from CCTBX rama8000 data files (Lovell et al. 2003)
into a single .pt file by scripts/extract_rama_tables.py.

Each table is a [180, 180] grid of probability values in 2-degree bins
covering [-180, 180) for both phi and psi.

Residue-type dispatch (matching CCTBX ramalyze.py):
    1. Glycine       → glycine table
    2. Proline       → cis_proline or trans_proline (based on omega angle)
    3. Pre-proline   → pre_proline table (residue before Pro, if not Gly/Pro)
    4. Ile or Val    → ile_val table (if not pre-Pro)
    5. Everything else → general table
"""

from pathlib import Path

import torch
from torch import Tensor

from protmetrics.backbone.constants import AA_GLY, AA_ILE, AA_PRO, AA_VAL, RAMA_THRESHOLDS

# Lazy-loaded tables
_RAMA_TABLES: dict[str, Tensor] | None = None
_RAMA_DATA_DIR = Path(__file__).parent / "rama_data"


def _load_tables(device: torch.device | None = None) -> dict[str, Tensor]:
    global _RAMA_TABLES
    if _RAMA_TABLES is not None:
        if device is not None:
            return {k: v.to(device) for k, v in _RAMA_TABLES.items()}
        return _RAMA_TABLES

    pt_path = _RAMA_DATA_DIR / "rama_tables.pt"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Rama lookup tables not found at {pt_path}. "
            "Run: python scripts/extract_rama_tables.py"
        )
    _RAMA_TABLES = torch.load(pt_path, weights_only=True)
    if device is not None:
        return {k: v.to(device) for k, v in _RAMA_TABLES.items()}
    return _RAMA_TABLES


def _angle_to_bin(angle: Tensor) -> Tensor:
    """Convert angle in degrees to bin index [0, 179].

    Matches CCTBX convention: bin = (angle + 179) // 2
    (from cctbx_project/mmtbx/validation/ramachandran/convert_from_text.py)
    """
    return ((angle + 179.0) / 2.0).long().clamp(0, 179)


def _angle_to_fractional_bin(angle: Tensor) -> Tensor:
    """Convert angle in degrees to fractional bin index for bilinear interpolation.

    Returns continuous values in [0, 180) with periodic wrapping via % 180.
    """
    return ((angle + 179.0) / 2.0) % 180.0


def _bilinear_lookup(table: Tensor, phi_frac: Tensor, psi_frac: Tensor) -> Tensor:
    """Bilinear interpolation into a [180, 180] rama table with periodic wrapping.

    Args:
        table: [180, 180] probability grid.
        phi_frac: [...] fractional bin indices for phi.
        psi_frac: [...] fractional bin indices for psi.

    Returns:
        [...] interpolated probability values.
    """
    phi_floor = phi_frac.long() % 180
    psi_floor = psi_frac.long() % 180
    phi_ceil = (phi_floor + 1) % 180
    psi_ceil = (psi_floor + 1) % 180

    phi_w = phi_frac - phi_frac.floor()  # weight for ceil
    psi_w = psi_frac - psi_frac.floor()

    # Four corners
    v00 = table[phi_floor, psi_floor]
    v01 = table[phi_floor, psi_ceil]
    v10 = table[phi_ceil, psi_floor]
    v11 = table[phi_ceil, psi_ceil]

    # Bilinear blend
    return (
        v00 * (1 - phi_w) * (1 - psi_w)
        + v01 * (1 - phi_w) * psi_w
        + v10 * phi_w * (1 - psi_w)
        + v11 * phi_w * psi_w
    )


def _classify_residues(
    aa_seq: Tensor,
    omega: Tensor | None,
    valid: Tensor,
    aa_index_offset: int,
) -> Tensor:
    """Assign each residue to a rama table index.

    Table indices: 0=general, 1=glycine, 2=cis_proline, 3=trans_proline,
                   4=pre_proline, 5=ile_val

    Dispatch order (matching CCTBX ramalyze.py):
        1. Gly → glycine
        2. Pro → cis or trans (omega-based)
        3. Pre-Pro (not Gly, not Pro, next is Pro) → pre_proline
        4. Ile/Val (not pre-Pro) → ile_val
        5. Everything else → general
    """
    B, L = aa_seq.shape
    aa = aa_seq - aa_index_offset  # normalize to 0-indexed

    table_idx = torch.zeros(B, L, dtype=torch.long, device=aa_seq.device)  # general

    is_gly = aa == AA_GLY
    is_pro = aa == AA_PRO
    is_ile_val = (aa == AA_ILE) | (aa == AA_VAL)

    # Pre-proline: residue i where i+1 is Pro, and i is not Gly or Pro
    is_pre_pro = torch.zeros_like(is_gly)
    is_pre_pro[:, :-1] = is_pro[:, 1:] & ~is_gly[:, :-1] & ~is_pro[:, :-1]

    # Cis vs trans proline: |omega| < 90° → cis, else trans
    is_cis_pro = torch.zeros_like(is_pro)
    if omega is not None:
        is_cis_pro = is_pro & ~torch.isnan(omega) & (omega.abs() < 90.0)
    is_trans_pro = is_pro & ~is_cis_pro

    # Assign in priority order (later assignments override earlier)
    table_idx[is_ile_val] = 5  # ile_val
    table_idx[is_pre_pro] = 4  # pre_proline (overrides ile_val if applicable)
    table_idx[is_trans_pro] = 3
    table_idx[is_cis_pro] = 2
    table_idx[is_gly] = 1

    return table_idx


# Ordered to match table_idx values from _classify_residues
_TABLE_NAMES = ["general", "glycine", "cis_proline", "trans_proline", "pre_proline", "ile_val"]


def ramachandran_metrics(
    phi: Tensor,
    psi: Tensor,
    aa_seq: Tensor | None = None,
    omega: Tensor | None = None,
    aa_index_offset: int = 0,
) -> dict[str, Tensor]:
    """Classify phi/psi angles using rama8000 tables.

    Args:
        phi: [B, L] phi angles in degrees (NaN = undefined).
        psi: [B, L] psi angles in degrees (NaN = undefined).
        aa_seq: [B, L] integer amino acid indices (optional).
            If None, uses the General table for all residues.
        omega: [B, L] omega dihedral angles in degrees (optional).
            Needed for cis/trans proline distinction. If None and aa_seq
            contains Pro, defaults to trans-proline table.
        aa_index_offset: Offset to subtract from aa_seq to get 0-indexed
            standard alphabetical order. Use 1 if your encoding reserves
            index 0 for padding (e.g. xymol convention).

    Returns:
        Flat dict with rama/favored_frac, rama/allowed_frac, rama/outlier_frac.
    """
    # Mask: both phi and psi must be defined (not NaN)
    valid = ~(torch.isnan(phi) | torch.isnan(psi))

    if valid.sum() == 0:
        nan = torch.tensor(float("nan"), device=phi.device)
        return {
            "rama/favored_frac": nan,
            "rama/allowed_frac": nan,
            "rama/outlier_frac": nan,
        }

    tables = _load_tables(phi.device)

    phi_frac = _angle_to_fractional_bin(phi)  # [B, L]
    psi_frac = _angle_to_fractional_bin(psi)

    # Determine per-residue table assignment
    if aa_seq is not None:
        table_idx = _classify_residues(aa_seq, omega, valid, aa_index_offset)
    else:
        table_idx = torch.zeros_like(phi, dtype=torch.long)  # all general

    # Look up probability and thresholds per residue (bilinear interpolation)
    probs = torch.zeros_like(phi)
    fav_thresh = torch.zeros_like(phi)
    allowed_thresh = torch.zeros_like(phi)

    for i, name in enumerate(_TABLE_NAMES):
        mask = (table_idx == i) & valid
        if mask.any():
            table = tables[name].to(phi.device)
            probs[mask] = _bilinear_lookup(table, phi_frac[mask], psi_frac[mask])
            fav_t, allowed_t = RAMA_THRESHOLDS[name]
            fav_thresh[mask] = fav_t
            allowed_thresh[mask] = allowed_t

    # Classify only valid residues
    probs_valid = probs[valid]
    fav_valid = fav_thresh[valid]
    allowed_valid = allowed_thresh[valid]

    favored = (probs_valid >= fav_valid).float()
    allowed = ((probs_valid >= allowed_valid) & (probs_valid < fav_valid)).float()
    outlier = (probs_valid < allowed_valid).float()

    return {
        "rama/favored_frac": favored.mean(),
        "rama/allowed_frac": allowed.mean(),
        "rama/outlier_frac": outlier.mean(),
    }
