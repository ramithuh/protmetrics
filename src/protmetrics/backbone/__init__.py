"""protmetrics.backbone — Backbone geometry metrics."""

from protmetrics.backbone.angles import bond_angle_metrics
from protmetrics.backbone.bonds import bond_length_metrics
from protmetrics.backbone.dihedrals import compute_dihedrals
from protmetrics.backbone.ramachandran import ramachandran_metrics

__all__ = [
    "bond_length_metrics",
    "bond_angle_metrics",
    "compute_dihedrals",
    "ramachandran_metrics",
    "compute_structural_metrics",
]


def compute_structural_metrics(
    backbone_coords,
    backbone_mask=None,
    aa_seq=None,
    aa_index_offset=0,
):
    """All-in-one structural metrics. Returns flat dict for self.log_dict().

    Args:
        backbone_coords: [B, L*3, 3] with repeating N, CA, C order.
        backbone_mask: [B, L] residue-level mask (1 = valid, 0 = padding).
        aa_seq: [B, L] integer amino acid indices for rama table dispatch.
        aa_index_offset: Offset for aa_seq (use 1 if 1-indexed/padding at 0).
    """
    metrics = {}
    metrics.update(bond_length_metrics(backbone_coords, backbone_mask, aa_seq, aa_index_offset))
    metrics.update(bond_angle_metrics(backbone_coords, backbone_mask))
    phi, psi, omega = compute_dihedrals(backbone_coords, backbone_mask)
    metrics.update(ramachandran_metrics(phi, psi, aa_seq, omega, aa_index_offset))
    return metrics
