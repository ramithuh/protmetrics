"""protmetrics — Pure PyTorch structural validation metrics for protein backbones.

Usage with PyTorch Lightning + W&B:

    import protmetrics

    class MyModel(L.LightningModule):
        def validation_step(self, batch, batch_idx):
            coords = self.predict(batch)       # [B, L*3, 3]
            mask = batch["mask"]               # [B, L]
            aa_seq = batch["aa_seq"]           # [B, L]

            metrics = protmetrics.compute_structural_metrics(
                coords, backbone_mask=mask, aa_seq=aa_seq
            )
            self.log_dict(metrics, prog_bar=False)
            # Logged keys include:
            #   bond/rmsz, bond/outlier_frac_4sigma, bond/violation_frac,
            #   angle/rmsz, angle/outlier_frac_4sigma, angle/violation_frac,
            #   rama/favored_frac, rama/allowed_frac, rama/outlier_frac,
            #   bond/N_CA_mean, bond/CA_C_mean, bond/C_N_mean, ...
"""

from protmetrics.backbone import (
    bond_angle_metrics,
    bond_length_metrics,
    compute_dihedrals,
    compute_structural_metrics,
    ramachandran_metrics,
)

__all__ = [
    "bond_length_metrics",
    "bond_angle_metrics",
    "compute_dihedrals",
    "ramachandran_metrics",
    "compute_structural_metrics",
]
