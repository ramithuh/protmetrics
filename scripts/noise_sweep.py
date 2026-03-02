#!/usr/bin/env python3
"""Noise sweep: compute structural metrics on a real PDB at increasing noise levels.

Validates that metrics degrade smoothly from ground truth → pure noise,
which is the signal we expect to see during flow matching training.

Usage:
    python scripts/noise_sweep.py [PDB_ID]
    python scripts/noise_sweep.py 1ubq
"""

import sys
import tempfile
import urllib.request
from pathlib import Path

import torch

from protmetrics import compute_structural_metrics
from protmetrics.backbone.dihedrals import compute_dihedrals
from protmetrics.backbone.ramachandran import ramachandran_metrics

# Standard alphabetical AA ordering (0-indexed)
_AA3_TO_IDX = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}


def download_and_extract(pdb_id: str):
    """Download PDB and extract backbone coords + aa_seq."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    tmp = Path(tempfile.mkdtemp()) / f"{pdb_id}.pdb"
    urllib.request.urlretrieve(url, tmp)

    coords_list = []
    aa_indices = []

    with open(tmp) as f:
        prev_resseq = None
        cur_atoms = {}
        cur_resname = None

        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                if line.startswith("ENDMDL"):
                    break
                continue

            # Only first chain, no altlocs
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue

            name = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21]
            resseq = line[22:27].strip()

            # First chain only
            if coords_list and chain != first_chain:
                break
            if not coords_list and name == "N":
                first_chain = chain

            if resseq != prev_resseq and prev_resseq is not None:
                # Emit previous residue
                if "N" in cur_atoms and "CA" in cur_atoms and "C" in cur_atoms and cur_resname in _AA3_TO_IDX:
                    coords_list.extend([cur_atoms["N"], cur_atoms["CA"], cur_atoms["C"]])
                    aa_indices.append(_AA3_TO_IDX[cur_resname])
                cur_atoms = {}

            prev_resseq = resseq
            cur_resname = resname
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            cur_atoms[name] = [x, y, z]

        # Last residue
        if "N" in cur_atoms and "CA" in cur_atoms and "C" in cur_atoms and cur_resname in _AA3_TO_IDX:
            coords_list.extend([cur_atoms["N"], cur_atoms["CA"], cur_atoms["C"]])
            aa_indices.append(_AA3_TO_IDX[cur_resname])

    coords = torch.tensor(coords_list, dtype=torch.float32).unsqueeze(0)
    aa_seq = torch.tensor(aa_indices, dtype=torch.long).unsqueeze(0)
    return coords, aa_seq


def main():
    pdb_id = sys.argv[1] if len(sys.argv) > 1 else "1ubq"
    print(f"Downloading {pdb_id.upper()}...")
    coords, aa_seq = download_and_extract(pdb_id)
    n_res = aa_seq.shape[1]
    print(f"Extracted {n_res} residues\n")

    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]

    # Header
    print(f"{'σ (Å)':>8s}  {'bond_viol':>10s}  {'angle_viol':>10s}  "
          f"{'rama_fav':>10s}  {'rama_out':>10s}  "
          f"{'N-CA mean':>10s}  {'CA-C mean':>10s}  {'C-N mean':>10s}")
    print("-" * 95)

    torch.manual_seed(42)
    for sigma in noise_levels:
        if sigma == 0.0:
            noisy = coords.clone()
        else:
            noisy = coords + torch.randn_like(coords) * sigma

        m = compute_structural_metrics(noisy, aa_seq=aa_seq)

        print(
            f"{sigma:8.2f}  "
            f"{m['bond/violation_frac'].item():10.4f}  "
            f"{m['angle/violation_frac'].item():10.4f}  "
            f"{m['rama/favored_frac'].item():10.4f}  "
            f"{m['rama/outlier_frac'].item():10.4f}  "
            f"{m['bond/N_CA_mean'].item():10.4f}  "
            f"{m['bond/CA_C_mean'].item():10.4f}  "
            f"{m['bond/C_N_mean'].item():10.4f}"
        )


if __name__ == "__main__":
    main()
