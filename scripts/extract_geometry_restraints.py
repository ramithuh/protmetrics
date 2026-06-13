#!/usr/bin/env python3
"""Extract per-residue sidechain covalent-geometry restraints into a .pt file.

Parses the CCP4/geostd monomer-library cif (``data_<RES>.cif``) for each of the
20 standard residues and pulls the heavy-atom bond, angle, chirality, and
planarity restraints — the same ideals CCTBX uses for geometry validation.
Atom names are mapped to canonical atom14 slot indices (constants.ATOM14_NAMES)
so the PyTorch metrics can gather coordinates directly.

A restraint is kept only if every atom it touches is a heavy atom present in
ATOM14_NAMES (drops all hydrogens). It is tagged ``sidechain=True`` when it
involves any atom beyond backbone N/CA/C/O (slot >= 4), i.e. CB or further
(so CA-CB and N/C-CA-CB are sidechain; pure peptide-backbone restraints are
flagged sidechain=False and can be filtered out downstream).

Usage:
    python scripts/extract_geometry_restraints.py --geostd /home/ruh/geostd

Source: github.com/phenix-project/geostd (CCP4 monomer library + neutron dists).
"""

import argparse
from pathlib import Path

import torch

from protmetrics.allatom.constants import ATOM14_NAMES, RESTYPES

_SIGN = {"negativ": -1, "positiv": 1, "both": 0}

# CA-CB is the one sidechain bond CCTBX adjusts by CDL (varies with backbone
# phi/psi). No static value reproduces it exactly; these per-restype (ideal, esd)
# are the MEAN of the CDL distribution (computed from CCTBX over ~1100 residues),
# the best static approximation. The CA-CB term of sidechain/bond_rmsz is thus
# CDL-approximate (+-~0.01 A phi/psi spread); all distal bonds are exact vs CCTBX.
CA_CB_CDL_MEAN = {
    "ALA": (1.529, 0.017), "ARG": (1.530, 0.017), "ASN": (1.530, 0.016),
    "ASP": (1.529, 0.016), "CYS": (1.530, 0.016), "GLN": (1.530, 0.017),
    "GLU": (1.530, 0.017), "HIS": (1.530, 0.016), "ILE": (1.540, 0.013),
    "LEU": (1.530, 0.016), "LYS": (1.530, 0.017), "MET": (1.530, 0.017),
    "PHE": (1.531, 0.018), "PRO": (1.533, 0.014), "SER": (1.530, 0.017),
    "THR": (1.532, 0.018), "TRP": (1.531, 0.017), "TYR": (1.531, 0.017),
    "VAL": (1.540, 0.014),
}


def _iter_loops(lines):
    """Yield (header_fields, list_of_row_token_lists) for each cif loop_ block."""
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() == "loop_":
            i += 1
            header = []
            while i < n and lines[i].lstrip().startswith("_"):
                header.append(lines[i].strip())
                i += 1
            rows = []
            while i < n:
                s = lines[i].strip()
                if s == "" or s == "loop_" or s.startswith("data_") or s.startswith(";"):
                    break
                rows.append(s.split())
                i += 1
            yield header, rows
        else:
            i += 1


def parse_residue(path: Path, res: str) -> dict:
    """Parse one data_<RES>.cif into atom14-indexed restraint lists."""
    name_to_slot = {nm: s for s, nm in enumerate(ATOM14_NAMES[res]) if nm}
    heavy = set(name_to_slot)
    lines = path.read_text().splitlines()

    bonds, angles, chirs, planes = [], [], [], []
    plane_groups: dict[str, list[int]] = {}

    for header, rows in _iter_loops(lines):
        cols = [h.split(".")[-1] for h in header]
        cat = header[0].split(".")[0] if header else ""

        if cat == "_chem_comp_bond":
            ci = {c: k for k, c in enumerate(cols)}
            for r in rows:
                a1, a2 = r[ci["atom_id_1"]], r[ci["atom_id_2"]]
                if a1 in heavy and a2 in heavy:
                    bonds.append((
                        name_to_slot[a1], name_to_slot[a2],
                        float(r[ci["value_dist"]]), float(r[ci["value_dist_esd"]]),
                    ))

        elif cat == "_chem_comp_angle":
            ci = {c: k for k, c in enumerate(cols)}
            for r in rows:
                a1, a2, a3 = r[ci["atom_id_1"]], r[ci["atom_id_2"]], r[ci["atom_id_3"]]
                if a1 in heavy and a2 in heavy and a3 in heavy:
                    angles.append((
                        name_to_slot[a1], name_to_slot[a2], name_to_slot[a3],
                        float(r[ci["value_angle"]]), float(r[ci["value_angle_esd"]]),
                    ))

        elif cat == "_chem_comp_chir":
            ci = {c: k for k, c in enumerate(cols)}
            for r in rows:
                c, a1, a2, a3 = (r[ci["atom_id_centre"]], r[ci["atom_id_1"]],
                                 r[ci["atom_id_2"]], r[ci["atom_id_3"]])
                sign = _SIGN.get(r[ci["volume_sign"]].lower(), 0)
                if sign != 0 and all(x in heavy for x in (c, a1, a2, a3)):
                    chirs.append((name_to_slot[c], name_to_slot[a1],
                                  name_to_slot[a2], name_to_slot[a3], sign))

        elif cat == "_chem_comp_plane_atom":
            ci = {c: k for k, c in enumerate(cols)}
            for r in rows:
                pid, atom = r[ci["plane_id"]], r[ci["atom_id"]]
                if atom in heavy:
                    plane_groups.setdefault(pid, []).append(name_to_slot[atom])

    for pid, slots in plane_groups.items():
        if len(slots) >= 4:  # 3 points are always coplanar
            planes.append(sorted(slots))

    def _beyond_cb(idxs):  # restraint reaches beyond Cbeta (slot >= 5, i.e. CG onward)
        # These angle/chir/plane ideals are static-cif and thus EXACT vs CCTBX.
        return any(i >= 5 for i in idxs)

    # Bonds: include every bond touching the sidechain (CB onward, slot >= 4) so
    # the CA-CB bond is in. CA-CB (slots 1,4) is the lone CDL-adjusted bond -> use
    # the CA_CB_CDL_MEAN override; all distal bonds keep their exact cif ideals.
    sc_bonds = []
    for i, j, d, e in bonds:
        if not any(s >= 4 for s in (i, j)):
            continue
        if tuple(sorted((i, j))) == (1, 4):  # CA-CB
            d, e = CA_CB_CDL_MEAN[res]
        sc_bonds.append((i, j, d, e))

    return {
        "bonds": sc_bonds,
        "angles": [a for a in angles if _beyond_cb(a[:3])],
        "chirs": [c for c in chirs if _beyond_cb(c[:4])],
        "planes": [p for p in planes if _beyond_cb(p)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geostd", required=True, help="geostd clone root")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent
                    / "src/protmetrics/allatom/geom_data/geom_restraints.pt"),
    )
    args = ap.parse_args()
    geostd = Path(args.geostd)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tables = {}
    for res in RESTYPES:
        path = geostd / res[0].lower() / f"data_{res}.cif"
        if not path.exists():
            raise FileNotFoundError(path)
        t = parse_residue(path, res)
        tables[res] = t
        print(f"  {res}: {len(t['bonds'])} bonds, {len(t['angles'])} angles, "
              f"{len(t['chirs'])} chir, {len(t['planes'])} plane(s)")

    torch.save(tables, out)
    print(f"\nSaved -> {out} ({out.stat().st_size/1e3:.1f} KB)")


if __name__ == "__main__":
    main()
