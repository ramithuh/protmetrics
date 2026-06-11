#!/usr/bin/env python3
"""Extract Top8000 rotamer contour grids into a packaged .pt file.

Parses the rota8000-*.data files (rlabduke/reference_data, Top8000 rotamer
percent-contour grids — the same data CCTBX rotalyze uses) into per-residue
n-dimensional probability grids, replicating mmtbx.rotamer.n_dim_table's
layout exactly so the PyTorch lookup matches CCTBX's NDimTable.valueAt.

Each .data file header gives, per chi dimension: lower_bound, upper_bound,
n_bins, wrapping. Data lines give bin-center coordinates + probability value.
Grids are stored C-order (row-major), matching flex.grid / bin2index.

Usage:
    python scripts/extract_rotamer_tables.py \
        --src /home/ruh/reference_data/Top8000/Top8000_rotamer_pct_contour_grids

Source: github.com/rlabduke/reference_data
Reference: Shapovalov & Dunbrack rotamer libs as packaged in Top8000.
"""

import argparse
import math
import re
from pathlib import Path

import torch

# CCTBX aminoAcids mapping (mmtbx/rotamer/rotamer_eval.py): aa -> file stem.
AA_TO_FILE = {
    "arg": "arg", "asn": "asn", "asp": "asp", "cys": "cys", "gln": "gln",
    "glu": "glu", "his": "his", "ile": "ile", "leu": "leu", "lys": "lys",
    "met": "met", "phe": "phetyr", "pro": "pro", "ser": "ser", "thr": "thr",
    "trp": "trp", "tyr": "phetyr", "val": "val",
}
FILE_STEMS = sorted(set(AA_TO_FILE.values()))

_DIM_RE = re.compile(r"#\s*x\d:\s*([-\d.]+)\s+([-\d.]+)\s+(\d+)\s+(\w+)")


def parse_data(path: Path) -> dict:
    """Parse one rota8000-*.data file into grid params + a C-order grid tensor."""
    minVal, maxVal, nBins, doWrap = [], [], [], []
    n_dim = None
    data_lines = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                if "Number of dimensions" in line:
                    n_dim = int(re.search(r"(\d+)\s*$", line).group(1))
                m = _DIM_RE.search(line)
                if m:
                    minVal.append(float(m.group(1)))
                    maxVal.append(float(m.group(2)))
                    nBins.append(int(m.group(3)))
                    doWrap.append(m.group(4).lower() in ("true", "yes", "on", "1"))
            elif line.strip():
                data_lines.append(line.split())

    assert n_dim == len(nBins), f"{path}: dim mismatch"
    wBin = [(maxVal[i] - minVal[i]) / nBins[i] for i in range(n_dim)]
    grid = torch.zeros(*nBins, dtype=torch.float32)

    for fields in data_lines:
        if len(fields) <= n_dim:
            continue
        coords = [float(x) for x in fields[:n_dim]]
        val = float(fields[n_dim])
        idx = tuple(
            min(int(math.floor((coords[i] - minVal[i]) / wBin[i])), nBins[i] - 1)
            for i in range(n_dim)
        )
        grid[idx] = val

    return {
        "n_dim": n_dim,
        "minVal": minVal,
        "wBin": wBin,
        "nBins": nBins,
        "doWrap": doWrap,
        "grid": grid,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir with rota8000-*.data")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent.parent
                    / "src/protmetrics/allatom/rota_data/rota_tables.pt"),
    )
    args = ap.parse_args()
    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tables = {}
    for stem in FILE_STEMS:
        path = src / f"rota8000-{stem}.data"
        tables[stem] = parse_data(path)
        g = tables[stem]
        print(f"  {stem:8s} nDim={g['n_dim']} nBins={g['nBins']} "
              f"nonzero={(g['grid'] > 0).sum().item()}/{g['grid'].numel()}")

    torch.save({"aa_to_file": AA_TO_FILE, "tables": tables}, out)
    print(f"\nSaved {len(tables)} grids -> {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
