#!/usr/bin/env python3
"""Run CCTBX ramalyze + clashscore on generated PDB files and output CSV.

Optionally compares CCTBX results against protmetrics values from eval_5ckpt.csv.

Usage:
    python scripts/validate_cctbx.py /path/to/results [--epochs GT,ep319] [--n N] [--out cctbx_eval.csv]
"""

import argparse
import csv
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# CCTBX imports + probe monkey-patch
# ---------------------------------------------------------------------------
from iotbx import pdb as iotbx_pdb
from mmtbx.validation import ramalyze

try:
    from mmtbx.validation.clashscore import clashscore as cctbx_clashscore_cls
    import libtbx
    if hasattr(libtbx, "env") and libtbx.env is not None:
        _orig_has_module = libtbx.env.has_module
        def _patched_has_module(name):
            result = _orig_has_module(name)
            if not result:
                result = shutil.which(name) is not None
            return result
        libtbx.env.has_module = _patched_has_module
    _HAS_PROBE = True
except ImportError:
    _HAS_PROBE = False

_HAS_REDUCE = shutil.which("phenix.reduce") is not None


def _first_model_hierarchy(hierarchy):
    if len(hierarchy.models()) > 1:
        sel = hierarchy.atom_selection_cache().selection(
            f"model_id {hierarchy.models()[0].id}"
        )
        hierarchy = hierarchy.select(sel)
    return hierarchy


def _reduce_add_hydrogens(pdb_path: str) -> str | None:
    """Run phenix.reduce to add hydrogens, matching MolProbity pipeline.

    MolProbity first strips H (reduce -trim -allalt), then re-adds with flips
    (reduce -build). Returns path to temp PDB with H, or None on failure.
    """
    if not _HAS_REDUCE:
        return None
    try:
        # Step 1: strip existing H (including all alt conformations)
        trim_result = subprocess.run(
            ["phenix.reduce", "-quiet", "-trim", "-allalt", pdb_path],
            capture_output=True, timeout=60,
        )
        trimmed = trim_result.stdout
        if trim_result.returncode != 0 or not trimmed:
            return None

        # Strip USER MOD records (MolProbity does this; fatal to some tools)
        trimmed = b"\n".join(
            line for line in trimmed.split(b"\n")
            if not line.startswith(b"USER  MOD")
        )

        # Step 2: re-add H with Asn/Gln/His flips
        build_result = subprocess.run(
            ["phenix.reduce", "-quiet", "-build"],
            input=trimmed, capture_output=True, timeout=120,
        )
        built = build_result.stdout
        if not built:
            return None

        # Write to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
        tmp.write(built)
        tmp.close()
        return tmp.name
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"  Warning: reduce failed on {pdb_path}: {e}", file=sys.stderr)
        return None


def run_cctbx_on_pdb(pdb_path: str) -> dict:
    """Run ramalyze + clashscore on a PDB, return metrics dict.

    Follows MolProbity's pipeline: ramalyze on the H-added structure,
    clashscore with b_factor_cutoff=40 and keep_hydrogens=True.
    """
    # Add hydrogens via Reduce (MolProbity Step 1+2)
    reduced_path = _reduce_add_hydrogens(pdb_path)
    analysis_path = reduced_path if reduced_path else pdb_path

    pdb_inp = iotbx_pdb.input(file_name=analysis_path)
    hierarchy = _first_model_hierarchy(pdb_inp.construct_hierarchy())

    # Ramachandran
    rama = ramalyze.ramalyze(pdb_hierarchy=hierarchy, outliers_only=False)
    n = rama.n_total
    if n > 0:
        rama_fav = rama.n_favored / n
        rama_alw = rama.n_allowed / n
        rama_out = rama.n_outliers / n
    else:
        rama_fav = rama_alw = rama_out = float("nan")

    # Clashscore — matching MolProbity: b_factor_cutoff=40, keep_hydrogens=True
    clashscore_val = float("nan")
    n_clashes = 0
    if _HAS_PROBE:
        try:
            clash = cctbx_clashscore_cls(
                pdb_hierarchy=hierarchy,
                b_factor_cutoff=40,
                keep_hydrogens=True,
                nuclear=False,
            )
            clashscore_val = clash.get_clashscore()
            n_clashes = len(clash.results)
        except Exception as e:
            print(f"  Warning: clashscore failed on {pdb_path}: {e}", file=sys.stderr)

    # Clean up temp file
    if reduced_path:
        try:
            Path(reduced_path).unlink()
        except OSError:
            pass

    return {
        "rama_favored": rama_fav,
        "rama_allowed": rama_alw,
        "rama_outlier": rama_out,
        "rama_n_total": n,
        "clashscore": clashscore_val,
        "n_clashes": n_clashes,
    }


def load_eval_csv(results_dir: Path) -> dict:
    """Load eval_5ckpt.csv (or eval.csv fallback) into {(label, system_id): row_dict}."""
    for name in ["eval_5ckpt.csv", "eval.csv"]:
        csv_path = results_dir / name
        if csv_path.exists():
            print(f"Loading protmetrics reference from {csv_path.name}")
            lookup = {}
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row["label"], row["system_id"])
                    lookup[key] = row
            return lookup
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run CCTBX validation on generated PDBs")
    parser.add_argument("results_dir", type=Path, help="Path to results directory")
    parser.add_argument("--epochs", default=None, help="Comma-separated labels (e.g. GT,ep319)")
    parser.add_argument("--n", type=int, default=None, help="Max samples per epoch")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: results_dir/cctbx_eval.csv)")
    args = parser.parse_args()

    results_dir = args.results_dir
    eval_lookup = load_eval_csv(results_dir)

    # Determine output path (never inside the results dir — write next to this script or user-specified)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(__file__).resolve().parent.parent / "cctbx_eval.csv"

    # Discover directories
    if args.epochs:
        epoch_labels = args.epochs.split(",")
    else:
        epoch_labels = []
        if (results_dir / "ground_truth").is_dir():
            epoch_labels.append("GT")
        for d in sorted(results_dir.iterdir()):
            if d.is_dir() and d.name.startswith("epoch_"):
                epoch_labels.append(d.name)

    # Build (label, dir_path, csv_label) triples
    dirs = []
    for label in epoch_labels:
        if label == "GT":
            dirs.append(("GT", results_dir / "ground_truth", "GT"))
        elif label.startswith("epoch_"):
            num = label.replace("epoch_", "")
            dirs.append((label, results_dir / label, f"ep{num}"))
        else:
            num = label.replace("ep", "")
            dirs.append((label, results_dir / f"epoch_{num}", label))

    # CSV fields
    fieldnames = [
        "label", "system_id",
        "cctbx_rama_favored", "cctbx_rama_allowed", "cctbx_rama_outlier", "cctbx_rama_n_total",
        "cctbx_clashscore", "cctbx_n_clashes",
        "pm_rama_favored", "pm_rama_outlier",
    ]

    all_rows = []
    t0 = time.time()

    for dir_label, dir_path, csv_label in dirs:
        if not dir_path.is_dir():
            print(f"Skipping {dir_label}: {dir_path} not found")
            continue

        pdbs = sorted(dir_path.glob("*.pdb"))
        if args.n is not None:
            pdbs = pdbs[:args.n]

        print(f"\n[{csv_label}] Processing {len(pdbs)} PDBs from {dir_path.name}...")

        for i, pdb_path in enumerate(pdbs):
            # Extract system_id: sample_0_sample_0.pdb -> sample_0
            fname = pdb_path.stem
            parts = fname.split("_")
            system_id = f"{parts[0]}_{parts[1]}"

            cctbx = run_cctbx_on_pdb(str(pdb_path))

            # Protmetrics reference (rama only — clash is handled by CCTBX)
            pm_row = eval_lookup.get((csv_label, system_id))
            pm_fav = float(pm_row["rama/favored_frac"]) if pm_row else float("nan")
            pm_out = float(pm_row["rama/outlier_frac"]) if pm_row else float("nan")

            all_rows.append({
                "label": csv_label,
                "system_id": system_id,
                "cctbx_rama_favored": f"{cctbx['rama_favored']:.6f}",
                "cctbx_rama_allowed": f"{cctbx['rama_allowed']:.6f}",
                "cctbx_rama_outlier": f"{cctbx['rama_outlier']:.6f}",
                "cctbx_rama_n_total": cctbx["rama_n_total"],
                "cctbx_clashscore": f"{cctbx['clashscore']:.4f}" if not math.isnan(cctbx['clashscore']) else "nan",
                "cctbx_n_clashes": cctbx["n_clashes"],
                "pm_rama_favored": f"{pm_fav:.6f}" if not math.isnan(pm_fav) else "",
                "pm_rama_outlier": f"{pm_out:.6f}" if not math.isnan(pm_out) else "",
            })

            if (i + 1) % 20 == 0 or i + 1 == len(pdbs):
                elapsed = time.time() - t0
                print(f"  [{csv_label}] {i+1}/{len(pdbs)} done  ({elapsed:.1f}s elapsed)")

    # Write CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    elapsed = time.time() - t0
    print(f"\nDone! {len(all_rows)} rows written to {out_path}  ({elapsed:.1f}s total)")

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  Summary (CCTBX means per epoch)")
    print(f"{'='*80}")
    print(f"{'label':>10s}  {'n':>4s}  {'rama_fav':>10s}  {'rama_out':>10s}  {'clashscore':>12s}")
    print("-" * 56)

    from collections import defaultdict
    by_label = defaultdict(list)
    for row in all_rows:
        by_label[row["label"]].append(row)

    for lbl in dict.fromkeys(r["label"] for r in all_rows):
        rows = by_label[lbl]
        n = len(rows)
        fav_vals = [float(r["cctbx_rama_favored"]) for r in rows if r["cctbx_rama_favored"] != "nan"]
        out_vals = [float(r["cctbx_rama_outlier"]) for r in rows if r["cctbx_rama_outlier"] != "nan"]
        clash_vals = [float(r["cctbx_clashscore"]) for r in rows if r["cctbx_clashscore"] != "nan"]

        mean_fav = sum(fav_vals) / len(fav_vals) if fav_vals else float("nan")
        mean_out = sum(out_vals) / len(out_vals) if out_vals else float("nan")
        mean_clash = sum(clash_vals) / len(clash_vals) if clash_vals else float("nan")

        print(f"{lbl:>10s}  {n:4d}  {mean_fav:10.4f}  {mean_out:10.4f}  {mean_clash:12.2f}")


if __name__ == "__main__":
    main()
