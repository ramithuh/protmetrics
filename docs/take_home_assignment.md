# Take-Home Assignment: Structural Validation of Unconditionally Generated Proteins

**Time estimate:** 4–6 hours (not counting environment setup or generation wallclock time)

---

## Background

RFdiffusion3 (RFd3) is an all-atom protein structure generator that can produce novel proteins from noise — a process called **unconditional generation**. Unlike earlier backbone-only generators, RFd3 outputs full atomic detail including side chains, making it possible to run comprehensive structural validation.

A key question after generation is: *are the structures physically plausible?* We assess this using **CCTBX** (Computational Crystallography Toolbox), the open-source library that implements MolProbity's validation algorithms. CCTBX's `mmtbx.validation` module provides Ramachandran analysis, rotamer validation, and steric clash detection — the same tools used by structural biologists to assess experimental crystal structures.

Your task is to generate unconditional protein structures with RFd3, validate them with CCTBX, and analyze the results.

---

## Task

### Part 1: Structure Generation

Generate **~100 unconditional all-atom structures** using RFdiffusion3, with lengths ranging from 100–300 residues.

RFd3 is distributed via the [RosettaCommons Foundry](https://github.com/RosettaCommons/foundry). Setup:

```bash
conda create -n foundry python=3.12
conda activate foundry
pip install torch --index-url https://download.pytorch.org/whl/cu130  # match your CUDA version
pip install "rc-foundry[rfd3]"
foundry install rfd3
```

Example generation command:

```bash
rfd3 design \
  out_dir=./rfd3_unconditional \
  inputs=null \
  +specification.length="100-300" \
  n_batches=13 \
  diffusion_batch_size=8
```

This produces 104 structures (13 batches x 8 per batch) as `.cif.gz` files.

**Notes:**
- Document any non-default inference parameters you used and why.
- RFd3 outputs compressed CIF files, not PDB. You'll need to handle this format in your validation pipeline.
- **We have provided 104 pre-generated structures** (lengths 100–300, default RFd3 parameters) in the accompanying `rfd3_unconditional/` directory. You may use these directly and skip generation. Running your own generation is optional bonus credit.

### Part 2: CCTBX Validation

For each generated structure, compute the following three metrics using CCTBX:

| Metric | CCTBX module | What it measures |
|--------|-------------|------------------|
| Ramachandran favored / outliers (%) | `mmtbx.validation.ramalyze` | Backbone phi/psi angle quality |
| Rotamer outliers (%) | `mmtbx.validation.rotalyze` | Side-chain conformation quality |
| Clashscore | `mmtbx.validation.clashscore` | Steric clashes per 1000 atoms (all-atom) |

These are the three components of the **MolProbity score** — the standard composite metric for protein structure quality.

**Setup hint:**
```bash
conda install -c conda-forge cctbx-base
```

`ramalyze` and `rotalyze` work out of the box. `clashscore` requires the `reduce` and `probe` binaries for hydrogen placement and contact analysis — recent `cctbx-base` builds include these, or they can be installed separately from the [Richardson Lab GitHub](https://github.com/rlabduke).

### Part 3: Analysis

Write a short report (1–2 pages, or a notebook) that includes:

1. **Summary statistics**: Distribution of each metric across your generated structures (mean, median, std, and histograms or violin plots).

2. **Length dependence**: Do any metrics correlate with chain length? Show scatter plots.

3. **Comparison to reference values**: How do your numbers compare to:
   - High-resolution experimental structures (e.g., PDB entries at <1.5 Å resolution)
   - Published metrics from other protein generators (La-Proteina is a good comparison point since it also reports full MolProbity metrics)

   You don't need to re-run baselines — literature values are fine.

4. **Interpretation**: In 2–3 paragraphs, discuss:
   - What do these metrics tell us about the quality of the generated structures?
   - What do these metrics *not* capture? What failure modes would they miss?
   - RFd3 is an all-atom generator. How does this change the validation picture compared to backbone-only generators like RFdiffusion v1, where side chains are packed separately by ProteinMPNN + Rosetta?

---

## Deliverables

1. **Code**: Scripts used for generation and validation. Should be runnable with clear instructions.
2. **Data**: A CSV file with one row per structure and columns for each metric (+ chain length and filename).
3. **Report**: The analysis described in Part 3 (Jupyter notebook, PDF, or Markdown).

---

## Evaluation Criteria

We are looking for:

- **Correctness**: Are the CCTBX metrics computed properly? Do the numbers make sense?
- **Code quality**: Is the code clean, readable, and reasonably documented?
- **Analysis depth**: Does the report show understanding of what the metrics mean and their limitations?
- **Scientific communication**: Are the plots clear? Is the writing concise and precise?

We are *not* looking for:
- Fancy visualizations or overproduced reports
- Novel research contributions
- Optimizing RFd3 hyperparameters

---

## Notes

- This is an open-book assignment. Use any resources you like.
- If you get stuck on environment setup, reach out — we'd rather evaluate your analysis skills than your ability to debug conda.
- If anything is unclear, ask. We won't penalize you for clarifying questions.
