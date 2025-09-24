# AGEB: Architecture Generation and Evaluation Benchmark

This repository packages the code and assets used in the paper "LLMs Are Not Yet Reliable for Generating Architectural Design Images: A Comprehensive Evaluation Framework and Benchmark Suite (AGEB)".

## What is AGEB?
AGEB is a domain-grounded, automated benchmark for evaluating architectural text-to-image systems along three pillars:
- Semantic fidelity (dual-expert chain-of-thought evaluation)
- Functional adequacy (graph-theoretic circulation analysis)
- Geometric consistency (perspective analysis)

## Repository Layout
- `src/` — Core scripts for data generation, figure creation, and evaluation
  - `orchestrator.py` — Entry point to run end-to-end experiments
  - `circulation_analysis.py` — Functional/circulation evaluation
  - `perspective_analysis.py` — Geometric/perspective evaluation
  - `create_*`, `fix_*`, `generate_*` — Utilities for figures and processes
- `src/analysis/` — Aggregation and analysis
  - `comprehensive_analysis.py` — Aggregates results and produces summary tables/figures
- `configs/` — (Optional) Place YAML/JSON configs here if you extend runs
- `docs/` — Extra documentation (e.g., analysis notes)
- `requirements.txt` — Python dependencies

## Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Minimal Reproduction
1) Run the core evaluators (modify inputs/paths inside scripts as needed):
```bash
python src/orchestrator.py
```
2) Run geometric and functional submodules directly if desired:
```bash
python src/perspective_analysis.py
python src/circulation_analysis.py
```
3) Aggregate results and build tables/figures:
```bash
python src/analysis/comprehensive_analysis.py
```

## Expected Inputs/Outputs
- Inputs: prompts, model outputs (images/metadata), and config paths referenced inside `orchestrator.py`.
- Outputs: JSON/CSV summaries, figures (e.g., perspective pipeline, performance comparisons).

## Notes
- Some scripts expect pre-generated model outputs (see your `*-results/` folders in the project root). Adjust paths in scripts accordingly.
- If you publish, pin exact package versions from `requirements.txt` and capture your commit hashes.

## Citation
If you use AGEB in academic work, please cite the paper.
