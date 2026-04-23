# Graphene/CuS ASE+GPAW Colab Pipeline

## 2-line Colab launcher (auto-runs everything from GitHub)
```python
!git clone https://github.com/<YOUR_GITHUB_USER>/GrapheneNanoplateIntegratedCovellite_CuS_nanocomposites.git && cd GrapheneNanoplateIntegratedCovellite_CuS_nanocomposites
!pip -q install ase gpaw gpaw-data numpy scipy matplotlib && python scripts/run_from_github.py --output-dir /content/results --profile publish --engine gpaw --adsorbate Pb2+
```

## Optional Quantum ESPRESSO backend
Use the same command with `--engine qe` (requires installed `pw.x` and pseudopotentials).

## Accelerator support
- CPU: supported.
- GPU (T4/L4/A100/H100/G4): detected and logged in `run_metadata.txt`; GPAW pip wheels typically run CPU mode.
- TPU (v5e/v6e): metadata detection only (DFT engines here are not TPU-native).

## What the pipeline exports
- `energies.txt`, `paper_summary_metrics.csv`
- `convergence_scan.csv`, `convergence_scan.png` (publish profile)
- structures (`.xyz`, `.traj`)
- electronic plots (`dos_total.png`, `pdos_cu_d.png`, `band_structure.png`)
- restart files (`.gpw` for GPAW)

## Suggested upstream repositories to clone/reference
1. GPAW main repository: `https://github.com/dijasila/GPAW`
2. ASE main repository: `https://github.com/rosswhitfield/ase`

## Publication planning
See `docs/publishing_strategy.md` for novelty pivot recommendations and literature overlap guidance.


## Manuscript draft
A complete paper draft template is provided at `docs/manuscript_draft_graphene_cus.md` for direct journal writing with placeholders for computed results.
