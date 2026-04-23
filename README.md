# Graphene/CuS ASE+GPAW Colab Pipeline

## 2-line Colab launcher (auto-runs everything from GitHub)
```python
![ -d GrapheneNanoplateIntegratedCovellite_CuS_nanocomposites ] || git clone https://github.com/<YOUR_GITHUB_USER>/GrapheneNanoplateIntegratedCovellite_CuS_nanocomposites.git
!cd GrapheneNanoplateIntegratedCovellite_CuS_nanocomposites && git pull --ff-only || true; pip -q install ase gpaw gpaw-data numpy scipy matplotlib && python scripts/run_from_github.py --output-dir /content/results --profile quick --engine gpaw --adsorbate Pb2+
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

## Data/code sharing guidance
See `docs/data_code_sharing_recommendations.md` for what to publish publicly vs archive externally (e.g., large `.gpw` files).


**Colab tip:** each `!` line runs in a new shell. Keep `cd ... && python ...` in the same line.

If the repo already exists, the first line safely skips clone. To refresh, run `!cd GrapheneNanoplateIntegratedCovellite_CuS_nanocomposites && git pull`.

The runner now auto-selects a compatible graphene/CuS supercell to avoid lattice-mismatch failures and logs the chosen supercells in `lattice_mismatch.txt`.


## Colab speed defaults
- `--profile quick` now uses **LCAO mode** for faster relaxations on Colab CPU.
- `--profile publish` keeps PW accuracy, but uses an **LCAO pre-relax** before PW refinement.
- Vacuum is reduced for quick profile to shrink FFT workload, and the runner now enforces compact z-cells and logs thickness/vacuum in `lattice_mismatch.txt`.

## Vacuum sanity check (optional)
If you want to verify your final z-height quickly:
```python
import numpy as np
from ase.io import read
atoms = read('/content/results/composite_relaxed.xyz')
z = atoms.positions[:, 2]
thickness = z.max() - z.min()
cell_z = atoms.cell[2, 2]
print(f'thickness={thickness:.2f} A, cell_z={cell_z:.2f} A, total_vacuum={cell_z-thickness:.2f} A')
```
