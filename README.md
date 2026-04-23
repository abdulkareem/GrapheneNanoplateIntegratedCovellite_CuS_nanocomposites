# Graphene/CuS ASE+GPAW Colab Pipeline

## 2-line Colab launcher
```python
!git clone https://github.com/<YOUR_GITHUB_USER>/GrapheneNanoplateIntegratedCovellite_CuS_nanocomposites.git && cd GrapheneNanoplateIntegratedCovellite_CuS_nanocomposites
!pip -q install ase gpaw gpaw-data numpy scipy matplotlib && python scripts/run_from_github.py --output-dir /content/drive/MyDrive/gpaw_cus_graphene_project/outputs --adsorbate Pb2+
```

That is all you run in Colab; the rest is automated from this GitHub repo.

## Files
- `src/gpaw_cus_graphene_pipeline.py`: Reusable functions for structure generation, GPAW setup, relaxation, DOS/PDOS, band structure, charge/adsorption analysis, and export.
- `scripts/run_from_github.py`: End-to-end CLI runner used by the 2-line Colab launcher.
- `notebooks/colab_gpaw_cus_graphene_pipeline.ipynb`: Expanded 10-cell notebook version.

## Suggested upstream repositories to clone in Colab
1. GPAW main repository: `https://github.com/dijasila/GPAW`
2. ASE main repository: `https://github.com/rosswhitfield/ase`
