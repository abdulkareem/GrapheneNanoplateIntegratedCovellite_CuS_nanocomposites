#!/usr/bin/env python3
"""One-command runner for the graphene/CuS ASE+GPAW pipeline.

Intended for Google Colab after cloning this repository.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ase import Atoms
from gpaw import GPAW

from src.gpaw_cus_graphene_pipeline import (
    add_adsorbate_to_composite,
    build_covellite_slab,
    build_graphene_nanoplate,
    compute_adsorption_energy,
    compute_band_structure,
    compute_binding_energy,
    compute_dos,
    compute_pdos,
    make_gpaw_calculator,
    plot_xy,
    relax_structure,
    save_artifacts,
    save_structure_images,
    single_point_energy,
)


def run(output_dir: Path, graphene_size=(4, 4, 1), spacing=2.5, adsorbate="Pb2+") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    graphene = build_graphene_nanoplate(size=graphene_size, vacuum=15.0)
    cus_slab = build_covellite_slab(layers=4, vacuum=15.0, supercell=(2, 2, 1))

    from src.gpaw_cus_graphene_pipeline import create_graphene_cus_composite

    composite, mismatch = create_graphene_cus_composite(graphene, cus_slab, spacing=spacing, max_strain=0.05)
    print("Lattice mismatch:", mismatch)

    calc_relax = make_gpaw_calculator(kpts=(3, 3, 1), ecut=450, xc="PBE", txt=str(output_dir / "relax.log"))
    composite_relaxed, e_comp = relax_structure(
        composite,
        calc_relax,
        traj_path=str(output_dir / "composite_relax.traj"),
        fmax=0.02,
        steps=200,
    )
    calc_relax.write(str(output_dir / "composite_relaxed.gpw"), mode="all")

    calc_g = make_gpaw_calculator(kpts=(3, 3, 1), ecut=450, xc="PBE", txt=str(output_dir / "graphene_sp.log"))
    calc_c = make_gpaw_calculator(kpts=(3, 3, 1), ecut=450, xc="PBE", txt=str(output_dir / "cus_sp.log"))
    e_graphene = single_point_energy(graphene, calc_g, gpw_out=str(output_dir / "graphene.gpw"))
    e_cus = single_point_energy(cus_slab, calc_c, gpw_out=str(output_dir / "cus.gpw"))
    e_bind = compute_binding_energy(e_comp, e_graphene, e_cus)

    calc_loaded = GPAW(str(output_dir / "composite_relaxed.gpw"), txt=None)
    energies, dos = compute_dos(calc_loaded, npts=1000, width=0.15)
    cu_indices = [i for i, a in enumerate(composite_relaxed) if a.symbol == "Cu"]
    energies_p, pdos_cu_d = compute_pdos(calc_loaded, atom_indices=cu_indices, angular="d", npts=1000, width=0.15)
    plot_xy(energies, dos, "Energy - E_F (eV)", "DOS (states/eV)", "Total DOS", str(output_dir / "dos_total.png"))
    plot_xy(
        energies_p,
        pdos_cu_d,
        "Energy - E_F (eV)",
        "PDOS (states/eV)",
        "Cu-d PDOS",
        str(output_dir / "pdos_cu_d.png"),
    )
    bs = compute_band_structure(calc_loaded, path="GMKG", npoints=80)
    bs.plot(filename=str(output_dir / "band_structure.png"), show=False, emin=-6, emax=4)

    ads_system = add_adsorbate_to_composite(composite_relaxed, adsorbate=adsorbate, height=2.4)
    calc_ads = make_gpaw_calculator(kpts=(3, 3, 1), ecut=450, xc="PBE", txt=str(output_dir / "ads_relax.log"))
    ads_relaxed, e_ads_total = relax_structure(
        ads_system,
        calc_ads,
        traj_path=str(output_dir / "adsorption_relax.traj"),
        fmax=0.02,
        steps=160,
    )

    if adsorbate == "Pb2+":
        ads_atom = Atoms("Pb", positions=[(0, 0, 0)], cell=[15, 15, 15], pbc=False)
    elif adsorbate == "Cd2+":
        ads_atom = Atoms("Cd", positions=[(0, 0, 0)], cell=[15, 15, 15], pbc=False)
    else:
        ads_atom = Atoms(adsorbate, positions=[(0, 0, 0)], cell=[15, 15, 15], pbc=False)

    calc_adsorbate = make_gpaw_calculator(kpts=(1, 1, 1), ecut=450, xc="PBE", txt=str(output_dir / "adsorbate_sp.log"))
    e_adsorbate = single_point_energy(ads_atom, calc_adsorbate, gpw_out=str(output_dir / "adsorbate.gpw"))
    e_ads = compute_adsorption_energy(e_ads_total, e_comp, e_adsorbate)

    save_structure_images(graphene, str(output_dir / "graphene_init.png"))
    save_structure_images(cus_slab, str(output_dir / "cus_slab_init.png"))
    save_structure_images(composite, str(output_dir / "composite_init.png"))
    save_structure_images(composite_relaxed, str(output_dir / "composite_relaxed.png"))
    save_structure_images(ads_relaxed, str(output_dir / "ads_relaxed.png"))

    energies_report = {
        "E_composite": e_comp,
        "E_graphene": e_graphene,
        "E_cus": e_cus,
        "E_binding": e_bind,
        "E_ads_total": e_ads_total,
        "E_adsorbate": e_adsorbate,
        "E_adsorption": e_ads,
    }
    structures = {
        "graphene": graphene,
        "cus_slab": cus_slab,
        "composite_init": composite,
        "composite_relaxed": composite_relaxed,
        "ads_relaxed": ads_relaxed,
    }
    save_artifacts(output_dir, structures, energies_report)
    print("Done. Results in:", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run graphene/CuS Colab pipeline from cloned repo")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adsorbate", type=str, default="Pb2+", choices=["Pb2+", "Cd2+"])
    parser.add_argument("--graphene-n", type=int, default=4, help="NxN graphene supercell (N x N x 1)")
    parser.add_argument("--spacing", type=float, default=2.5, help="Initial CuS-graphene gap (Å)")
    args = parser.parse_args()

    run(
        output_dir=args.output_dir,
        graphene_size=(args.graphene_n, args.graphene_n, 1),
        spacing=args.spacing,
        adsorbate=args.adsorbate,
    )


if __name__ == "__main__":
    main()
