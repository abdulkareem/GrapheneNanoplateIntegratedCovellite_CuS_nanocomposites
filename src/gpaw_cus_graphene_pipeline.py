"""Research-grade ASE+GPAW workflow for graphene/CuS nanocomposites.

Designed for Google Colab CPU runtime without MPI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, bulk, graphene, make_supercell, molecule, surface
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import BFGS
from gpaw import GPAW, PW, FermiDirac
from gpaw.dos import DOSCalculator


def ensure_dir(path: Path | str) -> Path:
    """Create directory if needed and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def build_graphene_nanoplate(
    size: Tuple[int, int, int] = (4, 4, 1),
    vacuum: float = 15.0,
    a: float = 2.46,
) -> Atoms:
    """Build graphene nanoplate with configurable supercell and vacuum."""
    g = graphene(formula="C2", a=a, size=size, vacuum=vacuum)
    g.center(axis=2)
    return g


def build_covellite_bulk(a: float = 3.79, c: float = 16.34) -> Atoms:
    """Approximate covellite CuS bulk from hexagonal prototype.

    Notes
    -----
    Covellite has P63/mmc symmetry and multiple atoms per unit cell.
    This helper uses an ASE hexagonal template and manual basis coordinates
    suited for quick Colab-scale studies; refine with experimental CIF for
    production-quality publications.
    """
    # fractional coordinates for a compact CuS model (2 Cu, 2 S)
    cell = np.array(
        [
            [a, 0.0, 0.0],
            [-a / 2.0, np.sqrt(3) * a / 2.0, 0.0],
            [0.0, 0.0, c],
        ]
    )
    symbols = ["Cu", "Cu", "S", "S"]
    scaled_positions = [
        (1 / 3, 2 / 3, 0.25),
        (2 / 3, 1 / 3, 0.75),
        (1 / 3, 2 / 3, 0.62),
        (2 / 3, 1 / 3, 0.38),
    ]
    cus = Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell, pbc=True)
    return cus


def build_covellite_slab(
    layers: int = 4,
    vacuum: float = 15.0,
    supercell: Tuple[int, int, int] = (2, 2, 1),
) -> Atoms:
    """Build CuS (001) slab from bulk model."""
    bulk_cus = build_covellite_bulk()
    slab = surface(bulk_cus, (0, 0, 1), layers=layers, vacuum=vacuum)
    if supercell != (1, 1, 1):
        slab = slab.repeat(supercell)
    slab.center(axis=2, vacuum=vacuum)
    return slab


def _inplane_cell_lengths(atoms: Atoms) -> Tuple[float, float]:
    a_vec, b_vec = atoms.cell[0], atoms.cell[1]
    return np.linalg.norm(a_vec), np.linalg.norm(b_vec)


def match_inplane_lattice(
    substrate: Atoms,
    overlayer: Atoms,
    max_strain: float = 0.05,
) -> Tuple[Atoms, Dict[str, float]]:
    """Scale overlayer in-plane lattice to substrate with strain checks."""
    sub_a, sub_b = _inplane_cell_lengths(substrate)
    ov_a, ov_b = _inplane_cell_lengths(overlayer)

    sx, sy = sub_a / ov_a, sub_b / ov_b
    strain_x, strain_y = abs(1.0 - sx), abs(1.0 - sy)
    if strain_x > max_strain or strain_y > max_strain:
        raise ValueError(
            f"Lattice mismatch too large: strain_x={strain_x:.3f}, strain_y={strain_y:.3f}. "
            "Adjust supercells before matching."
        )

    scaled = overlayer.copy()
    cell = scaled.cell.array.copy()
    cell[0] *= sx
    cell[1] *= sy
    scaled.set_cell(cell, scale_atoms=True)
    return scaled, {"strain_x": strain_x, "strain_y": strain_y}


def create_graphene_cus_composite(
    graphene_atoms: Atoms,
    cus_slab: Atoms,
    spacing: float = 2.5,
    max_strain: float = 0.05,
) -> Tuple[Atoms, Dict[str, float]]:
    """Place CuS slab on graphene with target interlayer spacing."""
    matched_cus, mismatch = match_inplane_lattice(graphene_atoms, cus_slab, max_strain=max_strain)

    g = graphene_atoms.copy()
    c = matched_cus.copy()

    g_top = g.positions[:, 2].max()
    c_bottom = c.positions[:, 2].min()
    c.positions[:, 2] += (g_top - c_bottom) + spacing

    comp = g + c
    comp.set_cell(g.cell, scale_atoms=False)
    comp.pbc = (True, True, True)
    comp.center(axis=2, vacuum=15.0)
    return comp, mismatch


def add_bottom_constraints(atoms: Atoms, thickness: float = 2.0) -> None:
    """Fix bottom slab atoms to improve relaxation stability."""
    z = atoms.positions[:, 2]
    zmin = z.min()
    mask = z < (zmin + thickness)
    atoms.set_constraint(FixAtoms(mask=mask))


def make_gpaw_calculator(
    kpts: Tuple[int, int, int] = (3, 3, 1),
    ecut: float = 450.0,
    xc: str = "PBE",
    txt: str = "gpaw.log",
    occupations_width: float = 0.1,
    mode_parallel: bool = False,
) -> GPAW:
    """Construct CPU-friendly GPAW calculator for Colab."""
    calc = GPAW(
        mode=PW(ecut),
        xc=xc,
        kpts=kpts,
        occupations=FermiDirac(occupations_width),
        convergence={"energy": 1e-5},
        txt=txt,
        parallel={"domain": 1, "band": 1} if mode_parallel else {},
    )
    return calc


def relax_structure(
    atoms: Atoms,
    calc: GPAW,
    traj_path: str,
    fmax: float = 0.02,
    steps: int = 200,
) -> Tuple[Atoms, float]:
    """Run BFGS geometry optimization and return relaxed atoms + energy."""
    atoms = atoms.copy()
    atoms.calc = calc
    opt = BFGS(atoms, trajectory=traj_path, logfile=traj_path.replace(".traj", ".opt.log"))
    opt.run(fmax=fmax, steps=steps)
    energy = atoms.get_potential_energy()
    return atoms, energy


def single_point_energy(atoms: Atoms, calc: GPAW, gpw_out: Optional[str] = None) -> float:
    """Compute single-point total energy and optionally save restart file."""
    atoms = atoms.copy()
    atoms.calc = calc
    e = atoms.get_potential_energy()
    if gpw_out is not None:
        calc.write(gpw_out, mode="all")
    return e


def compute_binding_energy(e_comp: float, e_graphene: float, e_cus: float) -> float:
    return e_comp - (e_graphene + e_cus)


def compute_dos(calc: GPAW, npts: int = 1200, width: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    dos_calc = DOSCalculator.from_calculator(calc)
    energies = dos_calc.get_energies(npoints=npts)
    dos = dos_calc.raw_dos(energies, width=width)
    return energies, dos


def compute_pdos(
    calc: GPAW,
    atom_indices: Sequence[int],
    angular: str = "d",
    npts: int = 1200,
    width: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Projected DOS using LCAO projector channels."""
    dos_calc = DOSCalculator.from_calculator(calc)
    energies = dos_calc.get_energies(npoints=npts)
    pdos = np.zeros_like(energies)
    for idx in atom_indices:
        pdos += dos_calc.raw_pdos(energies, a=idx, l=angular, width=width)
    return energies, pdos


def compute_band_structure(calc: GPAW, path: str = "GMKG", npoints: int = 80):
    bs = calc.band_structure(path=path, npoints=npoints)
    return bs


def density_difference(
    composite_calc: GPAW,
    graphene_calc: GPAW,
    cus_calc: GPAW,
) -> np.ndarray:
    """Electron density difference grid on composite cell."""
    rho_comp = composite_calc.get_all_electron_density(gridrefinement=2)
    rho_g = graphene_calc.get_all_electron_density(gridrefinement=2)
    rho_c = cus_calc.get_all_electron_density(gridrefinement=2)
    # NOTE: requires same grid dimensions; for rigorous work, interpolate onto common grid.
    return rho_comp - (rho_g + rho_c)


def add_adsorbate_to_composite(
    composite: Atoms,
    adsorbate: str = "H2O",
    height: float = 2.3,
    position: Optional[Tuple[float, float]] = None,
) -> Atoms:
    """Add adsorbate molecule/ion marker to top surface."""
    atoms = composite.copy()
    if adsorbate in {"Pb2+", "Cd2+"}:
        species = "Pb" if adsorbate.startswith("Pb") else "Cd"
        ads = Atoms(species)
    else:
        ads = molecule(adsorbate)

    if position is None:
        x = atoms.cell[0, 0] * 0.5
        y = atoms.cell[1, 1] * 0.5
        position = (x, y)

    add_adsorbate(atoms, ads, height=height, position=position)
    atoms.center(axis=2, vacuum=15.0)
    return atoms


def compute_adsorption_energy(e_total: float, e_surface: float, e_adsorbate: float) -> float:
    return e_total - (e_surface + e_adsorbate)


def plot_xy(
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    out_png: str,
    y2: Optional[np.ndarray] = None,
    label1: str = "DOS",
    label2: str = "PDOS",
) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label=label1, lw=1.8)
    if y2 is not None:
        plt.plot(x, y2, label=label2, lw=1.5)
    plt.axvline(0.0, color="k", lw=0.8, ls="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if y2 is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def save_structure_images(atoms: Atoms, out_png: str) -> None:
    """Save a simple structure snapshot using ASE's matplotlib backend."""
    from ase.visualize.plot import plot_atoms

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_atoms(atoms, ax, rotation=("90x,0y,0z"), radii=0.5)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def export_energy_report(energies: Dict[str, float], out_txt: str) -> None:
    lines = [f"{k}: {v:.8f} eV" for k, v in energies.items()]
    Path(out_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_artifacts(
    out_dir: Path | str,
    structures: Dict[str, Atoms],
    energies: Dict[str, float],
) -> None:
    out = ensure_dir(out_dir)
    for name, atoms in structures.items():
        write(out / f"{name}.xyz", atoms)
        write(out / f"{name}.traj", atoms)
    export_energy_report(energies, out / "energies.txt")
