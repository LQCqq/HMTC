#!/usr/bin/env python3
"""Relax CIFs with MatterSim; save CIFs and print all results to the job log."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


DEFAULT_MODEL = "MatterSim-v1.0.0-5M.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--fmax", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-structures", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.fmax) or args.fmax <= 0:
        parser.error("--fmax must be finite and positive")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.max_structures is not None and args.max_structures <= 0:
        parser.error("--max-structures must be positive")
    return args


def main() -> int:
    args = parse_args()
    print("HMTC MatterSim relaxation: version1", flush=True)

    import numpy as np
    import torch
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGSLineSearch
    from mattersim.forcefield import MatterSimCalculator
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.cif import CifWriter

    input_dir = args.cifs.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"CIF directory not found: {input_dir}")
    if (
        input_dir == output_dir
        or input_dir in output_dir.parents
        or output_dir in input_dir.parents
    ):
        raise ValueError("input and output directories must be separate, non-nested directories")
    all_paths = sorted(input_dir.rglob("*.cif"))
    paths = all_paths[args.start_index:]
    if args.max_structures is not None:
        paths = paths[:args.max_structures]
    if not paths:
        raise ValueError(f"no CIF files selected from {input_dir}")
    old_cifs = sorted(output_dir.glob("*.cif")) if output_dir.is_dir() else []
    if old_cifs and not args.overwrite:
        raise FileExistsError(f"{output_dir} already contains CIFs; choose a new directory or use --overwrite")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Request ngpus=1 in PBS, or use --device cpu.")

    print(f"Torch: {torch.__version__}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Device: {args.device}", flush=True)
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Input: {input_dir}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    print(f"Selected CIFs: {len(paths)} / available: {len(all_paths)}", flush=True)
    print(f"Cell and positions relaxation: fmax={args.fmax}, steps={args.steps}", flush=True)

    calculator = MatterSimCalculator(
        load_path=args.model,
        device=args.device,
        dtype=args.default_dtype,
        load_training_state=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in old_cifs:
            path.unlink()

    written = failed = converged_count = 0
    initial_energies: list[float] = []
    final_energies: list[float] = []
    energy_changes: list[float] = []
    final_forces: list[float] = []

    for local_index, cif_path in enumerate(paths):
        index = args.start_index + local_index
        temporary_path: Path | None = None
        try:
            structure = Structure.from_file(cif_path)
            atoms = AseAtomsAdaptor.get_atoms(structure)
            if len(atoms) == 0:
                raise ValueError("structure contains no atoms")
            if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
                raise ValueError("coordinates or lattice contain non-finite values")
            if atoms.get_volume() <= 1e-6:
                raise ValueError("cell volume must be positive")
            atoms.pbc = True
            atoms.calc = calculator
            initial_energy = float(atoms.get_potential_energy())
            if not math.isfinite(initial_energy):
                raise ValueError("initial potential energy is non-finite")

            optimizer = BFGSLineSearch(FrechetCellFilter(atoms), logfile=None)
            converged = bool(optimizer.run(fmax=args.fmax, steps=args.steps))
            final_energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces())
            stress = np.asarray(atoms.get_stress())
            if (
                not math.isfinite(final_energy)
                or not np.isfinite(forces).all()
                or not np.isfinite(stress).all()
                or not np.isfinite(atoms.positions).all()
                or not np.isfinite(atoms.cell.array).all()
                or atoms.get_volume() <= 1e-6
            ):
                raise ValueError("relaxation produced non-finite values or an invalid cell")
            max_force = float(np.linalg.norm(forces, axis=1).max())
            relaxed = AseAtomsAdaptor.get_structure(atoms)
            destination = output_dir / f"relaxed_{index:06d}_{cif_path.name}"
            temporary_path = destination.with_suffix(".cif.tmp")
            CifWriter(relaxed, symprec=None).write_file(str(temporary_path))
            temporary_path.replace(destination)

            n_atoms = len(atoms)
            initial_per_atom = initial_energy / n_atoms
            final_per_atom = final_energy / n_atoms
            delta_per_atom = (final_energy - initial_energy) / n_atoms
            written += 1
            converged_count += int(converged)
            initial_energies.append(initial_per_atom)
            final_energies.append(final_per_atom)
            energy_changes.append(delta_per_atom)
            final_forces.append(max_force)
            print(
                f"[{local_index + 1}/{len(paths)}] {cif_path.name} "
                f"N={n_atoms} converged={converged} steps={optimizer.get_number_of_steps()} "
                f"E0/N={initial_per_atom:.6f} E/N={final_per_atom:.6f} "
                f"deltaE/N={delta_per_atom:.6f} eV/atom "
                f"max_force={max_force:.6f} eV/A",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            print(f"[{local_index + 1}/{len(paths)}] FAILED {cif_path.name}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        if (local_index + 1) % 10 == 0 or local_index + 1 == len(paths):
            print(f"Progress: processed={local_index + 1}/{len(paths)} CIF_written={written} failed={failed} converged={converged_count}", flush=True)

    print("=" * 72, flush=True)
    print("Final MatterSim relaxation results", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Input CIFs selected: {len(paths)}", flush=True)
    print(f"CIFs successfully written: {written}", flush=True)
    print(f"Failed: {failed}", flush=True)
    print(f"Converged: {converged_count}", flush=True)
    print(f"Written but not converged: {written - converged_count}", flush=True)
    print(f"Convergence rate / selected: {converged_count / len(paths):.6f}", flush=True)
    if written:
        print(f"Convergence rate / written: {converged_count / written:.6f}", flush=True)
        print(f"Mean initial total energy / atom: {np.mean(initial_energies):.6f} eV/atom", flush=True)
        print(f"Mean final total energy / atom: {np.mean(final_energies):.6f} eV/atom", flush=True)
        print(f"Mean energy change / atom: {np.mean(energy_changes):.6f} eV/atom", flush=True)
        print(f"Mean final max atomic force: {np.mean(final_forces):.6f} eV/A", flush=True)
    print(f"CIF output directory: {output_dir}", flush=True)
    print("Energy values are total potential energies; formation energy and energy above hull are not calculated.", flush=True)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
