#!/usr/bin/env python3
"""MatterSim energies + MatterGen reference-hull metrics; CIFs and log only."""

from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cifs", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", default="MatterSim-v1.0.0-5M.pth")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--fmax", type=float, default=0.02)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--max-structures", type=int)
    p.add_argument("--energy-correction-scheme", choices=("MP2020", "TRI2024"), default="MP2020")
    p.add_argument("--reference-path", type=Path, help="Optional compatible reference LMDB .gz")
    p.add_argument("--stability-threshold", type=float, default=0.1, help="eV/atom above reference hull")
    p.add_argument("--no-relax", action="store_true", help="Evaluate MatterSim energies at input geometries")
    p.add_argument("--overwrite", action="store_true", help="Replace existing output CIFs")
    p.add_argument("--check-imports", action="store_true", help="Check imports and exit without downloading models/reference")
    a = p.parse_args()
    if not math.isfinite(a.fmax) or a.fmax <= 0 or a.steps <= 0:
        p.error("--fmax and --steps must be positive")
    if not math.isfinite(a.stability_threshold) or a.stability_threshold < 0:
        p.error("--stability-threshold must be finite and non-negative")
    if a.max_structures is not None and a.max_structures <= 0:
        p.error("--max-structures must be positive")
    return a


def main() -> int:
    args = parse_args()
    vendor = Path(__file__).resolve().parent / "mattergen_eval_vendor"
    if not (vendor / "mattergen/evaluation/metrics/evaluator.py").is_file():
        raise FileNotFoundError("Copy scripts/mattergen_eval_vendor along with this script.")
    sys.path.insert(0, str(vendor))

    import numpy as np
    import torch
    from ase.filters import FrechetCellFilter
    from ase.optimize import BFGSLineSearch
    from mattersim.forcefield import MatterSimCalculator
    from pymatgen.core import Structure
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.cif import CifWriter
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from mattergen.evaluation.metrics.evaluator import MetricsEvaluator
    from mattergen.evaluation.metrics.energy import (
        AvgEnergyAboveHullPerAtom, AvgRMSDFromRelaxation,
        EnergyMetricsCapability, FracNovelUniqueStableStructures, FracStableStructures,
    )
    from mattergen.evaluation.metrics.structure import (
        AvgCompValidity, AvgStructureValidity, FracNovelStructures, FracUniqueStructures,
    )
    from mattergen.evaluation.reference.correction_schemes import TRI110Compatibility2024
    from mattergen.evaluation.reference.presets import ReferenceMP2020Correction, ReferenceTRI2024Correction
    from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
    from mattergen.evaluation.utils.metrics_structure_summary import MetricsStructureSummary
    from mattergen.evaluation.utils.structure_matcher import DefaultDisorderedStructureMatcher
    from mattergen.evaluation.utils.vasprunlike import IdentityCorrectionScheme, VasprunLike

    print("HMTC MatterSim benchmark: version2", flush=True)
    print(f"Torch: {torch.__version__}", flush=True)
    if args.check_imports:
        print("MatterSim and MatterGen evaluation imports OK", flush=True)
        return 0

    input_dir, output_dir = args.cifs.resolve(), args.output_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"CIF directory not found: {input_dir}")
    if input_dir == output_dir or input_dir in output_dir.parents or output_dir in input_dir.parents:
        raise ValueError("Input/output directories must be separate and non-nested.")
    paths = sorted(input_dir.rglob("*.cif"))
    if args.max_structures is not None:
        paths = paths[:args.max_structures]
    if not paths:
        raise ValueError(f"No CIF files found: {input_dir}")
    old_cifs = list(output_dir.glob("*.cif")) if output_dir.exists() else []
    if old_cifs and not args.overwrite:
        raise FileExistsError("Output already contains CIFs; choose a new directory or use --overwrite.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: submit a GPU PBS job or use --device cpu.")

    print(f"Input: {input_dir}; selected CIFs: {len(paths)}", flush=True)
    print(f"Model: {args.model}; device: {args.device}; float32", flush=True)
    print(f"Correction: {args.energy_correction_scheme}; stable threshold: {args.stability_threshold} eV/atom", flush=True)
    print("Loading reference phase diagram data (first use downloads the reference)...", flush=True)
    correction = MaterialsProject2020Compatibility() if args.energy_correction_scheme == "MP2020" else TRI110Compatibility2024()
    if args.reference_path:
        reference = LMDBGZSerializer().deserialize(args.reference_path)
    else:
        reference = ReferenceMP2020Correction() if args.energy_correction_scheme == "MP2020" else ReferenceTRI2024Correction()
    expected_reference_name = {"MP2020": "MP2020correction", "TRI2024": "TRI2024correction"}[args.energy_correction_scheme]
    if reference.name in {"MP2020correction", "TRI2024correction"} and reference.name != expected_reference_name:
        raise ValueError(f"Reference {reference.name} is incompatible with correction {args.energy_correction_scheme}.")
    print(f"Reference: {reference.name}; entries: {len(reference)}", flush=True)

    calculator = MatterSimCalculator(load_path=args.model, device=args.device, dtype="float32", load_training_state=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in old_cifs:
            path.unlink()
    records = []
    for index, path in enumerate(paths):
        temp = None
        try:
            original = Structure.from_file(path)
            atoms = AseAtomsAdaptor.get_atoms(original)
            atoms.pbc = True
            atoms.calc = calculator
            if not len(atoms) or not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all() or atoms.get_volume() <= 1e-6:
                raise ValueError("Empty structure or invalid coordinates/lattice")
            initial_energy = float(atoms.get_potential_energy())
            if not math.isfinite(initial_energy):
                raise ValueError("Non-finite initial energy")
            converged = None
            if not args.no_relax:
                optimizer = BFGSLineSearch(FrechetCellFilter(atoms), logfile=None)
                converged = bool(optimizer.run(fmax=args.fmax, steps=args.steps))
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces())
            if not math.isfinite(energy) or not np.isfinite(forces).all() or not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all() or atoms.get_volume() <= 1e-6:
                raise ValueError("Non-finite relaxation result or invalid cell")
            structure = AseAtomsAdaptor.get_structure(atoms)
            # New pymatgen versions preserve the live calculator. Metrics copy
            # entries; detach it to avoid serializing/reloading the ML model.
            if hasattr(structure, "calc"):
                structure.calc = None
            destination = output_dir / f"mattersim_{index:06d}_{path.name}"
            temp = destination.with_suffix(".cif.tmp")
            CifWriter(structure, symprec=None).write_file(str(temp))
            temp.replace(destination)
            records.append((path, original, structure, energy, converged))
            print(f"[{index + 1}/{len(paths)}] {path.name} N={len(atoms)} converged={converged} E/N={energy / len(atoms):.6f} deltaE/N={(energy - initial_energy) / len(atoms):.6f} eV/atom Fmax={np.linalg.norm(forces, axis=1).max():.6f} eV/A", flush=True)
        except Exception as exc:
            print(f"[{index + 1}/{len(paths)}] FAILED {path.name}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
    print(f"CIFs written: {len(records)}; read/energy/relaxation failures: {len(paths) - len(records)}", flush=True)
    if not records:
        return 1

    summaries, evaluable_records = [], []
    for record in records:
        path, original, structure, energy, converged = record
        try:
            raw_entry = VasprunLike(structure, energy).get_computed_entry(energy_correction_scheme=IdentityCorrectionScheme())
            entry = correction.process_entry(raw_entry)
            if entry is None or not math.isfinite(entry.energy):
                raise ValueError("Energy correction rejected this structure")
            summary = MetricsStructureSummary(entry=entry, original_structure=None if args.no_relax else original)
            missing = EnergyMetricsCapability.check_missing_reference_terminal_systems([summary], reference)
            if missing.has_missing_data:
                raise ValueError(f"Missing reference terminal energies: {missing.missing_terminals + missing.missing_energy}")
            summaries.append(summary)
            evaluable_records.append(record)
        except Exception as exc:
            print(f"HULL_NOT_EVALUATED {path.name}: {type(exc).__name__}: {exc}", flush=True)
    if not summaries:
        print("No structures have compatible energies/reference data; no stability rate can be computed.", flush=True)
        return 1

    evaluator = MetricsEvaluator.from_structure_summaries(
        summaries, reference=reference, stability_threshold=args.stability_threshold,
        structure_matcher=DefaultDisorderedStructureMatcher(), n_failed_jobs=len(paths) - len(summaries),
    )
    requested = [AvgEnergyAboveHullPerAtom, FracStableStructures, FracUniqueStructures,
                 FracNovelStructures, FracNovelUniqueStableStructures, AvgStructureValidity, AvgCompValidity]
    if not args.no_relax:
        requested.append(AvgRMSDFromRelaxation)
    metrics = evaluator.compute_metrics(requested, save_as=None, pretty_print=False)
    hull = evaluator.energy_capability.energy_above_hull
    for record, ehull in zip(evaluable_records, hull):
        print(f"HULL {record[0].relative_to(input_dir)} E_hull={ehull:.6f} eV/atom stable={bool(ehull <= args.stability_threshold)} converged={record[4]}", flush=True)

    print("=" * 72, flush=True)
    print("Final MatterSim benchmark results", flush=True)
    print(f"Reference: {reference.name}; correction: {args.energy_correction_scheme}", flush=True)
    print(f"Input CIFs: {len(paths)}; written: {len(records)}; hull-evaluable: {len(summaries)}", flush=True)
    print(f"Hull coverage: {len(summaries) / len(paths):.6f}", flush=True)
    print("Stable/SUN denominator: all selected input CIFs; unavailable results do not count as successes.", flush=True)
    print("Official uniqueness/novelty/validity denominator: hull-evaluable structures.", flush=True)
    for key, value in metrics.items():
        print(f"{key:<40} {value:.6f}", flush=True)
    print(f"unique_count: {np.count_nonzero(evaluator.is_unique)}; novel_count: {np.count_nonzero(evaluator.is_novel)}", flush=True)
    print(f"median_energy_above_hull_per_atom         {np.median(hull):.6f}", flush=True)
    print(f"p90_energy_above_hull_per_atom            {np.percentile(hull, 90):.6f}", flush=True)
    for threshold in sorted(set([0.0, 0.05, 0.1, args.stability_threshold])):
        count = int(np.count_nonzero(hull <= threshold))
        print(f"stable_at_{threshold:g}_eV_per_atom: count={count} / all_inputs={count / len(paths):.6f} / hull_evaluable={count / len(summaries):.6f}", flush=True)
    if not args.no_relax:
        n_converged = sum(r[4] is True for r in records)
        stable_converged = sum(r[4] is True and e <= args.stability_threshold for r, e in zip(evaluable_records, hull))
        print(f"Converged: {n_converged}; written but unconverged: {len(records) - n_converged}", flush=True)
        print(f"Convergence rate / all inputs: {n_converged / len(paths):.6f}", flush=True)
        print(f"Stable AND converged / all inputs: {stable_converged / len(paths):.6f}", flush=True)
    count_hist = Counter(len(r[2]) for r in records)
    sg_hist = Counter()
    for record in records:
        try:
            sg_hist[SpacegroupAnalyzer(record[2], symprec=0.1, angle_tolerance=5).get_space_group_number()] += 1
        except Exception:
            sg_hist["undetermined"] += 1
    print(f"Atom count histogram: {dict(sorted(count_hist.items()))}", flush=True)
    print(f"Space group histogram (symprec=0.1 A, angle_tolerance=5 deg): {dict(sg_hist)}", flush=True)
    print(f"CIF output: {output_dir}", flush=True)
    print("RMSD, when enabled, is measured from this run's input CIFs; filtered-input scores describe that subset.", flush=True)
    print("Stability is a MatterSim/reference-hull estimate at the printed threshold; this is not a DFT validation.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
