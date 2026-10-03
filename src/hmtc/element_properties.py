"""Fixed element properties used by differentiable physical penalties."""

from __future__ import annotations

# Covalent radii in Angstrom for atomic numbers 1..100. Values through Cm
# (Z=96) are the Cordero et al. single-bond covalent radii used by pymatgen
# and ASE: Dalton Trans. 2008, 2832-2838, DOI 10.1039/B801115J.
# Cordero values are unavailable for Bk..Fm (Z=97..100), for which 1.50 A is
# used as an explicit conservative fallback.
COVALENT_RADII_ANGSTROM: tuple[float, ...] = (
    0.31, 0.28, 1.28, 0.96, 0.84, 0.73, 0.71, 0.66, 0.57, 0.58,
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06, 2.03, 1.76,
    1.70, 1.60, 1.53, 1.39, 1.50, 1.42, 1.38, 1.24, 1.32, 1.22,
    1.22, 1.20, 1.19, 1.20, 1.20, 1.16, 2.20, 1.95, 1.90, 1.75,
    1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44, 1.42, 1.39,
    1.39, 1.38, 1.39, 1.40, 2.44, 2.15, 2.07, 2.04, 2.03, 2.01,
    1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87,
    1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32,
    1.45, 1.46, 1.48, 1.40, 1.50, 1.50, 2.60, 2.21, 2.15, 2.06,
    2.00, 1.96, 1.90, 1.87, 1.80, 1.69, 1.50, 1.50, 1.50, 1.50,
)

