"""Interface for a structure-only energy evaluator.

An evaluator must not receive the generated energy factor as an input feature.
It may read elements, coordinates, lattice, and existence only.  Its parameters
can be frozen while retaining gradients with respect to geometry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn

from hmtc.types import StructureState


class StructureEnergyEvaluator(nn.Module, ABC):
    @abstractmethod
    def forward(self, structure: StructureState) -> Tensor:
        """Return standardized formation energy with shape ``[batch, 1]``."""

        raise NotImplementedError

    def freeze_parameters(self) -> StructureEnergyEvaluator:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self
