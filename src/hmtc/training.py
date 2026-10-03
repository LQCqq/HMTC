"""A small trainer component suitable for integration into a larger workflow."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from hmtc.config import DiffusionConfig, LossConfig
from hmtc.diagnostics import routing_diagnostics
from hmtc.diffusion.joint import JointDiffusion
from hmtc.losses import hmtf_loss
from hmtc.models.energy import StructureEnergyEvaluator
from hmtc.models.hmtf import HMTFModel
from hmtc.types import CrystalState


@dataclass
class TrainStepResult:
    loss: Tensor
    terms: dict[str, Tensor]
    time: Tensor
    diagnostics: dict[str, Tensor]


class HMTFTrainer:
    def __init__(
        self,
        model: HMTFModel,
        optimizer: torch.optim.Optimizer,
        diffusion_config: DiffusionConfig,
        loss_config: LossConfig,
        energy_evaluator: StructureEnergyEvaluator | None = None,
        gradient_clip: float = 1.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_config = loss_config
        self.diffusion = JointDiffusion(diffusion_config, model.config.num_elements)
        self.energy_evaluator = energy_evaluator
        self.gradient_clip = gradient_clip

    def train_step(
        self,
        clean: CrystalState,
        active_trees: set[str] | None = None,
        collect_diagnostics: bool = False,
    ) -> TrainStepResult:
        self.model.train()
        time = torch.rand(clean.batch_size, device=clean.energy.device)
        noisy, noise_record = self.diffusion.q_sample(clean, time)
        output = self.model(noisy, time, active_trees=active_trees)
        total, terms = hmtf_loss(
            output,
            clean,
            self.loss_config,
            energy_evaluator=self.energy_evaluator,
            element_mask=noise_record.element_mask,
        )
        diagnostics = {}
        if collect_diagnostics:
            active = clean.resolved_site_mask() & (clean.existence > 0.5)
            diagnostics = routing_diagnostics(output, active)
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        return TrainStepResult(
            loss=total.detach(),
            terms={key: value.detach() for key, value in terms.items()},
            time=time,
            diagnostics={key: value.detach() for key, value in diagnostics.items()},
        )
