"""Interfaces and registration for an open set of hierarchy modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from torch import Tensor, nn

from hmtc.config import TreeSpec


@dataclass
class TreePass:
    name: str
    root: Tensor
    levels: list[Tensor]
    children: list[Tensor]
    assignments: list[Tensor]
    masses: list[Tensor]
    leaf_mask: Tensor


@dataclass
class TreeResult:
    name: str
    root: Tensor
    leaves: Tensor
    assignments: list[Tensor]
    masses: list[Tensor]


class TreePlugin(nn.Module, ABC):
    """A hierarchy that reads one entity bank and exports a root and leaf view."""

    spec: TreeSpec

    @abstractmethod
    def upward(self, leaves: Tensor, leaf_mask: Tensor, context: Tensor) -> TreePass:
        raise NotImplementedError

    @abstractmethod
    def downward(self, state: TreePass, bridged_root: Tensor, context: Tensor) -> TreeResult:
        raise NotImplementedError


TreeBuilder = Callable[[TreeSpec, int], TreePlugin]
TREE_REGISTRY: dict[str, TreeBuilder] = {}


def register_tree(kind: str) -> Callable[[TreeBuilder], TreeBuilder]:
    def decorator(builder: TreeBuilder) -> TreeBuilder:
        if kind in TREE_REGISTRY:
            raise ValueError(f"tree kind {kind!r} already registered")
        TREE_REGISTRY[kind] = builder
        return builder

    return decorator


def build_tree_plugin(spec: TreeSpec, hidden_dim: int) -> TreePlugin:
    try:
        builder = TREE_REGISTRY[spec.kind]
    except KeyError as exc:
        available = ", ".join(sorted(TREE_REGISTRY))
        raise KeyError(f"unknown tree kind {spec.kind!r}; registered kinds: {available}") from exc
    return builder(spec, hidden_dim)
