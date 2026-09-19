"""Core hierarchy registrations."""

from hmtc.config import TreeSpec
from hmtc.plugins.base import TreePlugin, register_tree
from hmtc.plugins.hierarchical import LearnedHierarchy


@register_tree("chemical")
def build_chemical_tree(spec: TreeSpec, hidden_dim: int) -> TreePlugin:
    return LearnedHierarchy(spec, hidden_dim, view_bias=0.01)


@register_tree("geometry")
def build_geometry_tree(spec: TreeSpec, hidden_dim: int) -> TreePlugin:
    return LearnedHierarchy(spec, hidden_dim, view_bias=-0.01)


@register_tree("generic")
def build_generic_tree(spec: TreeSpec, hidden_dim: int) -> TreePlugin:
    """Default hook for future site-backed physics/property trees."""

    return LearnedHierarchy(spec, hidden_dim)
