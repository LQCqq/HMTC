"""Tree plugin registry.

New domains register a builder without modifying the HMTF core.  Defect and
recipe interfaces are declared in :mod:`hmtc.plugins.extensions`.
"""

from . import core as _core  # noqa: F401
from .base import TREE_REGISTRY, TreePlugin, build_tree_plugin, register_tree

__all__ = ["TREE_REGISTRY", "TreePlugin", "build_tree_plugin", "register_tree"]
