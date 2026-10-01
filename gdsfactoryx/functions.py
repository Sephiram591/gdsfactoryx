"""Differentiable wrapper of `gdsfactory.functions` (see `gdsfactoryx.wrap`)."""

from gdsfactoryx._wrap import module_dir, module_getattr

__getattr__ = module_getattr("gdsfactory.functions")
__dir__ = module_dir("gdsfactory.functions")
