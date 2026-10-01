"""Differentiable wrapper of `gdsfactory.path` (see `gdsfactoryx.wrap`)."""

from gdsfactoryx._wrap import module_dir, module_getattr

__getattr__ = module_getattr("gdsfactory.path")
__dir__ = module_dir("gdsfactory.path")
