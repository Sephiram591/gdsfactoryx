"""Differentiable wrapper of `gdsfactory.routing` (see `gdsfactoryx.wrap`)."""

from gdsfactoryx._wrap import module_dir, module_getattr

__getattr__ = module_getattr("gdsfactory.routing")
__dir__ = module_dir("gdsfactory.routing")
