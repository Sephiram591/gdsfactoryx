"""Differentiable wrapper of `gdsfactory.components` (see `gdsfactoryx.wrap`)."""

from gdsfactoryx._wrap import module_dir, module_getattr

__getattr__ = module_getattr("gdsfactory.components")
__dir__ = module_dir("gdsfactory.components")
