"""Differentiable wrapper of `gdsfactory.containers` (see `gdsfactoryx.wrap`)."""

from gdsfactoryx._wrap import module_dir, module_getattr

__getattr__ = module_getattr("gdsfactory.containers")
__dir__ = module_dir("gdsfactory.containers")
