"""Differentiable wrapper of `gdsfactory.samples` (see `gdsfactoryx.wrap`)."""

from gdsfactoryx._wrap import module_dir, module_getattr

__getattr__ = module_getattr("gdsfactory.samples")
__dir__ = module_dir("gdsfactory.samples")
