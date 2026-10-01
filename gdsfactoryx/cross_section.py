"""Differentiable wrapper of `gdsfactory.cross_section` (see `gdsfactoryx.wrap`)."""

from gdsfactoryx._wrap import module_dir, module_getattr

__getattr__ = module_getattr("gdsfactory.cross_section")
__dir__ = module_dir("gdsfactory.cross_section")
