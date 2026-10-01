import gdsfactory as gf

import gdsfactoryx as gfx
from gdsfactoryx.components import straight


def test_submodule_import_and_identity():
    assert straight is gfx.components.straight
    assert straight.func is gf.components.straight


def test_nested_modules_are_wrapped():
    assert isinstance(gfx.routing, gfx.WrappedModule) or hasattr(gfx.routing, "__getattr__")
    assert isinstance(gfx.gpdk, gfx.WrappedModule)
    assert gfx.get_layer("WG") == gf.get_layer("WG")


def test_dir_lists_gdsfactory_names():
    assert "straight" in dir(gfx.components)
    assert "Component" in dir(gfx)
