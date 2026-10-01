import jax

jax.config.update("jax_enable_x64", True)

import gdsfactoryx as gfx  # noqa: E402

# Fine database unit -> accurate finite-difference gradients (see activate_pdk).
gfx.activate_pdk(dbu=1e-5)
