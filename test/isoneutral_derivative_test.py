"""The isoneutral mixing tensor: reference value, derivative held fixed by default."""

import os

import pytest

if os.environ.get("VEROS_BACKEND") != "jax":
    pytest.skip("derivatives exist only under the JAX backend", allow_module_level=True)

TEST_SETTINGS = dict(nx=12, ny=10, nz=8, dt_tracer=3600, dt_mom=3600, enable_neutral_diffusion=True, K_iso_steep=1.0)


def _state(enable_derivative):
    import jax

    jax.config.update("jax_enable_x64", True)
    from veros.pyom_compat import get_random_state

    return get_random_state(extra_settings=dict(TEST_SETTINGS, enable_isoneutral_tensor_derivative=enable_derivative))


def _set_derivative(state, enable_derivative):
    with state.settings.unlock():
        state.settings.enable_isoneutral_tensor_derivative = enable_derivative


def _tensor_of_temperature(state):
    """K_33 summed, as a function of the temperature field."""
    from veros.core import isoneutral

    vs = state.variables
    temp0 = vs.temp

    def f(temp):
        vs.temp = temp
        try:
            return isoneutral.isoneutral_diffusion_pre(state).K_33.sum()
        finally:
            vs.temp = temp0

    return f, temp0


def test_value_is_unchanged():
    import numpy as np

    from veros.core import isoneutral

    state = _state(True)
    reference = isoneutral.isoneutral_diffusion_pre(state)
    _set_derivative(state, False)
    held = isoneutral.isoneutral_diffusion_pre(state)
    for name in ("Ai_ez", "Ai_nz", "Ai_bx", "Ai_by", "K_11", "K_22", "K_33"):
        np.testing.assert_array_equal(np.asarray(getattr(held, name)), np.asarray(getattr(reference, name)))


def test_tensor_derivative_is_held_fixed_by_default():
    import jax
    import jax.numpy as jnp

    state = _state(False)
    f, temp = _tensor_of_temperature(state)
    assert jnp.all(jax.grad(f)(temp) == 0.0)
    _set_derivative(state, True)
    assert jnp.any(jax.grad(f)(temp) != 0.0)


def test_flux_is_still_differentiated_with_respect_to_the_tracer():
    """Holding the tensor fixed leaves the isoneutral flux a linear function of the tracer it mixes."""
    import jax
    import jax.numpy as jnp

    from veros.core import isoneutral
    from veros.core.isoneutral.diffusion import isoneutral_diffusion_kernel

    state = _state(False)
    vs = state.variables
    vs.update(isoneutral.isoneutral_diffusion_pre(state))
    temp0 = vs.temp

    def tendency(temp):
        vs.temp = temp
        try:
            return (isoneutral_diffusion_kernel(state, vs.temp, True, iso=True).dtemp_iso ** 2).sum()
        finally:
            vs.temp = temp0

    assert jnp.any(jax.grad(tendency)(temp0) != 0.0)
