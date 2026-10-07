"""The TKE Prandtl number: reference value, surrogate derivative."""

import os

import pytest

if os.environ.get("VEROS_BACKEND") != "jax":
    pytest.skip("surrogate derivatives exist only under the JAX backend", allow_module_level=True)


def _setup():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from veros.core import tke

    return jax, jnp, tke


def test_value_is_the_reference_formula():
    jax, jnp, tke = _setup()
    nsqr = jnp.array([-1e-6, 0.0, 1e-13, 1e-9, 1e-6, 1e-4])
    shear = jnp.array([0.0, 0.0, 0.0, 1e-6, 1e-6, 1e-8])
    reference = jnp.maximum(1.0, jnp.minimum(10.0, 6.6 * nsqr / jnp.maximum(shear, 1e-12)))
    for floor in (0.0, 1e-7):
        assert jnp.array_equal(tke.prandtl_number(nsqr, shear, None, floor, 0.5), reference)


def test_derivative_resolves_the_quiescent_switch():
    """No shear: the reference derivative is ~1e12 in a 1e-12 window; the surrogate's is bounded."""
    jax, jnp, tke = _setup()

    def pr(nsqr, floor):
        return tke.prandtl_number(nsqr, jnp.zeros(()), None, floor, 0.5)

    # N^2 = 5e-13: Ri = 0.5 against the 1e-12 floor, inside the reference ramp.
    reference_slope = jax.grad(pr)(jnp.array(5e-13), 0.0)
    surrogate_slope = jax.grad(pr)(jnp.array(5e-13), 1e-7)
    assert reference_slope == pytest.approx(6.6e12)
    assert 0.0 < surrogate_slope < 1e9
    # Inside the switch window the surrogate's slope is the slope of its own smooth formula.
    nsqr = jnp.array(5e-8)
    expected = jax.grad(lambda n: tke._prandtl_from_richardson(n / tke.utilities.smooth_maximum(0.0, 1e-7, 1e-7), 0.5))(nsqr)
    assert jax.grad(pr)(nsqr, 1e-7) == pytest.approx(float(expected))


def test_derivative_is_bounded_where_the_shear_is_weak():
    """Weak but resolved shear: the reference ramp is steep; the surrogate's slope stays below 6.6 / floor."""
    jax, jnp, tke = _setup()
    floor = 3e-7
    shear = jnp.array(1e-7)  # a few mm/s across 10 m: well above the 1e-12 floor, below the surrogate's
    nsqr = jnp.array(5.5 / 6.6 * 1e-7)  # 6.6 Ri = 5.5, mid-way up the reference ramp

    def pr(n, floor):
        return tke.prandtl_number(n, shear, None, floor, 0.5)

    assert jax.grad(pr)(nsqr, 0.0) == pytest.approx(6.6e7)
    assert 0.0 < jax.grad(pr)(nsqr, floor) < 6.6 / floor


def test_derivative_is_the_reference_where_the_shear_is_resolved():
    """Away from the floor and the clip corners the surrogate's slope is the reference's.

    To within the tails of the clip's rounding: mid-way between the corners
    (6.6 Ri = 5.5) a 0.5 rounding width changes the slope by ~0.6%.
    """
    jax, jnp, tke = _setup()
    shear = jnp.array(1e-5)
    nsqr = jnp.array(5.5 / 6.6 * 1e-5)  # 6.6 Ri = 5.5, mid-way between the clip corners

    def pr(n, floor):
        return tke.prandtl_number(n, shear, None, floor, 0.5)

    assert jax.grad(pr)(nsqr, 1e-7) == pytest.approx(float(jax.grad(pr)(nsqr, 0.0)), rel=1e-2)
