from veros.core.operators import numpy as npx

from veros import veros_kernel
from veros.core.operators import update, at, solve_tridiagonal


@veros_kernel(static_args=("enable_cyclic_x", "local"))
def enforce_boundaries(arr, enable_cyclic_x, local=False):
    from veros import runtime_state as rst
    from veros.routines import CURRENT_CONTEXT

    if rst.proc_num == 1 or not CURRENT_CONTEXT.is_dist_safe or local:
        if enable_cyclic_x:
            arr = update(arr, at[-2:, ...], arr[2:4, ...])
            arr = update(arr, at[:2, ...], arr[-4:-2, ...])
        return arr

    from veros.distributed import exchange_overlap

    arr = exchange_overlap(arr, ["xt", "yt"], cyclic=enable_cyclic_x)
    return arr


@veros_kernel
def sqrt_singularity_removed(x, eps=1e-12):
    """
    AD-safe replacement for `sqrt(maximum(0, x))`.

    Two problems arise when naively differentiating through `sqrt` near zero:
      1. `sqrt(x)` has an infinite derivative at x=0, so AD blows up to
         inf/nan wherever x reaches exactly 0 (e.g. masked cells, zero
         initial conditions like `tke`/`eke`).
      2. `sqrt(maximum(0, x))` guards the primal but keeps the kink at 0,
         causing the same inf tangent via JAX's `maximum` JVP rule
         (which picks the sqrt branch at the tie point).

    Clamping x away from the singularity with `maximum(x, eps)` (eps > 0)
    fixes both: the denominator is at least sqrt(eps) > 0, so sqrt's
    derivative is at most 1/(2*sqrt(eps)) -- finite; and for x < eps the
    gradient is zero (the clamp is active), not the dangerous 0*inf product.

    Note: `sqrt(x + eps)` (the Huber-shift approach) is NOT safe when x can
    be genuinely negative (e.g. Nsqr in convectively unstable regions, or
    tke/eke with numerical undershoot from the implicit solver). For those
    inputs, `sqrt((x + eps))` evaluates sqrt of a negative number -- NaN in
    both primal and tangent. `sqrt(maximum(x, eps))` is safe for any sign of
    x: the clamp floor keeps the argument positive.
    """
    return npx.sqrt(npx.maximum(x, eps))


@veros_kernel
def pad_z_edges(array):
    """
    Pads the z-axis of an array by repeating its edge values
    """
    if array.ndim == 1:
        newarray = npx.pad(array, 1, mode="edge")
    elif array.ndim >= 3:
        newarray = npx.pad(array, ((0, 0), (0, 0), (1, 1)), mode="edge")
    else:
        raise ValueError("Array to pad needs to have 1 or at least 3 dimensions")
    return newarray


@veros_kernel(static_args=("nz"))
def create_water_masks(ks, nz):
    ks = ks - 1
    land_mask = ks >= 0
    water_mask = npx.logical_and(
        land_mask[:, :, npx.newaxis], npx.arange(nz)[npx.newaxis, npx.newaxis, :] >= ks[:, :, npx.newaxis]
    )
    edge_mask = npx.logical_and(
        land_mask[:, :, npx.newaxis], npx.arange(nz)[npx.newaxis, npx.newaxis, :] == ks[:, :, npx.newaxis]
    )
    return land_mask, water_mask, edge_mask


@veros_kernel
def solve_implicit(a, b, c, d, water_mask, edge_mask, b_edge=None, d_edge=None):
    if b_edge is not None:
        b = npx.where(edge_mask, b_edge, b)

    if d_edge is not None:
        d = npx.where(edge_mask, d_edge, d)

    return solve_tridiagonal(a, b, c, d, water_mask, edge_mask)


def with_surrogate_gradient(exact, surrogate):
    """
    Return a function with `exact`'s value and `surrogate`'s derivatives.

    The same construction as jax-gcm's `jcm.physics.surrogate_gradient`:
    the value is `exact(*args)` bit for bit, so the forward model is
    unchanged, while every derivative (forward and reverse mode, any order)
    is that of the smooth function `surrogate`, taken by differentiating it.
    Use it where the reference derivative carries no usable information --
    a threshold switch whose ramp is far narrower than any perturbation of
    interest, a plateau, a singular point -- not for an ordinary bounded kink.

    Under the NumPy backend there are no derivatives, and `exact` itself is
    returned.
    """
    from veros import runtime_settings

    if runtime_settings.backend != "jax":
        return exact

    import jax

    @jax.custom_jvp
    def f(*args):
        return exact(*args)

    @f.defjvp
    def f_jvp(primals, tangents):
        # The value comes from `f`, not `exact`, so that differentiating
        # twice meets this rule again rather than `exact`'s kinks.
        _, tangent_out = jax.jvp(surrogate, primals, tangents)
        return f(*primals), tangent_out

    return f


def smooth_maximum(a, b, width):
    """`max(a, b)` with its corner rounded over `width`: `(a + b + sqrt((a - b)**2 + width**2)) / 2`."""
    return 0.5 * (a + b + npx.sqrt((a - b) ** 2 + width**2))


def smooth_minimum(a, b, width):
    """`min(a, b)` with its corner rounded over `width`."""
    return -smooth_maximum(-a, -b, width)

