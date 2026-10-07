from veros import veros_kernel, veros_routine, KernelOutput
from veros.variables import allocate
from veros.core import advection, utilities
from veros.core.operators import update, update_add, at, for_loop, numpy as npx


@veros_routine
def set_tke_diffusivities(state):
    vs = state.variables
    settings = state.settings

    if settings.enable_tke:
        tke_diff_out = set_tke_diffusivities_kernel(state)
        vs.update(tke_diff_out)
    else:
        vs.kappaM = update(vs.kappaM, at[...], settings.kappaM_0)
        vs.kappaH = npx.where(vs.Nsqr[..., vs.tau] < 0.0, 1.0, settings.kappaH_0)


def _prandtl_from_richardson(richardson, clip=None):
    """Prandtl number `6.6 * Ri` clipped to [1, 10]; rounded over `clip` when given."""
    if clip is None:
        return npx.maximum(1.0, npx.minimum(10.0, 6.6 * richardson))
    return utilities.smooth_maximum(1.0, utilities.smooth_minimum(10.0, 6.6 * richardson, clip), clip)


def prandtl_number(Nsqr, shear_squared, internal_wave_bound, shear_floor, clip_width):
    """
    The TKE closure's Prandtl number, with a derivative that resolves its switch.

    The value is the reference formulation, `clip(6.6 * Ri, 1, 10)` with
    `Ri = Nsqr / max(shear_squared, 1e-12)` (further bounded by the IDEMIX
    term when IDEMIX is on). Where the water column has no shear -- an ocean
    started from rest, or any quiescent column -- `Ri` is `Nsqr * 1e12`, so
    the Prandtl number switches from 1 (convective) to 10 (stable) as `Nsqr`
    crosses a window about 1e-12 s^-2 wide around zero. That switch is real,
    but its derivative is ~1e12 inside the window and zero outside: a cell
    whose stratification happens to fall in the window has a sensitivity
    four or five orders of magnitude above its neighbours that no
    perturbation larger than ~1e-4 K reproduces, and a multi-step adjoint
    carries it along.

    The derivative is therefore taken from a surrogate
    (`utilities.with_surrogate_gradient`, the construction of jax-gcm's
    `jcm.physics.surrogate_gradient`): the same formula with the shear floor
    raised smoothly to `shear_floor` and the clip rounded over `clip_width`.
    Wherever the shear well exceeds `shear_floor` its slope is the
    reference's to within the tails of the clip's hyperbolic rounding (under
    1% mid-way between the corners for the default width); where the shear
    is below the floor it spreads the switch over `0 < Nsqr < ~1.5 *
    shear_floor`. `shear_floor = 0`
    selects the reference derivative.
    """

    def exact(Nsqr, shear_squared, internal_wave_bound):
        richardson = Nsqr / npx.maximum(shear_squared, 1e-12)
        if internal_wave_bound is not None:
            richardson = npx.minimum(richardson, internal_wave_bound)
        return _prandtl_from_richardson(richardson)

    if shear_floor <= 0:
        return exact(Nsqr, shear_squared, internal_wave_bound)

    def surrogate(Nsqr, shear_squared, internal_wave_bound):
        richardson = Nsqr / utilities.smooth_maximum(shear_squared, shear_floor, shear_floor)
        if internal_wave_bound is not None:
            richardson = utilities.smooth_minimum(richardson, internal_wave_bound, clip_width / 6.6)
        return _prandtl_from_richardson(richardson, clip_width)

    return utilities.with_surrogate_gradient(exact, surrogate)(Nsqr, shear_squared, internal_wave_bound)


@veros_kernel
def set_tke_diffusivities_kernel(state):
    """
    set vertical diffusivities based on TKE model
    """
    vs = state.variables
    settings = state.settings

    vs.sqrttke = utilities.sqrt_singularity_removed(vs.tke[:, :, :, vs.tau])
    """
    calculate buoyancy length scale
    """
    vs.mxl = npx.sqrt(2) * vs.sqrttke / utilities.sqrt_singularity_removed(vs.Nsqr[:, :, :, vs.tau]) * vs.maskW

    """
    apply limits for mixing length
    """
    if settings.tke_mxl_choice == 1:
        """
        bounded by the distance to surface/bottom
        """
        vs.mxl = npx.minimum(
            npx.minimum(vs.mxl, -vs.zw[npx.newaxis, npx.newaxis, :] + vs.dzw[npx.newaxis, npx.newaxis, :] * 0.5),
            vs.ht[:, :, npx.newaxis] + vs.zw[npx.newaxis, npx.newaxis, :],
        )
        vs.mxl = npx.maximum(vs.mxl, settings.mxl_min)
    elif settings.tke_mxl_choice == 2:
        """
        bound length scale as in mitgcm/OPA code
        """
        nz = state.dimensions["zt"]

        def backwards_pass(kinv, mxl):
            k = nz - kinv - 1
            return update(mxl, at[:, :, k], npx.minimum(mxl[:, :, k], mxl[:, :, k + 1] + vs.dzt[k + 1]))

        vs.mxl = for_loop(1, nz, backwards_pass, vs.mxl)
        vs.mxl = update(vs.mxl, at[:, :, -1], npx.minimum(vs.mxl[:, :, -1], settings.mxl_min + vs.dzt[-1]))

        def forwards_pass(k, mxl):
            return update(mxl, at[:, :, k], npx.minimum(mxl[:, :, k], mxl[:, :, k - 1] + vs.dzt[k]))

        vs.mxl = for_loop(1, nz, forwards_pass, vs.mxl)
        vs.mxl = npx.maximum(vs.mxl, settings.mxl_min)
    else:
        raise ValueError("unknown mixing length choice in tke_mxl_choice")

    """
    calculate viscosity and diffusivity based on Prandtl number
    """
    vs.K_diss_v = utilities.enforce_boundaries(vs.K_diss_v, settings.enable_cyclic_x)
    vs.kappaM = update(vs.kappaM, at[...], npx.minimum(settings.kappaM_max, settings.c_k * vs.mxl * vs.sqrttke))
    shear_squared = vs.K_diss_v / npx.maximum(1e-12, vs.kappaM)
    if settings.enable_idemix:
        internal_wave_bound = (
            vs.kappaM * vs.Nsqr[:, :, :, vs.tau] / npx.maximum(1e-12, vs.alpha_c * vs.E_iw[:, :, :, vs.tau] ** 2)
        )
    else:
        internal_wave_bound = None

    if settings.enable_Prandtl_tke:
        vs.Prandtlnumber = prandtl_number(
            vs.Nsqr[:, :, :, vs.tau],
            shear_squared,
            internal_wave_bound,
            settings.tke_prandtl_surrogate_shear_floor,
            settings.tke_prandtl_surrogate_width,
        )
    else:
        vs.Prandtlnumber = update(vs.Prandtlnumber, at[...], settings.Prandtl_tke0)

    vs.kappaH = npx.maximum(settings.kappaH_min, vs.kappaM / vs.Prandtlnumber)

    if settings.enable_kappaH_profile:
        # Correct diffusivity according to
        # Bryan, K., and L. J. Lewis, 1979:
        # A water mass model of the world ocean. J. Geophys. Res., 84, 2503–2517.
        # It mainly modifies kappaH within 20S - 20N deg. belt
        vs.kappaH = npx.maximum(
            vs.kappaH,
            (0.8 + 1.05 / settings.pi * npx.arctan((-vs.zw[npx.newaxis, npx.newaxis, :] - 2500.0) / 222.2)) * 1e-4,
        )

    vs.kappaM = npx.maximum(settings.kappaM_min, vs.kappaM)

    return KernelOutput(
        sqrttke=vs.sqrttke,
        mxl=vs.mxl,
        kappaM=vs.kappaM,
        kappaH=vs.kappaH,
        Prandtlnumber=vs.Prandtlnumber,
        K_diss_v=vs.K_diss_v,
    )


@veros_routine
def integrate_tke(state):
    vs = state.variables
    tke_out = integrate_tke_kernel(state)
    vs.update(tke_out)


@veros_kernel
def integrate_tke_kernel(state):
    """
    integrate Tke equation on W grid with surface flux boundary condition
    """
    vs = state.variables
    settings = state.settings

    conditional_outputs = {}

    flux_east = allocate(state.dimensions, ("xt", "yt", "zt"))
    flux_north = allocate(state.dimensions, ("xt", "yt", "zt"))
    flux_top = allocate(state.dimensions, ("xt", "yt", "zt"))

    dt_tke = settings.dt_mom  # use momentum time step to prevent spurious oscillations

    """
    Sources and sinks by vertical friction, vertical mixing, and non-conservative advection
    """
    forc = vs.K_diss_v - vs.P_diss_v - vs.P_diss_adv

    """
    store transfer due to vertical mixing from dyn. enthalpy by non-linear eq.of
    state either to TKE or to heat
    """
    if not settings.enable_store_cabbeling_heat:
        forc = forc - vs.P_diss_nonlin

    """
    transfer part of dissipation of EKE to TKE
    """
    if settings.enable_eke:
        forc = forc + vs.eke_diss_tke

    if settings.enable_idemix:
        """
        transfer dissipation of internal waves to TKE
        """
        forc = forc + vs.iw_diss
        """
        store bottom friction either in TKE or internal waves
        """
        if settings.enable_store_bottom_friction_tke:
            forc = forc + vs.K_diss_bot

    else:  # short-cut without idemix
        if settings.enable_eke:
            forc = forc + vs.eke_diss_iw

        else:  # and without EKE model
            if settings.enable_store_cabbeling_heat:
                forc = forc + vs.K_diss_gm + vs.K_diss_h - vs.P_diss_skew - vs.P_diss_hmix - vs.P_diss_iso
            else:
                forc = forc + vs.K_diss_gm + vs.K_diss_h - vs.P_diss_skew

        forc = forc + vs.K_diss_bot

    """
    vertical mixing and dissipation of TKE
    """
    _, water_mask, edge_mask = utilities.create_water_masks(vs.kbot[2:-2, 2:-2], settings.nz)

    a_tri, b_tri, c_tri, d_tri, delta = (
        allocate(state.dimensions, ("xt", "yt", "zt"))[2:-2, 2:-2, :] for _ in range(5)
    )

    delta = update(
        delta,
        at[:, :, :-1],
        dt_tke
        / vs.dzt[npx.newaxis, npx.newaxis, 1:]
        * settings.alpha_tke
        * 0.5
        * (vs.kappaM[2:-2, 2:-2, :-1] + vs.kappaM[2:-2, 2:-2, 1:]),
    )

    a_tri = update(a_tri, at[:, :, 1:-1], -delta[:, :, :-2] / vs.dzw[npx.newaxis, npx.newaxis, 1:-1])
    a_tri = update(a_tri, at[:, :, -1], -delta[:, :, -2] / (0.5 * vs.dzw[-1]))

    b_tri = update(
        b_tri,
        at[:, :, 1:-1],
        1
        + (delta[:, :, 1:-1] + delta[:, :, :-2]) / vs.dzw[npx.newaxis, npx.newaxis, 1:-1]
        + dt_tke * settings.c_eps * vs.sqrttke[2:-2, 2:-2, 1:-1] / vs.mxl[2:-2, 2:-2, 1:-1],
    )
    b_tri = update(
        b_tri,
        at[:, :, -1],
        1
        + delta[:, :, -2] / (0.5 * vs.dzw[-1])
        + dt_tke * settings.c_eps / vs.mxl[2:-2, 2:-2, -1] * vs.sqrttke[2:-2, 2:-2, -1],
    )
    b_tri_edge = (
        1
        + delta / vs.dzw[npx.newaxis, npx.newaxis, :]
        + dt_tke * settings.c_eps / vs.mxl[2:-2, 2:-2, :] * vs.sqrttke[2:-2, 2:-2, :]
    )

    c_tri = update(c_tri, at[:, :, :-1], -delta[:, :, :-1] / vs.dzw[npx.newaxis, npx.newaxis, :-1])

    d_tri = update(d_tri, at[...], vs.tke[2:-2, 2:-2, :, vs.tau] + dt_tke * forc[2:-2, 2:-2, :])
    d_tri = update_add(d_tri, at[:, :, -1], dt_tke * vs.forc_tke_surface[2:-2, 2:-2] / (0.5 * vs.dzw[-1]))

    sol = utilities.solve_implicit(a_tri, b_tri, c_tri, d_tri, water_mask, b_edge=b_tri_edge, edge_mask=edge_mask)
    vs.tke = update(vs.tke, at[2:-2, 2:-2, :, vs.taup1], npx.where(water_mask, sol, vs.tke[2:-2, 2:-2, :, vs.taup1]))

    """
    Clamp sub-surface levels to non-negative, extending to depth the same
    floor already applied at the surface below (via tke_surf_corr). The
    implicit solve can leave small negative numerical-undershoot values at
    depth (observed ~-1e-7 against typical tke magnitudes of ~1e-4) -- a
    well-known artifact of implicit discretizations of nominally-positive
    quantities, and negligible for the forward solution. Left unclamped,
    these values reach sqrt_singularity_removed via vs.tke[..., vs.tau] on
    the following step and produce NaN tangents under jax.jvp, since sqrt's
    derivative blows up at/near zero.
    """
    vs.tke = update(
        vs.tke,
        at[2:-2, 2:-2, :-1, vs.taup1],
        npx.maximum(0.0, vs.tke[2:-2, 2:-2, :-1, vs.taup1]),
    )

    """
    store tke dissipation for diagnostics
    """
    vs.tke_diss = settings.c_eps / vs.mxl * vs.sqrttke * vs.tke[:, :, :, vs.taup1]

    """
    Add TKE if surface density flux drains TKE in uppermost box
    """
    mask = vs.tke[2:-2, 2:-2, -1, vs.taup1] < 0.0
    vs.tke_surf_corr = update(
        vs.tke_surf_corr,
        at[2:-2, 2:-2],
        npx.where(mask, -vs.tke[2:-2, 2:-2, -1, vs.taup1] * 0.5 * vs.dzw[-1] / dt_tke, 0.0),
    )
    vs.tke = update(vs.tke, at[2:-2, 2:-2, -1, vs.taup1], npx.maximum(0.0, vs.tke[2:-2, 2:-2, -1, vs.taup1]))

    if settings.enable_tke_hor_diffusion:
        """
        add tendency due to lateral diffusion
        """
        flux_east = update(
            flux_east,
            at[:-1, :, :],
            settings.K_h_tke
            * (vs.tke[1:, :, :, vs.tau] - vs.tke[:-1, :, :, vs.tau])
            / (vs.cost[npx.newaxis, :, npx.newaxis] * vs.dxu[:-1, npx.newaxis, npx.newaxis])
            * vs.maskU[:-1, :, :],
        )

        flux_north = update(
            flux_north,
            at[:, :-1, :],
            settings.K_h_tke
            * (vs.tke[:, 1:, :, vs.tau] - vs.tke[:, :-1, :, vs.tau])
            / vs.dyu[npx.newaxis, :-1, npx.newaxis]
            * vs.maskV[:, :-1, :]
            * vs.cosu[npx.newaxis, :-1, npx.newaxis],
        )
        flux_north = update(flux_north, at[:, -1, :], 0.0)

        vs.tke = update_add(
            vs.tke,
            at[2:-2, 2:-2, :, vs.taup1],
            dt_tke
            * vs.maskW[2:-2, 2:-2, :]
            * (
                (flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
                / (vs.cost[npx.newaxis, 2:-2, npx.newaxis] * vs.dxt[2:-2, npx.newaxis, npx.newaxis])
                + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
                / (vs.cost[npx.newaxis, 2:-2, npx.newaxis] * vs.dyt[npx.newaxis, 2:-2, npx.newaxis])
            ),
        )

    """
    add tendency due to advection
    """
    if settings.enable_tke_superbee_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_superbee_wgrid(state, vs.tke[:, :, :, vs.tau])

    if settings.enable_tke_upwind_advection:
        flux_east, flux_north, flux_top = advection.adv_flux_upwind_wgrid(state, vs.tke[:, :, :, vs.tau])

    if settings.enable_tke_superbee_advection or settings.enable_tke_upwind_advection:
        vs.dtke = update(
            vs.dtke,
            at[2:-2, 2:-2, :, vs.tau],
            vs.maskW[2:-2, 2:-2, :]
            * (
                -(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
                / (vs.cost[npx.newaxis, 2:-2, npx.newaxis] * vs.dxt[2:-2, npx.newaxis, npx.newaxis])
                - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
                / (vs.cost[npx.newaxis, 2:-2, npx.newaxis] * vs.dyt[npx.newaxis, 2:-2, npx.newaxis])
            ),
        )
        vs.dtke = update_add(vs.dtke, at[:, :, 0, vs.tau], -flux_top[:, :, 0] / vs.dzw[0])
        vs.dtke = update_add(
            vs.dtke, at[:, :, 1:-1, vs.tau], -(flux_top[:, :, 1:-1] - flux_top[:, :, :-2]) / vs.dzw[1:-1]
        )
        vs.dtke = update_add(
            vs.dtke, at[:, :, -1, vs.tau], -(flux_top[:, :, -1] - flux_top[:, :, -2]) / (0.5 * vs.dzw[-1])
        )

        """
        Adam Bashforth time stepping
        """
        vs.tke = update_add(
            vs.tke,
            at[:, :, :, vs.taup1],
            settings.dt_tracer
            * (
                (1.5 + settings.AB_eps) * vs.dtke[:, :, :, vs.tau]
                - (0.5 + settings.AB_eps) * vs.dtke[:, :, :, vs.taum1]
            ),
        )

        conditional_outputs.update(dtke=vs.dtke)

    return KernelOutput(tke=vs.tke, tke_surf_corr=vs.tke_surf_corr, tke_diss=vs.tke_diss, **conditional_outputs)
