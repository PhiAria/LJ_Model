from __future__ import annotations

from dataclasses import replace
import numpy as np

from .config import build_user_params, make_default_params, select_solvent
from .evaporation import diffusion_limited_flux, hertz_knudsen_flux
from .jet_model import compute_nozzle_diagnostics, local_instability_growth_rate, solve_jet
from .nucleation import compute_survival
from .properties import liquid_dynamic_viscosity, liquid_heat_capacity, liquid_surface_tension
from .solvents import SOLVENT_DATABASE


def run_smoke_tests() -> list[str]:
    def expect_value_error(**kwargs: float) -> None:
        try:
            build_user_params(**kwargs)
        except ValueError:
            return
        raise AssertionError(f'Expected ValueError for build_user_params({kwargs}).')

    params = make_default_params()

    cp_293 = liquid_heat_capacity(293.15, params)
    cp_273 = liquid_heat_capacity(273.15, params)
    mu_293 = liquid_dynamic_viscosity(293.15, params)
    sigma_293 = liquid_surface_tension(293.15, params)
    assert 4100.0 < cp_293 < 4250.0
    assert 4150.0 < cp_273 < 4250.0
    assert 8.0e-4 < mu_293 < 1.3e-3
    assert 0.05 < sigma_293 < 0.09

    hk_flux = hertz_knudsen_flux(params.T_nozzle, params)
    diff_flux = diffusion_limited_flux(params.T_nozzle, params.r_nozzle, params.P_chamber, params)
    assert hk_flux > 0.0
    assert diff_flux > 0.0

    solution = solve_jet(params)
    assert solution.termination_reason == 'breakup'
    assert solution.z[-1] <= solution.breakup_length + 1e-9
    assert solution.breakup_source.startswith('computed')
    assert solution.nozzle_diagnostics.breakup_model_name.startswith('Sterling-Sleicher')

    fixed_mode = solve_jet(
        params.with_updates(
            switches=replace(params.switches, use_breakup_length_model=False),
            fixed_breakup_length=2.5e-3,
        )
    )
    assert fixed_mode.breakup_source.startswith('fixed')
    assert abs(fixed_mode.breakup_length - 2.5e-3) < 1e-12
    assert fixed_mode.termination_reason == 'breakup'
    assert abs(fixed_mode.nozzle_diagnostics.computed_breakup_length - fixed_mode.breakup_length) > 1e-6

    user_params = build_user_params(
        selected_solvent='EtOH',
        use_computed_breakup_length=False,
        fixed_breakup_length_mm=3.5,
        breakup_calibration_factor=0.72,
        breakup_viscous_coefficient=2.5,
    )
    user_diag = compute_nozzle_diagnostics(user_params)
    assert user_params.solvent == 'EtOH'
    assert not user_params.switches.use_breakup_length_model
    assert abs(user_params.fixed_breakup_length - 3.5e-3) < 1e-12
    assert abs(user_params.breakup_calibration_factor - 0.72) < 1e-12
    assert abs(user_params.breakup_viscous_coefficient - 2.5) < 1e-12
    assert user_diag.breakup_mode == 'fixed user value'
    assert user_diag.computed_breakup_length > 0.0

    for invalid_length_mm in (-1.0, 0.0):
        expect_value_error(fixed_breakup_length_mm=invalid_length_mm)
    for invalid_calibration in (-0.1, 0.0):
        expect_value_error(breakup_calibration_factor=invalid_calibration)
    expect_value_error(breakup_viscous_coefficient=-0.1)

    for solvent in SOLVENT_DATABASE.values():
        solvent_params = select_solvent(params, solvent.name)
        assert solvent_params.rho_l > 500.0
        assert solvent_params.M > 0.0
        assert liquid_dynamic_viscosity(solvent_params.T_nozzle, solvent_params) > 0.0
        assert liquid_surface_tension(solvent_params.T_nozzle, solvent_params) > 0.0

    nucleation = compute_survival(params, solution)
    survival_delta = np.diff(nucleation.P_survival)
    assert np.all(survival_delta <= 1e-12)

    # --- instability_growth mode tests ---
    ig_params = build_user_params(
        selected_solvent='water',
        breakup_mode='instability_growth',
        breakup_growth_prefactor=0.34,
        breakup_viscous_damping_coefficient=3.0,
        breakup_initial_amplitude_fraction=1e-4,
        breakup_final_amplitude_fraction=0.3,
    )
    assert ig_params.switches.breakup_mode == 'instability_growth'
    ig_nozzle = compute_nozzle_diagnostics(ig_params)
    assert ig_nozzle.instability_growth_mode is True
    assert ig_nozzle.breakup_mode == 'instability_growth'
    assert ig_nozzle.breakup_model_name.startswith('Local instability')
    assert ig_nozzle.computed_breakup_length > 0.0

    ig_solution = solve_jet(ig_params)
    assert ig_solution.termination_reason == 'breakup', (
        f'Instability-growth mode did not terminate by breakup; got {ig_solution.termination_reason!r}'
    )
    assert 0.0 < ig_solution.breakup_length < 0.05, (
        f'Instability-growth breakup length out of expected range: {ig_solution.breakup_length * 1e3:.3f} mm'
    )
    assert ig_solution.breakup_source == 'instability-growth amplitude threshold'
    assert ig_solution.nozzle_diagnostics.instability_growth_mode is True

    # Verify growth rate function returns a positive value with typical water conditions
    ig_omega = local_instability_growth_rate(
        r=10e-6, sigma=0.073, mu=1e-3, rho=998.0, params=ig_params
    )
    assert ig_omega > 0.0

    # Verify build_user_params raises for invalid breakup_mode
    try:
        build_user_params(breakup_mode='bad_mode')
        raise AssertionError('Expected ValueError for invalid breakup_mode.')
    except ValueError:
        pass

    # Verify radial-profile off + instability_growth mode also terminates by breakup
    ig_no_profile = build_user_params(breakup_mode='instability_growth').with_updates(
        switches=replace(ig_params.switches, use_radial_profile=False)
    )
    ig_no_profile_sol = solve_jet(ig_no_profile)
    assert ig_no_profile_sol.termination_reason == 'breakup'
    assert 0.0 < ig_no_profile_sol.breakup_length < 0.05

    return [
        'Property correlations return plausible values at 293 K and 273 K.',
        'Hertz-Knudsen and continuum diffusion reference fluxes are positive.',
        'Default ODE integration terminates at or before selected breakup length.',
        'Breakup model switch supports computed and fixed-length operation.',
        'Supported solvents expose positive transport/thermo properties.',
        'CNT survival curve is monotonically non-increasing.',
        'Instability-growth mode terminates by amplitude threshold and reports finite breakup position.',
        'Instability-growth mode works with and without radial profile.',
    ]
