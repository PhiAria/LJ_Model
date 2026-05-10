from __future__ import annotations

from dataclasses import replace
import numpy as np

from .config import build_user_params, make_default_params, select_solvent
from .evaporation import diffusion_limited_flux, hertz_knudsen_flux
from .jet_model import compute_nozzle_diagnostics, solve_jet
from .nucleation import compute_survival
from .properties import liquid_dynamic_viscosity, liquid_heat_capacity, liquid_surface_tension
from .solvents import SOLVENT_DATABASE


def run_smoke_tests() -> list[str]:
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
        try:
            build_user_params(fixed_breakup_length_mm=invalid_length_mm)
        except ValueError:
            pass
        else:
            raise AssertionError(f'fixed_breakup_length_mm={invalid_length_mm} should raise ValueError.')
    for invalid_calibration in (-0.1, 0.0):
        try:
            build_user_params(breakup_calibration_factor=invalid_calibration)
        except ValueError:
            pass
        else:
            raise AssertionError(f'breakup_calibration_factor={invalid_calibration} should raise ValueError.')
    try:
        build_user_params(breakup_viscous_coefficient=-0.1)
    except ValueError:
        pass
    else:
        raise AssertionError('Negative breakup_viscous_coefficient should raise ValueError.')

    for solvent in SOLVENT_DATABASE.values():
        solvent_params = select_solvent(params, solvent.name)
        assert solvent_params.rho_l > 500.0
        assert solvent_params.M > 0.0
        assert liquid_dynamic_viscosity(solvent_params.T_nozzle, solvent_params) > 0.0
        assert liquid_surface_tension(solvent_params.T_nozzle, solvent_params) > 0.0

    nucleation = compute_survival(params, solution)
    survival_delta = np.diff(nucleation.P_survival)
    assert np.all(survival_delta <= 1e-12)

    return [
        'Property correlations return plausible values at 293 K and 273 K.',
        'Hertz-Knudsen and continuum diffusion reference fluxes are positive.',
        'Default ODE integration terminates at or before selected breakup length.',
        'Breakup model switch supports computed and fixed-length operation.',
        'Supported solvents expose positive transport/thermo properties.',
        'CNT survival curve is monotonically non-increasing.',
    ]
