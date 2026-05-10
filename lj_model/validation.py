from __future__ import annotations

import numpy as np

from .config import make_default_params
from .evaporation import diffusion_limited_flux, hertz_knudsen_flux
from .jet_model import solve_jet
from .nucleation import compute_survival
from .properties import liquid_dynamic_viscosity, liquid_heat_capacity, liquid_surface_tension


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

    nucleation = compute_survival(params, solution)
    survival_delta = np.diff(nucleation.P_survival)
    assert np.all(survival_delta <= 1e-12)

    return [
        'Property correlations return plausible values at 293 K and 273 K.',
        'Hertz-Knudsen and continuum diffusion reference fluxes are positive.',
        'Default ODE integration terminates at or before Rayleigh breakup.',
        'CNT survival curve is monotonically non-increasing.',
    ]
