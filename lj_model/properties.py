from __future__ import annotations

import numpy as np

from .config import JetParams


def clamp_temperature(T: float, params: JetParams, low: float | None = None, high: float | None = None) -> float:
    low_val = params.T_guard_min if low is None else low
    high_val = params.T_guard_max if high is None else high
    return float(np.clip(T, low_val, high_val))


def liquid_heat_capacity(T: float, params: JetParams) -> float:
    if not params.switches.use_temp_dependent_properties:
        return 4200.0
    T_c = clamp_temperature(T, params, params.T_property_min, params.T_property_max) - 273.15
    cp = (
        4217.4
        - 3.720283 * T_c
        + 0.1412855 * T_c**2
        - 2.654387e-3 * T_c**3
        + 2.093236e-5 * T_c**4
    )
    return max(float(cp), 3000.0)


def liquid_surface_tension(T: float, params: JetParams) -> float:
    if not params.switches.use_temp_dependent_properties:
        return 0.072
    T_eval = clamp_temperature(T, params, params.T_property_min, params.T_property_critical_max)
    tau = max(1.0 - T_eval / params.T_critical, 1e-9)
    sigma = 0.2358 * tau**1.256 * (1.0 - 0.625 * tau)
    return max(float(sigma), params.sigma_min)


def liquid_thermal_conductivity(T: float, params: JetParams) -> float:
    if not params.switches.use_temp_dependent_properties:
        return 0.6
    T_c = clamp_temperature(T, params, params.T_property_min, params.T_property_max) - 273.15
    k_val = 0.561 + 1.93e-3 * T_c - 6.35e-6 * T_c**2
    return max(float(k_val), params.k_thermal_min)


def liquid_dynamic_viscosity(T: float, params: JetParams) -> float:
    if not params.switches.use_temp_dependent_properties:
        return 1.0e-3
    T_c = clamp_temperature(T, params, params.T_property_min, params.T_property_max) - 273.15
    return float(2.414e-5 * 10 ** (247.8 / (T_c + 133.15)))


def vapor_pressure(T: float, params: JetParams) -> float:
    T_eval = clamp_temperature(T, params, 123.0, 332.0)
    ln_p = (
        54.842763
        - 6763.22 / T_eval
        - 4.210 * np.log(T_eval)
        + 0.000367 * T_eval
        + np.tanh(0.0415 * (T_eval - 218.8))
        * (53.878 - 1331.22 / T_eval - 9.44523 * np.log(T_eval) + 0.014025 * T_eval)
    )
    return float(np.exp(ln_p))
