from __future__ import annotations

import numpy as np

from .config import JetParams
from .solvents import canonical_solvent_name, get_solvent_properties


def clamp_temperature(T: float, params: JetParams, low: float | None = None, high: float | None = None) -> float:
    low_val = params.T_guard_min if low is None else low
    high_val = params.T_guard_max if high is None else high
    return float(np.clip(T, low_val, high_val))


def liquid_heat_capacity(T: float, params: JetParams) -> float:
    solvent = get_solvent_properties(params.solvent)
    is_water = canonical_solvent_name(params.solvent) == 'water'
    if not params.switches.use_temp_dependent_properties:
        return solvent.cp_ref_J_per_kg_K
    T_eval = clamp_temperature(T, params, params.T_property_min, params.T_property_max)
    if is_water:
        T_c = T_eval - 273.15
        cp = (
            4217.4
            - 3.720283 * T_c
            + 0.1412855 * T_c**2
            - 2.654387e-3 * T_c**3
            + 2.093236e-5 * T_c**4
        )
        return max(float(cp), 3000.0)
    cp = solvent.cp_ref_J_per_kg_K + solvent.cp_temp_slope_J_per_kg_K2 * (T_eval - params.T_ref)
    return max(float(cp), 1200.0)


def liquid_surface_tension(T: float, params: JetParams) -> float:
    solvent = get_solvent_properties(params.solvent)
    is_water = canonical_solvent_name(params.solvent) == 'water'
    if not params.switches.use_temp_dependent_properties:
        return max(solvent.sigma_ref_N_per_m, params.sigma_min)
    t_high = max(params.T_property_min + 1e-6, min(params.T_property_critical_max, params.T_critical - 1.0))
    T_eval = clamp_temperature(T, params, params.T_property_min, t_high)
    if is_water:
        tau = max(1.0 - T_eval / params.T_critical, 1e-9)
        sigma = 0.2358 * tau**1.256 * (1.0 - 0.625 * tau)
    else:
        sigma = solvent.sigma_ref_N_per_m - solvent.sigma_temp_coeff_N_per_m_K * (T_eval - params.T_ref)
    return max(float(sigma), params.sigma_min)


def liquid_thermal_conductivity(T: float, params: JetParams) -> float:
    solvent = get_solvent_properties(params.solvent)
    is_water = canonical_solvent_name(params.solvent) == 'water'
    if not params.switches.use_temp_dependent_properties:
        return max(solvent.k_ref_W_per_m_K, params.k_thermal_min)
    T_eval = clamp_temperature(T, params, params.T_property_min, params.T_property_max)
    if is_water:
        T_c = T_eval - 273.15
        k_val = 0.561 + 1.93e-3 * T_c - 6.35e-6 * T_c**2
    else:
        k_val = solvent.k_ref_W_per_m_K + solvent.k_temp_coeff_W_per_m_K2 * (T_eval - params.T_ref)
    return max(float(k_val), params.k_thermal_min)


def liquid_dynamic_viscosity(T: float, params: JetParams) -> float:
    solvent = get_solvent_properties(params.solvent)
    is_water = canonical_solvent_name(params.solvent) == 'water'
    if not params.switches.use_temp_dependent_properties:
        return solvent.mu_ref_Pa_s
    T_eval = clamp_temperature(T, params, params.T_property_min, params.T_property_max)
    if is_water:
        T_c = T_eval - 273.15
        return float(2.414e-5 * 10 ** (247.8 / (T_c + 133.15)))
    mu = solvent.mu_ref_Pa_s * np.exp(solvent.mu_activation_K * (1.0 / T_eval - 1.0 / params.T_ref))
    return float(max(mu, 5e-5))


def vapor_pressure(T: float, params: JetParams) -> float:
    solvent = get_solvent_properties(params.solvent)
    T_c = float(np.clip(T - 273.15, solvent.antoine_min_C, solvent.antoine_max_C))
    # Antoine constants use log10(P_mmHg). This is an engineering-range approximation.
    p_mmHg = 10 ** (solvent.antoine_A - solvent.antoine_B / (solvent.antoine_C + T_c))
    return float(p_mmHg * 133.322368)
