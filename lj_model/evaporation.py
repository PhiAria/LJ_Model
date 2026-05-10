from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import JetParams
from .properties import clamp_temperature, vapor_pressure

SQRT_TWO = np.sqrt(2.0)


@dataclass(frozen=True)
class FluxComponents:
    net: float
    hk: float
    diffusion: float
    diffusion_raw: float
    diffusion_applied: bool
    kn: float
    regime: str


def mean_free_path(T: float, P_total: float, params: JetParams) -> float:
    T_eval = clamp_temperature(T, params)
    P_eval = max(P_total, 1e-9)
    return params.kB * T_eval / (SQRT_TWO * np.pi * params.d_molecule**2 * P_eval)


def diffusion_coefficient_vapor(T: float, P_total: float, params: JetParams) -> float:
    T_eval = clamp_temperature(T, params)
    P_eval = max(P_total, 1e-9)
    return params.D_v_ref * (T_eval / params.T_ref) ** 1.75 * (params.P_ref / P_eval)


def hertz_knudsen_flux(T: float, params: JetParams, alpha: float | None = None, P_vapor: float = 0.0) -> float:
    T_eval = clamp_temperature(T, params)
    alpha_eff = params.alpha_evap if alpha is None else alpha
    P_sat = vapor_pressure(T_eval, params)
    delta_p = max(P_sat - max(P_vapor, 0.0), 0.0)
    return float(alpha_eff * delta_p * np.sqrt(params.M / (2.0 * np.pi * params.R_gas * T_eval)))


def diffusion_limited_flux(
    T_surface: float,
    r: float,
    P_total: float,
    params: JetParams,
    P_vapor: float = 0.0,
    T_gas_local: float | None = None,
    sh: float = 1.0,
) -> float:
    T_s = clamp_temperature(T_surface, params)
    T_inf = clamp_temperature(T_s if T_gas_local is None else T_gas_local, params)
    P_total = max(P_total, 1e-9)
    P_vapor = float(np.clip(P_vapor, 0.0, P_total))
    T_film = 0.5 * (T_s + T_inf)
    D_v = diffusion_coefficient_vapor(T_film, P_total, params)
    rho_v_surface = vapor_pressure(T_s, params) * params.M / (params.R_gas * T_s)
    rho_v_inf = P_vapor * params.M / (params.R_gas * T_inf)
    delta_rho = max(rho_v_surface - rho_v_inf, 0.0)
    return float(sh * D_v * delta_rho / max(r, params.r_guard_min))


def knudsen_number(T: float, P_total: float, r: float, params: JetParams) -> float:
    return mean_free_path(T, P_total, params) / max(r, params.r_guard_min)


def classify_knudsen_regime(Kn: float) -> str:
    if Kn > 10.0:
        return 'free-molecular'
    if Kn > 0.1:
        return 'transitional'
    return 'continuum'


def evaporation_flux_components(
    T_surface: float,
    r: float,
    params: JetParams,
    alpha: float | None = None,
    P_total: float | None = None,
    P_vapor: float | None = None,
    T_gas_local: float | None = None,
) -> FluxComponents:
    P_total_eff = max(params.P_chamber if P_total is None else P_total, 1e-9)
    P_vapor_input = params.P_back if P_vapor is None else P_vapor
    P_vapor_eff = P_vapor_input if params.switches.use_back_pressure else 0.0
    gamma_hk = hertz_knudsen_flux(T_surface, params, alpha=alpha, P_vapor=P_vapor_eff)
    gamma_diff_raw = diffusion_limited_flux(
        T_surface,
        r,
        P_total_eff,
        params,
        P_vapor=P_vapor_eff,
        T_gas_local=params.T_gas if T_gas_local is None else T_gas_local,
    )
    kn = knudsen_number(params.T_gas if T_gas_local is None else T_gas_local, P_total_eff, r, params)
    diffusion_applicable = params.switches.use_diffusion_limit and kn < params.diffusion_limit_kn_threshold
    if diffusion_applicable and gamma_hk > 0.0 and gamma_diff_raw > 0.0:
        gamma_net = 1.0 / (1.0 / gamma_hk + 1.0 / gamma_diff_raw)
        gamma_diff = gamma_diff_raw
    elif diffusion_applicable:
        gamma_net = 0.0
        gamma_diff = gamma_diff_raw
    else:
        gamma_net = gamma_hk
        gamma_diff = float('nan') if params.switches.use_diffusion_limit else gamma_diff_raw
    return FluxComponents(
        net=float(gamma_net),
        hk=float(gamma_hk),
        diffusion=float(gamma_diff),
        diffusion_raw=float(gamma_diff_raw),
        diffusion_applied=diffusion_applicable,
        kn=float(kn),
        regime=classify_knudsen_regime(kn),
    )
