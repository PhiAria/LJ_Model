from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .config import JetParams
from .evaporation import FluxComponents, classify_knudsen_regime, evaporation_flux_components, knudsen_number
from .properties import (
    clamp_temperature,
    liquid_dynamic_viscosity,
    liquid_heat_capacity,
    liquid_surface_tension,
    liquid_thermal_conductivity,
)

MIN_BREAKUP_GROWTH_RATE = 1e-12
BREAKUP_MODEL_NAME = 'Sterling-Sleicher laminar capillary correlation'
BREAKUP_MODEL_FORMULA = 'Lb/D = C_cal × 13 × sqrt(We) × (1 + C_Oh Oh)'
BREAKUP_MODEL_LIMITATIONS = 'Laminar capillary breakup only; aerodynamic breakup and measured nozzle forcing are not modeled.'
INSTABILITY_GROWTH_MODEL_NAME = 'Local instability-growth amplitude threshold'
INSTABILITY_GROWTH_MODEL_FORMULA = 'da/dz = (C_omega * sqrt(sigma/(rho*r^3)) / (1 + C_mu*Oh)) / v * a;  breakup when a >= beta_c * r'
INSTABILITY_GROWTH_MODEL_LIMITATIONS = 'Reduced-order local Rayleigh instability; calibrate C_omega, C_mu, a0/r, beta_c to experiment.'


@dataclass(frozen=True)
class NozzleDiagnostics:
    breakup_length: float
    breakup_time: float
    breakup_source: str
    breakup_mode: str
    breakup_model_name: str
    breakup_model_formula: str
    breakup_model_limitations: str
    computed_breakup_length: float
    computed_breakup_time: float
    reynolds_number: float
    weber_number: float
    ohnesorge_number: float
    cp_nozzle: float
    sigma_nozzle: float
    k_nozzle: float
    flux: FluxComponents
    cooling_rate_1d: float
    fourier_breakup: float
    delta_relax_length: float
    delta_t_quasisteady: float
    kn_nozzle: float
    adiabatic_freeze_fraction: float
    instability_growth_mode: bool = False
    breakup_initial_amplitude_fraction: float = 1e-4
    breakup_final_amplitude_fraction: float = 0.3
    breakup_growth_prefactor: float = 0.34
    breakup_viscous_damping_coefficient: float = 3.0

    def report_lines(self, params: JetParams) -> list[str]:
        diffusion_line = (
            f'Diffusion limit      : {self.flux.diffusion:.2e} kg/m2/s'
            if self.flux.diffusion_applied
            else f'Diffusion limit      : gated off in {self.flux.regime} regime (raw continuum value {self.flux.diffusion_raw:.2e})'
        )
        if self.instability_growth_mode:
            breakup_len_line = (
                f'Max integration len  : {self.breakup_length * 1e3:.2f} mm  (instability-growth mode; actual Lb set by amplitude event)'
            )
            ig_lines = [
                f'IG growth prefactor  : C_omega = {self.breakup_growth_prefactor:.4f}',
                f'IG viscous damping   : C_mu = {self.breakup_viscous_damping_coefficient:.2f}',
                f'IG initial ampl.     : a0/r = {self.breakup_initial_amplitude_fraction:.2e}',
                f'IG breakup threshold : beta_c = {self.breakup_final_amplitude_fraction:.2f}  (breakup when a >= beta_c * r)',
            ]
        else:
            breakup_len_line = (
                f'Selected breakup len   : {self.breakup_length * 1e3:.2f} mm  ({self.breakup_source}, {self.breakup_time * 1e6:.1f} us)'
            )
            ig_lines = []
        return [
            '=' * 60,
            'DIAGNOSTICS',
            '=' * 60,
            f'Net evap flux at nozzle: {self.flux.net:.2e} kg/m2/s',
            f'  HK limit             : {self.flux.hk:.2e} kg/m2/s',
            f'  {diffusion_line}',
            f'Cooling rate (1-D)     : {self.cooling_rate_1d:.2e} K/m',
            f'Re={self.reynolds_number:.0f}  We={self.weber_number:.2f}  Oh={self.ohnesorge_number:.4f}',
            f'Breakup mode           : {self.breakup_mode}',
            f'Breakup model          : {self.breakup_model_name}',
            f'Breakup correlation    : {self.breakup_model_formula}',
            f'Breakup tuning         : C_cal={params.breakup_calibration_factor:.2f}, C_Oh={params.breakup_viscous_coefficient:.2f}',
            breakup_len_line,
            f'Computed reference Lb  : {self.computed_breakup_length * 1e3:.2f} mm  ({self.computed_breakup_time * 1e6:.1f} us)',
            *ig_lines,
            f'Breakup caveat         : {self.breakup_model_limitations}',
            f'Fourier no. at breakup : {self.fourier_breakup:.3f}  (>1=isothermal, <0.1=strong gradient)',
            f'Delta relaxation len   : {self.delta_relax_length * 1e3:.2f} mm',
            f'Quasi-steady dT_surf   : {self.delta_t_quasisteady:.1f} K below mean',
            f'Knudsen number         : {self.kn_nozzle:.2e} ({classify_knudsen_regime(self.kn_nozzle)})',
            f'sigma(T_nozzle)        : {self.sigma_nozzle:.4f} N/m',
            f'mu(T_nozzle)           : {liquid_dynamic_viscosity(params.T_nozzle, params):.3e} Pa s',
            f'cp(T_nozzle)           : {self.cp_nozzle:.1f} J/kg/K',
            f'k(T_nozzle)            : {self.k_nozzle:.3f} W/m/K',
            f'Adiabatic freeze frac. : {self.adiabatic_freeze_fraction:.3f} of the jet if freezing starts at {params.T_freeze:.2f} K',
            'Velocity model         : constant axial speed; this is continuity-consistent for incompressible surface recession.',
            f'Freeze interpretation  : {params.switches.freeze_model} at {params.T_freeze:.2f} K; CNT is reported only on the liquid branch.',
        ]


@dataclass(frozen=True)
class SimulationResult:
    z: np.ndarray
    T_mean: np.ndarray
    r: np.ndarray
    Delta: np.ndarray
    T_surface: np.ndarray
    T_center: np.ndarray
    gamma_net: np.ndarray
    gamma_hk: np.ndarray
    gamma_diff: np.ndarray
    gamma_diff_raw: np.ndarray
    wave_factor: np.ndarray
    kn_profile: np.ndarray
    velocity: np.ndarray
    local_cooling_rate: np.ndarray
    m_evap_rate: np.ndarray
    m_evap_rate_hk: np.ndarray
    m_evap_rate_diff: np.ndarray
    M_dot_evap: float
    breakup_length: float
    breakup_source: str
    termination_reason: str
    termination_position: float
    freeze_position: float | None
    radius_guard_hit: bool
    temperature_guard_hit: bool
    delta_guard_hit: bool
    within_breakup: bool
    nozzle_diagnostics: NozzleDiagnostics

    @property
    def broke(self) -> bool:
        return self.termination_reason == 'breakup'

    @property
    def froze(self) -> bool:
        return self.termination_reason == 'freeze'

    def report_lines(self, params: JetParams) -> list[str]:
        reason_map = {
            'freeze': f'Jet reached empirical freeze onset at z = {self.termination_position * 1e3:.2f} mm',
            'breakup': f'Integration stopped at breakup: z = {self.termination_position * 1e3:.2f} mm',
            'radius_guard': f'Integration stopped at radius guard: z = {self.termination_position * 1e3:.2f} mm',
            'temperature_guard': f'Integration stopped at temperature guard: z = {self.termination_position * 1e3:.2f} mm',
            'delta_guard': f'Integration stopped at delta guard: z = {self.termination_position * 1e3:.2f} mm',
            'max_length': f'Reached max integration length: z = {self.termination_position * 1e3:.2f} mm',
        }
        return [
            '',
            '=' * 60,
            'SOLVING JET ODE',
            '=' * 60,
            reason_map[self.termination_reason],
            '',
            f'At z = {self.z[-1] * 1e3:.2f} mm:',
            f'  T_mean    = {self.T_mean[-1]:.2f} K',
            f'  T_surface = {self.T_surface[-1]:.2f} K',
            f'  T_center  = {self.T_center[-1]:.2f} K',
            f'  Delta     = {self.Delta[-1]:.2f} K',
            f'  r         = {self.r[-1] * 1e6:.3f} μm',
            '',
            f'Total evaporation rate: {self.M_dot_evap * 1e9:.3f} ng/s',
            f'Breakup mode         : {self.nozzle_diagnostics.breakup_mode}',
            f'Breakup model        : {self.nozzle_diagnostics.breakup_model_name}',
            f'Breakup source       : {self.breakup_source}',
            f'Breakup length       : {self.breakup_length * 1e3:.3f} mm',
        ]


def adiabatic_freeze_fraction(params: JetParams, T_surface: float) -> float:
    cp_local = liquid_heat_capacity(max(T_surface, params.T_freeze), params)
    sensible_deficit = max(273.15 - T_surface, 0.0) * cp_local
    return float(np.clip(sensible_deficit / params.h_fus, 0.0, 1.0))


def rayleigh_breakup_length(params: JetParams, T_ref_local: float | None = None) -> tuple[float, float, float, float]:
    """
    Laminar capillary breakup estimate using a Sterling-Sleicher style correlation.

    Assumptions:
    - Laminar, capillary-dominated jet breakup (Rayleigh/Tomotika regime).
    - Breakup length scales with sqrt(We) and increases mildly with Oh.
    - The baseline coefficient is params.breakup_correlation_coefficient (13 by default),
      matching the nominal Sterling-Sleicher laminar-jet pre-factor before calibration.
    - A user-visible calibration factor captures nozzle-specific disturbance level.
    - External aerodynamic atomization is out of scope.
    """
    T_eval = params.T_nozzle if T_ref_local is None else T_ref_local
    sigma_local = liquid_surface_tension(T_eval, params)
    mu_local = liquid_dynamic_viscosity(T_eval, params)
    if params.d_nozzle <= 0.0:
        raise ValueError(f'Laminar breakup correlation requires a positive nozzle diameter, got {params.d_nozzle}.')
    if params.rho_l <= 0.0:
        raise ValueError(f'Laminar breakup correlation requires a positive liquid density, got {params.rho_l}.')
    if sigma_local <= 0.0:
        raise ValueError(f'Laminar breakup correlation requires positive surface tension, got {sigma_local}.')
    We = params.rho_l * params.v_nozzle**2 * (2.0 * params.r_nozzle) / sigma_local
    if We <= 0.0:
        raise ValueError(f'Laminar breakup correlation requires a positive Weber number, got {We}.')
    Oh = mu_local / np.sqrt(params.rho_l * sigma_local * 2.0 * params.r_nozzle)
    breakup_length = (
        params.breakup_calibration_factor
        * params.breakup_correlation_coefficient
        * params.d_nozzle
        * np.sqrt(We)
        * (1.0 + params.breakup_viscous_coefficient * Oh)
    )
    breakup_time = breakup_length / max(params.v_nozzle, params.v_guard_min)
    return float(breakup_length), float(breakup_time), float(We), float(Oh)


def surface_wave_area_factor(params: JetParams, r: float, z: float, sigma_local: float) -> float:
    if not params.switches.use_surface_waves:
        return 1.0
    r_eff = max(r, params.r_guard_min)
    k_star = 0.697 / r_eff
    omega = np.sqrt(sigma_local / (params.rho_l * r_eff**3)) * 0.343
    eps = min(
        params.wave_seed_fraction * r_eff * np.exp(omega * z / max(params.v_nozzle, params.v_guard_min)),
        params.wave_amplitude_max * r_eff,
    )
    return float(1.0 + 0.5 * (eps * k_star) ** 2)


def compute_nozzle_diagnostics(params: JetParams) -> NozzleDiagnostics:
    computed_breakup_length, computed_breakup_time, We, Oh = rayleigh_breakup_length(params)
    breakup_mode_switch = params.switches.breakup_mode

    if breakup_mode_switch == 'instability_growth':
        # Actual breakup position is determined during ODE integration by amplitude event.
        # Use the max integration span as the positional fallback limit.
        breakup_length = 0.05
        breakup_time = breakup_length / max(params.v_nozzle, params.v_guard_min)
        breakup_source = 'instability-growth amplitude threshold'
        breakup_mode_str = 'instability_growth'
        model_name = INSTABILITY_GROWTH_MODEL_NAME
        model_formula = INSTABILITY_GROWTH_MODEL_FORMULA
        model_limitations = INSTABILITY_GROWTH_MODEL_LIMITATIONS
        instability_growth_mode = True
    elif breakup_mode_switch == 'fixed' or not params.switches.use_breakup_length_model:
        breakup_length = params.fixed_breakup_length
        breakup_time = breakup_length / max(params.v_nozzle, params.v_guard_min)
        breakup_source = 'fixed user value'
        breakup_mode_str = 'fixed user value'
        model_name = BREAKUP_MODEL_NAME
        model_formula = BREAKUP_MODEL_FORMULA
        model_limitations = BREAKUP_MODEL_LIMITATIONS
        instability_growth_mode = False
    else:  # 'correlation' (default) with use_breakup_length_model=True
        breakup_length = computed_breakup_length
        breakup_time = computed_breakup_time
        breakup_source = 'computed Sterling-Sleicher correlation'
        breakup_mode_str = 'computed correlation'
        model_name = BREAKUP_MODEL_NAME
        model_formula = BREAKUP_MODEL_FORMULA
        model_limitations = BREAKUP_MODEL_LIMITATIONS
        instability_growth_mode = False

    Re = params.rho_l * params.v_nozzle * params.d_nozzle / liquid_dynamic_viscosity(params.T_nozzle, params)
    cp_nozzle = liquid_heat_capacity(params.T_nozzle, params)
    sigma_nozzle = liquid_surface_tension(params.T_nozzle, params)
    k_nozzle = liquid_thermal_conductivity(params.T_nozzle, params)
    flux = evaporation_flux_components(params.T_nozzle, params.r_nozzle, params)
    alpha_th = k_nozzle / (params.rho_l * cp_nozzle)
    fourier_breakup = alpha_th * (breakup_length / params.v_nozzle) / params.r_nozzle**2
    delta_relax_length = params.rho_l * cp_nozzle * params.v_nozzle * params.r_nozzle**2 / (8.0 * k_nozzle)
    delta_t_quasisteady = flux.net * params.h_vap * params.r_nozzle / (4.0 * k_nozzle)
    cooling_rate_1d = 2.0 * flux.net * params.h_vap / (params.rho_l * params.v_nozzle * params.r_nozzle * cp_nozzle)
    kn_nozzle = knudsen_number(params.T_gas, params.P_chamber, params.r_nozzle, params)
    freeze_fraction = adiabatic_freeze_fraction(params, params.T_freeze)
    return NozzleDiagnostics(
        breakup_length=breakup_length,
        breakup_time=breakup_time,
        breakup_source=breakup_source,
        breakup_mode=breakup_mode_str,
        breakup_model_name=model_name,
        breakup_model_formula=model_formula,
        breakup_model_limitations=model_limitations,
        computed_breakup_length=computed_breakup_length,
        computed_breakup_time=computed_breakup_time,
        reynolds_number=float(Re),
        weber_number=We,
        ohnesorge_number=Oh,
        cp_nozzle=cp_nozzle,
        sigma_nozzle=sigma_nozzle,
        k_nozzle=k_nozzle,
        flux=flux,
        cooling_rate_1d=float(cooling_rate_1d),
        fourier_breakup=float(fourier_breakup),
        delta_relax_length=float(delta_relax_length),
        delta_t_quasisteady=float(delta_t_quasisteady),
        kn_nozzle=float(kn_nozzle),
        adiabatic_freeze_fraction=freeze_fraction,
        instability_growth_mode=instability_growth_mode,
        breakup_initial_amplitude_fraction=params.breakup_initial_amplitude_fraction,
        breakup_final_amplitude_fraction=params.breakup_final_amplitude_fraction,
        breakup_growth_prefactor=params.breakup_growth_prefactor,
        breakup_viscous_damping_coefficient=params.breakup_viscous_damping_coefficient,
    )


def unpack_state(y: np.ndarray, params: JetParams) -> tuple[float, float, float]:
    T_mean = float(y[0])
    r = float(y[1])
    Delta = float(y[2]) if params.switches.use_radial_profile else 0.0
    return T_mean, r, Delta


def _amplitude_state_index(params: JetParams) -> int:
    """Return the index of the disturbance amplitude in the state vector."""
    return 3 if params.switches.use_radial_profile else 2


def unpack_amplitude(y: np.ndarray, params: JetParams) -> float:
    """Extract disturbance amplitude from state vector; returns 0 if not present."""
    idx = _amplitude_state_index(params)
    return float(y[idx]) if idx < len(y) else 0.0


def local_instability_growth_rate(r: float, sigma: float, mu: float, rho: float, params: JetParams) -> float:
    """
    Reduced-order local Rayleigh instability growth rate with viscous correction.

    Uses the most-amplified Rayleigh mode prefactor calibrated by C_omega, and a
    viscous damping term proportional to the Ohnesorge number.

    Parameters
    ----------
    r      : local jet radius (m)
    sigma  : local surface tension (N/m)
    mu     : local dynamic viscosity (Pa s)
    rho    : liquid density (kg/m^3)
    params : JetParams carrying breakup_growth_prefactor and breakup_viscous_damping_coefficient

    Returns
    -------
    omega : growth rate (1/s), lower-bounded by MIN_BREAKUP_GROWTH_RATE
    """
    r_eff = max(r, params.r_guard_min)
    sigma_eff = max(sigma, params.sigma_min)
    denom_Oh = max(np.sqrt(rho * sigma_eff * r_eff), 1e-30)
    Oh = mu / denom_Oh
    omega_inv = params.breakup_growth_prefactor * np.sqrt(sigma_eff / (rho * r_eff**3))
    omega = omega_inv / max(1.0 + params.breakup_viscous_damping_coefficient * Oh, 1e-30)
    return max(float(omega), MIN_BREAKUP_GROWTH_RATE)


def surface_temperature_from_state(y: np.ndarray, params: JetParams) -> float:
    T_mean, _, Delta = unpack_state(y, params)
    return T_mean - Delta / 2.0


def guard_state(y: np.ndarray, params: JetParams) -> tuple[float, float, float, float, float]:
    T_mean, r, Delta = unpack_state(y, params)
    T_mean = clamp_temperature(np.nan_to_num(T_mean, nan=params.T_guard_min, posinf=params.T_guard_max, neginf=params.T_guard_min), params)
    r = float(np.nan_to_num(r, nan=params.r_guard_min, posinf=params.r_guard_min, neginf=params.r_guard_min))
    r = max(r, params.r_guard_min)
    Delta = float(np.nan_to_num(Delta, nan=0.0, posinf=params.delta_guard_max, neginf=0.0))
    Delta = float(np.clip(Delta, 0.0, params.delta_guard_max))
    T_surface = clamp_temperature(T_mean - Delta / 2.0, params)
    T_center = clamp_temperature(T_mean + Delta / 2.0, params)
    return T_mean, r, Delta, T_surface, T_center


def build_jet_ode(params: JetParams) -> Callable[[float, np.ndarray], list[float]]:
    instability_mode = params.switches.breakup_mode == 'instability_growth'

    def jet_ode(z: float, y: np.ndarray) -> list[float]:
        T_mean, r, Delta, T_surface, _ = guard_state(y, params)
        cp_local = liquid_heat_capacity(T_mean, params)
        k_local = liquid_thermal_conductivity(T_mean, params)
        sigma_local = liquid_surface_tension(T_surface, params)
        wave = surface_wave_area_factor(params, r, z, sigma_local)
        flux = evaporation_flux_components(T_surface, r, params)
        gamma = flux.net * wave
        q_s = gamma * params.h_vap
        v_local = params.v_nozzle
        dTmeandz = -2.0 * q_s / (params.rho_l * v_local * r * cp_local)
        drdz = -gamma / (params.rho_l * v_local)
        if params.switches.use_radial_profile:
            dDeltadz = (
                2.0 * q_s / (params.rho_l * cp_local * v_local * r)
                - 8.0 * k_local * Delta / (params.rho_l * cp_local * v_local * r**2)
            )
            result: list[float] = [float(dTmeandz), float(drdz), float(dDeltadz)]
        else:
            result = [float(dTmeandz), float(drdz)]

        if instability_mode:
            a = max(unpack_amplitude(y, params), 0.0)
            mu_local = liquid_dynamic_viscosity(T_surface, params)
            omega = local_instability_growth_rate(r, sigma_local, mu_local, params.rho_l, params)
            v_safe = max(v_local, params.v_guard_min)
            dadz = (omega / v_safe) * a
            result.append(float(dadz))

        return result

    return jet_ode


def build_events(params: JetParams, breakup_length: float) -> list[Callable[[float, np.ndarray], float]]:
    def freeze_event(z: float, y: np.ndarray) -> float:
        return surface_temperature_from_state(y, params) - params.T_freeze

    freeze_event.terminal = True
    freeze_event.direction = -1

    def breakup_event(z: float, y: np.ndarray) -> float:
        return z - breakup_length

    breakup_event.terminal = True
    breakup_event.direction = 1

    def radius_guard_event(z: float, y: np.ndarray) -> float:
        return unpack_state(y, params)[1] - params.r_guard_min

    radius_guard_event.terminal = True
    radius_guard_event.direction = -1

    def temperature_guard_event(z: float, y: np.ndarray) -> float:
        return surface_temperature_from_state(y, params) - params.T_guard_min

    temperature_guard_event.terminal = True
    temperature_guard_event.direction = -1

    def delta_guard_event(z: float, y: np.ndarray) -> float:
        return params.delta_guard_max - unpack_state(y, params)[2]

    delta_guard_event.terminal = True
    delta_guard_event.direction = -1

    events = [freeze_event, breakup_event, radius_guard_event, temperature_guard_event, delta_guard_event]

    if params.switches.breakup_mode == 'instability_growth':
        def amplitude_breakup_event(z: float, y: np.ndarray) -> float:
            a = unpack_amplitude(y, params)
            r = max(unpack_state(y, params)[1], params.r_guard_min)
            return a - params.breakup_final_amplitude_fraction * r

        amplitude_breakup_event.terminal = True
        amplitude_breakup_event.direction = 1
        events.append(amplitude_breakup_event)

    return events


def solve_jet(params: JetParams) -> SimulationResult:
    nozzle = compute_nozzle_diagnostics(params)
    instability_mode = params.switches.breakup_mode == 'instability_growth'

    if params.switches.use_radial_profile:
        y0_base = [params.T_nozzle, params.r_nozzle, 0.0]
    else:
        y0_base = [params.T_nozzle, params.r_nozzle]

    if instability_mode:
        a0 = params.breakup_initial_amplitude_fraction * params.r_nozzle
        y0 = y0_base + [a0]
    else:
        y0 = y0_base

    sol = solve_ivp(
        build_jet_ode(params),
        (0.0, 0.05),
        y0,
        events=build_events(params, nozzle.breakup_length),
        max_step=5e-6,
        dense_output=True,
        rtol=1e-6,
        atol=1e-9,
    )

    z = sol.t
    T_mean = sol.y[0]
    r = np.maximum(sol.y[1], params.r_guard_min)
    if params.switches.use_radial_profile:
        Delta = np.clip(sol.y[2], 0.0, params.delta_guard_max)
        T_surface = np.clip(T_mean - Delta / 2.0, params.T_guard_min, params.T_guard_max)
        T_center = np.clip(T_mean + Delta / 2.0, params.T_guard_min, params.T_guard_max)
    else:
        Delta = np.zeros_like(T_mean)
        T_surface = np.clip(T_mean, params.T_guard_min, params.T_guard_max)
        T_center = T_surface.copy()

    wave_factor = np.array([
        surface_wave_area_factor(params, rr, zz, liquid_surface_tension(Ts, params))
        for zz, rr, Ts in zip(z, r, T_surface)
    ])
    flux_components = [evaporation_flux_components(Ts, rr, params) for Ts, rr in zip(T_surface, r)]
    gamma_hk = wave_factor * np.array([fc.hk for fc in flux_components])
    gamma_net = wave_factor * np.array([fc.net for fc in flux_components])
    gamma_diff_raw = wave_factor * np.array([fc.diffusion_raw for fc in flux_components])
    gamma_diff = wave_factor * np.array([
        fc.diffusion if np.isfinite(fc.diffusion) else np.nan for fc in flux_components
    ])
    kn_profile = np.array([fc.kn for fc in flux_components])
    velocity = np.full_like(z, params.v_nozzle)
    local_cooling_rate = -np.gradient(T_surface, z, edge_order=1) if len(z) > 1 else np.zeros_like(z)

    m_evap_rate = 2.0 * np.pi * r * gamma_net
    m_evap_rate_hk = 2.0 * np.pi * r * gamma_hk
    m_evap_rate_diff = 2.0 * np.pi * r * gamma_diff_raw
    M_dot_evap = float(np.trapezoid(m_evap_rate, z))

    froze = sol.status == 1 and len(sol.t_events[0]) > 0
    broke_by_position = sol.status == 1 and len(sol.t_events[1]) > 0
    radius_guard_hit = sol.status == 1 and len(sol.t_events[2]) > 0
    temperature_guard_hit = sol.status == 1 and len(sol.t_events[3]) > 0
    delta_guard_hit = sol.status == 1 and len(sol.t_events[4]) > 0
    broke_by_amplitude = instability_mode and sol.status == 1 and len(sol.t_events[5]) > 0
    broke = broke_by_position or broke_by_amplitude

    if froze:
        termination_reason = 'freeze'
        termination_position = float(sol.t_events[0][0])
        freeze_position = termination_position
        result_breakup_length = nozzle.breakup_length
        result_breakup_source = nozzle.breakup_source
    elif broke_by_amplitude:
        termination_reason = 'breakup'
        termination_position = float(sol.t_events[5][0])
        freeze_position = None
        result_breakup_length = termination_position
        result_breakup_source = 'instability-growth amplitude threshold'
    elif broke_by_position:
        termination_reason = 'breakup'
        termination_position = float(sol.t_events[1][0])
        freeze_position = None
        result_breakup_length = nozzle.breakup_length
        result_breakup_source = nozzle.breakup_source
    elif radius_guard_hit:
        termination_reason = 'radius_guard'
        termination_position = float(z[-1])
        freeze_position = None
        result_breakup_length = nozzle.breakup_length
        result_breakup_source = nozzle.breakup_source
    elif temperature_guard_hit:
        termination_reason = 'temperature_guard'
        termination_position = float(z[-1])
        freeze_position = None
        result_breakup_length = nozzle.breakup_length
        result_breakup_source = nozzle.breakup_source
    elif delta_guard_hit:
        termination_reason = 'delta_guard'
        termination_position = float(z[-1])
        freeze_position = None
        result_breakup_length = nozzle.breakup_length
        result_breakup_source = nozzle.breakup_source
    else:
        termination_reason = 'max_length'
        termination_position = float(z[-1])
        freeze_position = None
        result_breakup_length = nozzle.breakup_length
        result_breakup_source = nozzle.breakup_source

    within_breakup = (
        termination_reason == 'breakup'
        if instability_mode
        else float(z[-1]) <= nozzle.breakup_length + 1e-12
    )

    return SimulationResult(
        z=z,
        T_mean=T_mean,
        r=r,
        Delta=Delta,
        T_surface=T_surface,
        T_center=T_center,
        gamma_net=gamma_net,
        gamma_hk=gamma_hk,
        gamma_diff=gamma_diff,
        gamma_diff_raw=gamma_diff_raw,
        wave_factor=wave_factor,
        kn_profile=kn_profile,
        velocity=velocity,
        local_cooling_rate=local_cooling_rate,
        m_evap_rate=m_evap_rate,
        m_evap_rate_hk=m_evap_rate_hk,
        m_evap_rate_diff=m_evap_rate_diff,
        M_dot_evap=M_dot_evap,
        breakup_length=result_breakup_length,
        breakup_source=result_breakup_source,
        termination_reason=termination_reason,
        termination_position=termination_position,
        freeze_position=freeze_position,
        radius_guard_hit=radius_guard_hit,
        temperature_guard_hit=temperature_guard_hit,
        delta_guard_hit=delta_guard_hit,
        within_breakup=within_breakup,
        nozzle_diagnostics=nozzle,
    )
