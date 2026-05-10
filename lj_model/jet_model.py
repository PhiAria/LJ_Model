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


@dataclass(frozen=True)
class NozzleDiagnostics:
    breakup_length: float
    breakup_time: float
    breakup_source: str
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

    def report_lines(self, params: JetParams) -> list[str]:
        diffusion_line = (
            f'Diffusion limit      : {self.flux.diffusion:.2e} kg/m2/s'
            if self.flux.diffusion_applied
            else f'Diffusion limit      : gated off in {self.flux.regime} regime (raw continuum value {self.flux.diffusion_raw:.2e})'
        )
        return [
            '=' * 60,
            'DIAGNOSTICS',
            '=' * 60,
            f'Net evap flux at nozzle: {self.flux.net:.2e} kg/m2/s',
            f'  HK limit             : {self.flux.hk:.2e} kg/m2/s',
            f'  {diffusion_line}',
            f'Cooling rate (1-D)     : {self.cooling_rate_1d:.2e} K/m',
            f'Re={self.reynolds_number:.0f}  We={self.weber_number:.2f}  Oh={self.ohnesorge_number:.4f}',
            f'Breakup length ({self.breakup_source}) : {self.breakup_length * 1e3:.2f} mm  ({self.breakup_time * 1e6:.1f} us)',
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
            f'Breakup source       : {self.breakup_source}',
        ]


def adiabatic_freeze_fraction(params: JetParams, T_surface: float) -> float:
    cp_local = liquid_heat_capacity(max(T_surface, params.T_freeze), params)
    sensible_deficit = max(273.15 - T_surface, 0.0) * cp_local
    return float(np.clip(sensible_deficit / params.h_fus, 0.0, 1.0))


def rayleigh_breakup_length(params: JetParams, T_ref_local: float | None = None) -> tuple[float, float, float, float]:
    """
    Convective Rayleigh breakup estimate with viscous damping.

    Assumptions:
    - Axisymmetric capillary mode near kR ≈ 0.697 (Tomotika/Rayleigh fastest mode).
    - Disturbance grows exponentially while convecting with the mean jet speed.
    - Viscosity reduces the inviscid growth rate using 1/(1 + C_oh*Oh).
    - Initial/final disturbance amplitudes are user-tunable fractions of jet radius.
    """
    T_eval = params.T_nozzle if T_ref_local is None else T_ref_local
    sigma_local = liquid_surface_tension(T_eval, params)
    mu_local = liquid_dynamic_viscosity(T_eval, params)
    We = params.rho_l * params.v_nozzle**2 * (2.0 * params.r_nozzle) / sigma_local
    Oh = mu_local / np.sqrt(params.rho_l * sigma_local * 2.0 * params.r_nozzle)
    omega_capillary = np.sqrt(sigma_local / (params.rho_l * params.r_nozzle**3))
    omega_fastest = 0.343 * omega_capillary / (1.0 + params.breakup_viscous_coefficient * Oh)
    seed = np.clip(params.breakup_initial_amplitude_fraction, 1e-8, 0.9)
    final = np.clip(params.breakup_final_amplitude_fraction, seed + 1e-8, 0.95)
    growth_log = np.log(final / seed)
    tb = growth_log / max(omega_fastest, 1e-12)
    return float(params.v_nozzle * tb), float(tb), float(We), float(Oh)


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
    if params.switches.use_breakup_length_model:
        breakup_length = computed_breakup_length
        breakup_time = computed_breakup_time
        breakup_source = 'computed Rayleigh-Tomotika'
    else:
        breakup_length = params.fixed_breakup_length
        breakup_time = breakup_length / max(params.v_nozzle, params.v_guard_min)
        breakup_source = f'fixed user value ({params.fixed_breakup_length * 1e3:.2f} mm)'
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
    )


def unpack_state(y: np.ndarray, params: JetParams) -> tuple[float, float, float]:
    if params.switches.use_radial_profile:
        T_mean, r, Delta = y
    else:
        T_mean, r = y
        Delta = 0.0
    return float(T_mean), float(r), float(Delta)


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
            return [float(dTmeandz), float(drdz), float(dDeltadz)]
        return [float(dTmeandz), float(drdz)]

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

    return [freeze_event, breakup_event, radius_guard_event, temperature_guard_event, delta_guard_event]


def solve_jet(params: JetParams) -> SimulationResult:
    nozzle = compute_nozzle_diagnostics(params)
    y0 = [params.T_nozzle, params.r_nozzle, 0.0] if params.switches.use_radial_profile else [params.T_nozzle, params.r_nozzle]
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
    broke = sol.status == 1 and len(sol.t_events[1]) > 0
    radius_guard_hit = sol.status == 1 and len(sol.t_events[2]) > 0
    temperature_guard_hit = sol.status == 1 and len(sol.t_events[3]) > 0
    delta_guard_hit = sol.status == 1 and len(sol.t_events[4]) > 0
    if froze:
        termination_reason = 'freeze'
        termination_position = float(sol.t_events[0][0])
        freeze_position = termination_position
    elif broke:
        termination_reason = 'breakup'
        termination_position = float(sol.t_events[1][0])
        freeze_position = None
    elif radius_guard_hit:
        termination_reason = 'radius_guard'
        termination_position = float(z[-1])
        freeze_position = None
    elif temperature_guard_hit:
        termination_reason = 'temperature_guard'
        termination_position = float(z[-1])
        freeze_position = None
    elif delta_guard_hit:
        termination_reason = 'delta_guard'
        termination_position = float(z[-1])
        freeze_position = None
    else:
        termination_reason = 'max_length'
        termination_position = float(z[-1])
        freeze_position = None

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
        breakup_length=nozzle.breakup_length,
        breakup_source=nozzle.breakup_source,
        termination_reason=termination_reason,
        termination_position=termination_position,
        freeze_position=freeze_position,
        radius_guard_hit=radius_guard_hit,
        temperature_guard_hit=temperature_guard_hit,
        delta_guard_hit=delta_guard_hit,
        within_breakup=float(z[-1]) <= nozzle.breakup_length + 1e-12,
        nozzle_diagnostics=nozzle,
    )
