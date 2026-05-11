from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Union
from .solvents import SOLVENT_DATABASE, get_solvent_properties


@dataclass(frozen=True)
class ModelSwitches:
    use_radial_profile: bool = True
    use_surface_waves: bool = True
    use_temp_dependent_properties: bool = True
    use_back_pressure: bool = True
    use_diffusion_limit: bool = True
    use_breakup_length_model: bool = True
    freeze_model: str = 'empirical_backstop'
    breakup_mode: str = 'correlation'  # 'correlation', 'fixed', or 'instability_growth'


ParamUpdateValue = Union[float, int, bool, str, ModelSwitches]


@dataclass(frozen=True)
class JetParams:
    switches: ModelSwitches = field(default_factory=ModelSwitches)
    solvent: str = 'water'

    # Fundamental constants
    R_gas: float = 8.314
    kB: float = 1.380649e-23
    M: float = 0.01801528
    rho_l: float = 998.2
    h_vap: float = 2.45e6
    h_fus: float = 3.34e5
    T_critical: float = 647.096
    T_ref: float = 293.15
    P_ref: float = 101325.0
    D_v_ref: float = 2.6e-5
    d_molecule: float = 3.7e-10

    # Model validity / guards
    T_freeze: float = 263.15
    T_guard_min: float = 200.0
    T_guard_max: float = 373.15
    T_nucl_min: float = 233.15
    T_property_min: float = 235.0
    T_property_max: float = 373.15
    r_guard_min: float = 0.05e-6
    delta_guard_max: float = 80.0
    T_critical_offset: float = 1e-6
    sigma_min: float = 1e-4
    k_thermal_min: float = 0.15
    wave_seed_fraction: float = 0.01
    wave_amplitude_max: float = 0.9
    breakup_initial_amplitude_fraction: float = 1e-4
    breakup_final_amplitude_fraction: float = 0.3
    breakup_correlation_coefficient: float = 13.0
    breakup_calibration_factor: float = 0.65
    breakup_viscous_coefficient: float = 3.0
    breakup_growth_prefactor: float = 0.34
    breakup_viscous_damping_coefficient: float = 3.0
    fixed_breakup_length: float = 3e-3
    v_guard_min: float = 1e-12
    diffusion_limit_kn_threshold: float = 0.1

    # Operating conditions
    T_nozzle: float = 293.15
    d_nozzle: float = 20e-6
    Q_flow: float = 0.6e-6 / 60.0
    P_chamber: float = 1e-5 * 100.0
    P_back: float = 0.0
    T_gas: float = 293.15
    alpha_evap: float = 0.9

    def with_updates(self, **kwargs: ParamUpdateValue) -> 'JetParams':
        return replace(self, **kwargs)

    @property
    def T_property_critical_max(self) -> float:
        return self.T_critical - self.T_critical_offset

    @property
    def r_nozzle(self) -> float:
        return self.d_nozzle / 2.0

    @property
    def A_nozzle(self) -> float:
        return 3.141592653589793 * self.r_nozzle**2

    @property
    def v_nozzle(self) -> float:
        return self.Q_flow / self.A_nozzle

    @property
    def m_dot_total(self) -> float:
        return self.rho_l * self.Q_flow


def make_default_params(solvent: str = 'water') -> JetParams:
    return select_solvent(JetParams(), solvent)


def available_solvent_names() -> tuple[str, ...]:
    return tuple(props.name for props in SOLVENT_DATABASE.values())


def select_solvent(params: JetParams, solvent: str) -> JetParams:
    props = get_solvent_properties(solvent)
    return replace(
        params,
        solvent=props.name,
        M=props.molar_mass_kg_per_mol,
        rho_l=props.density_kg_per_m3,
        h_vap=props.latent_heat_vap_J_per_kg,
        h_fus=props.latent_heat_fus_J_per_kg,
        T_critical=props.critical_temperature_K,
        T_freeze=props.melting_point_K,
        D_v_ref=props.vapor_diffusivity_ref_m2_per_s,
        d_molecule=props.molecule_diameter_m,
        T_property_min=max(params.T_guard_min, props.melting_point_K - 25.0),
        T_property_max=min(params.T_guard_max, props.critical_temperature_K - 5.0),
        T_nucl_min=max(params.T_guard_min, props.melting_point_K - 40.0),
    )


_VALID_BREAKUP_MODES = frozenset({'correlation', 'fixed', 'instability_growth'})


def build_user_params(
    selected_solvent: str = 'water',
    use_computed_breakup_length: bool = True,
    fixed_breakup_length_mm: float = 3.0,
    breakup_calibration_factor: float = 0.65,
    breakup_viscous_coefficient: float = 3.0,
    breakup_mode: str = 'correlation',
    breakup_growth_prefactor: float = 0.34,
    breakup_viscous_damping_coefficient: float = 3.0,
    breakup_initial_amplitude_fraction: float = 1e-4,
    breakup_final_amplitude_fraction: float = 0.3,
) -> JetParams:
    if fixed_breakup_length_mm <= 0.0:
        raise ValueError('fixed_breakup_length_mm must be positive.')
    if breakup_calibration_factor <= 0.0:
        raise ValueError('breakup_calibration_factor must be positive.')
    if breakup_viscous_coefficient < 0.0:
        raise ValueError('breakup_viscous_coefficient must be non-negative.')
    if breakup_mode not in _VALID_BREAKUP_MODES:
        raise ValueError(f"breakup_mode must be one of {sorted(_VALID_BREAKUP_MODES)}, got {breakup_mode!r}.")
    if breakup_growth_prefactor <= 0.0:
        raise ValueError('breakup_growth_prefactor must be positive.')
    if breakup_viscous_damping_coefficient < 0.0:
        raise ValueError('breakup_viscous_damping_coefficient must be non-negative.')
    if not (0.0 < breakup_initial_amplitude_fraction < 1.0):
        raise ValueError('breakup_initial_amplitude_fraction must be strictly between 0 and 1.')
    if not (0.0 < breakup_final_amplitude_fraction <= 1.0):
        raise ValueError('breakup_final_amplitude_fraction must be between 0 (exclusive) and 1.')
    params = make_default_params(selected_solvent)
    return params.with_updates(
        switches=replace(params.switches, use_breakup_length_model=use_computed_breakup_length, breakup_mode=breakup_mode),
        fixed_breakup_length=fixed_breakup_length_mm * 1e-3,
        breakup_calibration_factor=breakup_calibration_factor,
        breakup_viscous_coefficient=breakup_viscous_coefficient,
        breakup_growth_prefactor=breakup_growth_prefactor,
        breakup_viscous_damping_coefficient=breakup_viscous_damping_coefficient,
        breakup_initial_amplitude_fraction=breakup_initial_amplitude_fraction,
        breakup_final_amplitude_fraction=breakup_final_amplitude_fraction,
    )


def initial_condition_lines(params: JetParams) -> list[str]:
    return [
        '=' * 60,
        'INITIAL CONDITIONS',
        '=' * 60,
        f'Solvent             : {params.solvent}',
        f'Nozzle diameter     : {params.d_nozzle * 1e6:.1f} um',
        f'Flow rate           : {params.Q_flow * 60 * 1e6:.2f} uL/min',
        f'Jet velocity        : {params.v_nozzle:.1f} m/s',
        f'Chamber pressure    : {params.P_chamber:.2e} Pa ({params.P_chamber / 100:.2e} mbar)',
        f'Vapor back pressure : {params.P_back:.2e} Pa',
        f'Nozzle temp         : {params.T_nozzle:.2f} K',
        f'Gas temp            : {params.T_gas:.2f} K',
    ]


def model_switch_lines(params: JetParams) -> list[str]:
    switches = params.switches
    bmode = switches.breakup_mode
    if bmode == 'instability_growth':
        breakup_mode_desc = 'instability_growth (amplitude-threshold; overrides use_breakup_length_model)'
    elif bmode == 'fixed':
        breakup_mode_desc = f'fixed ({params.fixed_breakup_length * 1e3:.2f} mm; overrides use_breakup_length_model)'
    elif switches.use_breakup_length_model:
        breakup_mode_desc = 'computed from laminar-jet correlation'
    else:
        breakup_mode_desc = f'fixed user value ({params.fixed_breakup_length * 1e3:.2f} mm)'
    return [
        f'use_radial_profile            : {switches.use_radial_profile}',
        f'use_surface_waves             : {switches.use_surface_waves}',
        f'use_temp_dependent_properties : {switches.use_temp_dependent_properties}',
        f'use_back_pressure             : {switches.use_back_pressure}',
        f'use_diffusion_limit           : {switches.use_diffusion_limit} (applies only for Kn < {params.diffusion_limit_kn_threshold:g})',
        f'breakup_mode                  : {breakup_mode_desc}',
        f'use_breakup_length_model      : {switches.use_breakup_length_model} (fixed fallback = {params.fixed_breakup_length * 1e3:.2f} mm)',
        f'breakup_correlation           : Lb/D = {params.breakup_calibration_factor:.2f} × {params.breakup_correlation_coefficient:.1f} × sqrt(We) × (1 + {params.breakup_viscous_coefficient:.1f} Oh)',
        f'breakup_limitations           : laminar capillary regime only; aerodynamic breakup and nozzle-forcing spectra are not modeled',
        f'instability_growth_model      : C_omega={params.breakup_growth_prefactor:.3f}, C_mu={params.breakup_viscous_damping_coefficient:.2f},'
        f' a0/r={params.breakup_initial_amplitude_fraction:.2e}, beta_c={params.breakup_final_amplitude_fraction:.2f}',
        f'freeze_model                  : {switches.freeze_model} (empirical freeze onset at {params.T_freeze:.2f} K; CNT remains a liquid-branch diagnostic)',
        f'velocity_assumption           : constant axial speed from incompressible surface recession continuity',
        f'alpha_evap                    : {params.alpha_evap}',
    ]
