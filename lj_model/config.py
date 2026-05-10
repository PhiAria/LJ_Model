from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class ModelSwitches:
    use_radial_profile: bool = True
    use_surface_waves: bool = True
    use_temp_dependent_properties: bool = True
    use_back_pressure: bool = True
    use_diffusion_limit: bool = True
    freeze_model: str = 'empirical_backstop'


@dataclass(frozen=True)
class JetParams:
    switches: ModelSwitches = field(default_factory=ModelSwitches)

    # Fundamental constants
    R_gas: float = 8.314
    kB: float = 1.380649e-23
    M: float = 0.018
    rho_l: float = 1000.0
    h_vap: float = 2.5e6
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

    def with_updates(self, **kwargs: float) -> 'JetParams':
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


def make_default_params() -> JetParams:
    return JetParams()


def initial_condition_lines(params: JetParams) -> list[str]:
    return [
        '=' * 60,
        'INITIAL CONDITIONS',
        '=' * 60,
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
    return [
        f'use_radial_profile            : {switches.use_radial_profile}',
        f'use_surface_waves             : {switches.use_surface_waves}',
        f'use_temp_dependent_properties : {switches.use_temp_dependent_properties}',
        f'use_back_pressure             : {switches.use_back_pressure}',
        f'use_diffusion_limit           : {switches.use_diffusion_limit} (applies only for Kn < {params.diffusion_limit_kn_threshold:g})',
        f'freeze_model                  : {switches.freeze_model} (empirical freeze onset at {params.T_freeze:.2f} K; CNT remains a liquid-branch diagnostic)',
        f'velocity_assumption           : constant axial speed from incompressible surface recession continuity',
        f'alpha_evap                    : {params.alpha_evap}',
    ]
