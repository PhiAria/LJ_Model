from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import cumulative_trapezoid

from .config import JetParams
from .jet_model import SimulationResult, solve_jet


@dataclass(frozen=True)
class NucleationResult:
    J_arr: np.ndarray
    integrand: np.ndarray
    cumulative_integral: np.ndarray
    P_survival: np.ndarray
    z_frozen_10: float
    z_frozen_50: float
    z_frozen_90: float
    freeze_mechanism_label: str

    def report_lines(self) -> list[str]:
        return [
            '',
            '=' * 60,
            'NUCLEATION STATISTICS  (CNT, homogeneous)',
            '=' * 60,
            f'z_frozen_10 (10% frozen) : {fmt_mm(self.z_frozen_10)}',
            f'z_frozen_50 (median)     : {fmt_mm(self.z_frozen_50)}',
            f'z_frozen_90 (90% frozen) : {fmt_mm(self.z_frozen_90)}',
            f'P_survival at end : {self.P_survival[-1]:.4f}',
            f'Interpretation     : {self.freeze_mechanism_label}',
        ]


@dataclass(frozen=True)
class ParametricStudyResult:
    T_range: np.ndarray
    z50_arr_mm: np.ndarray
    z_hard_arr_mm: np.ndarray
    breakup_arr_mm: np.ndarray
    end_surface_temp_arr_K: np.ndarray
    end_reason: tuple[str, ...]

    def report_lines(self) -> list[str]:
        return [
            f'Parametric study done. CNT z50 range: {fmt_range_mm(self.z50_arr_mm)}',
            f'Parametric end-surface range at termination: {self.end_surface_temp_arr_K.min():.2f} - {self.end_surface_temp_arr_K.max():.2f} K',
        ]


def nucleation_rate(T: float, params: JetParams) -> float:
    if T >= 273.15:
        return 0.0
    T_eval = max(T, params.T_nucl_min)
    log10_J = -906.7 + 8.502 * T_eval - 0.02657 * T_eval**2 + 2.766e-5 * T_eval**3
    return float(10 ** log10_J * 1e6)


def find_z_at_prob(z: np.ndarray, survival: np.ndarray, P_target: float) -> float:
    idx = np.searchsorted(-survival, -P_target)
    return float(z[idx]) if idx < len(z) else float('nan')


def fmt_mm(val: float) -> str:
    return f'{val * 1e3:.2f} mm' if np.isfinite(val) else 'not reached'


def fmt_range_mm(values_mm: np.ndarray) -> str:
    finite = values_mm[np.isfinite(values_mm)]
    if finite.size == 0:
        return 'not reached'
    return f'{finite.min():.2f} - {finite.max():.2f} mm'


def freeze_mechanism_text(solution: SimulationResult, nucleation_result: 'NucleationResult') -> str:
    if np.isfinite(nucleation_result.z_frozen_50) and nucleation_result.z_frozen_50 <= solution.z[-1] + 1e-12:
        return f'CNT median nucleation predicted by z ≈ {nucleation_result.z_frozen_50 * 1e3:.2f} mm'
    if solution.termination_reason == 'freeze':
        return 'Ended by empirical freeze onset before CNT median nucleation.'
    if solution.termination_reason == 'breakup':
        return f'Ended by breakup threshold ({solution.breakup_source}) before empirical freeze onset or CNT median nucleation.'
    return f'Ended by {solution.termination_reason.replace("_", " ")}.'


def compute_survival(params: JetParams, solution: SimulationResult) -> NucleationResult:
    J_arr = np.array([nucleation_rate(Ts, params) for Ts in solution.T_surface])
    integrand = J_arr * np.pi * solution.r**2 / np.maximum(solution.velocity, params.v_guard_min)
    cumulative_integral = np.zeros_like(solution.z)
    if len(solution.z) > 1:
        cumulative_integral[1:] = cumulative_trapezoid(integrand, solution.z)
    P_survival = np.exp(-cumulative_integral)
    provisional = NucleationResult(
        J_arr=J_arr,
        integrand=integrand,
        cumulative_integral=cumulative_integral,
        P_survival=P_survival,
        z_frozen_10=find_z_at_prob(solution.z, P_survival, 0.90),
        z_frozen_50=find_z_at_prob(solution.z, P_survival, 0.50),
        z_frozen_90=find_z_at_prob(solution.z, P_survival, 0.10),
        freeze_mechanism_label='',
    )
    return NucleationResult(
        J_arr=J_arr,
        integrand=integrand,
        cumulative_integral=cumulative_integral,
        P_survival=P_survival,
        z_frozen_10=provisional.z_frozen_10,
        z_frozen_50=provisional.z_frozen_50,
        z_frozen_90=provisional.z_frozen_90,
        freeze_mechanism_label=freeze_mechanism_text(solution, provisional),
    )


def run_parametric_study(base_params: JetParams, T_range: np.ndarray | None = None) -> ParametricStudyResult:
    T_values = np.linspace(274.0, 293.0, 25) if T_range is None else np.asarray(T_range, dtype=float)
    z50_arr_mm = []
    z_hard_arr_mm = []
    breakup_arr_mm = []
    end_surface_temp_arr_K = []
    reasons: list[str] = []

    for T_init in T_values:
        case_params = base_params.with_updates(T_nozzle=float(T_init))
        solution = solve_jet(case_params)
        nucleation = compute_survival(case_params, solution)
        breakup_arr_mm.append(solution.breakup_length * 1e3)
        end_surface_temp_arr_K.append(solution.T_surface[-1])
        z50_arr_mm.append(nucleation.z_frozen_50 * 1e3 if np.isfinite(nucleation.z_frozen_50) else float('nan'))
        if solution.freeze_position is not None:
            z_hard_arr_mm.append(solution.freeze_position * 1e3)
        else:
            z_hard_arr_mm.append(float('nan'))
        reasons.append(solution.termination_reason)

    return ParametricStudyResult(
        T_range=T_values,
        z50_arr_mm=np.array(z50_arr_mm),
        z_hard_arr_mm=np.array(z_hard_arr_mm),
        breakup_arr_mm=np.array(breakup_arr_mm),
        end_surface_temp_arr_K=np.array(end_surface_temp_arr_K),
        end_reason=tuple(reasons),
    )
