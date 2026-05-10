from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .config import JetParams
from .evaporation import classify_knudsen_regime
from .jet_model import SimulationResult
from .nucleation import NucleationResult, ParametricStudyResult, fmt_mm, fmt_range_mm


def status_line(label: str, ok: bool, ok_detail: str, warn_detail: str | None = None) -> str:
    detail = ok_detail if ok else (warn_detail or ok_detail)
    tag = 'OK' if ok else 'WARN'
    return f'{tag:<4} {label:<18} {detail}'


def severity_verdict(solution: SimulationResult, nucleation: NucleationResult) -> str:
    if solution.radius_guard_hit or solution.temperature_guard_hit or solution.delta_guard_hit:
        return 'ERROR (model violated numerical guard)'
    if solution.termination_reason == 'freeze':
        return 'OK (empirical freeze onset reached before breakup)'
    if np.isfinite(nucleation.z_frozen_50) and nucleation.z_frozen_50 <= solution.breakup_length + 1e-12:
        return 'OK (CNT median nucleation predicted before breakup)'
    if solution.within_breakup:
        return 'WARNING (no freezing or CNT median nucleation before breakup)'
    return 'ERROR (solution exceeded breakup window)'


def termination_label(solution: SimulationResult, nucleation: NucleationResult) -> str:
    if np.isfinite(nucleation.z_frozen_50) and nucleation.z_frozen_50 <= solution.z[-1] + 1e-12:
        return 'Termination context: CNT median nucleation reached on liquid branch'
    if solution.termination_reason == 'freeze':
        return 'Termination context: empirical freeze onset backstop'
    if solution.termination_reason == 'breakup':
        return f'Termination context: breakup threshold ({solution.breakup_source})'
    return f'Termination context: {solution.termination_reason.replace("_", " ")}'


def build_summary(params: JetParams, solution: SimulationResult, nucleation: NucleationResult, parametric: ParametricStudyResult) -> str:
    nozzle = solution.nozzle_diagnostics
    kn_end = solution.kn_profile[-1]
    property_temp_ok = solution.T_surface.min() >= params.T_property_min and solution.T_surface.max() <= params.T_property_max
    nucleation_fit_ok = solution.T_surface.min() >= params.T_nucl_min
    pressure_inputs_ok = 0.0 <= params.P_back <= max(params.P_chamber, 0.0)
    guards_ok = not any([solution.radius_guard_hit, solution.temperature_guard_hit, solution.delta_guard_hit])
    diffusion_ok = not params.switches.use_diffusion_limit or solution.nozzle_diagnostics.flux.diffusion_applied
    pressure_ratio_text = f'{params.P_back / params.P_chamber:.2f}' if params.P_chamber > 0 else 'N/A'

    validity_lines = [
        status_line('Breakup window', solution.within_breakup, f'end point {solution.z[-1] * 1e3:.2f} mm is within {solution.breakup_length * 1e3:.2f} mm', f'end point {solution.z[-1] * 1e3:.2f} mm exceeds {solution.breakup_length * 1e3:.2f} mm'),
        status_line('Property ranges', property_temp_ok, f'T_surface in [{solution.T_surface.min():.1f}, {solution.T_surface.max():.1f}] K', f'T_surface left [{params.T_property_min:.0f}, {params.T_property_max:.0f}] K fit range'),
        status_line('Nucleation fit', nucleation_fit_ok, f'min T_surface = {solution.T_surface.min():.1f} K', f'T_surface went below {params.T_nucl_min:.1f} K fit bound'),
        status_line('Pressure inputs', pressure_inputs_ok, f'P_back/P_chamber = {pressure_ratio_text}', 'Require 0 <= P_back <= P_chamber'),
        status_line('Solver guards', guards_ok, 'no numerical guard triggered', 'radius/temperature/Delta guard triggered'),
        status_line('Diffusion gate', diffusion_ok, f'continuum diffusion correction active at nozzle', f'Kn = {solution.nozzle_diagnostics.kn_nozzle:.2e} ({classify_knudsen_regime(solution.nozzle_diagnostics.kn_nozzle)}), so net flux falls back to HK'),
        status_line('Gas regime @ nozzle', True, f'Kn = {solution.nozzle_diagnostics.kn_nozzle:.2e} ({classify_knudsen_regime(solution.nozzle_diagnostics.kn_nozzle)})'),
        status_line('Gas regime @ end', True, f'Kn = {kn_end:.2e} ({classify_knudsen_regime(kn_end)})'),
    ]

    summary_lines = [
        'RESULTS SUMMARY',
        '',
        f'Verdict              : {severity_verdict(solution, nucleation)}',
        f'Termination          : {termination_label(solution, nucleation)}',
        f'Solvent              : {params.solvent}',
        f'Breakup mode         : {nozzle.breakup_mode}',
        f'Breakup model        : {nozzle.breakup_model_name}',
        f'Breakup correlation  : {nozzle.breakup_model_formula}',
        f'Breakup tuning       : C_cal={params.breakup_calibration_factor:.2f}, C_Oh={params.breakup_viscous_coefficient:.2f}',
        f'Breakup caveat       : {nozzle.breakup_model_limitations}',
        f'T_surface (end)      : {solution.T_surface[-1]:.2f} K',
        f'Breakup length       : {solution.breakup_length * 1e3:.2f} mm ({solution.breakup_source})',
        f'Computed ref. Lb     : {nozzle.computed_breakup_length * 1e3:.2f} mm',
        '',
        f'T_nozzle             : {params.T_nozzle:.2f} K',
        f'P_chamber            : {params.P_chamber:.2e} Pa',
        f'P_back               : {params.P_back:.2e} Pa',
        f'T_mean (end)         : {solution.T_mean[-1]:.2f} K',
        f'Delta (end)          : {solution.Delta[-1]:.2f} K',
        f'r (end)              : {solution.r[-1] * 1e6:.3f} μm',
        '',
        f'CNT z_frozen_50      : {fmt_mm(nucleation.z_frozen_50)}',
        f'CNT z_frozen_10      : {fmt_mm(nucleation.z_frozen_10)}',
        f'CNT z_frozen_90      : {fmt_mm(nucleation.z_frozen_90)}',
        f'Parametric CNT z50   : {fmt_range_mm(parametric.z50_arr_mm)}',
        f'P(liquid) at end     : {nucleation.P_survival[-1]:.4f}',
        '',
        f'Net evap rate        : {solution.M_dot_evap * 1e9:.3f} ng/s',
        f'Adiabatic freeze frac: {solution.nozzle_diagnostics.adiabatic_freeze_fraction:.3f}',
        '',
        'MODEL VALIDITY',
        *validity_lines,
    ]
    return '\n'.join(summary_lines)


def _add_breakup_marker(ax: plt.Axes, breakup_mm: float) -> None:
    ax.axvline(breakup_mm, color='black', ls='--', lw=1.2, alpha=0.8)
    xmin, xmax = ax.get_xlim()
    xmax = max(xmax, breakup_mm * 1.08)
    ax.set_xlim(min(0.0, xmin), xmax)
    if xmax > breakup_mm:
        ax.axvspan(breakup_mm, xmax, color='lightgray', alpha=0.18)
        ylim = ax.get_ylim()
        ax.text(
            breakup_mm + 0.02 * (xmax - ax.get_xlim()[0]),
            ylim[1] - 0.08 * (ylim[1] - ylim[0]),
            'post-breakup\n(model invalid)',
            fontsize=8,
            color='dimgray',
            va='top',
        )
        ax.set_ylim(ylim)


def create_figure(
    params: JetParams,
    solution: SimulationResult,
    nucleation: NucleationResult,
    parametric: ParametricStudyResult,
    output_path: str = 'jet_analysis.png',
) -> tuple[plt.Figure, np.ndarray, str]:
    summary = build_summary(params, solution, nucleation, parametric)
    nozzle = solution.nozzle_diagnostics
    fig, axes = plt.subplots(3, 3, figsize=(17, 13))
    fig.suptitle(
        f'Liquid jet model ({params.solvent}) | P_chamber={params.P_chamber:.2e} Pa | P_back={params.P_back:.2e} Pa | '
        f'radial profile={params.switches.use_radial_profile} | surface waves={params.switches.use_surface_waves} | '
        f'T-dependent props={params.switches.use_temp_dependent_properties} | breakup mode={nozzle.breakup_mode} | '
        f'model={nozzle.breakup_model_name}',
        fontsize=11,
        y=1.01,
    )

    z_mm = solution.z * 1e3
    breakup_mm = solution.breakup_length * 1e3

    ax = axes[0, 0]
    ax.plot(z_mm, solution.T_mean, 'b-', lw=2, label='Mean liquid T')
    ax.plot(z_mm, solution.T_surface, 'r-', lw=2, label='Surface T')
    ax.plot(z_mm, solution.T_center, 'b--', lw=1.5, alpha=0.6, label='Centerline T')
    ax.axhline(params.T_freeze, color='k', ls=':', lw=1.5, label=f'Empirical freeze onset ({params.T_freeze:.0f} K)')
    ax.axhline(273.15, color='gray', ls='--', lw=1, alpha=0.5, label='273.15 K')
    ax.set_xlabel('Axial distance, z (mm)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Liquid temperature evolution')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _add_breakup_marker(ax, breakup_mm)
    ax.text(0.02, 0.05, termination_label(solution, nucleation), transform=ax.transAxes, fontsize=8, va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax = axes[0, 1]
    ax.plot(z_mm, solution.Delta, 'purple', lw=2, label='Δ = T_center − T_surface')
    ax.fill_between(z_mm, 0.0, solution.Delta, alpha=0.15, color='purple')
    ax.set_xlabel('Axial distance, z (mm)')
    ax.set_ylabel('ΔT (K)')
    ax.set_title('Radial temperature contrast')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    if not params.switches.use_radial_profile:
        ax.text(0.5, 0.5, 'Radial profile OFF', transform=ax.transAxes, ha='center', va='center', color='gray', fontsize=12)
    _add_breakup_marker(ax, breakup_mm)

    ax = axes[0, 2]
    ax.plot(z_mm, solution.r * 1e6, 'g-', lw=2)
    ax.set_xlabel('Axial distance, z (mm)')
    ax.set_ylabel('Radius (μm)')
    ax.set_title('Jet radius evolution')
    ax.grid(alpha=0.3)
    _add_breakup_marker(ax, breakup_mm)

    ax = axes[1, 0]
    ax.plot(z_mm, nucleation.P_survival, 'darkorange', lw=2.5, label='P(liquid, not nucleated)')
    ax.axhline(0.5, color='k', ls='--', lw=1, label='50% frozen')
    ax.axhline(0.1, color='k', ls=':', lw=1, label='90% frozen')
    for zv, lab in [
        (nucleation.z_frozen_10, 'z_frozen_10'),
        (nucleation.z_frozen_50, 'z_frozen_50'),
        (nucleation.z_frozen_90, 'z_frozen_90'),
    ]:
        if np.isfinite(zv):
            ax.axvline(zv * 1e3, color='darkorange', ls=':', lw=1, alpha=0.8)
            ax.text(zv * 1e3 + 0.05, 0.55, lab, fontsize=8, color='darkorange')
    ax.set_xlabel('Axial distance, z (mm)')
    ax.set_ylabel('Survival probability (−)')
    ax.set_title('CNT survival on the liquid branch')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    _add_breakup_marker(ax, breakup_mm)
    ax.text(0.02, 0.05, nucleation.freeze_mechanism_label, transform=ax.transAxes, fontsize=8, va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax = axes[1, 1]
    mask = nucleation.J_arr > 0
    if mask.any():
        ax.semilogy(z_mm[mask], nucleation.J_arr[mask], 'darkorange', lw=2)
    else:
        ax.text(0.5, 0.5, 'CNT fit window not reached before termination', transform=ax.transAxes, ha='center', va='center', fontsize=10, color='dimgray')
    ax.set_xlabel('Axial distance, z (mm)')
    ax.set_ylabel('J (m⁻³ s⁻¹)')
    ax.set_title('Homogeneous nucleation rate')
    ax.grid(alpha=0.3, which='both')
    _add_breakup_marker(ax, breakup_mm)

    ax = axes[1, 2]
    finite_z50 = np.isfinite(parametric.z50_arr_mm)
    if finite_z50.any():
        ax.plot(parametric.T_range, parametric.z50_arr_mm, 'darkorange', lw=2, marker='o', ms=4, label='CNT median z50')
    else:
        ax.text(
            0.5,
            0.72,
            'CNT z₅₀ not reached — surface temperature stays above\nthe homogeneous nucleation window before termination.',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=9,
            color='dimgray',
        )
    ax.plot(parametric.T_range, parametric.z_hard_arr_mm, 'k--', lw=1.5, marker='s', ms=3, label=f'Empirical freeze onset ({params.T_freeze:.0f} K)')
    ax.plot(parametric.T_range, parametric.breakup_arr_mm, color='gray', ls=':', lw=1.5, label='Selected breakup length')
    ax.set_xlabel('Nozzle temperature, T_nozzle (K)')
    ax.set_ylabel('Axial position (mm)')
    ax.set_title('Freeze position vs nozzle temperature')
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(parametric.T_range, parametric.end_surface_temp_arr_K, color='steelblue', lw=1.5, alpha=0.8, label='T_surface at termination')
    ax2.set_ylabel('T_surface at termination (K)', color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, loc='best')

    ax = axes[2, 0]
    ax.plot(z_mm, solution.m_evap_rate * 1e9, 'teal', lw=2, label='Net')
    ax.plot(z_mm, solution.m_evap_rate_hk * 1e9, color='slateblue', ls='--', lw=1.5, label='Hertz-Knudsen limit')
    ax.plot(z_mm, solution.m_evap_rate_diff * 1e9, color='gray', ls=':', lw=1.5, label='Continuum diffusion reference')
    ax.set_xlabel('Axial distance, z (mm)')
    ax.set_ylabel('dṁ/dz (ng s⁻¹ m⁻¹)')
    ax.set_title('Local evaporation-rate limits')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _add_breakup_marker(ax, breakup_mm)

    ax = axes[2, 1]
    ax.semilogy(z_mm, solution.kn_profile, color='teal', lw=2)
    ax.axhline(0.1, color='gray', ls='--', lw=1, label='Kn = 0.1')
    ax.axhline(10.0, color='gray', ls=':', lw=1, label='Kn = 10')
    ax.set_xlabel('Axial distance, z (mm)')
    ax.set_ylabel('Knudsen number (−)')
    ax.set_title('Gas-regime evolution along the jet')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
    _add_breakup_marker(ax, breakup_mm)

    ax = axes[2, 2]
    ax.axis('off')
    ax.set_title('Results and model validity')
    ax.text(0.02, 0.98, summary, fontsize=8.5, family='monospace', va='top', transform=ax.transAxes)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig, axes, summary
