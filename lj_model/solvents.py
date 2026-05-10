from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SolventProperties:
    name: str
    aliases: tuple[str, ...]
    molar_mass_kg_per_mol: float
    density_kg_per_m3: float
    latent_heat_vap_J_per_kg: float
    latent_heat_fus_J_per_kg: float
    critical_temperature_K: float
    melting_point_K: float
    cp_ref_J_per_kg_K: float
    cp_temp_slope_J_per_kg_K2: float
    mu_ref_Pa_s: float
    mu_activation_K: float
    sigma_ref_N_per_m: float
    sigma_temp_coeff_N_per_m_K: float
    k_ref_W_per_m_K: float
    k_temp_coeff_W_per_m_K2: float
    antoine_A: float
    antoine_B: float
    antoine_C: float
    antoine_min_C: float
    antoine_max_C: float
    molecule_diameter_m: float
    vapor_diffusivity_ref_m2_per_s: float


# Approximate transport/thermo properties around room temperature.
# Organic-solvent h_fus and property slopes are lightweight engineering approximations
# to keep the model practical outside water while preserving explicit units.
SOLVENT_DATABASE: dict[str, SolventProperties] = {
    'water': SolventProperties(
        name='water',
        aliases=('water', 'h2o'),
        molar_mass_kg_per_mol=0.01801528,
        density_kg_per_m3=998.2,
        latent_heat_vap_J_per_kg=2.45e6,
        latent_heat_fus_J_per_kg=3.34e5,
        critical_temperature_K=647.096,
        melting_point_K=273.15,
        cp_ref_J_per_kg_K=4180.0,
        cp_temp_slope_J_per_kg_K2=0.5,
        mu_ref_Pa_s=1.002e-3,
        mu_activation_K=1800.0,
        sigma_ref_N_per_m=0.0720,
        sigma_temp_coeff_N_per_m_K=1.5e-4,
        k_ref_W_per_m_K=0.598,
        k_temp_coeff_W_per_m_K2=-1.5e-3,
        antoine_A=8.07131,
        antoine_B=1730.63,
        antoine_C=233.426,
        antoine_min_C=1.0,
        antoine_max_C=100.0,
        molecule_diameter_m=3.7e-10,
        vapor_diffusivity_ref_m2_per_s=2.6e-5,
    ),
    'acn': SolventProperties(
        name='ACN',
        aliases=('acn', 'acetonitrile'),
        molar_mass_kg_per_mol=0.04105,
        density_kg_per_m3=786.0,
        latent_heat_vap_J_per_kg=5.25e5,
        latent_heat_fus_J_per_kg=1.2e5,
        critical_temperature_K=545.5,
        melting_point_K=229.3,
        cp_ref_J_per_kg_K=2200.0,
        cp_temp_slope_J_per_kg_K2=2.0,
        mu_ref_Pa_s=3.5e-4,
        mu_activation_K=1300.0,
        sigma_ref_N_per_m=0.0280,
        sigma_temp_coeff_N_per_m_K=8.5e-5,
        k_ref_W_per_m_K=0.19,
        k_temp_coeff_W_per_m_K2=-3.0e-4,
        antoine_A=6.99464,
        antoine_B=1264.37,
        antoine_C=216.432,
        antoine_min_C=-10.0,
        antoine_max_C=90.0,
        molecule_diameter_m=4.6e-10,
        vapor_diffusivity_ref_m2_per_s=1.2e-5,
    ),
    'etoh': SolventProperties(
        name='EtOH',
        aliases=('etoh', 'ethanol'),
        molar_mass_kg_per_mol=0.04607,
        density_kg_per_m3=789.0,
        latent_heat_vap_J_per_kg=8.41e5,
        latent_heat_fus_J_per_kg=1.08e5,
        critical_temperature_K=514.0,
        melting_point_K=159.0,
        cp_ref_J_per_kg_K=2450.0,
        cp_temp_slope_J_per_kg_K2=2.2,
        mu_ref_Pa_s=1.08e-3,
        mu_activation_K=1600.0,
        sigma_ref_N_per_m=0.0223,
        sigma_temp_coeff_N_per_m_K=7.0e-5,
        k_ref_W_per_m_K=0.171,
        k_temp_coeff_W_per_m_K2=-3.0e-4,
        antoine_A=8.20417,
        antoine_B=1642.89,
        antoine_C=230.300,
        antoine_min_C=-30.0,
        antoine_max_C=80.0,
        molecule_diameter_m=4.6e-10,
        vapor_diffusivity_ref_m2_per_s=1.1e-5,
    ),
    'acetone': SolventProperties(
        name='Acetone',
        aliases=('acetone',),
        molar_mass_kg_per_mol=0.05808,
        density_kg_per_m3=784.0,
        latent_heat_vap_J_per_kg=5.18e5,
        latent_heat_fus_J_per_kg=9.5e4,
        critical_temperature_K=508.1,
        melting_point_K=178.5,
        cp_ref_J_per_kg_K=2160.0,
        cp_temp_slope_J_per_kg_K2=2.0,
        mu_ref_Pa_s=3.2e-4,
        mu_activation_K=1200.0,
        sigma_ref_N_per_m=0.0232,
        sigma_temp_coeff_N_per_m_K=7.0e-5,
        k_ref_W_per_m_K=0.16,
        k_temp_coeff_W_per_m_K2=-2.8e-4,
        antoine_A=7.11714,
        antoine_B=1210.595,
        antoine_C=229.664,
        antoine_min_C=-20.0,
        antoine_max_C=80.0,
        molecule_diameter_m=4.7e-10,
        vapor_diffusivity_ref_m2_per_s=1.1e-5,
    ),
    'meoh': SolventProperties(
        name='MeOH',
        aliases=('meoh', 'methanol'),
        molar_mass_kg_per_mol=0.03204,
        density_kg_per_m3=792.0,
        latent_heat_vap_J_per_kg=1.10e6,
        latent_heat_fus_J_per_kg=1.0e5,
        critical_temperature_K=512.6,
        melting_point_K=175.6,
        cp_ref_J_per_kg_K=2540.0,
        cp_temp_slope_J_per_kg_K2=2.3,
        mu_ref_Pa_s=5.4e-4,
        mu_activation_K=1450.0,
        sigma_ref_N_per_m=0.0225,
        sigma_temp_coeff_N_per_m_K=7.5e-5,
        k_ref_W_per_m_K=0.203,
        k_temp_coeff_W_per_m_K2=-3.2e-4,
        antoine_A=8.08097,
        antoine_B=1582.271,
        antoine_C=239.726,
        antoine_min_C=-20.0,
        antoine_max_C=90.0,
        molecule_diameter_m=4.0e-10,
        vapor_diffusivity_ref_m2_per_s=1.3e-5,
    ),
    'cyclohexane': SolventProperties(
        name='Cyclohexane',
        aliases=('cyclohexane',),
        molar_mass_kg_per_mol=0.08416,
        density_kg_per_m3=779.0,
        latent_heat_vap_J_per_kg=4.30e5,
        latent_heat_fus_J_per_kg=8.0e4,
        critical_temperature_K=553.5,
        melting_point_K=279.7,
        cp_ref_J_per_kg_K=1850.0,
        cp_temp_slope_J_per_kg_K2=1.5,
        mu_ref_Pa_s=1.02e-3,
        mu_activation_K=1650.0,
        sigma_ref_N_per_m=0.0253,
        sigma_temp_coeff_N_per_m_K=8.0e-5,
        k_ref_W_per_m_K=0.12,
        k_temp_coeff_W_per_m_K2=-2.5e-4,
        antoine_A=6.84930,
        antoine_B=1203.835,
        antoine_C=222.650,
        antoine_min_C=0.0,
        antoine_max_C=120.0,
        molecule_diameter_m=5.6e-10,
        vapor_diffusivity_ref_m2_per_s=8.5e-6,
    ),
}


def canonical_solvent_name(name: str) -> str:
    key = name.strip().lower()
    for canonical, props in SOLVENT_DATABASE.items():
        if key == canonical or key in props.aliases:
            return canonical
    supported = ', '.join(['ACN', 'water', 'EtOH', 'Acetone', 'MeOH', 'Cyclohexane'])
    raise ValueError(f'Unsupported solvent "{name}". Supported solvents: {supported}.')


def get_solvent_properties(name: str) -> SolventProperties:
    return SOLVENT_DATABASE[canonical_solvent_name(name)]
