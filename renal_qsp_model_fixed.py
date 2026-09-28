"""Fixed renal QSP model wrapper for GSA.

This file keeps the original modular kernels from renal_qsp_modular_kernels.ipynb
for the Hallow renal QSP model and only adds a light DataFrame/UACR/GSA wrapper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

STATE_NAMES = [
    'AngI','AngII','AT1_bound_AngII','AT2_bound_AngII','plasma_renin_concentration',
    'blood_volume_L','extracellular_fluid_volume','sodium_amount','ECF_sodium_amount',
    'tubulo_glomerular_feedback_effect','normalized_aldosterone_level_delayed',
    'preafferent_pressure_autoreg_signal','glomerular_pressure_autoreg_signal',
    'cardiac_output_delayed','CO_error','Na_concentration_error',
    'normalized_vasopressin_concentration_delayed','F0_TGF','P_bowmans',
    'oncotic_pressure_difference','SN_macula_densa_Na_flow_delayed','serum_creatinine'
]
IDX = {k:i for i,k in enumerate(STATE_NAMES)}

def safe_exp(x):
    return np.exp(np.clip(x, -60.0, 60.0))

def safe_log(x):
    return np.log(max(x, 1e-12))

def safe_log10(x):
    return np.log10(max(x, 1e-12))

def calc_nom_params():
    p = {}
    # constants and conversions first
    p.update(dict(
        nL_mL=1e6, dl_ml=0.01, L_dL=10, L_mL=1000, L_m3=0.001,
        g_mg=0.001, ng_mg=1e-6, secs_mins=60, min_hr=60, hr_day=24,
        min_day=1440, MW_creatinine=113.12, Pi=3.1416, pi=3.14,
        viscosity_length_constant=1.5e-9, gamma=1.16667e-5,
        mmHg_Nperm2_conv=133.32,
        ECF_scale_species=1, BV_scale_species=1, water_intake_species_scale=1,
        CO_scale_species=1,
    ))
    # systemic
    p.update(dict(
        nominal_map_setpoint=93, CO_nom=5, ECF_nom=15, blood_volume_nom=5,
        Na_intake_rate=100/24/60, nom_water_intake=2.1, ref_Na_concentration=140,
        plasma_protein_concentration=7, equilibrium_serum_creatinine=0.92,
        P_venous=4, R_venous=3.4, nom_right_atrial_pressure=0.87,
        nom_mean_filling_pressure=7, venous_compliance=0.13,
    ))
    # renal
    p.update(dict(
        nom_renal_blood_flow_L_min=1, baseline_nephrons=2e6, nom_Kf=3.9,
        nom_oncotic_pressure_difference=28, P_renal_vein=4,
        nom_preafferent_arteriole_resistance=19,
        nom_afferent_diameter=1.5e-5, nom_efferent_diameter=1.1e-5,
        Dc_pt_nom=27e-6, Dc_lh=17e-6, Dc_dt=17e-6, Dc_cd=22e-6,
        L_pt_s1_nom=0.005, L_pt_s2_nom=0.005, L_pt_s3_nom=0.004,
        L_lh_des=0.01, L_lh_asc=0.01, L_dct=0.005,
        tubular_compliance=0.2, Pc_pt_mmHg=14, Pc_lh_des_mmHg=10.5,
        Pc_lh_asc_mmHg=7, Pc_dt_mmHg=3, Pc_cd_mmHg=2,
        P_interstitial_mmHg=5, nominal_pt_na_reabsorption=0.7,
        nominal_loh_na_reabsorption=0.8, nominal_dt_na_reabsorption=0.5,
        LoH_flow_dependence=1,
    ))
    p['L_cd'] = p['L_lh_des']
    # RAAS pathway nominal
    p.update(dict(
        concentration_to_renin_activity_conversion_plasma=61,
        nominal_equilibrium_PRA=1000,
        nominal_equilibrium_AngI=7.5,
        nominal_equilibrium_AngII=4.75,
        nominal_renin_half_life=0.1733,
        nominal_AngI_half_life=0.5/60,
        nominal_AngII_half_life=0.66/60,
        nominal_AT1_bound_AngII_half_life=12/60,
        nominal_AT2_bound_AngII_half_life=12/60,
        ACE_chymase_fraction=0.95,
        fraction_AT1_bound_AngII=0.75,
    ))
    # equilibrium-derived parameters
    p['nom_preafferent_pressure'] = p['nominal_map_setpoint'] - p['nom_renal_blood_flow_L_min'] * p['nom_preafferent_arteriole_resistance']
    p['nom_glomerular_pressure'] = p['nom_preafferent_pressure'] - p['nom_renal_blood_flow_L_min'] * (p['L_m3'] * p['viscosity_length_constant'] / (p['nom_afferent_diameter']**4) / p['baseline_nephrons'])
    p['nom_postglomerular_pressure'] = p['nom_preafferent_pressure'] - p['nom_renal_blood_flow_L_min'] * (p['L_m3'] * p['viscosity_length_constant'] * (1/p['nom_afferent_diameter']**4 + 1/p['nom_efferent_diameter']**4) / p['baseline_nephrons'])
    p['RIHP0'] = p['nom_postglomerular_pressure']
    p['nom_GFR'] = p['nom_Kf'] * (p['nom_glomerular_pressure'] - p['nom_oncotic_pressure_difference'] - (p['Pc_pt_mmHg'] + p['P_interstitial_mmHg'])) / p['nL_mL'] * p['baseline_nephrons']
    p['nom_filtered_sodium_load'] = p['nom_GFR'] / p['L_mL'] * p['ref_Na_concentration']
    p['nom_PT_Na_outflow'] = p['nom_filtered_sodium_load'] * (1-p['nominal_pt_na_reabsorption'])
    p['nom_Na_in_AscLoH'] = p['nom_PT_Na_outflow'] / p['baseline_nephrons']
    p['AscLoH_Reab_Rate_nominal'] = (2*p['nominal_loh_na_reabsorption']*p['nom_Na_in_AscLoH'])/p['L_lh_des']
    p['nom_LoH_Na_outflow'] = p['nom_PT_Na_outflow'] * (1-p['nominal_loh_na_reabsorption'])
    p['nom_DT_Na_outflow'] = p['nom_LoH_Na_outflow'] * (1-p['nominal_dt_na_reabsorption'])
    p['nominal_cd_na_reabsorption'] = 1 - p['Na_intake_rate']/p['nom_DT_Na_outflow']
    p['nom_RVR'] = (p['nominal_map_setpoint'] - p['P_venous'])/p['nom_renal_blood_flow_L_min']
    p['nom_peritubular_resistance'] = p['nom_RVR'] - (p['nom_preafferent_arteriole_resistance'] + p['L_m3']*p['viscosity_length_constant']*(1/p['nom_afferent_diameter']**4+1/p['nom_efferent_diameter']**4)/p['baseline_nephrons'])
    p['nom_TPR'] = p['nominal_map_setpoint']/p['CO_nom']
    p['nom_systemic_arterial_resistance'] = p['nom_TPR'] - p['R_venous']
    p['creatinine_synthesis_rate'] = p['equilibrium_serum_creatinine'] * p['dl_ml'] * p['nom_GFR']
    p['nominal_equilibrium_PRC'] = p['nominal_equilibrium_PRA']/p['concentration_to_renin_activity_conversion_plasma']
    p['nominal_AngI_degradation_rate'] = np.log(2)/p['nominal_AngI_half_life']
    p['nominal_AngII_degradation_rate'] = np.log(2)/p['nominal_AngII_half_life']
    p['nominal_AT1_bound_AngII_degradation_rate'] = np.log(2)/p['nominal_AT1_bound_AngII_half_life']
    p['nominal_AT2_bound_AngII_degradation_rate'] = np.log(2)/p['nominal_AT2_bound_AngII_half_life']
    p['nominal_ACE_activity'] = p['ACE_chymase_fraction']*(p['nominal_equilibrium_PRA'] - p['nominal_AngI_degradation_rate']*p['nominal_equilibrium_AngI'])/p['nominal_equilibrium_AngI']
    p['nominal_chymase_activity'] = (1-p['ACE_chymase_fraction'])*(p['nominal_equilibrium_PRA'] - p['nominal_AngI_degradation_rate']*p['nominal_equilibrium_AngI'])/p['nominal_equilibrium_AngI']
    p['nominal_AT1_receptor_binding_rate'] = p['fraction_AT1_bound_AngII']*(p['nominal_equilibrium_AngI']*(p['nominal_ACE_activity']+p['nominal_chymase_activity']) - p['nominal_AngII_degradation_rate']*p['nominal_equilibrium_AngII'])/p['nominal_equilibrium_AngII']
    p['nominal_AT2_receptor_binding_rate'] = (1-p['fraction_AT1_bound_AngII'])*(p['nominal_equilibrium_AngI']*(p['nominal_ACE_activity']+p['nominal_chymase_activity']) - p['nominal_AngII_degradation_rate']*p['nominal_equilibrium_AngII'])/p['nominal_equilibrium_AngII']
    p['nominal_equilibrium_AT1_bound_AngII'] = p['nominal_equilibrium_AngII']*p['nominal_AT1_receptor_binding_rate']/p['nominal_AT1_bound_AngII_degradation_rate']
    p['nominal_equilibrium_AT2_bound_AngII'] = p['nominal_equilibrium_AngII']*p['nominal_AT2_receptor_binding_rate']/p['nominal_AT2_bound_AngII_degradation_rate']
    # fitted / regulatory params
    p.update(dict(
        AT1_svr_slope=0, AT1_preaff_scale=0.5, AT1_preaff_slope=7,
        AT1_aff_scale=0.5, AT1_aff_slope=7, AT1_eff_scale=0.3, AT1_eff_slope=7,
        AT1_PT_scale=0.1, AT1_PT_slope=7, AT1_aldo_slope=0.05,
        nominal_aldosterone_concentration=85, aldo_DCT_scale=0, aldo_DCT_slope=0.5,
        aldo_CD_scale=0.3, aldo_CD_slope=0.5, aldo_renin_slope=-0.05,
        Q_water=1, Q_Na=1, Na_controller_gain=2, Kp_VP=0.05, Ki_VP=0.00002,
        nom_ADH_urea_permeability=0.98, nom_ADH_water_permeability=0.98,
        nominal_vasopressin_conc=4, water_intake_vasopressin_scale=0,
        water_intake_vasopressin_slope=-0.5,
        S_tubulo_glomerular_feedback=0.7, F_md_scale_tubulo_glomerular_feedback=6,
        MD_Na_concentration_setpoint=62.4,
        md_renin_A=1, md_renin_tau=2,
        preaff_diameter_range=0.25, afferent_diameter_range=1.2e-5,
        efferent_diameter_range=3e-6, preaff_signal_nonlin_scale=3,
        afferent_signal_nonlin_scale=3, efferent_signal_nonlin_scale=3,
        AngI_half_life=0.008333, AngII_half_life=0.011, AT1_bound_AngII_half_life=0.2,
        AT1_PRC_slope=-1.2, AT1_PRC_yint=0, AT2_bound_AngII_half_life=0.2,
        nominal_ACE_activity=48.9, nominal_AT1_receptor_binding_rate=12.1,
        nominal_AT2_receptor_binding_rate=4.0, nominal_chymase_activity=1.25,
        nominal_equilibrium_AT1_bound_AngII=16.63, nominal_equilibrium_PRC=16.4,
        renin_half_life=0.1733,
        C_aldo_secretion=1000, C_P_bowmans=1000, C_P_oncotic=1000,
        C_tgf_reset=0, C_cardiac_output_delayed=0.001, C_co_error=0.00001,
        C_vasopressin_delay=1, C_md_flow=0.001, C_tgf=1,
        C_na_excretion_na_amount=-1, C_na_intake_na_amount=1,
        C_urine_flow_ecf_volume=-1, C_water_intake_ecf_volume=1,
        C_Na_error=1/6, C_serum_creatinine=1,
        HCTZ_effect_on_DT_Na_reabs=1, HCTZ_effect_on_renin_secretion=1,
        DRI_effect_on_PRA=1, CCB_effect_on_preafferent_resistance=1,
        CCB_effect_on_afferent_resistance=1, CCB_effect_on_efferent_resistance=1,
        MR_antagonist_effect_on_aldo_MR=1, pct_target_inhibition_ARB=0,
        pct_target_inhibition_ACEi=0,
        tissue_autoreg_scale=1, Kp_CO=1.5, Ki_CO=30,
        K_Na_ratio_effect_on_aldo=1, gp_autoreg_scale=0,
        preaff_autoreg_scale=0.5, myogenic_steepness=2,
        max_pt_reabs_rate=0.995, pressure_natriuresis_PT_scale=3,
        pressure_natriuresis_PT_slope=1, pressure_natriuresis_LoH_scale=3,
        pressure_natriuresis_LoH_slope=1, pressure_natriuresis_DCT_scale=3,
        pressure_natriuresis_DCT_slope=1, max_cd_reabs_rate=0.995,
        pressure_natriuresis_CD_scale=3, pressure_natriuresis_CD_slope=1,
    ))
    return p

def initial_conditions(p):
    y0 = np.zeros(len(STATE_NAMES))
    initial = dict(
        AngI=8.164,
        AngII=5.17,
        AT1_bound_AngII=16.6,
        AT2_bound_AngII=5.5,
        plasma_renin_concentration=17.845,
        blood_volume_L=p['blood_volume_nom'],
        extracellular_fluid_volume=p['ECF_nom'],
        sodium_amount=p['blood_volume_nom']*p['ref_Na_concentration'],
        ECF_sodium_amount=p['ECF_nom']*p['ref_Na_concentration'],
        tubulo_glomerular_feedback_effect=1,
        normalized_aldosterone_level_delayed=1,
        preafferent_pressure_autoreg_signal=1,
        glomerular_pressure_autoreg_signal=1,
        cardiac_output_delayed=p['CO_nom'],
        CO_error=0,
        Na_concentration_error=0,
        normalized_vasopressin_concentration_delayed=1,
        F0_TGF=p['nom_LoH_Na_outflow'],
        P_bowmans=p['Pc_pt_mmHg'],
        oncotic_pressure_difference=p['nom_oncotic_pressure_difference'],
        SN_macula_densa_Na_flow_delayed=p['nom_LoH_Na_outflow']/p['baseline_nephrons'],
        serum_creatinine=p['equilibrium_serum_creatinine']*p['blood_volume_nom'],
    )
    for k,v in initial.items():
        y0[IDX[k]] = v
    return y0

def protocol_value(t, p, protocol, name, default=1.0):
    if protocol is None or name not in protocol:
        return default
    v = protocol[name]
    return v(t) if callable(v) else v



def unpack_state_and_protocol(t, y, p, protocol=None):
    """
    Unpack the ODE state vector y into named variables,
    apply numerical guards, and add protocol-dependent inputs.

    Inputs:
        t        : current time, minutes
        y        : state vector
        p        : parameter dictionary
        protocol : optional perturbation dictionary

    Returns:
        ctx      : shared context dictionary used by all modules
    """

    ctx = {
        "t": t,
        "y": y,
        "p": p,
        "protocol": protocol,
    }

    # RAAS states
    AngI = max(y[IDX["AngI"]], 1e-12)
    AngII = max(y[IDX["AngII"]], 1e-12)
    AT1_bound_AngII = max(y[IDX["AT1_bound_AngII"]], 1e-12)
    AT2_bound_AngII = max(y[IDX["AT2_bound_AngII"]], 1e-12)
    plasma_renin_concentration = max(
        y[IDX["plasma_renin_concentration"]],
        1e-12,
    )

    # Body fluid and sodium states
    blood_volume_L = max(y[IDX["blood_volume_L"]], 1e-6)
    extracellular_fluid_volume = max(
        y[IDX["extracellular_fluid_volume"]],
        1e-6,
    )
    sodium_amount = max(y[IDX["sodium_amount"]], 1e-9)
    ECF_sodium_amount = max(y[IDX["ECF_sodium_amount"]], 1e-9)

    # Feedback and delay states
    tubulo_glomerular_feedback_effect = max(
        y[IDX["tubulo_glomerular_feedback_effect"]],
        1e-6,
    )

    normalized_aldosterone_level_delayed = max(
        y[IDX["normalized_aldosterone_level_delayed"]],
        1e-9,
    )

    preafferent_pressure_autoreg_signal = max(
        y[IDX["preafferent_pressure_autoreg_signal"]],
        1e-6,
    )

    glomerular_pressure_autoreg_signal = max(
        y[IDX["glomerular_pressure_autoreg_signal"]],
        1e-6,
    )

    cardiac_output_delayed = y[IDX["cardiac_output_delayed"]]
    CO_error = y[IDX["CO_error"]]
    Na_concentration_error = y[IDX["Na_concentration_error"]]

    normalized_vasopressin_concentration_delayed = max(
        y[IDX["normalized_vasopressin_concentration_delayed"]],
        1e-6,
    )

    # Filtration-related dynamic states
    F0_TGF = y[IDX["F0_TGF"]]

    P_bowmans = max(
        y[IDX["P_bowmans"]],
        1e-6,
    )

    oncotic_pressure_difference = max(
        y[IDX["oncotic_pressure_difference"]],
        1e-6,
    )

    SN_macula_densa_Na_flow_delayed = max(
        y[IDX["SN_macula_densa_Na_flow_delayed"]],
        1e-18,
    )

    # Creatinine state
    serum_creatinine = max(
        y[IDX["serum_creatinine"]],
        1e-12,
    )

    # Protocol-dependent input.
    # Default is baseline Na intake.
    Na_intake_rate = p["Na_intake_rate"] * protocol_value(
        t,
        p,
        protocol,
        "Na_intake_multiplier",
        1.0,
    )

    ctx.update(
        {
            "AngI": AngI,
            "AngII": AngII,
            "AT1_bound_AngII": AT1_bound_AngII,
            "AT2_bound_AngII": AT2_bound_AngII,
            "plasma_renin_concentration": plasma_renin_concentration,
            "blood_volume_L": blood_volume_L,
            "extracellular_fluid_volume": extracellular_fluid_volume,
            "sodium_amount": sodium_amount,
            "ECF_sodium_amount": ECF_sodium_amount,
            "tubulo_glomerular_feedback_effect": tubulo_glomerular_feedback_effect,
            "normalized_aldosterone_level_delayed": normalized_aldosterone_level_delayed,
            "preafferent_pressure_autoreg_signal": preafferent_pressure_autoreg_signal,
            "glomerular_pressure_autoreg_signal": glomerular_pressure_autoreg_signal,
            "cardiac_output_delayed": cardiac_output_delayed,
            "CO_error": CO_error,
            "Na_concentration_error": Na_concentration_error,
            "normalized_vasopressin_concentration_delayed": normalized_vasopressin_concentration_delayed,
            "F0_TGF": F0_TGF,
            "P_bowmans": P_bowmans,
            "oncotic_pressure_difference": oncotic_pressure_difference,
            "SN_macula_densa_Na_flow_delayed": SN_macula_densa_Na_flow_delayed,
            "serum_creatinine": serum_creatinine,
            "Na_intake_rate": Na_intake_rate,
        }
    )

    return ctx



def compute_systemic_hemodynamics(ctx):
    """
    Systemic hemodynamics module.

    Inputs required in ctx:
        p
        AT1_bound_AngII
        cardiac_output_delayed
        CO_error
        blood_volume_L

    Outputs added to ctx:
        tissue_autoregulation_signal
        AT1_bound_AngII_effect_on_SVR
        systemic_arterial_resistance
        resistance_to_venous_return
        mean_filling_pressure
        cardiac_output
        total_peripheral_resistance
        mean_arterial_pressure_MAP
    """

    p = ctx["p"]

    AT1_bound_AngII = ctx["AT1_bound_AngII"]
    cardiac_output_delayed = ctx["cardiac_output_delayed"]
    CO_error = ctx["CO_error"]
    blood_volume_L = ctx["blood_volume_L"]

    # Whole-body autoregulation of systemic vascular resistance.
    # This tries to keep cardiac output near CO_nom.
    tissue_autoregulation_signal = max(
        0.1,
        1
        + p["tissue_autoreg_scale"]
        * (
            (p["Kp_CO"] / p["CO_scale_species"])
            * (cardiac_output_delayed - p["CO_nom"])
            + (p["Ki_CO"] / p["CO_scale_species"])
            * CO_error
        ),
    )

    # AT1-bound AngII effect on systemic vascular resistance.
    AT1_svr_int = (
        1
        - p["AT1_svr_slope"]
        * p["nominal_equilibrium_AT1_bound_AngII"]
    )

    AT1_bound_AngII_effect_on_SVR = (
        AT1_svr_int
        + p["AT1_svr_slope"] * AT1_bound_AngII
    )

    # Systemic arterial resistance.
    systemic_arterial_resistance = (
        p["nom_systemic_arterial_resistance"]
        * tissue_autoregulation_signal
        * AT1_bound_AngII_effect_on_SVR
    )

    # Resistance to venous return.
    resistance_to_venous_return = (
        8 * p["R_venous"] + systemic_arterial_resistance
    ) / 31

    # Blood volume determines mean filling pressure.
    mean_filling_pressure = (
        p["nom_mean_filling_pressure"]
        + (
            blood_volume_L / p["BV_scale_species"]
            - p["blood_volume_nom"]
        )
        / p["venous_compliance"]
    )

    # Cardiac output.
    cardiac_output = mean_filling_pressure / resistance_to_venous_return

    # Total peripheral resistance and MAP.
    total_peripheral_resistance = (
        systemic_arterial_resistance
        + p["R_venous"]
    )

    mean_arterial_pressure_MAP = (
        cardiac_output
        * total_peripheral_resistance
    )

    # Save outputs back into ctx.
    ctx.update(
        {
            "tissue_autoregulation_signal": tissue_autoregulation_signal,
            "AT1_svr_int": AT1_svr_int,
            "AT1_bound_AngII_effect_on_SVR": AT1_bound_AngII_effect_on_SVR,
            "systemic_arterial_resistance": systemic_arterial_resistance,
            "resistance_to_venous_return": resistance_to_venous_return,
            "mean_filling_pressure": mean_filling_pressure,
            "cardiac_output": cardiac_output,
            "total_peripheral_resistance": total_peripheral_resistance,
            "mean_arterial_pressure_MAP": mean_arterial_pressure_MAP,
        }
    )

    return ctx



def compute_renal_vasculature(ctx):
    """
    Renal vasculature module.

    Inputs required in ctx:
        p
        AT1_bound_AngII
        tubulo_glomerular_feedback_effect
        preafferent_pressure_autoreg_signal
        glomerular_pressure_autoreg_signal
        mean_arterial_pressure_MAP

    Outputs added to ctx:
        AT1_effect_on_preaff
        AT1_effect_on_aff
        AT1_effect_on_eff
        preafferent_arteriole_resistance
        afferent_arteriole_resistance
        efferent_arteriole_resistance
        renal_vascular_resistance
        renal_blood_flow_L_min
        renal_blood_flow_ml_hr
        preafferent_pressure
        glomerular_pressure
        postglomerular_pressure
        preafferent_pressure_autoreg_function
        glomerular_pressure_autoreg_function
    """

    p = ctx["p"]

    AT1_bound_AngII = ctx["AT1_bound_AngII"]
    tubulo_glomerular_feedback_effect = ctx["tubulo_glomerular_feedback_effect"]
    preafferent_pressure_autoreg_signal = ctx["preafferent_pressure_autoreg_signal"]
    glomerular_pressure_autoreg_signal = ctx["glomerular_pressure_autoreg_signal"]
    mean_arterial_pressure_MAP = ctx["mean_arterial_pressure_MAP"]

    # ------------------------------------------------------------
    # 1. AT1-bound AngII effects on renal arterioles
    # ------------------------------------------------------------

    AT1_preaff_int = 1 - p["AT1_preaff_scale"] / 2

    AT1_effect_on_preaff = (
        AT1_preaff_int
        + p["AT1_preaff_scale"]
        / (
            1
            + safe_exp(
                -(
                    AT1_bound_AngII
                    - p["nominal_equilibrium_AT1_bound_AngII"]
                )
                / p["AT1_preaff_slope"]
            )
        )
    )

    AT1_aff_int = 1 - p["AT1_aff_scale"] / 2

    AT1_effect_on_aff = (
        AT1_aff_int
        + p["AT1_aff_scale"]
        / (
            1
            + safe_exp(
                -(
                    AT1_bound_AngII
                    - p["nominal_equilibrium_AT1_bound_AngII"]
                )
                / p["AT1_aff_slope"]
            )
        )
    )

    AT1_eff_int = 1 - p["AT1_eff_scale"] / 2

    AT1_effect_on_eff = (
        AT1_eff_int
        + p["AT1_eff_scale"]
        / (
            1
            + safe_exp(
                -(
                    AT1_bound_AngII
                    - p["nominal_equilibrium_AT1_bound_AngII"]
                )
                / p["AT1_eff_slope"]
            )
        )
    )

    # ------------------------------------------------------------
    # 2. Preafferent resistance
    # ------------------------------------------------------------

    preaff_arteriole_signal_multiplier = (
        AT1_effect_on_preaff
        * preafferent_pressure_autoreg_signal
        * p["CCB_effect_on_preafferent_resistance"]
    )

    preaff_arteriole_adjusted_signal_multiplier = (
        1
        / (
            1
            + safe_exp(
                p["preaff_signal_nonlin_scale"]
                * (1 - preaff_arteriole_signal_multiplier)
            )
        )
        + 0.5
    )

    preafferent_arteriole_resistance = (
        p["nom_preafferent_arteriole_resistance"]
        * preaff_arteriole_adjusted_signal_multiplier
    )

    # ------------------------------------------------------------
    # 3. Afferent arteriole resistance
    # ------------------------------------------------------------

    nom_afferent_arteriole_resistance = (
        p["L_m3"]
        * p["viscosity_length_constant"]
        / (p["nom_afferent_diameter"] ** 4)
    )

    afferent_arteriole_signal_multiplier = (
        tubulo_glomerular_feedback_effect
        * AT1_effect_on_aff
        * glomerular_pressure_autoreg_signal
        * p["CCB_effect_on_afferent_resistance"]
    )

    afferent_arteriole_adjusted_signal_multiplier = (
        1
        / (
            1
            + safe_exp(
                p["afferent_signal_nonlin_scale"]
                * (1 - afferent_arteriole_signal_multiplier)
            )
        )
        + 0.5
    )

    afferent_arteriole_resistance = (
        nom_afferent_arteriole_resistance
        * afferent_arteriole_adjusted_signal_multiplier
    )

    # ------------------------------------------------------------
    # 4. Efferent arteriole resistance
    # ------------------------------------------------------------

    nom_efferent_arteriole_resistance = (
        p["L_m3"]
        * p["viscosity_length_constant"]
        / (p["nom_efferent_diameter"] ** 4)
    )

    efferent_arteriole_signal_multiplier = (
        AT1_effect_on_eff
        * p["CCB_effect_on_efferent_resistance"]
    )

    efferent_arteriole_adjusted_signal_multiplier = (
        1
        / (
            1
            + safe_exp(
                p["efferent_signal_nonlin_scale"]
                * (1 - efferent_arteriole_signal_multiplier)
            )
        )
        + 0.5
    )

    efferent_arteriole_resistance = (
        nom_efferent_arteriole_resistance
        * efferent_arteriole_adjusted_signal_multiplier
    )

    # ------------------------------------------------------------
    # 5. Whole-kidney renal vascular resistance
    # ------------------------------------------------------------

    peritubular_resistance = (
        p["nom_peritubular_resistance"]
        * p["baseline_nephrons"]
    )

    renal_vascular_resistance = (
        preafferent_arteriole_resistance
        + (
            afferent_arteriole_resistance
            + efferent_arteriole_resistance
            + peritubular_resistance
        )
        / p["baseline_nephrons"]
    )

    # ------------------------------------------------------------
    # 6. Renal blood flow and renal vascular pressures
    # ------------------------------------------------------------

    renal_blood_flow_L_min = (
        mean_arterial_pressure_MAP
        - p["P_venous"]
    ) / renal_vascular_resistance

    renal_blood_flow_L_min = max(
        renal_blood_flow_L_min,
        1e-9,
    )

    renal_blood_flow_ml_hr = renal_blood_flow_L_min * 1000 * 60

    preafferent_pressure = (
        mean_arterial_pressure_MAP
        - renal_blood_flow_L_min
        * preafferent_arteriole_resistance
    )

    glomerular_pressure = (
        mean_arterial_pressure_MAP
        - renal_blood_flow_L_min
        * (
            preafferent_arteriole_resistance
            + afferent_arteriole_resistance
            / p["baseline_nephrons"]
        )
    )

    postglomerular_pressure = (
        mean_arterial_pressure_MAP
        - renal_blood_flow_L_min
        * (
            preafferent_arteriole_resistance
            + (
                afferent_arteriole_resistance
                + efferent_arteriole_resistance
            )
            / p["baseline_nephrons"]
        )
    )

    # ------------------------------------------------------------
    # 7. Myogenic autoregulation target signals
    # ------------------------------------------------------------

    preaff_autoreg_int = 1 - p["preaff_autoreg_scale"] / 2

    preafferent_pressure_autoreg_function = (
        preaff_autoreg_int
        + p["preaff_autoreg_scale"]
        / (
            1
            + safe_exp(
                (
                    p["nom_preafferent_pressure"]
                    - preafferent_pressure
                )
                / p["myogenic_steepness"]
            )
        )
    )

    gp_autoreg_int = 1 - p["gp_autoreg_scale"] / 2

    glomerular_pressure_autoreg_function = (
        gp_autoreg_int
        + p["gp_autoreg_scale"]
        / (
            1
            + safe_exp(
                (
                    p["nom_glomerular_pressure"]
                    - glomerular_pressure
                )
                / p["myogenic_steepness"]
            )
        )
    )

    # ------------------------------------------------------------
    # 8. Save outputs back into ctx
    # ------------------------------------------------------------

    ctx.update(
        {
            "AT1_preaff_int": AT1_preaff_int,
            "AT1_effect_on_preaff": AT1_effect_on_preaff,
            "AT1_aff_int": AT1_aff_int,
            "AT1_effect_on_aff": AT1_effect_on_aff,
            "AT1_eff_int": AT1_eff_int,
            "AT1_effect_on_eff": AT1_effect_on_eff,
            "preaff_arteriole_signal_multiplier": preaff_arteriole_signal_multiplier,
            "preaff_arteriole_adjusted_signal_multiplier": preaff_arteriole_adjusted_signal_multiplier,
            "preafferent_arteriole_resistance": preafferent_arteriole_resistance,
            "nom_afferent_arteriole_resistance": nom_afferent_arteriole_resistance,
            "afferent_arteriole_signal_multiplier": afferent_arteriole_signal_multiplier,
            "afferent_arteriole_adjusted_signal_multiplier": afferent_arteriole_adjusted_signal_multiplier,
            "afferent_arteriole_resistance": afferent_arteriole_resistance,
            "nom_efferent_arteriole_resistance": nom_efferent_arteriole_resistance,
            "efferent_arteriole_signal_multiplier": efferent_arteriole_signal_multiplier,
            "efferent_arteriole_adjusted_signal_multiplier": efferent_arteriole_adjusted_signal_multiplier,
            "efferent_arteriole_resistance": efferent_arteriole_resistance,
            "peritubular_resistance": peritubular_resistance,
            "renal_vascular_resistance": renal_vascular_resistance,
            "renal_blood_flow_L_min": renal_blood_flow_L_min,
            "renal_blood_flow_ml_hr": renal_blood_flow_ml_hr,
            "preafferent_pressure": preafferent_pressure,
            "glomerular_pressure": glomerular_pressure,
            "postglomerular_pressure": postglomerular_pressure,
            "preaff_autoreg_int": preaff_autoreg_int,
            "preafferent_pressure_autoreg_function": preafferent_pressure_autoreg_function,
            "gp_autoreg_int": gp_autoreg_int,
            "glomerular_pressure_autoreg_function": glomerular_pressure_autoreg_function,
        }
    )

    return ctx



def compute_filtration_and_oncotic_pressure(ctx):
    """
    Glomerular filtration and oncotic pressure module.

    Inputs required in ctx:
        p
        glomerular_pressure
        oncotic_pressure_difference
        P_bowmans
        serum_creatinine
        blood_volume_L
        renal_blood_flow_L_min

    Outputs added to ctx:
        glomerular_hydrostatic_conductance_Kf
        net_filtration_pressure
        SNGFR_nL_min_raw
        SNGFR_nL_min
        GFR
        GFR_ml_min
        serum_creatinine_concentration
        creatinine_clearance_rate
        Oncotic_pressure_in
        SNRBF_nl_min
        plasma_protein_concentration_out
        Oncotic_pressure_out
        oncotic_pressure_avg
    """

    p = ctx["p"]

    glomerular_pressure = ctx["glomerular_pressure"]
    oncotic_pressure_difference = ctx["oncotic_pressure_difference"]
    P_bowmans = ctx["P_bowmans"]
    serum_creatinine = ctx["serum_creatinine"]
    blood_volume_L = ctx["blood_volume_L"]
    renal_blood_flow_L_min = ctx["renal_blood_flow_L_min"]

    # ------------------------------------------------------------
    # 1. Starling filtration equation
    # ------------------------------------------------------------

    glomerular_hydrostatic_conductance_Kf = p["nom_Kf"]

    net_filtration_pressure = (
        glomerular_pressure
        - oncotic_pressure_difference
        - P_bowmans
    )

    SNGFR_nL_min_raw = (
        glomerular_hydrostatic_conductance_Kf
        * net_filtration_pressure
    )

    # Numerical guard: do not allow negative filtration in this reduced model.
    SNGFR_nL_min = max(SNGFR_nL_min_raw, 1e-9)

    # Convert single-nephron GFR to whole-kidney GFR.
    GFR = (
        SNGFR_nL_min
        / 1000
        / 1000000
        * p["baseline_nephrons"]
    )

    GFR_ml_min = GFR * 1000

    # ------------------------------------------------------------
    # 2. Creatinine clearance
    # ------------------------------------------------------------

    serum_creatinine_concentration = (
        serum_creatinine
        / blood_volume_L
    )

    creatinine_clearance_rate = (
        GFR_ml_min
        * p["dl_ml"]
        * serum_creatinine_concentration
    )

    # ------------------------------------------------------------
    # 3. Oncotic pressure
    # ------------------------------------------------------------

    Oncotic_pressure_in = (
        1.629 * p["plasma_protein_concentration"]
        + 0.2935 * (p["plasma_protein_concentration"] ** 2)
    )

    # Single-nephron renal blood flow, nL/min.
    SNRBF_nl_min = (
        1e6
        * 1000
        * renal_blood_flow_L_min
        / p["baseline_nephrons"]
    )

    # Avoid division by zero when filtration approaches plasma flow.
    denom = max(
        SNRBF_nl_min - SNGFR_nL_min,
        1e-9,
    )

    plasma_protein_concentration_out = (
        SNRBF_nl_min
        * p["plasma_protein_concentration"]
        / denom
    )

    Oncotic_pressure_out = (
        1.629 * plasma_protein_concentration_out
        + 0.2935 * (plasma_protein_concentration_out ** 2)
    )

    oncotic_pressure_avg = (
        Oncotic_pressure_in
        + Oncotic_pressure_out
    ) / 2

    # ------------------------------------------------------------
    # 4. Save outputs back into ctx
    # ------------------------------------------------------------

    ctx.update(
        {
            "glomerular_hydrostatic_conductance_Kf": glomerular_hydrostatic_conductance_Kf,
            "net_filtration_pressure": net_filtration_pressure,
            "SNGFR_nL_min_raw": SNGFR_nL_min_raw,
            "SNGFR_nL_min": SNGFR_nL_min,
            "GFR": GFR,
            "GFR_ml_min": GFR_ml_min,
            "serum_creatinine_concentration": serum_creatinine_concentration,
            "creatinine_clearance_rate": creatinine_clearance_rate,
            "Oncotic_pressure_in": Oncotic_pressure_in,
            "SNRBF_nl_min": SNRBF_nl_min,
            "denom": denom,
            "plasma_protein_concentration_out": plasma_protein_concentration_out,
            "Oncotic_pressure_out": Oncotic_pressure_out,
            "oncotic_pressure_avg": oncotic_pressure_avg,
        }
    )

    return ctx



def compute_sodium_vasopressin(ctx):
    """
    Sodium concentration and vasopressin/ADH module.

    This version stores both:
        normalized_vasopressin_concentration_raw
        normalized_vasopressin_concentration

    The second one is clipped to avoid unphysical negative or explosive VP.
    """

    p = ctx["p"]

    sodium_amount = ctx["sodium_amount"]
    blood_volume_L = ctx["blood_volume_L"]
    ECF_sodium_amount = ctx["ECF_sodium_amount"]
    extracellular_fluid_volume = ctx["extracellular_fluid_volume"]
    Na_concentration_error = ctx["Na_concentration_error"]
    normalized_vasopressin_concentration_delayed = ctx[
        "normalized_vasopressin_concentration_delayed"
    ]

    Na_concentration = sodium_amount / blood_volume_L

    ECF_Na_concentration = (
        ECF_sodium_amount
        / extracellular_fluid_volume
    )

    Na_water_controller = (
        p["Na_controller_gain"]
        * (
            p["Kp_VP"]
            * (Na_concentration - p["ref_Na_concentration"])
            + p["Ki_VP"]
            * Na_concentration_error
        )
    )

    normalized_vasopressin_concentration_raw = (
        1
        + Na_water_controller
    )

    # Physiological/numerical bounds.
    # The upper bound does not remove ADH saturation, because CD permeability
    # is still capped later. It mainly prevents integral wind-up from exploding.
    VP_MIN = p.get("VP_MIN", 0.05)
    VP_MAX = p.get("VP_MAX", 3.0)

    normalized_vasopressin_concentration = min(
        VP_MAX,
        max(VP_MIN, normalized_vasopressin_concentration_raw),
    )

    vasopressin_concentration = (
        p["nominal_vasopressin_conc"]
        * normalized_vasopressin_concentration
    )

    water_intake_vasopressin_int = (
        1
        - p["water_intake_vasopressin_scale"] / 2
    )

    water_intake = (
        p["water_intake_species_scale"]
        * (p["nom_water_intake"] / 60 / 24)
        * (
            water_intake_vasopressin_int
            + p["water_intake_vasopressin_scale"]
            / (
                1
                + safe_exp(
                    (
                        normalized_vasopressin_concentration_delayed
                        - 1
                    )
                    / p["water_intake_vasopressin_slope"]
                )
            )
        )
    )

    daily_water_intake = water_intake * 24 * 60

    ctx.update(
        {
            "Na_concentration": Na_concentration,
            "ECF_Na_concentration": ECF_Na_concentration,
            "Na_water_controller": Na_water_controller,
            "normalized_vasopressin_concentration_raw": normalized_vasopressin_concentration_raw,
            "normalized_vasopressin_concentration": normalized_vasopressin_concentration,
            "vasopressin_concentration": vasopressin_concentration,
            "water_intake_vasopressin_int": water_intake_vasopressin_int,
            "water_intake": water_intake,
            "daily_water_intake": daily_water_intake,
        }
    )

    return ctx



def compute_tubular_transport(ctx):
    """
    Tubular transport module.

    This module computes:
        filtered Na load
        pressure natriuresis effects
        PT Na and water reabsorption
        LoH Na and water handling
        macula densa Na concentration
        TGF signal
        DCT Na handling
        collecting duct Na and water handling
        urine flow and Na excretion

    Inputs required in ctx:
        p
        SNGFR_nL_min
        Na_concentration
        postglomerular_pressure
        AT1_bound_AngII
        normalized_aldosterone_level_delayed
        normalized_vasopressin_concentration
        Na_intake_rate
        daily_water_intake

    Outputs added to ctx:
        many tubular transport and urine variables
    """

    p = ctx["p"]

    SNGFR_nL_min = ctx["SNGFR_nL_min"]
    Na_concentration = ctx["Na_concentration"]
    postglomerular_pressure = ctx["postglomerular_pressure"]
    AT1_bound_AngII = ctx["AT1_bound_AngII"]
    normalized_aldosterone_level_delayed = ctx[
        "normalized_aldosterone_level_delayed"
    ]
    normalized_vasopressin_concentration = ctx[
        "normalized_vasopressin_concentration"
    ]
    Na_intake_rate = ctx["Na_intake_rate"]
    daily_water_intake = ctx["daily_water_intake"]

    # ------------------------------------------------------------
    # 1. Tubule geometry and filtered sodium load
    # ------------------------------------------------------------

    Dc_pt = p["Dc_pt_nom"]

    L_pt = (
        p["L_pt_s1_nom"]
        + p["L_pt_s2_nom"]
        + p["L_pt_s3_nom"]
    )

    # Single-nephron filtered sodium load.
    # SNGFR: nL/min
    # Convert nL/min to L/min, then multiply by plasma Na concentration.
    SN_filtered_Na_load = (
        SNGFR_nL_min
        / 1000
        / 1000000
        * Na_concentration
    )

    # Whole-kidney filtered sodium load.
    filtered_Na_load = (
        SN_filtered_Na_load
        * p["baseline_nephrons"]
    )

    # ------------------------------------------------------------
    # 2. Pressure natriuresis effects
    # ------------------------------------------------------------

    pressure_natriuresis_PT_int = (
        1
        - p["pressure_natriuresis_PT_scale"] / 2
    )

    pressure_natriuresis_PT_effect = max(
        0.001,
        pressure_natriuresis_PT_int
        + p["pressure_natriuresis_PT_scale"]
        / (
            1
            + safe_exp(
                (
                    postglomerular_pressure
                    - p["RIHP0"]
                )
                / p["pressure_natriuresis_PT_slope"]
            )
        ),
    )

    pressure_natriuresis_LoH_int = (
        1
        - p["pressure_natriuresis_LoH_scale"] / 2
    )

    pressure_natriuresis_LoH_effect = max(
        0.001,
        pressure_natriuresis_LoH_int
        + p["pressure_natriuresis_LoH_scale"]
        / (
            1
            + safe_exp(
                (
                    postglomerular_pressure
                    - p["RIHP0"]
                )
                / p["pressure_natriuresis_LoH_slope"]
            )
        ),
    )

    pressure_natriuresis_DCT_magnitude = max(
        0,
        p["pressure_natriuresis_DCT_scale"],
    )

    pressure_natriuresis_DCT_int = (
        1
        - pressure_natriuresis_DCT_magnitude / 2
    )

    pressure_natriuresis_DCT_effect = max(
        0.001,
        pressure_natriuresis_DCT_int
        + pressure_natriuresis_DCT_magnitude
        / (
            1
            + safe_exp(
                (
                    postglomerular_pressure
                    - p["RIHP0"]
                )
                / p["pressure_natriuresis_DCT_slope"]
            )
        ),
    )

    pressure_natriuresis_CD_magnitude = max(
        0,
        p["pressure_natriuresis_CD_scale"],
    )

    pressure_natriuresis_CD_int = (
        1
        - pressure_natriuresis_CD_magnitude / 2
    )

    pressure_natriuresis_CD_effect = max(
        0.001,
        pressure_natriuresis_CD_int
        + pressure_natriuresis_CD_magnitude
        / (
            1
            + safe_exp(
                (
                    postglomerular_pressure
                    - p["RIHP0"]
                )
                / p["pressure_natriuresis_CD_slope"]
            )
        ),
    )

    # ------------------------------------------------------------
    # 3. AngII and aldosterone effects on tubular Na reabsorption
    # ------------------------------------------------------------

    AT1_PT_int = 1 - p["AT1_PT_scale"] / 2

    AT1_effect_on_PT = (
        AT1_PT_int
        + p["AT1_PT_scale"]
        / (
            1
            + safe_exp(
                -(
                    AT1_bound_AngII
                    - p["nominal_equilibrium_AT1_bound_AngII"]
                )
                / p["AT1_PT_slope"]
            )
        )
    )

    aldosterone_concentration = (
        normalized_aldosterone_level_delayed
        * p["nominal_aldosterone_concentration"]
    )

    Aldo_MR_normalised_effect = (
        normalized_aldosterone_level_delayed
        * p["MR_antagonist_effect_on_aldo_MR"]
    )

    aldo_DCT_int = 1 - p["aldo_DCT_scale"] / 2

    aldo_effect_on_DCT = (
        aldo_DCT_int
        + p["aldo_DCT_scale"]
        / (
            1
            + safe_exp(
                (
                    1
                    - Aldo_MR_normalised_effect
                )
                / p["aldo_DCT_slope"]
            )
        )
    )

    aldo_CD_int = 1 - p["aldo_CD_scale"] / 2

    aldo_effect_on_CD = (
        aldo_CD_int
        + p["aldo_CD_scale"]
        / (
            1
            + safe_exp(
                (
                    1
                    - Aldo_MR_normalised_effect
                )
                / p["aldo_CD_slope"]
            )
        )
    )

    # Segmental fractional Na reabsorption rates.
    e_pt_sodreab = min(
        0.999999,
        p["nominal_pt_na_reabsorption"]
        * AT1_effect_on_PT
        * pressure_natriuresis_PT_effect,
    )

    e_dct_sodreab = min(
        0.999999,
        p["nominal_dt_na_reabsorption"]
        * aldo_effect_on_DCT
        * pressure_natriuresis_DCT_effect
        * p["HCTZ_effect_on_DT_Na_reabs"],
    )

    e_cd_sodreab = min(
        0.999999,
        p["nominal_cd_na_reabsorption"]
        * aldo_effect_on_CD
        * pressure_natriuresis_CD_effect,
    )

    # ------------------------------------------------------------
    # 4. Proximal tubule
    # ------------------------------------------------------------

    Na_reabs_per_unit_length = (
        -safe_log(1 - e_pt_sodreab)
        / L_pt
    )

    Na_pt_out = (
        SN_filtered_Na_load
        * safe_exp(-Na_reabs_per_unit_length * L_pt)
    )

    water_out_pt = (
        (
            SNGFR_nL_min
            / 1000
            / 1000000
        )
        / max(SN_filtered_Na_load, 1e-18)
        * Na_pt_out
    )

    PT_Na_reabs_fraction = (
        1
        - Na_pt_out
        / max(SN_filtered_Na_load, 1e-18)
    )

    PT_water_reabs_fraction = (
        1
        - water_out_pt
        / max(
            SNGFR_nL_min / 1000 / 1000000,
            1e-18,
        )
    )

    Na_concentration_out_pt = (
        Na_pt_out
        / max(water_out_pt, 1e-18)
    )

    PT_Na_outflow = (
        Na_pt_out
        * p["baseline_nephrons"]
    )

    # ------------------------------------------------------------
    # 5. Descending loop of Henle
    # ------------------------------------------------------------

    water_in_DescLoH = water_out_pt
    Na_in_DescLoH = Na_pt_out
    Na_concentration_in_DescLoH = Na_concentration_out_pt

    # No solute reabsorption in descending limb.
    Na_out_DescLoH = Na_in_DescLoH

    # ------------------------------------------------------------
    # 6. Ascending loop of Henle
    # ------------------------------------------------------------

    deltaLoH_NaFlow = (
        p["LoH_flow_dependence"]
        * (
            Na_out_DescLoH
            - p["nom_Na_in_AscLoH"]
        )
    )

    AscLoH_Reab_Rate = (
        p["nominal_loh_na_reabsorption"]
        * (
            p["nom_Na_in_AscLoH"]
            + deltaLoH_NaFlow
        )
        / p["L_lh_des"]
    )

    effective_AscLoH_Reab_Rate = (
        AscLoH_Reab_Rate
        * pressure_natriuresis_LoH_effect
    )

    loh_reab_amount = min(
        effective_AscLoH_Reab_Rate
        * p["L_lh_des"],
        Na_in_DescLoH,
    )

    denom_desc = max(
        water_in_DescLoH
        * Na_concentration_in_DescLoH,
        1e-18,
    )

    Na_concentration_out_DescLoH = (
        Na_concentration_in_DescLoH
        * safe_exp(loh_reab_amount / denom_desc)
    )

    water_out_DescLoH = (
        water_in_DescLoH
        * Na_concentration_in_DescLoH
        / max(Na_concentration_out_DescLoH, 1e-18)
    )

    Na_in_AscLoH = Na_out_DescLoH
    Na_concentration_in_AscLoH = Na_concentration_out_DescLoH
    water_in_AscLoH = water_out_DescLoH

    Na_concentration_out_AscLoH = (
        Na_concentration_in_AscLoH
        - loh_reab_amount
        * (
            safe_exp(loh_reab_amount / denom_desc)
            / max(water_in_DescLoH, 1e-18)
        )
    )

    Na_reabsorbed_AscLoH = (
        Na_concentration_in_AscLoH
        - Na_concentration_out_AscLoH
    ) * water_in_AscLoH

    Na_out_AscLoH = max(
        0,
        Na_in_AscLoH
        - Na_reabsorbed_AscLoH,
    )

    water_out_AscLoH = water_in_AscLoH

    Na_concentration_out_AscLoH = (
        Na_out_AscLoH
        / max(water_out_AscLoH, 1e-18)
    )

    LoH_reabs_fraction = (
        1
        - Na_out_AscLoH
        / max(Na_in_AscLoH, 1e-18)
    )

    # ------------------------------------------------------------
    # 7. Macula densa and TGF signal
    # ------------------------------------------------------------

    SN_macula_densa_Na_flow = Na_out_AscLoH

    MD_Na_concentration = Na_concentration_out_AscLoH

    TGF0_tubulo_glomerular_feedback = (
        1
        - p["S_tubulo_glomerular_feedback"] / 2
    )

    tubulo_glomerular_feedback_signal = (
        TGF0_tubulo_glomerular_feedback
        + p["S_tubulo_glomerular_feedback"]
        / (
            1
            + safe_exp(
                (
                    p["MD_Na_concentration_setpoint"]
                    - MD_Na_concentration
                )
                / p["F_md_scale_tubulo_glomerular_feedback"]
            )
        )
    )

    # ------------------------------------------------------------
    # 8. Distal convoluted tubule
    # ------------------------------------------------------------

    water_in_DCT = water_out_AscLoH
    Na_in_DCT = Na_out_AscLoH
    Na_concentration_in_DCT = Na_concentration_out_AscLoH

    # DCT is assumed impermeable to water.
    water_out_DCT = water_in_DCT

    R_dct = (
        -safe_log(1 - e_dct_sodreab)
        / p["L_dct"]
    )

    Na_out_DCT = (
        Na_in_DCT
        * safe_exp(-R_dct * p["L_dct"])
    )

    Na_concentration_out_DCT = (
        Na_out_DCT
        / max(water_out_DCT, 1e-18)
    )

    DCT_Na_reabs_fraction = (
        1
        - Na_out_DCT
        / max(Na_in_DCT, 1e-18)
    )

    # ------------------------------------------------------------
    # 9. Collecting duct
    # ------------------------------------------------------------

    water_in_CD = water_out_DCT
    Na_in_CD = Na_out_DCT
    Na_concentration_in_CD = Na_concentration_out_DCT

    R_cd = (
        -safe_log(1 - e_cd_sodreab)
        / p["L_cd"]
    )

    Na_out_CD = (
        Na_in_CD
        * safe_exp(-R_cd * p["L_cd"])
    )

    CD_Na_reabs_fraction = (
        1
        - Na_out_CD
        / max(Na_in_CD, 1e-18)
    )

    ADH_water_permeability = min(
        1,
        max(
            0,
            p["nom_ADH_water_permeability"]
            * normalized_vasopressin_concentration,
        ),
    )

    max_water_reabs_CD = (
        water_in_CD
        - (
            Na_concentration_in_CD
            * water_in_CD
            - Na_in_CD
            * (
                1
                - safe_exp(-R_cd * p["L_cd"])
            )
        )
        / max(Na_concentration_in_AscLoH, 1e-18)
    )
    max_water_reabs_CD = max(0.0, max_water_reabs_CD)

    water_out_CD = max(
        0,
        water_in_CD
        - ADH_water_permeability
        * max_water_reabs_CD,
    )

    # ------------------------------------------------------------
    # 10. Urine output and balances
    # ------------------------------------------------------------

    urine_flow_rate = (
        water_out_CD
        * p["baseline_nephrons"]
    )

    daily_urine_flow = (
        urine_flow_rate
        * 60
        * 24
    )

    Na_excretion_via_urine = (
        Na_out_CD
        * p["baseline_nephrons"]
    )

    Na_balance = (
        Na_intake_rate
        - Na_excretion_via_urine
    )

    water_balance = (
        daily_water_intake
        - daily_urine_flow
    )

    FENA = (
        Na_excretion_via_urine
        / max(filtered_Na_load, 1e-18)
    )

    # ------------------------------------------------------------
    # 11. Save outputs back into ctx
    # ------------------------------------------------------------

    ctx.update(
        {
            "Dc_pt": Dc_pt,
            "L_pt": L_pt,
            "SN_filtered_Na_load": SN_filtered_Na_load,
            "filtered_Na_load": filtered_Na_load,

            "pressure_natriuresis_PT_int": pressure_natriuresis_PT_int,
            "pressure_natriuresis_PT_effect": pressure_natriuresis_PT_effect,
            "pressure_natriuresis_LoH_int": pressure_natriuresis_LoH_int,
            "pressure_natriuresis_LoH_effect": pressure_natriuresis_LoH_effect,
            "pressure_natriuresis_DCT_magnitude": pressure_natriuresis_DCT_magnitude,
            "pressure_natriuresis_DCT_int": pressure_natriuresis_DCT_int,
            "pressure_natriuresis_DCT_effect": pressure_natriuresis_DCT_effect,
            "pressure_natriuresis_CD_magnitude": pressure_natriuresis_CD_magnitude,
            "pressure_natriuresis_CD_int": pressure_natriuresis_CD_int,
            "pressure_natriuresis_CD_effect": pressure_natriuresis_CD_effect,

            "AT1_PT_int": AT1_PT_int,
            "AT1_effect_on_PT": AT1_effect_on_PT,
            "aldosterone_concentration": aldosterone_concentration,
            "Aldo_MR_normalised_effect": Aldo_MR_normalised_effect,
            "aldo_DCT_int": aldo_DCT_int,
            "aldo_effect_on_DCT": aldo_effect_on_DCT,
            "aldo_CD_int": aldo_CD_int,
            "aldo_effect_on_CD": aldo_effect_on_CD,

            "e_pt_sodreab": e_pt_sodreab,
            "e_dct_sodreab": e_dct_sodreab,
            "e_cd_sodreab": e_cd_sodreab,

            "Na_reabs_per_unit_length": Na_reabs_per_unit_length,
            "Na_pt_out": Na_pt_out,
            "water_out_pt": water_out_pt,
            "PT_Na_reabs_fraction": PT_Na_reabs_fraction,
            "PT_water_reabs_fraction": PT_water_reabs_fraction,
            "Na_concentration_out_pt": Na_concentration_out_pt,
            "PT_Na_outflow": PT_Na_outflow,

            "water_in_DescLoH": water_in_DescLoH,
            "Na_in_DescLoH": Na_in_DescLoH,
            "Na_concentration_in_DescLoH": Na_concentration_in_DescLoH,
            "Na_out_DescLoH": Na_out_DescLoH,

            "deltaLoH_NaFlow": deltaLoH_NaFlow,
            "AscLoH_Reab_Rate": AscLoH_Reab_Rate,
            "effective_AscLoH_Reab_Rate": effective_AscLoH_Reab_Rate,
            "loh_reab_amount": loh_reab_amount,
            "denom_desc": denom_desc,
            "Na_concentration_out_DescLoH": Na_concentration_out_DescLoH,
            "water_out_DescLoH": water_out_DescLoH,

            "Na_in_AscLoH": Na_in_AscLoH,
            "Na_concentration_in_AscLoH": Na_concentration_in_AscLoH,
            "water_in_AscLoH": water_in_AscLoH,
            "Na_concentration_out_AscLoH": Na_concentration_out_AscLoH,
            "Na_reabsorbed_AscLoH": Na_reabsorbed_AscLoH,
            "Na_out_AscLoH": Na_out_AscLoH,
            "water_out_AscLoH": water_out_AscLoH,
            "LoH_reabs_fraction": LoH_reabs_fraction,

            "SN_macula_densa_Na_flow": SN_macula_densa_Na_flow,
            "MD_Na_concentration": MD_Na_concentration,
            "TGF0_tubulo_glomerular_feedback": TGF0_tubulo_glomerular_feedback,
            "tubulo_glomerular_feedback_signal": tubulo_glomerular_feedback_signal,

            "water_in_DCT": water_in_DCT,
            "Na_in_DCT": Na_in_DCT,
            "Na_concentration_in_DCT": Na_concentration_in_DCT,
            "water_out_DCT": water_out_DCT,
            "R_dct": R_dct,
            "Na_out_DCT": Na_out_DCT,
            "Na_concentration_out_DCT": Na_concentration_out_DCT,
            "DCT_Na_reabs_fraction": DCT_Na_reabs_fraction,

            "water_in_CD": water_in_CD,
            "Na_in_CD": Na_in_CD,
            "Na_concentration_in_CD": Na_concentration_in_CD,
            "R_cd": R_cd,
            "Na_out_CD": Na_out_CD,
            "CD_Na_reabs_fraction": CD_Na_reabs_fraction,
            "ADH_water_permeability": ADH_water_permeability,
            "max_water_reabs_CD": max_water_reabs_CD,
            "water_out_CD": water_out_CD,

            "urine_flow_rate": urine_flow_rate,
            "daily_urine_flow": daily_urine_flow,
            "Na_excretion_via_urine": Na_excretion_via_urine,
            "Na_balance": Na_balance,
            "water_balance": water_balance,
            "FENA": FENA,
        }
    )

    return ctx



def compute_tubular_pressure(ctx):
    """
    Tubular pressure module.

    This module computes hydrostatic pressure along the tubule:
        CD -> DCT -> ascending LoH -> descending LoH -> PT

    The final output P_in_pt_mmHg is used as the dynamic target
    for Bowman's space pressure P_bowmans in rhs().
    """

    p = ctx["p"]

    # Inputs from tubular transport module
    water_in_CD = ctx["water_in_CD"]
    water_out_CD = ctx["water_out_CD"]
    water_in_DCT = ctx["water_in_DCT"]
    water_in_AscLoH = ctx["water_in_AscLoH"]
    water_in_DescLoH = ctx["water_in_DescLoH"]
    Na_concentration_in_DescLoH = ctx["Na_concentration_in_DescLoH"]
    effective_AscLoH_Reab_Rate = ctx["effective_AscLoH_Reab_Rate"]
    Na_reabs_per_unit_length = ctx["Na_reabs_per_unit_length"]
    SN_filtered_Na_load = ctx["SN_filtered_Na_load"]
    L_pt = ctx["L_pt"]
    Dc_pt = ctx["Dc_pt"]
    SNGFR_nL_min = ctx["SNGFR_nL_min"]

    # ------------------------------------------------------------
    # 1. Unit conversion and control pressures
    # ------------------------------------------------------------

    mmHg_Nperm2_conv = p["mmHg_Nperm2_conv"]

    Pc_pt = p["Pc_pt_mmHg"] * mmHg_Nperm2_conv
    Pc_lh_des = p["Pc_lh_des_mmHg"] * mmHg_Nperm2_conv
    Pc_lh_asc = p["Pc_lh_asc_mmHg"] * mmHg_Nperm2_conv
    Pc_dt = p["Pc_dt_mmHg"] * mmHg_Nperm2_conv
    Pc_cd = p["Pc_cd_mmHg"] * mmHg_Nperm2_conv

    P_interstitial = (
        p["P_interstitial_mmHg"]
        * mmHg_Nperm2_conv
    )

    B1 = (
        (4 * p["tubular_compliance"] + 1)
        * 128
        * p["gamma"]
        / p["pi"]
    )

    # ------------------------------------------------------------
    # 2. Collecting duct pressure
    # ------------------------------------------------------------

    mean_cd_water_flow = (
        water_in_CD
        - water_out_CD
    ) / 2

    B2_cd = (
        Pc_cd ** (4 * p["tubular_compliance"])
        / (p["Dc_cd"] ** 4)
    )

    if mean_cd_water_flow > 0:
        P_in_cd = (
            B1
            * B2_cd
            * (mean_cd_water_flow / 1e3)
            * p["L_cd"]
        ) ** (1 / (4 * p["tubular_compliance"] + 1))
    else:
        P_in_cd = 0

    P_in_cd_mmHg = (
        P_in_cd
        + P_interstitial
    ) / mmHg_Nperm2_conv

    # ------------------------------------------------------------
    # 3. Distal convoluted tubule pressure
    # ------------------------------------------------------------

    B2_dt = (
        Pc_dt ** (4 * p["tubular_compliance"])
        / (p["Dc_dt"] ** 4)
    )

    P_in_dt = (
        P_in_cd ** (4 * p["tubular_compliance"] + 1)
        + B1
        * B2_dt
        * (water_in_DCT / 1e3)
        * p["L_dct"]
    ) ** (1 / (4 * p["tubular_compliance"] + 1))

    P_in_dt_mmHg = (
        P_in_dt
        + P_interstitial
    ) / mmHg_Nperm2_conv

    # ------------------------------------------------------------
    # 4. Ascending loop of Henle pressure
    # ------------------------------------------------------------

    B2_lh_asc = (
        Pc_lh_asc ** (4 * p["tubular_compliance"])
        / (p["Dc_lh"] ** 4)
    )

    P_in_lh_asc = (
        P_in_dt ** (4 * p["tubular_compliance"] + 1)
        + B1
        * B2_lh_asc
        * (water_in_AscLoH / 1e3)
        * p["L_lh_asc"]
    ) ** (1 / (4 * p["tubular_compliance"] + 1))

    P_in_lh_asc_mmHg = (
        P_in_lh_asc
        + P_interstitial
    ) / mmHg_Nperm2_conv

    # ------------------------------------------------------------
    # 5. Descending loop of Henle pressure
    # ------------------------------------------------------------

    A_lh_des = (
        effective_AscLoH_Reab_Rate
        / max(
            water_in_DescLoH
            * Na_concentration_in_DescLoH,
            1e-18,
        )
    )

    B2_lh_des = (
        Pc_lh_des ** (4 * p["tubular_compliance"])
        * (water_in_DescLoH / 1e3)
        / (
            (p["Dc_lh"] ** 4)
            * max(A_lh_des, 1e-18)
        )
    )

    P_in_lh_des = (
        P_in_lh_asc ** (4 * p["tubular_compliance"] + 1)
        + B1
        * B2_lh_des
        * (
            1
            - safe_exp(-A_lh_des * p["L_lh_des"])
        )
    ) ** (1 / (4 * p["tubular_compliance"] + 1))

    P_in_lh_des_mmHg = (
        P_in_lh_des
        + P_interstitial
    ) / mmHg_Nperm2_conv

    # ------------------------------------------------------------
    # 6. Proximal tubule pressure
    # ------------------------------------------------------------

    A_na = Na_reabs_per_unit_length

    flow_integral_pt = (
        SN_filtered_Na_load
        / max(A_na, 1e-18)
        * (
            1
            - safe_exp(-A_na * L_pt)
        )
    )

    B2_pt = (
        Pc_pt ** (4 * p["tubular_compliance"])
        / (Dc_pt ** 4)
    )

    B3_pt = (
        (SNGFR_nL_min / 1e12)
        / max(SN_filtered_Na_load, 1e-18)
    )

    P_in_pt = (
        P_in_lh_des ** (4 * p["tubular_compliance"] + 1)
        + B1
        * B2_pt
        * B3_pt
        * flow_integral_pt
    ) ** (1 / (4 * p["tubular_compliance"] + 1))

    P_in_pt_mmHg = (
        P_in_pt
        + P_interstitial
    ) / mmHg_Nperm2_conv

    # ------------------------------------------------------------
    # 7. Save outputs back into ctx
    # ------------------------------------------------------------

    ctx.update(
        {
            "mmHg_Nperm2_conv": mmHg_Nperm2_conv,
            "Pc_pt": Pc_pt,
            "Pc_lh_des": Pc_lh_des,
            "Pc_lh_asc": Pc_lh_asc,
            "Pc_dt": Pc_dt,
            "Pc_cd": Pc_cd,
            "P_interstitial": P_interstitial,
            "B1": B1,

            "mean_cd_water_flow": mean_cd_water_flow,
            "B2_cd": B2_cd,
            "P_in_cd": P_in_cd,
            "P_in_cd_mmHg": P_in_cd_mmHg,

            "B2_dt": B2_dt,
            "P_in_dt": P_in_dt,
            "P_in_dt_mmHg": P_in_dt_mmHg,

            "B2_lh_asc": B2_lh_asc,
            "P_in_lh_asc": P_in_lh_asc,
            "P_in_lh_asc_mmHg": P_in_lh_asc_mmHg,

            "A_lh_des": A_lh_des,
            "B2_lh_des": B2_lh_des,
            "P_in_lh_des": P_in_lh_des,
            "P_in_lh_des_mmHg": P_in_lh_des_mmHg,

            "A_na": A_na,
            "flow_integral_pt": flow_integral_pt,
            "B2_pt": B2_pt,
            "B3_pt": B3_pt,
            "P_in_pt": P_in_pt,
            "P_in_pt_mmHg": P_in_pt_mmHg,
        }
    )

    return ctx



def compute_raas(ctx):
    """
    RAAS module.

    This module computes:
        aldosterone level
        macula-densa effect on renin secretion
        AT1-bound AngII negative feedback on renin
        plasma renin activity
        renin secretion rate
        degradation rates
        ACE / chymase activity
        AT1 / AT2 receptor binding rates

    Inputs required in ctx:
        p
        AT1_bound_AngII
        SN_macula_densa_Na_flow_delayed
        plasma_renin_concentration

    Outputs added to ctx:
        normalized_aldosterone_level
        md_effect_on_renin_secretion
        AT1_bound_AngII_effect_on_PRA
        aldo_effect_on_renin_secretion
        plasma_renin_activity
        renin_secretion_rate
        degradation rates
        ACE_activity
        chymase_activity
        receptor binding rates
    """

    p = ctx["p"]

    AT1_bound_AngII = ctx["AT1_bound_AngII"]
    SN_macula_densa_Na_flow_delayed = ctx[
        "SN_macula_densa_Na_flow_delayed"
    ]
    plasma_renin_concentration = ctx[
        "plasma_renin_concentration"
    ]

    # ------------------------------------------------------------
    # 1. AT1-bound AngII effect on aldosterone
    # ------------------------------------------------------------

    AT1_aldo_int = (
        1
        - p["AT1_aldo_slope"]
        * p["nominal_equilibrium_AT1_bound_AngII"]
    )

    AngII_effect_on_aldo = (
        AT1_aldo_int
        + p["AT1_aldo_slope"]
        * AT1_bound_AngII
    )

    normalized_aldosterone_level = (
        p["K_Na_ratio_effect_on_aldo"]
        * AngII_effect_on_aldo
    )

    # ------------------------------------------------------------
    # 2. Macula densa sodium-flow effect on renin secretion
    # ------------------------------------------------------------

    md_effect_on_renin_secretion = (
        p["md_renin_A"]
        * safe_exp(
            -p["md_renin_tau"]
            * (
                SN_macula_densa_Na_flow_delayed
                * p["baseline_nephrons"]
                - p["nom_LoH_Na_outflow"]
            )
        )
    )

    # ------------------------------------------------------------
    # 3. AT1-bound AngII negative feedback on PRA / renin
    # ------------------------------------------------------------

    AT1_bound_AngII_effect_on_PRA = (
        10
        ** (
            p["AT1_PRC_slope"]
            * safe_log10(
                AT1_bound_AngII
                / p["nominal_equilibrium_AT1_bound_AngII"]
            )
            + p["AT1_PRC_yint"]
        )
    )

    # ------------------------------------------------------------
    # 4. Aldosterone effect on renin secretion
    # ------------------------------------------------------------

    aldo_renin_intercept = (
        1
        - p["aldo_renin_slope"]
    )

    aldo_effect_on_renin_secretion = (
        p["aldo_renin_slope"]
        * normalized_aldosterone_level
        + aldo_renin_intercept
    )

    # ------------------------------------------------------------
    # 5. Plasma renin activity and renin secretion
    # ------------------------------------------------------------

    plasma_renin_activity = (
        p["concentration_to_renin_activity_conversion_plasma"]
        * plasma_renin_concentration
        * p["DRI_effect_on_PRA"]
    )

    renin_secretion_rate = (
        np.log(2)
        / p["renin_half_life"]
        * p["nominal_equilibrium_PRC"]
        * AT1_bound_AngII_effect_on_PRA
        * md_effect_on_renin_secretion
        * p["HCTZ_effect_on_renin_secretion"]
        * aldo_effect_on_renin_secretion
    )

    # ------------------------------------------------------------
    # 6. RAAS degradation rates
    # ------------------------------------------------------------

    renin_degradation_rate = (
        np.log(2)
        / p["renin_half_life"]
    )

    AngI_degradation_rate = (
        np.log(2)
        / p["AngI_half_life"]
    )

    AngII_degradation_rate = (
        np.log(2)
        / p["AngII_half_life"]
    )

    AT1_bound_AngII_degradation_rate = (
        np.log(2)
        / p["AT1_bound_AngII_half_life"]
    )

    AT2_bound_AngII_degradation_rate = (
        np.log(2)
        / p["AT2_bound_AngII_half_life"]
    )

    # ------------------------------------------------------------
    # 7. ACE / chymase activity and receptor binding
    # ------------------------------------------------------------

    ACE_activity = (
        p["nominal_ACE_activity"]
        * (1 - p["pct_target_inhibition_ACEi"])
    )

    chymase_activity = p["nominal_chymase_activity"]

    AT1_receptor_binding_rate = (
        p["nominal_AT1_receptor_binding_rate"]
        * (1 - p["pct_target_inhibition_ARB"])
    )

    AT2_receptor_binding_rate = (
        p["nominal_AT2_receptor_binding_rate"]
    )

    # ------------------------------------------------------------
    # 8. Save outputs back into ctx
    # ------------------------------------------------------------

    ctx.update(
        {
            "AT1_aldo_int": AT1_aldo_int,
            "AngII_effect_on_aldo": AngII_effect_on_aldo,
            "normalized_aldosterone_level": normalized_aldosterone_level,
            "md_effect_on_renin_secretion": md_effect_on_renin_secretion,
            "AT1_bound_AngII_effect_on_PRA": AT1_bound_AngII_effect_on_PRA,
            "aldo_renin_intercept": aldo_renin_intercept,
            "aldo_effect_on_renin_secretion": aldo_effect_on_renin_secretion,
            "plasma_renin_activity": plasma_renin_activity,
            "renin_secretion_rate": renin_secretion_rate,
            "renin_degradation_rate": renin_degradation_rate,
            "AngI_degradation_rate": AngI_degradation_rate,
            "AngII_degradation_rate": AngII_degradation_rate,
            "AT1_bound_AngII_degradation_rate": AT1_bound_AngII_degradation_rate,
            "AT2_bound_AngII_degradation_rate": AT2_bound_AngII_degradation_rate,
            "ACE_activity": ACE_activity,
            "chymase_activity": chymase_activity,
            "AT1_receptor_binding_rate": AT1_receptor_binding_rate,
            "AT2_receptor_binding_rate": AT2_receptor_binding_rate,
        }
    )

    return ctx



def derived(t, y, p, protocol=None):
    ctx = unpack_state_and_protocol(t, y, p, protocol)
    compute_systemic_hemodynamics(ctx)
    compute_renal_vasculature(ctx)
    compute_filtration_and_oncotic_pressure(ctx)
    compute_sodium_vasopressin(ctx)
    compute_tubular_transport(ctx)
    compute_tubular_pressure(ctx)
    compute_raas(ctx)
    out = ctx.copy()
    for k in ['y', 'p', 'protocol']:
        out.pop(k, None)
    return out

def rhs(t, y, p, protocol=None):
    z = derived(t, y, p, protocol)
    dy = np.zeros(len(STATE_NAMES))
    dy[IDX['AngI']] = z['plasma_renin_activity'] - z['AngI']*(z['chymase_activity']+z['ACE_activity']) - z['AngI']*z['AngI_degradation_rate']
    dy[IDX['AngII']] = z['AngI']*(z['chymase_activity']+z['ACE_activity']) - z['AngII']*z['AngII_degradation_rate'] - z['AngII']*z['AT1_receptor_binding_rate'] - z['AngII']*z['AT2_receptor_binding_rate']
    dy[IDX['AT1_bound_AngII']] = z['AngII']*z['AT1_receptor_binding_rate'] - z['AT1_bound_AngII_degradation_rate']*z['AT1_bound_AngII']
    dy[IDX['AT2_bound_AngII']] = z['AngII']*z['AT2_receptor_binding_rate'] - z['AT2_bound_AngII_degradation_rate']*z['AT2_bound_AngII']
    dy[IDX['plasma_renin_concentration']] = z['renin_secretion_rate'] - z['plasma_renin_concentration']*z['renin_degradation_rate']
    dy[IDX['blood_volume_L']] = p['C_water_intake_ecf_volume']*z['water_intake'] + p['C_urine_flow_ecf_volume']*z['urine_flow_rate'] + p['Q_water']*(z['Na_concentration']-z['ECF_Na_concentration'])
    dy[IDX['extracellular_fluid_volume']] = p['Q_water']*(z['ECF_Na_concentration']-z['Na_concentration'])
    dy[IDX['sodium_amount']] = p['C_na_excretion_na_amount']*z['Na_excretion_via_urine'] + p['C_na_intake_na_amount']*z['Na_intake_rate'] + p['Q_Na']*(z['ECF_Na_concentration']-z['Na_concentration'])
    dy[IDX['ECF_sodium_amount']] = p['Q_Na']*(z['Na_concentration']-z['ECF_Na_concentration'])
    dy[IDX['tubulo_glomerular_feedback_effect']] = p['C_tgf']*(z['tubulo_glomerular_feedback_signal']-z['tubulo_glomerular_feedback_effect'])
    dy[IDX['normalized_aldosterone_level_delayed']] = p['C_aldo_secretion']*(z['normalized_aldosterone_level']-z['normalized_aldosterone_level_delayed'])
    dy[IDX['preafferent_pressure_autoreg_signal']] = 500*(z['preafferent_pressure_autoreg_function']-z['preafferent_pressure_autoreg_signal'])
    dy[IDX['glomerular_pressure_autoreg_signal']] = 500*(z['glomerular_pressure_autoreg_function']-z['glomerular_pressure_autoreg_signal'])
    dy[IDX['cardiac_output_delayed']] = p['C_cardiac_output_delayed']*(z['cardiac_output']-z['cardiac_output_delayed'])
    dy[IDX['CO_error']] = p['C_co_error']*(z['cardiac_output']-p['CO_nom'])
     # ------------------------------------------------------------
    # Anti-windup for sodium/vasopressin PI controller
    # ------------------------------------------------------------

    na_error = z["Na_concentration"] - p["ref_Na_concentration"]

    vp_raw = z.get(
        "normalized_vasopressin_concentration_raw",
        z["normalized_vasopressin_concentration"],
    )

    VP_MIN = p.get("VP_MIN", 0.05)
    VP_MAX = p.get("VP_MAX", 3.0)

    vp_saturated_high = vp_raw >= VP_MAX and na_error > 0
    vp_saturated_low = vp_raw <= VP_MIN and na_error < 0

    if vp_saturated_high or vp_saturated_low:
        dy[IDX["Na_concentration_error"]] = 0.0
    else:
        dy[IDX["Na_concentration_error"]] = (
            p["C_Na_error"]
            * na_error
        )
    dy[IDX['normalized_vasopressin_concentration_delayed']] = p['C_vasopressin_delay']*(z['normalized_vasopressin_concentration']-z['normalized_vasopressin_concentration_delayed'])
    dy[IDX['F0_TGF']] = p['C_tgf_reset']*(z['SN_macula_densa_Na_flow']*p['baseline_nephrons'] - z['F0_TGF'])
    dy[IDX['P_bowmans']] = p['C_P_bowmans']*(z['P_in_pt_mmHg']-z['P_bowmans'])
    dy[IDX['oncotic_pressure_difference']] = p['C_P_oncotic']*(z['oncotic_pressure_avg']-z['oncotic_pressure_difference'])
    dy[IDX['SN_macula_densa_Na_flow_delayed']] = p['C_md_flow']*(z['SN_macula_densa_Na_flow']-z['SN_macula_densa_Na_flow_delayed'])
    dy[IDX['serum_creatinine']] = p['creatinine_synthesis_rate'] - z['creatinine_clearance_rate']
    return dy

def simulate(p=None, y0=None, days=20, dt_min=20, protocol=None, rtol=1e-4, atol=1e-4):
    if p is None:
        p = calc_nom_params()
    if y0 is None:
        y0 = initial_conditions(p)
    t_end = days*1440
    t_eval = np.arange(0, t_end+dt_min, dt_min)
    sol = solve_ivp(lambda t,y: rhs(t,y,p,protocol), (0,t_end), y0,
                    t_eval=t_eval, method='LSODA', rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol

def solution_to_frame(sol, p, protocol=None, thin=1):
    rows = []
    for j in range(0, len(sol.t), thin):
        z = derived(sol.t[j], sol.y[:,j], p, protocol)
        row = {k: z.get(k, np.nan) for k in z.keys() if np.isscalar(z.get(k, np.nan))}
        row['time_min'] = sol.t[j]
        row['time_days'] = sol.t[j]/1440
        for name, value in zip(STATE_NAMES, sol.y[:,j]):
            row[name] = value
        rows.append(row)
    return pd.DataFrame(rows)




# ---------------------------------------------------------------------
# Compatibility aliases and GSA wrappers
# ---------------------------------------------------------------------
from pathlib import Path
import copy
import time
import json

EPS = 1e-12

# Keep the original solver function under a separate name.
simulate_solution = simulate


def get_nominal_params():
    """Compatibility wrapper: return the original baseline parameters."""
    return calc_nom_params()


def compute_equilibrium_derived_params(p):
    """Compatibility wrapper.

    In the original modular Python reconstruction, `calc_nom_params()` already
    computes the Hallow equilibrium-derived baseline quantities. For GSA, this
    function intentionally returns a copy and does not recalibrate after each
    perturbation unless the caller explicitly rebuilds parameters.
    """
    return dict(p)


def initial_conditions_with_counters(p):
    return initial_conditions(p)


def _uacr_observation_from_row(row, p, baseline):
    baseline_gfr = max(float(baseline.get("GFR_ml_min", row["GFR_ml_min"])), EPS)
    baseline_pgc = float(baseline.get("glomerular_pressure", row["glomerular_pressure"]))
    baseline_ccr = max(float(baseline.get("creatinine_clearance_rate", row["creatinine_clearance_rate"])), EPS)
    baseline_ucr_g_day = float(baseline.get("ucr_g_day", baseline_ccr * 1440.0 / 1000.0))

    ucr_g_day = max(float(row["creatinine_clearance_rate"]) * 1440.0 / 1000.0, EPS)
    pressure_term = np.exp(np.clip(p.get("beta_pressure_albumin", 0.05) * (float(row["glomerular_pressure"]) - baseline_pgc), -60, 60))
    gfr_term = (max(float(row["GFR_ml_min"]), EPS) / baseline_gfr) ** p.get("beta_gfr_albumin", 0.5)
    aer = (
        p.get("uacr0_mg_g", 10.0)
        * baseline_ucr_g_day
        * p.get("barrier_leakiness", 1.0)
        * pressure_term
        * gfr_term
        / max(p.get("pt_albumin_uptake_capacity", 1.0), EPS)
    )
    uacr = aer / max(ucr_g_day * p.get("ucr_denominator_factor", 1.0), EPS)
    return {
        "ucr_g_day": ucr_g_day,
        "AER_mg_day": float(aer),
        "UACR_mg_g": float(uacr),
        "log10_UACR": float(np.log10(max(uacr, EPS))),
    }


def _add_default_uacr_params(p):
    p = dict(p)
    p.setdefault("uacr0_mg_g", 10.0)
    p.setdefault("barrier_leakiness", 1.0)
    p.setdefault("beta_pressure_albumin", 0.05)
    p.setdefault("beta_gfr_albumin", 0.5)
    p.setdefault("pt_albumin_uptake_capacity", 1.0)
    p.setdefault("ucr_denominator_factor", 1.0)
    return p


def make_steady_initial_conditions(p, days=80, dt_min=120, rtol=1e-5, atol=1e-7):
    """Run a pre-equilibration and return the final state vector.

    This reproduces the original R-style workflow: run long, take the final
    state, then use it as the new initial condition for checking/sampling.
    """
    sol = simulate_solution(p=p, y0=initial_conditions(p), days=days, dt_min=dt_min, rtol=rtol, atol=atol)
    return sol.y[:, -1].copy()


def simulate(p=None, y0=None, days=60, dt_min=60, protocol=None, rtol=1e-5, atol=1e-7,
             add_uacr=True, baseline_for_uacr=None, return_solution=False):
    """Run the original ODE solver and return a DataFrame by default.

    This wrapper intentionally keeps the original model equations. It only
    converts the solution to a DataFrame and adds the UACR observation layer.
    """
    if p is None:
        p = get_nominal_params()
    p = _add_default_uacr_params(p)
    if y0 is None:
        y0 = initial_conditions(p)
    sol = simulate_solution(p=p, y0=y0, days=days, dt_min=dt_min, protocol=protocol, rtol=rtol, atol=atol)
    if return_solution:
        return sol
    df = solution_to_frame(sol, p, protocol=protocol)
    if add_uacr:
        # If no external baseline is supplied, use the last-day average of this
        # run as the observation reference. For GSA samples, pass the baseline
        # summary to keep UACR anchored to the fixed baseline framework.
        if baseline_for_uacr is None:
            cutoff = df["time_days"].max() - 1.0
            tail = df[df["time_days"] >= cutoff]
            baseline_for_uacr = {
                "GFR_ml_min": float(tail["GFR_ml_min"].mean()),
                "glomerular_pressure": float(tail["glomerular_pressure"].mean()),
                "creatinine_clearance_rate": float(tail["creatinine_clearance_rate"].mean()),
            }
            baseline_for_uacr["ucr_g_day"] = baseline_for_uacr["creatinine_clearance_rate"] * 1440.0 / 1000.0
        uacr_rows = [_uacr_observation_from_row(row, p, baseline_for_uacr) for _, row in df.iterrows()]
        uacr_df = pd.DataFrame(uacr_rows)
        df = pd.concat([df.reset_index(drop=True), uacr_df.reset_index(drop=True)], axis=1)
    return df


def summarize_simulation(df, last_days=1):
    cutoff = df["time_days"].max() - float(last_days)
    tail = df[df["time_days"] >= cutoff]
    cols = [
        "GFR_ml_min", "UACR_mg_g", "log10_UACR",
        "serum_creatinine_concentration", "daily_urine_flow",
        "mean_arterial_pressure_MAP", "Na_concentration", "FENA",
        "blood_volume_L", "glomerular_pressure", "renal_blood_flow_L_min",
        "renal_vascular_resistance", "P_bowmans", "oncotic_pressure_difference",
        "filtered_Na_load", "MD_Na_concentration", "PT_Na_reabs_fraction",
        "LoH_reabs_fraction", "DCT_Na_reabs_fraction", "CD_Na_reabs_fraction",
        "creatinine_clearance_rate", "ucr_g_day", "AER_mg_day",
    ]
    return {c: float(tail[c].mean()) for c in cols if c in tail.columns}


def check_physiology(summary):
    reasons = []
    def finite_between(name, lo, hi):
        v = summary.get(name, np.nan)
        if not np.isfinite(v) or v < lo or v > hi:
            reasons.append(f"{name} outside [{lo}, {hi}]: {v}")
    finite_between("GFR_ml_min", 1.0, 250.0)
    if not np.isfinite(summary.get("UACR_mg_g", np.nan)) or summary.get("UACR_mg_g", 0) <= 0:
        reasons.append(f"UACR_mg_g non-positive/non-finite: {summary.get('UACR_mg_g')}")
    finite_between("mean_arterial_pressure_MAP", 50.0, 180.0)
    finite_between("Na_concentration", 125.0, 160.0)
    if "blood_volume_L" in summary:
        finite_between("blood_volume_L", 2.0, 8.0)
    if not np.isfinite(summary.get("renal_blood_flow_L_min", np.nan)) or summary.get("renal_blood_flow_L_min", 0) <= 0:
        reasons.append(f"renal_blood_flow_L_min not positive/finite: {summary.get('renal_blood_flow_L_min')}")
    if not np.isfinite(summary.get("daily_urine_flow", np.nan)) or summary.get("daily_urine_flow", 0) <= 0:
        reasons.append(f"daily_urine_flow not positive/finite: {summary.get('daily_urine_flow')}")
    return len(reasons) == 0, reasons


def apply_sample_to_params(base_params, sample, recalibrate_setpoints=False):
    p = copy.deepcopy(base_params)
    for key, value in sample.items():
        value = float(value)
        if key == "functional_nephron_fraction":
            p["baseline_nephrons"] = base_params["baseline_nephrons"] * value
        elif key == "afferent_diameter_multiplier":
            p["nom_afferent_diameter"] = base_params["nom_afferent_diameter"] * value
        elif key == "efferent_diameter_multiplier":
            p["nom_efferent_diameter"] = base_params["nom_efferent_diameter"] * value
        elif key == "Na_intake_mmol_day":
            p["Na_intake_rate"] = value / 24.0 / 60.0
        elif key == "MAP_target":
            p["nom_systemic_arterial_resistance"] = value / p["CO_nom"] - p["R_venous"]
        elif key == "pressure_natriuresis_distal_scale":
            p["pressure_natriuresis_LoH_scale"] = value
            p["pressure_natriuresis_DCT_scale"] = value
            p["pressure_natriuresis_CD_scale"] = value
        elif key == "log10_barrier_leakiness":
            p["barrier_leakiness"] = 10.0 ** value
        elif key == "log10_pt_albumin_uptake_capacity":
            p["pt_albumin_uptake_capacity"] = 10.0 ** value
        else:
            p[key] = value
    # Default: do not recalibrate Hallow baseline setpoints after perturbation.
    if recalibrate_setpoints:
        p = calc_nom_params() | p
    return _add_default_uacr_params(p)


def run_one_sample(sample, base_params, baseline_summary, sample_id=None):
    start = time.time()
    row = {"sample_id": sample_id, **sample}
    try:
        p = apply_sample_to_params(base_params, sample, recalibrate_setpoints=False)
        y0 = make_steady_initial_conditions(p, days=20, dt_min=120)  # shorter per sample; adjust if needed
        df = simulate(p=p, y0=y0, days=20, dt_min=120, baseline_for_uacr=baseline_summary)
        summary = summarize_simulation(df)
        ok, reasons = check_physiology(summary)
        row.update(summary)
        row.update(pass_physiology=ok, failure_reasons="; ".join(reasons), runtime_sec=time.time() - start)
    except Exception as exc:
        row.update(pass_physiology=False, failure_reasons=str(exc), runtime_sec=time.time() - start)
    return row


if __name__ == "__main__":
    out = Path("outputs/baseline")
    out.mkdir(parents=True, exist_ok=True)
    p0 = compute_equilibrium_derived_params(get_nominal_params())
    y_ss = make_steady_initial_conditions(p0, days=80, dt_min=120)
    df = simulate(p0, y0=y_ss, days=60, dt_min=60)
    summary = summarize_simulation(df)
    ok, reasons = check_physiology(summary)
    df.to_csv(out / "baseline_timeseries.csv", index=False)
    pd.DataFrame([summary]).to_csv(out / "baseline_summary.csv", index=False)
    print(pd.Series(summary).to_string())
    print("Baseline physiology pass:", ok)
    print("Reasons:", reasons)
