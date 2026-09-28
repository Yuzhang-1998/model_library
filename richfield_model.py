"""Richfield-style steady-state GFR and UACR forward model.

This file defines the first-stage interface for OAT and GSA:

    sampled parameters -> glomerular steady-state outputs -> GFR_ml_min,
    log10_UACR

The current implementation keeps the model modular and inspectable. The
vascular geometry is controlled by shared multipliers, so segment-level
Richfield geometry can be plugged into `run_richfield_kernel` later without
changing the OAT/GSA interface.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.integrate import solve_ivp
except ImportError:  # pragma: no cover - only needed for richfield_ode mode.
    solve_ivp = None


EPS = 1e-12
PRIMARY_OUTPUTS = ["GFR_ml_min", "log10_UACR"]
MACULA_DENSA_MODES = {"proxy", "richfield_ode"}


def safe_exp(x: float, lo: float = -60.0, hi: float = 60.0) -> float:
    return float(np.exp(np.clip(x, lo, hi)))


def safe_log10(x: float) -> float:
    return float(np.log10(max(float(x), EPS)))


def oncotic_pressure_from_protein(protein_g_dl: float) -> float:
    """Landis-Pappenheimer style plasma oncotic pressure approximation."""

    c = max(float(protein_g_dl), EPS)
    return float(2.1 * c + 0.16 * c**2 + 0.009 * c**3)


def get_nominal_params() -> dict[str, float]:
    """Return baseline Richfield + UACR parameters.

    Values here are anchors for the interface and first smoke tests. Formal
    OAT/GSA ranges will be stored separately in the parameter evidence table.
    """

    return {
        # Boundary pressure and blood composition
        "MAP": 100.0,
        "efferent_outlet_pressure": 15.0,
        "Bowmans_pressure": 13.0,
        "external_pressure": 5.0,
        "plasma_protein_concentration": 5.94,
        "hematocrit": 0.40,
        "plasma_viscosity_scale": 1.0,
        "oncotic_pressure_scale": 1.0,
        # Unified vascular geometry / resistance
        "afferent_diameter_scale": 1.0,
        "afferent_length_scale": 1.0,
        "efferent_diameter_scale": 1.0,
        "efferent_length_scale": 1.0,
        "glomerular_capillary_diameter_scale": 1.0,
        "glomerular_capillary_length_scale": 1.0,
        "preafferent_resistance_scale": 1.0,
        "vascular_compliance_scale": 1.0,
        # Richfield-like baseline geometric anchors
        "DAA_0_um": 7.0,
        "DAA_pass_um": 22.0,
        "L_AA_um": 106.6698,
        "DEA_0_um": 7.3,
        "L_EA_um": 106.6698,
        "Rra_baseline": 1.09,
        # Filtration and glomerular wall parameters
        "Kf_scale": 1.0,
        "hydraulic_permeability_scale": 1.0,
        "podocyte_height_scale": 1.0,
        "podocyte_width_scale": 1.0,
        "capillary_recruitment_fraction": 1.0,
        "filtration_pressure_scale": 1.0,
        "glomerular_wall_tension_scale": 1.0,
        "glomerular_compliance_scale": 1.0,
        "baseline_single_nephron_Kf_nL_min_mmHg": 1.45,
        # Autoregulation parameters
        "passive_tension_scale": 1.0,
        "active_tension_scale": 1.0,
        "myogenic_sensitivity_scale": 1.0,
        "myogenic_setpoint_shift": 0.0,
        "myogenic_min_tone_scale": 1.0,
        "myogenic_max_tone_scale": 1.0,
        "TGF_sensitivity_scale": 1.0,
        "TGF_setpoint_shift": 0.0,
        "TGF_min_tone_scale": 1.0,
        "TGF_max_tone_scale": 1.0,
        "myogenic_TGF_interaction_scale": 1.0,
        "autoregulation_strength_scale": 1.0,
        # Tubule / macula densa module
        "proximal_transport_Vmax_scale": 1.0,
        "ascending_limb_transport_Vmax_scale": 1.0,
        "tubular_Km_scale": 1.0,
        "proximal_water_permeability_scale": 1.0,
        "descending_limb_water_permeability_scale": 1.0,
        "tubule_radius_scale": 1.0,
        "proximal_tubule_length_scale": 1.0,
        "loop_tubule_length_scale": 1.0,
        "macula_densa_signal_scale": 1.0,
        # Richfield tubule ODE anchors used when macula_densa_mode="richfield_ode".
        "richfield_tubule_Vmax": 16.6e-3,
        "richfield_tubule_Km": 70.0,
        "richfield_tubule_r_Tub": 10e-4,
        "richfield_tubule_L_Tub": 0.5,
        "richfield_tubule_P_Tub": 1.5e-5,
        "richfield_tubule_Co": 275.0,
        "richfield_tubule_r": 1.25e-3,
        "richfield_tubule_L_PT": 0.5,
        "richfield_tubule_L_DHL": 0.4,
        "richfield_tubule_Pv_PT": 1.5e-5,
        "richfield_tubule_Pv_DHL": 1.5e-5,
        "richfield_tubule_Vmax_PT": 28e-3,
        "richfield_tubule_Vmax_DHL": 0.0,
        "richfield_tubule_Ce_L_Tub": 150.0,
        "richfield_tubule_A3_Tub": 2.0,
        "richfield_tubule_Ce_L_PT": 650.0,
        "richfield_tubule_A3_PT": 2.0,
        "richfield_ode_C_MD_ref": 356.24459406886,
        # Whole-kidney scaling
        "functional_nephron_fraction": 1.0,
        "total_nephron_number": 2.0e6,
        "single_nephron_to_kidney_scale": 1.0,
        "kidney_mass_scale": 1.0,
        "body_surface_area_scale": 1.0,
        # Creatinine / UACR denominator
        "serum_creatinine_mg_dl": 0.9,
        "creatinine_generation_rate_scale": 1.0,
        "urine_creatinine_excretion_g_day": 1.0,
        "ucr_denominator_factor": 1.0,
        # Albumin / UACR observation layer
        "uacr0_mg_g": 10.0,
        "barrier_leakiness": 1.0,
        "beta_pressure_albumin": 0.05,
        "beta_gfr_albumin": 0.5,
        "pt_albumin_uptake_capacity": 1.0,
    }


def get_default_gsa_inputs() -> list[dict[str, Any]]:
    """Return candidate OAT/GSA inputs with ranges left for literature review."""

    rows = [
        # Boundary pressure and blood composition
        ("MAP", "boundary_pressure_blood", "mmHg", "linear", "Renal perfusion pressure / arterial pressure driver"),
        ("efferent_outlet_pressure", "boundary_pressure_blood", "mmHg", "linear", "Postglomerular outlet pressure"),
        ("Bowmans_pressure", "boundary_pressure_blood", "mmHg", "linear", "Bowman's space hydrostatic pressure"),
        ("external_pressure", "boundary_pressure_blood", "mmHg", "linear", "External/interstitial pressure"),
        ("plasma_protein_concentration", "boundary_pressure_blood", "g/dL", "linear", "Inlet plasma protein concentration"),
        ("hematocrit", "boundary_pressure_blood", "fraction", "linear", "Systemic hematocrit"),
        ("plasma_viscosity_scale", "boundary_pressure_blood", "multiplier", "linear", "Plasma viscosity multiplier"),
        ("oncotic_pressure_scale", "boundary_pressure_blood", "multiplier", "linear", "Oncotic pressure multiplier"),
        # Unified vascular geometry
        ("afferent_diameter_scale", "unified_vascular_geometry", "multiplier", "linear", "Shared afferent diameter multiplier"),
        ("afferent_length_scale", "unified_vascular_geometry", "multiplier", "linear", "Shared afferent length multiplier"),
        ("efferent_diameter_scale", "unified_vascular_geometry", "multiplier", "linear", "Shared efferent diameter multiplier"),
        ("efferent_length_scale", "unified_vascular_geometry", "multiplier", "linear", "Shared efferent length multiplier"),
        ("glomerular_capillary_diameter_scale", "unified_vascular_geometry", "multiplier", "linear", "Shared glomerular capillary diameter multiplier"),
        ("glomerular_capillary_length_scale", "unified_vascular_geometry", "multiplier", "linear", "Shared glomerular capillary length multiplier"),
        ("preafferent_resistance_scale", "unified_vascular_geometry", "multiplier", "linear", "Preafferent/upstream resistance multiplier"),
        ("vascular_compliance_scale", "unified_vascular_geometry", "multiplier", "linear", "Vascular compliance multiplier"),
        # Filtration and wall
        ("Kf_scale", "filtration_wall", "multiplier", "linear", "Single-nephron filtration coefficient multiplier"),
        ("hydraulic_permeability_scale", "filtration_wall", "multiplier", "linear", "Capillary hydraulic permeability multiplier"),
        ("podocyte_height_scale", "filtration_wall", "multiplier", "linear", "Podocyte/slit geometry proxy"),
        ("podocyte_width_scale", "filtration_wall", "multiplier", "linear", "Podocyte/slit geometry proxy"),
        ("capillary_recruitment_fraction", "filtration_wall", "fraction", "linear", "Available filtration surface fraction"),
        ("filtration_pressure_scale", "filtration_wall", "multiplier", "linear", "Net filtration pressure correction"),
        ("glomerular_wall_tension_scale", "filtration_wall", "multiplier", "linear", "Glomerular wall tension multiplier"),
        ("glomerular_compliance_scale", "filtration_wall", "multiplier", "linear", "Glomerular compliance multiplier"),
        # Autoregulation
        ("passive_tension_scale", "autoregulation", "multiplier", "linear", "Passive tension multiplier"),
        ("active_tension_scale", "autoregulation", "multiplier", "linear", "Active tension multiplier"),
        ("myogenic_sensitivity_scale", "autoregulation", "multiplier", "linear", "Myogenic sensitivity multiplier"),
        ("myogenic_setpoint_shift", "autoregulation", "mmHg", "linear", "Myogenic setpoint shift"),
        ("myogenic_min_tone_scale", "autoregulation", "multiplier", "linear", "Lower myogenic tone bound multiplier"),
        ("myogenic_max_tone_scale", "autoregulation", "multiplier", "linear", "Upper myogenic tone bound multiplier"),
        ("TGF_sensitivity_scale", "autoregulation", "multiplier", "linear", "TGF sensitivity multiplier"),
        ("TGF_setpoint_shift", "autoregulation", "mM", "linear", "Macula densa/TGF setpoint shift"),
        ("TGF_min_tone_scale", "autoregulation", "multiplier", "linear", "Lower TGF tone bound multiplier"),
        ("TGF_max_tone_scale", "autoregulation", "multiplier", "linear", "Upper TGF tone bound multiplier"),
        ("myogenic_TGF_interaction_scale", "autoregulation", "multiplier", "linear", "Myogenic/TGF interaction multiplier"),
        ("autoregulation_strength_scale", "autoregulation", "multiplier", "linear", "Global autoregulation strength multiplier"),
        # Tubule / macula densa
        ("proximal_transport_Vmax_scale", "tubule_macula_densa", "multiplier", "linear", "Proximal transport capacity multiplier"),
        ("ascending_limb_transport_Vmax_scale", "tubule_macula_densa", "multiplier", "linear", "Ascending limb transport capacity multiplier"),
        ("tubular_Km_scale", "tubule_macula_densa", "multiplier", "linear", "Tubular transport Km multiplier"),
        ("proximal_water_permeability_scale", "tubule_macula_densa", "multiplier", "linear", "Proximal water permeability multiplier"),
        ("descending_limb_water_permeability_scale", "tubule_macula_densa", "multiplier", "linear", "Descending limb water permeability multiplier"),
        ("tubule_radius_scale", "tubule_macula_densa", "multiplier", "linear", "Tubule radius multiplier"),
        ("proximal_tubule_length_scale", "tubule_macula_densa", "multiplier", "linear", "Proximal tubule length multiplier"),
        ("loop_tubule_length_scale", "tubule_macula_densa", "multiplier", "linear", "Loop/ascending limb length multiplier"),
        ("macula_densa_signal_scale", "tubule_macula_densa", "multiplier", "linear", "Macula densa signal multiplier"),
        # Whole kidney
        ("functional_nephron_fraction", "whole_kidney_scaling", "fraction", "linear", "Functioning nephron fraction"),
        ("total_nephron_number", "whole_kidney_scaling", "count", "linear", "Total nephron number"),
        ("single_nephron_to_kidney_scale", "whole_kidney_scaling", "multiplier", "linear", "SNGFR-to-kidney scale multiplier"),
        ("kidney_mass_scale", "whole_kidney_scaling", "multiplier", "linear", "Kidney size/mass proxy"),
        ("body_surface_area_scale", "whole_kidney_scaling", "multiplier", "linear", "Body size/BSA proxy"),
        # Creatinine / denominator
        ("serum_creatinine_mg_dl", "creatinine_uacr_denominator", "mg/dL", "linear", "Serum creatinine proxy"),
        ("creatinine_generation_rate_scale", "creatinine_uacr_denominator", "multiplier", "linear", "Creatinine generation multiplier"),
        ("urine_creatinine_excretion_g_day", "creatinine_uacr_denominator", "g/day", "linear", "Urinary creatinine denominator"),
        ("ucr_denominator_factor", "creatinine_uacr_denominator", "multiplier", "linear", "UACR denominator nuisance factor"),
        # Albumin / UACR
        ("uacr0_mg_g", "albumin_uacr_layer", "mg/g", "linear", "Baseline UACR anchor"),
        ("log10_barrier_leakiness", "albumin_uacr_layer", "log10 multiplier", "log10", "Filtration barrier leakiness"),
        ("beta_pressure_albumin", "albumin_uacr_layer", "1/mmHg", "linear", "Pressure sensitivity of albumin excretion"),
        ("beta_gfr_albumin", "albumin_uacr_layer", "unitless", "linear", "GFR sensitivity of albumin excretion"),
        ("log10_pt_albumin_uptake_capacity", "albumin_uacr_layer", "log10 multiplier", "log10", "Proximal tubular albumin uptake capacity"),
    ]
    out = []
    for name, group, unit, scale, description in rows:
        out.append(
            {
                "parameter": name,
                "mechanism_group": group,
                "baseline_value": get_nominal_params().get(name, np.nan),
                "lower_bound": np.nan,
                "upper_bound": np.nan,
                "unit": unit,
                "sampling_scale": scale,
                "evidence_source": "",
                "evidence_type": "range_pending",
                "description": description,
                "include_in_gsa": True,
            }
        )
    return out


def apply_sample_to_params(base_params: dict[str, float], sample: dict[str, float]) -> dict[str, float]:
    """Copy baseline parameters and apply one sampled parameter set."""

    p = copy.deepcopy(base_params)
    for key, value in sample.items():
        value = float(value)
        if key == "log10_barrier_leakiness":
            p["barrier_leakiness"] = 10.0**value
        elif key == "log10_pt_albumin_uptake_capacity":
            p["pt_albumin_uptake_capacity"] = 10.0**value
        else:
            p[key] = value
    return p


def compute_macula_densa_proxy(p: dict[str, float], previous_sngfr: float | None = None) -> dict[str, float]:
    """Fast algebraic proxy for macula densa concentration."""

    baseline_sngfr = 40.0
    if previous_sngfr is None:
        previous_sngfr = baseline_sngfr

    md = (
        30.0
        * (previous_sngfr / baseline_sngfr) ** 0.35
        * p["macula_densa_signal_scale"]
        * p["ascending_limb_transport_Vmax_scale"]
        / max(p["tubular_Km_scale"], EPS) ** 0.15
    )
    return {
        "macula_densa_concentration": float(md),
        "loop_flow": float(previous_sngfr / 1e6),
        "macula_densa_mode": "proxy",
    }


def _richfield_tubule_params(p: dict[str, float]) -> dict[str, float]:
    """Return tubule ODE parameters after applying GSA scale factors."""

    return {
        "Vmax": p["richfield_tubule_Vmax"] * p["ascending_limb_transport_Vmax_scale"],
        "Km": p["richfield_tubule_Km"] * p["tubular_Km_scale"],
        "r_Tub": p["richfield_tubule_r_Tub"] * p["tubule_radius_scale"],
        "L_Tub": p["richfield_tubule_L_Tub"] * p["loop_tubule_length_scale"],
        "P_Tub": p["richfield_tubule_P_Tub"],
        "Co": p["richfield_tubule_Co"],
        "r": p["richfield_tubule_r"] * p["tubule_radius_scale"],
        "L_PT": p["richfield_tubule_L_PT"] * p["proximal_tubule_length_scale"],
        "L_DHL": p["richfield_tubule_L_DHL"] * p["loop_tubule_length_scale"],
        "Pv_PT": p["richfield_tubule_Pv_PT"] * p["proximal_water_permeability_scale"],
        "Pv_DHL": p["richfield_tubule_Pv_DHL"] * p["descending_limb_water_permeability_scale"],
        "Vmax_PT": p["richfield_tubule_Vmax_PT"] * p["proximal_transport_Vmax_scale"],
        "Vmax_DHL": p["richfield_tubule_Vmax_DHL"],
        "Ce_L_Tub": p["richfield_tubule_Ce_L_Tub"],
        "A3_Tub": p["richfield_tubule_A3_Tub"],
        "Ce_L_PT": p["richfield_tubule_Ce_L_PT"],
        "A3_PT": p["richfield_tubule_A3_PT"],
    }


def _richfield_tubule_area(tp: dict[str, float]) -> float:
    return float(np.pi * tp["r"] ** 2)


def ce_calc_tubule(x: float, tp: dict[str, float]) -> float:
    """External concentration profile from the Richfield tubule kernel."""

    if x < tp["L_PT"]:
        return float(tp["Co"])
    if x <= tp["L_PT"] + tp["L_DHL"]:
        a1 = (1 - tp["Ce_L_PT"] / tp["Co"]) / (1 - np.exp(tp["A3_PT"]))
        return float(tp["Co"] * (a1 * np.exp(tp["A3_PT"] * (x - tp["L_PT"]) / tp["L_DHL"]) + 1 - a1))
    ce_join = ce_calc_tubule(tp["L_PT"] + tp["L_DHL"], tp)
    a1 = (1 - tp["Ce_L_Tub"] / ce_join) / (1 - np.exp(-tp["A3_Tub"]))
    return float(ce_join * (a1 * np.exp(-tp["A3_Tub"] * (x - tp["L_PT"] - tp["L_DHL"]) / tp["L_Tub"]) + 1 - a1))


def _richfield_tubule_rhs(tp: dict[str, float], Pv: float, Vmax: float):
    def rhs(x: float, y: np.ndarray) -> list[float]:
        C, v = max(float(y[0]), EPS), max(float(y[1]), EPS)
        pv_func = Pv / (1 + np.exp(-(v - 0.03) * 350))
        area = _richfield_tubule_area(tp)
        J_v = pv_func * (C - ce_calc_tubule(float(x), tp)) / area
        J_s = -J_v * C / v - Vmax * C / (C + tp["Km"]) / v / area
        return [float(2 * np.pi * tp["r"] * J_s), float(2 * np.pi * tp["r"] * J_v)]

    return rhs


def _ascending_limb_rhs(tp: dict[str, float], loop_flow: float):
    def rhs(x: float, y: np.ndarray) -> list[float]:
        C = max(float(y[0]), EPS)
        flow = max(float(loop_flow), EPS)
        dCdx = -2 * np.pi * tp["r_Tub"] * (
            tp["Vmax"] * C / (tp["Km"] + C) + tp["P_Tub"] * (C - ce_calc_tubule(float(x), tp))
        ) / flow
        return [float(dCdx)]

    return rhs


def _integrate_ode_fixed_grid(rhs, x_grid: np.ndarray, y0: list[float]) -> np.ndarray:
    """Small RK4 fallback used when scipy.solve_ivp is unavailable."""

    y = np.zeros((len(y0), len(x_grid)), dtype=float)
    y[:, 0] = np.asarray(y0, dtype=float)
    for i in range(1, len(x_grid)):
        x0 = float(x_grid[i - 1])
        h = float(x_grid[i] - x_grid[i - 1])
        yi = y[:, i - 1]
        k1 = np.asarray(rhs(x0, yi), dtype=float)
        k2 = np.asarray(rhs(x0 + 0.5 * h, yi + 0.5 * h * k1), dtype=float)
        k3 = np.asarray(rhs(x0 + 0.5 * h, yi + 0.5 * h * k2), dtype=float)
        k4 = np.asarray(rhs(x0 + h, yi + h * k3), dtype=float)
        y[:, i] = yi + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        y[:, i] = np.maximum(y[:, i], EPS)
        if not np.all(np.isfinite(y[:, i])):
            raise RuntimeError(f"fixed-grid ODE integration produced non-finite state at x={x_grid[i]}")
    return y


def _integrate_tubule_segment(rhs, x_grid: np.ndarray, y0: list[float], method: str, rtol: float, atol: float) -> np.ndarray:
    if solve_ivp is None:
        raise RuntimeError("scipy is required for macula_densa_mode='richfield_ode'")
    out = solve_ivp(
        rhs,
        (x_grid[0], x_grid[-1]),
        y0,
        t_eval=x_grid,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not out.success:
        raise RuntimeError(out.message)
    return out.y


def compute_macula_densa_richfield_ode(
    p: dict[str, float],
    previous_sngfr: float | None = None,
    method: str = "LSODA",
    rtol: float = 1e-1,
    atol: float = 1e-8,
    n_eval: int = 1001,
) -> dict[str, float]:
    """Compute C_MD and loop flow using the Richfield tubule ODE submodel."""

    if previous_sngfr is None:
        previous_sngfr = 40.0

    tp = _richfield_tubule_params(p)
    area = _richfield_tubule_area(tp)
    vo = max(float(previous_sngfr), EPS) / area / 1e6 / 60

    x_pt = np.linspace(0, tp["L_PT"], n_eval)
    try:
        y_pt = _integrate_tubule_segment(
            _richfield_tubule_rhs(tp, tp["Pv_PT"], tp["Vmax_PT"]),
            x_pt,
            [tp["Co"], vo],
            method,
            rtol,
            atol,
        )
    except Exception as exc:
        raise RuntimeError(f"PT tubule ODE failed: {exc}") from exc

    x_dhl = np.linspace(tp["L_PT"], tp["L_PT"] + tp["L_DHL"], n_eval)
    try:
        y_dhl = _integrate_tubule_segment(
            _richfield_tubule_rhs(tp, tp["Pv_DHL"], tp["Vmax_DHL"]),
            x_dhl,
            y_pt[:, -1].tolist(),
            method,
            rtol,
            atol,
        )
    except Exception as exc:
        raise RuntimeError(f"DHL tubule ODE failed: {exc}") from exc

    loop_flow = float(y_dhl[1, -1] * area)
    x_al = np.linspace(tp["L_PT"] + tp["L_DHL"], tp["L_PT"] + tp["L_DHL"] + tp["L_Tub"], n_eval)
    try:
        y_al = _integrate_tubule_segment(
            _ascending_limb_rhs(tp, loop_flow),
            x_al,
            [float(y_dhl[0, -1])],
            method,
            rtol,
            atol,
        )
    except Exception as exc:
        raise RuntimeError(f"ascending limb ODE failed: {exc}") from exc

    md = float(y_al[0, -1] * p["macula_densa_signal_scale"])
    return {
        "macula_densa_concentration": md,
        "loop_flow": loop_flow,
        "macula_densa_mode": "richfield_ode",
    }


def compute_macula_densa_state(
    p: dict[str, float],
    previous_sngfr: float | None = None,
    mode: str = "proxy",
) -> dict[str, float]:
    """Dispatch macula densa calculation to the proxy or Richfield ODE mode."""

    if mode == "proxy":
        return compute_macula_densa_proxy(p, previous_sngfr)
    if mode == "richfield_ode":
        return compute_macula_densa_richfield_ode(p, previous_sngfr)
    raise ValueError(f"Unknown macula_densa_mode={mode!r}; expected one of {sorted(MACULA_DENSA_MODES)}")


def compute_autoregulation_state(
    p: dict[str, float],
    inlet_pressure: float,
    previous_sngfr: float | None = None,
    macula_densa_mode: str = "proxy",
) -> dict[str, float]:
    """Compute a compact Richfield-style myogenic/TGF tone state."""

    md_state = compute_macula_densa_state(p, previous_sngfr, mode=macula_densa_mode)
    md = md_state["macula_densa_concentration"]
    myogenic_signal = (
        p["myogenic_sensitivity_scale"]
        * ((inlet_pressure - (100.0 + p["myogenic_setpoint_shift"])) / 20.0)
    )
    tgf_signal = (
        p["TGF_sensitivity_scale"]
        * (
            (
                md
                - (
                    (p["richfield_ode_C_MD_ref"] if macula_densa_mode == "richfield_ode" else 30.0)
                    + p["TGF_setpoint_shift"]
                )
            )
            / 15.0
        )
    )
    interaction = p["myogenic_TGF_interaction_scale"] * myogenic_signal * tgf_signal
    total_tone = p["autoregulation_strength_scale"] * (myogenic_signal + tgf_signal + 0.15 * interaction)
    total_tone = float(np.clip(total_tone, -4.0, 4.0))
    diameter_factor = float(np.clip(1.0 / (1.0 + 0.08 * total_tone), 0.55, 1.45))
    return {
        **md_state,
        "myogenic_signal": float(myogenic_signal),
        "TGF_signal": float(tgf_signal),
        "total_tone_signal": float(total_tone),
        "autoregulation_diameter_factor": diameter_factor,
    }


def _relative_resistance(length_scale: float, diameter_scale: float, viscosity_scale: float) -> float:
    return float(length_scale * viscosity_scale / max(diameter_scale, EPS) ** 4)


def run_richfield_kernel(p: dict[str, float], macula_densa_mode: str = "proxy") -> dict[str, float]:
    """Run one steady-state glomerular solve.

    This is the replaceable Richfield kernel wrapper. The first implementation
    uses Richfield-style pressure, geometry, resistance, autoregulation, and
    Starling filtration relationships while exposing the final interface needed
    for OAT/GSA.
    """

    inlet_pressure = float(p["MAP"])
    outlet_pressure = float(p["efferent_outlet_pressure"])
    pressure_drop = max(inlet_pressure - outlet_pressure, EPS)

    auto = compute_autoregulation_state(p, inlet_pressure, macula_densa_mode=macula_densa_mode)
    aff_diam = (
        p["DAA_0_um"]
        * p["afferent_diameter_scale"]
        * auto["autoregulation_diameter_factor"]
        * p["vascular_compliance_scale"] ** 0.10
    )
    eff_diam = p["DEA_0_um"] * p["efferent_diameter_scale"]
    cap_diam_scale = p["glomerular_capillary_diameter_scale"]

    viscosity = p["plasma_viscosity_scale"] * (1.0 + 0.7 * (p["hematocrit"] - 0.40))
    pre_r = p["preafferent_resistance_scale"] * p["Rra_baseline"]
    aff_r = _relative_resistance(p["afferent_length_scale"], aff_diam / p["DAA_0_um"], viscosity)
    eff_r = _relative_resistance(p["efferent_length_scale"], eff_diam / p["DEA_0_um"], viscosity)
    cap_r = _relative_resistance(
        p["glomerular_capillary_length_scale"],
        cap_diam_scale,
        viscosity,
    )
    upstream_r = pre_r + aff_r + 0.5 * cap_r
    downstream_r = eff_r + 0.5 * cap_r
    total_r = max(upstream_r + downstream_r, EPS)

    baseline_total_r = 1.09 + 1.0 + 1.0 + 1.0
    single_nephron_blood_flow = 180.0 * pressure_drop / 85.0 * baseline_total_r / total_r
    single_nephron_plasma_flow = single_nephron_blood_flow * (1.0 - p["hematocrit"])

    glomerular_pressure = outlet_pressure + pressure_drop * downstream_r / total_r
    glomerular_pressure *= p["glomerular_compliance_scale"] ** 0.05

    onc_in = oncotic_pressure_from_protein(p["plasma_protein_concentration"])
    filtration_fraction_guess = 0.20
    protein_out = p["plasma_protein_concentration"] / max(1.0 - filtration_fraction_guess, 0.2)
    onc_out = oncotic_pressure_from_protein(protein_out)
    onc_avg = 0.5 * (onc_in + onc_out) * p["oncotic_pressure_scale"]

    net_filtration_pressure = (
        glomerular_pressure - p["Bowmans_pressure"] - onc_avg
    ) * p["filtration_pressure_scale"]
    net_filtration_pressure = max(float(net_filtration_pressure), 0.0)

    effective_kf = (
        p["baseline_single_nephron_Kf_nL_min_mmHg"]
        * p["Kf_scale"]
        * p["hydraulic_permeability_scale"]
        * p["capillary_recruitment_fraction"]
        * p["podocyte_height_scale"]
        / max(p["podocyte_width_scale"], EPS)
    )
    sngfr = max(effective_kf * net_filtration_pressure, EPS)
    sngfr = min(sngfr, 0.95 * max(single_nephron_plasma_flow, EPS))

    # One light fixed-point update lets the TGF module respond to the SNGFR.
    auto = compute_autoregulation_state(
        p,
        inlet_pressure,
        previous_sngfr=sngfr,
        macula_densa_mode=macula_densa_mode,
    )

    filtration_fraction = sngfr / max(single_nephron_plasma_flow, EPS)
    renal_blood_flow_ml_min = single_nephron_blood_flow * p["total_nephron_number"] / 1e6
    renal_plasma_flow_ml_min = single_nephron_plasma_flow * p["total_nephron_number"] / 1e6

    return {
        **auto,
        "afferent_diameter": float(aff_diam),
        "efferent_diameter": float(eff_diam),
        "afferent_resistance": float(aff_r),
        "efferent_resistance": float(eff_r),
        "preafferent_resistance": float(pre_r),
        "glomerular_capillary_resistance": float(cap_r),
        "total_renal_resistance": float(total_r),
        "single_nephron_blood_flow_nL_min": float(single_nephron_blood_flow),
        "single_nephron_plasma_flow_nL_min": float(single_nephron_plasma_flow),
        "renal_blood_flow_ml_min": float(renal_blood_flow_ml_min),
        "renal_plasma_flow_ml_min": float(renal_plasma_flow_ml_min),
        "glomerular_inlet_pressure": float(inlet_pressure),
        "postglomerular_pressure": float(outlet_pressure),
        "glomerular_pressure": float(glomerular_pressure),
        "Bowmans_pressure": float(p["Bowmans_pressure"]),
        "external_pressure": float(p["external_pressure"]),
        "oncotic_pressure_in": float(onc_in),
        "oncotic_pressure_out": float(onc_out),
        "oncotic_pressure_avg": float(onc_avg),
        "plasma_protein_concentration_out": float(protein_out),
        "net_filtration_pressure": float(net_filtration_pressure),
        "SNGFR_nL_min": float(sngfr),
        "filtration_fraction": float(filtration_fraction),
        "mean_viscosity": float(viscosity),
        "mean_hematocrit": float(p["hematocrit"]),
        "mean_shear": float(single_nephron_blood_flow / max(aff_diam**3, EPS)),
        "mean_hoop_stress": float(aff_diam * max(glomerular_pressure - p["external_pressure"], 0.0) / 2.0),
    }


def compute_whole_kidney_outputs(glom: dict[str, float], p: dict[str, float]) -> dict[str, float]:
    """Scale single-nephron Richfield outputs to kidney-level outputs."""

    functional_nephrons = (
        p["total_nephron_number"]
        * p["functional_nephron_fraction"]
        * p["kidney_mass_scale"]
        * p["body_surface_area_scale"]
    )
    gfr_ml_min = (
        glom["SNGFR_nL_min"]
        * functional_nephrons
        / 1e6
        * p["single_nephron_to_kidney_scale"]
    )
    creatinine_clearance_rate = (
        gfr_ml_min
        * p["serum_creatinine_mg_dl"]
        / 100.0
        * p["creatinine_generation_rate_scale"]
    )
    ucr = p["urine_creatinine_excretion_g_day"] * p["creatinine_generation_rate_scale"]
    return {
        "functional_nephron_number": float(functional_nephrons),
        "GFR_ml_min": float(gfr_ml_min),
        "GFR_ml_day": float(gfr_ml_min * 1440.0),
        "serum_creatinine_mg_dl": float(p["serum_creatinine_mg_dl"]),
        "creatinine_clearance_rate": float(creatinine_clearance_rate),
        "urine_creatinine_excretion_g_day": float(max(ucr, EPS)),
    }


def compute_uacr_observation(
    outputs: dict[str, float],
    p: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    """Compute the semi-mechanistic UACR observation layer."""

    baseline_gfr = max(float(baseline.get("GFR_ml_min", outputs["GFR_ml_min"])), EPS)
    baseline_pgc = float(baseline.get("glomerular_pressure", outputs["glomerular_pressure"]))
    baseline_ucr = max(
        float(baseline.get("urine_creatinine_excretion_g_day", outputs["urine_creatinine_excretion_g_day"])),
        EPS,
    )
    ucr = max(outputs["urine_creatinine_excretion_g_day"], EPS)
    pressure_term = safe_exp(p["beta_pressure_albumin"] * (outputs["glomerular_pressure"] - baseline_pgc))
    gfr_term = (max(outputs["GFR_ml_min"], EPS) / baseline_gfr) ** p["beta_gfr_albumin"]
    aer = (
        p["uacr0_mg_g"]
        * baseline_ucr
        * p["barrier_leakiness"]
        * pressure_term
        * gfr_term
        / max(p["pt_albumin_uptake_capacity"], EPS)
    )
    uacr = aer / max(ucr * p["ucr_denominator_factor"], EPS)
    return {
        "albumin_excreted_load": float(aer),
        "AER_mg_day": float(aer),
        "UACR_mg_g": float(uacr),
        "log10_UACR": safe_log10(uacr),
    }


def predict(
    p: dict[str, float] | None = None,
    baseline: dict[str, float] | None = None,
    macula_densa_mode: str = "proxy",
) -> dict[str, float]:
    """Run the full steady-state model for one parameter set."""

    if p is None:
        p = get_nominal_params()
    glom = run_richfield_kernel(p, macula_densa_mode=macula_densa_mode)
    kidney = compute_whole_kidney_outputs(glom, p)
    combined = {**glom, **kidney}
    if baseline is None:
        baseline = combined
    uacr = compute_uacr_observation(combined, p, baseline)
    return {**combined, **uacr}


def summarize_prediction(outputs: dict[str, float]) -> dict[str, float]:
    """Return a flat summary with primary outputs and mediators."""

    cols = [
        "GFR_ml_min",
        "log10_UACR",
        "UACR_mg_g",
        "AER_mg_day",
        "SNGFR_nL_min",
        "glomerular_pressure",
        "Bowmans_pressure",
        "net_filtration_pressure",
        "filtration_fraction",
        "renal_blood_flow_ml_min",
        "renal_plasma_flow_ml_min",
        "single_nephron_plasma_flow_nL_min",
        "afferent_diameter",
        "efferent_diameter",
        "afferent_resistance",
        "efferent_resistance",
        "preafferent_resistance",
        "total_renal_resistance",
        "oncotic_pressure_in",
        "oncotic_pressure_out",
        "oncotic_pressure_avg",
        "macula_densa_concentration",
        "macula_densa_mode",
        "loop_flow",
        "myogenic_signal",
        "TGF_signal",
        "total_tone_signal",
        "functional_nephron_number",
        "serum_creatinine_mg_dl",
        "creatinine_clearance_rate",
        "urine_creatinine_excretion_g_day",
        "albumin_excreted_load",
    ]
    summary: dict[str, float | str] = {}
    for c in cols:
        if c not in outputs:
            continue
        if c == "macula_densa_mode":
            summary[c] = str(outputs[c])
        else:
            summary[c] = float(outputs[c])
    return summary


def check_physiology(summary: dict[str, float]) -> tuple[bool, list[str]]:
    """Return pass/fail and reasons for one model output row."""

    reasons: list[str] = []

    def finite_between(name: str, lo: float, hi: float) -> None:
        value = summary.get(name, np.nan)
        if not np.isfinite(value) or value < lo or value > hi:
            reasons.append(f"{name} outside [{lo}, {hi}]: {value}")

    finite_between("GFR_ml_min", 1.0, 250.0)
    finite_between("log10_UACR", -3.0, 7.0)
    finite_between("glomerular_pressure", 10.0, 100.0)
    finite_between("SNGFR_nL_min", 0.01, 200.0)
    finite_between("filtration_fraction", 0.001, 0.95)
    if not np.isfinite(summary.get("UACR_mg_g", np.nan)) or summary.get("UACR_mg_g", 0.0) <= 0:
        reasons.append(f"UACR_mg_g non-positive/non-finite: {summary.get('UACR_mg_g')}")
    if not np.isfinite(summary.get("renal_blood_flow_ml_min", np.nan)) or summary.get("renal_blood_flow_ml_min", 0.0) <= 0:
        reasons.append(f"renal_blood_flow_ml_min non-positive/non-finite: {summary.get('renal_blood_flow_ml_min')}")
    if not np.isfinite(summary.get("macula_densa_concentration", np.nan)) or summary.get("macula_densa_concentration", 0.0) <= 0:
        reasons.append(f"macula_densa_concentration non-positive/non-finite: {summary.get('macula_densa_concentration')}")
    return len(reasons) == 0, reasons


def run_one_sample(
    sample: dict[str, float],
    base_params: dict[str, float],
    baseline_summary: dict[str, float],
    sample_id: Any = None,
    macula_densa_mode: str = "proxy",
) -> dict[str, Any]:
    """Apply one sample, run the model, summarize, and return a CSV-ready row."""

    start = time.time()
    row: dict[str, Any] = {"sample_id": sample_id, **sample}
    try:
        p = apply_sample_to_params(base_params, sample)
        outputs = predict(p, baseline=baseline_summary, macula_densa_mode=macula_densa_mode)
        summary = summarize_prediction(outputs)
        ok, reasons = check_physiology(summary)
        row.update(summary)
        row.update(
            pass_physiology=ok,
            failure_reasons="; ".join(reasons),
            macula_densa_mode=macula_densa_mode,
            runtime_sec=time.time() - start,
        )
    except Exception as exc:
        row.update(
            GFR_ml_min=np.nan,
            log10_UACR=np.nan,
            pass_physiology=False,
            failure_reasons=str(exc),
            macula_densa_mode=macula_densa_mode,
            runtime_sec=time.time() - start,
        )
    return row


def write_input_parameter_table(path: str | Path = "parameter_tables/richfield_gsa_inputs.csv") -> Path:
    """Write the candidate input parameter list for later range curation."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(get_default_gsa_inputs()).to_csv(out, index=False)
    return out


if __name__ == "__main__":
    out = Path("outputs/baseline")
    out.mkdir(parents=True, exist_ok=True)
    base = get_nominal_params()
    baseline_outputs = predict(base)
    baseline_summary = summarize_prediction(baseline_outputs)
    ok, reasons = check_physiology(baseline_summary)
    pd.DataFrame([baseline_summary]).to_csv(out / "richfield_baseline_summary.csv", index=False)
    table_path = write_input_parameter_table()
    print(pd.Series({k: baseline_summary[k] for k in PRIMARY_OUTPUTS}).to_string())
    print("Baseline physiology pass:", ok)
    print("Reasons:", reasons)
    print("Input parameter table:", table_path)
