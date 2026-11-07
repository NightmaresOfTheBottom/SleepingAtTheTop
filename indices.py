"""
indices.py — CPAP Indices & Descriptive Statistics Module
SleepingAtTheTop — NightmaresOfTheBottom / AirSense 11

Computes:
- Standard descriptive statistics for any signal (compute_stats)
- Per-night indices: AHI estimate, ODI, PCI, flow limitation, snore
- CTQI (CPAP Therapy Quality Index) — original composite score (0–100)
- Per-signal stats for BRP, PLD, SA2
- Longitudinal stats across nights (from STR or batch summaries)

Dependencies: numpy, (pandas optional for longitudinal output)
No pyedflib, no MNE.
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# 8.1  Standard Statistics Block
# ---------------------------------------------------------------------------

STANDARD_STATS_KEYS = [
    "mean", "median", "mode",
    "min", "max",
    "p0_5", "p1", "p5", "p95", "p99", "p99_5",
    "variance", "std", "cv_pct",
    "n_valid", "n_total", "pct_valid",
]

# Suggested bin widths for binned mode, keyed by signal category
MODE_BIN_WIDTHS = {
    "flow": 0.1,
    "pressure": 0.5,
    "spo2": 1.0,
    "pulse": 2.0,
    "default": 0.1,
}


def _binned_mode(arr, bin_width):
    # type: (np.ndarray, float) -> float
    """Compute mode via binning. Returns midpoint of most-populated bin."""
    if arr.size == 0:
        return float("nan")
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if lo == hi:
        return lo
    n_bins = max(1, int(np.ceil((hi - lo) / bin_width)))
    counts, edges = np.histogram(arr, bins=n_bins, range=(lo, hi + bin_width))
    idx = int(np.argmax(counts))
    return round((edges[idx] + edges[idx + 1]) / 2.0, 4)


def compute_stats(arr, label="", bin_width=None):
    # type: (np.ndarray, str, Optional[float]) -> dict
    """Compute full descriptive statistics block for a signal.

    Parameters
    ----------
    arr : np.ndarray
        1-D signal array. May contain NaN.
    label : str
        Signal name (stored in output for reference).
    bin_width : float, optional
        Bin width for mode. If None, inferred from label.

    Returns
    -------
    Flat dict with all STANDARD_STATS_KEYS plus 'label'.
    All values are float or int. Returns NaN for stats when
    insufficient valid data.
    """
    n_total = arr.size
    valid = arr[~np.isnan(arr)] if n_total > 0 else np.array([])
    n_valid = valid.size
    pct_valid = round(100.0 * n_valid / n_total, 2) if n_total > 0 else 0.0

    nan = float("nan")
    out = {"label": label, "n_total": n_total, "n_valid": n_valid,
           "pct_valid": pct_valid}

    if n_valid == 0:
        for k in STANDARD_STATS_KEYS:
            if k not in out:
                out[k] = nan
        return out

    out["mean"] = round(float(np.mean(valid)), 4)
    out["median"] = round(float(np.median(valid)), 4)
    out["min"] = round(float(np.min(valid)), 4)
    out["max"] = round(float(np.max(valid)), 4)
    out["variance"] = round(float(np.var(valid, ddof=0)), 6)
    out["std"] = round(float(np.std(valid, ddof=0)), 4)
    out["cv_pct"] = round(
        100.0 * out["std"] / abs(out["mean"]), 2
    ) if abs(out["mean"]) > 1e-9 else nan

    for pct, key in [(0.5, "p0_5"), (1, "p1"), (5, "p5"),
                     (95, "p95"), (99, "p99"), (99.5, "p99_5")]:
        out[key] = round(float(np.percentile(valid, pct)), 4)

    # Binned mode
    if bin_width is None:
        lbl = label.lower()
        if "flow" in lbl or "lim" in lbl:
            bin_width = MODE_BIN_WIDTHS["flow"]
        elif "press" in lbl:
            bin_width = MODE_BIN_WIDTHS["pressure"]
        elif "spo2" in lbl or "o2" in lbl:
            bin_width = MODE_BIN_WIDTHS["spo2"]
        elif "pulse" in lbl or "hr" in lbl:
            bin_width = MODE_BIN_WIDTHS["pulse"]
        else:
            bin_width = MODE_BIN_WIDTHS["default"]
    out["mode"] = _binned_mode(valid, bin_width)

    return out


# ---------------------------------------------------------------------------
# 8.2  Per-Signal Statistics for a Single Night
# ---------------------------------------------------------------------------

def compute_brp_stats(session, ramp_minutes=0):
    # type: (dict, float) -> Dict[str, dict]
    """Compute stats for BRP signals.

    Parameters
    ----------
    session : dict
        Session dict with 'brp' sub-dict containing numpy arrays.
    ramp_minutes : float
        Ramp period to exclude from pressure stats.

    Returns
    -------
    Dict keyed by signal name, each value is a compute_stats() result.
    """
    brp = session.get("brp")
    if brp is None:
        return {}

    out = {}
    flow = brp.get("Flow.40ms")
    fs = brp.get("fs", 25.0)

    if flow is not None and flow.size > 0:
        out["Flow.40ms"] = compute_stats(flow, "Flow.40ms")
        # Inspiratory (positive) and expiratory (negative) separately
        insp = flow[flow > 0]
        exp_abs = np.abs(flow[flow < 0])
        out["Flow.40ms_insp"] = compute_stats(insp, "Flow.40ms_insp")
        out["Flow.40ms_exp"] = compute_stats(exp_abs, "Flow.40ms_exp")

    press = brp.get("Press.40ms")
    if press is not None and press.size > 0:
        ramp_samples = int(ramp_minutes * 60 * fs)
        p = press[ramp_samples:] if ramp_samples < press.size else press
        out["Press.40ms"] = compute_stats(p, "Press.40ms")

    return out


def compute_pld_stats(session):
    # type: (dict,) -> Dict[str, dict]
    """Compute stats for all PLD signals with extra flags."""
    pld = session.get("pld")
    if pld is None:
        return {}

    PLD_SIGNALS = [
        "MaskPress.2s", "Press.2s", "EprPress.2s", "Leak.2s",
        "RespRate.2s", "TidVol.2s", "MinVent.2s", "Snore.2s", "FlowLim.2s",
    ]
    out = {}
    for sig in PLD_SIGNALS:
        arr = pld.get(sig)
        if arr is not None and arr.size > 0:
            s = compute_stats(arr, sig)

            # Extra flags per spec
            if sig == "Leak.2s":
                s["pct_time_above_0.4"] = round(
                    100.0 * float(np.sum(arr > 0.4)) / arr.size, 2)
            elif sig == "Snore.2s":
                s["snore_frac_elevated"] = round(
                    float(np.sum(arr > 1.0)) / arr.size, 4)
                s["snore_frac_heavy"] = round(
                    float(np.sum(arr > 2.5)) / arr.size, 4)
            elif sig == "FlowLim.2s":
                s["fl_frac_elevated"] = round(
                    float(np.sum(arr > 0.3)) / arr.size, 4)
                s["fl_frac_heavy"] = round(
                    float(np.sum(arr > 0.5)) / arr.size, 4)
            elif sig == "RespRate.2s":
                valid = arr[~np.isnan(arr)]
                if valid.size > 0:
                    s["pct_out_of_range"] = round(
                        100.0 * float(np.sum((valid < 6) | (valid > 30))) / valid.size, 2)
            elif sig == "TidVol.2s":
                valid = arr[~np.isnan(arr)]
                if valid.size > 0:
                    s["pct_out_of_range"] = round(
                        100.0 * float(np.sum((valid < 0.1) | (valid > 2.0))) / valid.size, 2)

            out[sig] = s
    return out


def compute_sa2_stats(session):
    # type: (dict,) -> Dict[str, dict]
    """Compute stats for SA2 signals (filter zeros first)."""
    sa2 = session.get("sa2")
    if sa2 is None:
        return {}

    out = {}
    for sig in ["SpO2.1s", "Pulse.1s"]:
        arr = sa2.get(sig)
        if arr is None or arr.size == 0:
            continue
        # Filter zeros/sentinels
        filtered = arr.astype(np.float64).copy()
        filtered[filtered <= 0] = np.nan
        s = compute_stats(filtered, sig)

        if sig == "SpO2.1s":
            valid = filtered[~np.isnan(filtered)]
            fs = sa2.get("fs", 1.0)
            if valid.size > 0:
                s["T90"] = round(float(np.sum(valid < 90)) / fs / 60.0, 2)
                s["T88"] = round(float(np.sum(valid < 88)) / fs / 60.0, 2)
                s["T85"] = round(float(np.sum(valid < 85)) / fs / 60.0, 2)
        elif sig == "Pulse.1s":
            valid = filtered[~np.isnan(filtered)]
            if valid.size > 0:
                s["pulse_range"] = round(float(np.max(valid) - np.min(valid)), 1)

        out[sig] = s
    return out


def compute_all_signal_stats(session, ramp_minutes=0):
    # type: (dict, float) -> Dict[str, dict]
    """Compute descriptive stats for all signals in a session.

    Returns
    -------
    Dict keyed by signal name, values are compute_stats() results.
    """
    out = {}
    out.update(compute_brp_stats(session, ramp_minutes))
    out.update(compute_pld_stats(session))
    out.update(compute_sa2_stats(session))
    return out


# ---------------------------------------------------------------------------
# 6.  CTQI — CPAP Therapy Quality Index  (0–100)
# ---------------------------------------------------------------------------

def _piecewise_linear(value, breakpoints):
    # type: (float, List[Tuple[float, float]]) -> float
    """Interpolate a piecewise-linear function.

    breakpoints: list of (x, y) sorted by x ascending.
    Clamps to first/last y outside range.
    """
    if np.isnan(value):
        return float("nan")
    if value <= breakpoints[0][0]:
        return breakpoints[0][1]
    if value >= breakpoints[-1][0]:
        return breakpoints[-1][1]
    for i in range(len(breakpoints) - 1):
        x0, y0 = breakpoints[i]
        x1, y1 = breakpoints[i + 1]
        if x0 <= value <= x1:
            t = (value - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return breakpoints[-1][1]


# Breakpoint tables from PROJECT_GUIDE Section 6
_AHI_BP = [(0, 100), (1, 90), (5, 75), (10, 50), (20, 20), (30, 0)]
_SPO2_BP = [(88, 0), (90, 25), (92, 55), (94, 80), (96, 100)]
_LEAK_BP = [(0.05, 100), (0.10, 80), (0.20, 55), (0.40, 20), (0.60, 0)]
_FL_BP = [(0.10, 100), (0.20, 80), (0.30, 55), (0.50, 20), (0.70, 0)]
_PCI_BP = [(0, 100), (5, 80), (10, 55), (20, 20), (35, 0)]

# Weights: with SA2 / without SA2
_WEIGHTS_FULL = {"ahi": 0.25, "spo2": 0.25, "leak": 0.20, "fl": 0.20, "pci": 0.10}
_WEIGHTS_NO_SA2 = {"ahi": 0.40, "spo2": 0.0, "leak": 0.30, "fl": 0.30, "pci": 0.00}

_GRADES = [(90, "Excellent"), (75, "Good"), (55, "Fair"), (35, "Poor")]


def _ctqi_grade(score):
    # type: (float,) -> str
    if np.isnan(score):
        return "N/A"
    for threshold, label in _GRADES:
        if score >= threshold:
            return label
    return "Very Poor"


def compute_ctqi(ahi, spo2_median, leak_median, fl_mean, pci,
                 t90=0.0, odi_3=0.0, has_sa2=True,
                 session_minutes=480.0, min_session_minutes=60.0):
    # type: (...) -> dict
    """Compute the CPAP Therapy Quality Index.

    Parameters
    ----------
    ahi : float
        Apnea-Hypopnea Index (events/hr). From STR or custom detector.
    spo2_median : float
        Median SpO2 for the night.
    leak_median : float
        Median leak (Leak.50) in L/s.
    fl_mean : float
        Mean FlowLim.2s across the session.
    pci : float
        Pulse Change Index (events/hr) at 6 bpm threshold.
    t90 : float
        Minutes with SpO2 < 90%.
    odi_3 : float
        3% Oxygen Desaturation Index.
    has_sa2 : bool
        Whether oximetry data is available.
    session_minutes : float
        Session duration in minutes.
    min_session_minutes : float
        Minimum session for valid CTQI.

    Returns
    -------
    Dict with: ctqi, grade, components (sub-scores), weights used.
    """
    nan = float("nan")
    result = {"ctqi": nan, "grade": "N/A", "components": {},
              "weights": {}, "insufficient": False}

    if session_minutes < min_session_minutes:
        result["insufficient"] = True
        return result

    # Component sub-scores
    s_ahi = _piecewise_linear(ahi, _AHI_BP)
    s_spo2 = _piecewise_linear(spo2_median, _SPO2_BP) if has_sa2 else nan
    s_leak = _piecewise_linear(leak_median, _LEAK_BP)
    s_fl = _piecewise_linear(fl_mean, _FL_BP)
    s_pci = _piecewise_linear(pci, _PCI_BP) if has_sa2 else nan

    # SpO2 penalties
    if has_sa2 and not np.isnan(s_spo2):
        if t90 > 5.0:
            s_spo2 = max(0, s_spo2 - 10)
        if odi_3 > 10.0:
            s_spo2 = max(0, s_spo2 - 5)

    result["components"] = {
        "ahi_score": round(s_ahi, 1) if not np.isnan(s_ahi) else nan,
        "spo2_score": round(s_spo2, 1) if not np.isnan(s_spo2) else nan,
        "leak_score": round(s_leak, 1) if not np.isnan(s_leak) else nan,
        "fl_score": round(s_fl, 1) if not np.isnan(s_fl) else nan,
        "pci_score": round(s_pci, 1) if not np.isnan(s_pci) else nan,
    }

    # Select weights
    weights = _WEIGHTS_FULL if has_sa2 else _WEIGHTS_NO_SA2
    result["weights"] = dict(weights)

    # Handle individual NaN components (PCI may be NaN even with SA2
    # if pulse data is insufficient)
    scores = {"ahi": s_ahi, "spo2": s_spo2, "leak": s_leak,
              "fl": s_fl, "pci": s_pci}
    active_w = {}
    for k, w in weights.items():
        if not np.isnan(scores[k]) and w > 0:
            active_w[k] = w

    if len(active_w) == 0:
        return result

    # Renormalize weights
    total_w = sum(active_w.values())
    ctqi = sum(scores[k] * (w / total_w) for k, w in active_w.items())
    ctqi = max(0.0, min(100.0, ctqi))
    result["ctqi"] = round(ctqi, 1)
    result["grade"] = _ctqi_grade(ctqi)

    return result


# ---------------------------------------------------------------------------
# Per-Night Index Computation (orchestrator)
# ---------------------------------------------------------------------------

def compute_night_indices(session, detection_result, str_row=None):
    # type: (dict, dict, Optional[dict]) -> dict
    """Compute all per-night indices from session data and detection results.

    Parameters
    ----------
    session : dict
        Output from resmed_edf.read_session().
    detection_result : dict
        Output from event_detection.detect_all().
    str_row : dict, optional
        Single row from STR DataFrame for this night (for ResMed AHI, RIN, etc.).

    Returns
    -------
    Dict with:
      - event_counts: per-type event counts
      - ahi_custom: custom AHI estimate
      - indices: ODI, PCI, fl_index, snore_index
      - ctqi: CTQI result dict
      - signal_stats: per-signal descriptive statistics
      - uars: UARS assessment
    """
    result = {}
    duration_s = session.get("meta", {}).get("duration_s", 0)
    duration_h = duration_s / 3600.0
    duration_min = duration_s / 60.0

    # --- Event counts ---
    n_apnea = len(detection_result.get("events_apnea", []))
    n_hypopnea = len(detection_result.get("events_hypopnea", []))
    # Hypopneas with no usable oximetry: reported, not counted in custom AHI
    n_hypopnea_flow_only = len(detection_result.get("events_hypopnea_flow_only", []))
    n_desat = len(detection_result.get("events_desat", []))
    n_pulse = len(detection_result.get("events_pulse_surge", []))
    n_snore = len(detection_result.get("events_snore", []))
    n_fl = len(detection_result.get("events_flow_lim", []))

    result["event_counts"] = {
        "apnea": n_apnea,
        "hypopnea": n_hypopnea,
        "hypopnea_flow_only": n_hypopnea_flow_only,
        "desaturation": n_desat,
        "pulse_surge": n_pulse,
        "snore": n_snore,
        "flow_limitation": n_fl,
        "total_respiratory": n_apnea + n_hypopnea,
    }

    # --- Custom AHI ---
    ahi_custom = (n_apnea + n_hypopnea) / duration_h if duration_h > 0 else float("nan")
    result["ahi_custom"] = round(ahi_custom, 2)

    # --- Indices from detection metrics ---
    fl_m = detection_result.get("fl_metrics", {})
    ds_m = detection_result.get("desat_metrics", {})
    ps_m = detection_result.get("pulse_metrics", {})
    sn_m = detection_result.get("snore_metrics", {})

    result["indices"] = {
        "ODI_3": ds_m.get("ODI_3", float("nan")),
        "ODI_4": ds_m.get("ODI_4", float("nan")),
        "T90": ds_m.get("T90", float("nan")),
        "T88": ds_m.get("T88", float("nan")),
        "spo2_nadir": ds_m.get("spo2_nadir", float("nan")),
        "PCI_6bpm": ps_m.get("PCI_6bpm", float("nan")),
        "PCI_8bpm": ps_m.get("PCI_8bpm", float("nan")),
        "fl_mean": fl_m.get("fl_mean", float("nan")),
        "fl_index": fl_m.get("fl_index", float("nan")),
        "fl_frac_elevated": fl_m.get("fl_frac_elevated", float("nan")),
        "snore_index": sn_m.get("snore_index", float("nan")),
        "snore_heavy_index": sn_m.get("snore_heavy_index", float("nan")),
    }

    # --- Glasgow cumulative ---
    result["glasgow"] = detection_result.get("glasgow_cumulative", {})

    # --- UARS ---
    result["uars"] = detection_result.get("uars_assessment", {})

    # --- Signal stats ---
    ramp_min = 0
    if str_row is not None:
        ramp_min = str_row.get("S.RampTime", 0)
        if np.isnan(ramp_min):
            ramp_min = 0
    result["signal_stats"] = compute_all_signal_stats(session, ramp_minutes=ramp_min)

    # --- CTQI ---
    # Gather inputs — prefer STR values, fall back to custom/computed
    ahi_for_ctqi = ahi_custom
    if str_row is not None and not np.isnan(str_row.get("AHI", float("nan"))):
        ahi_for_ctqi = str_row["AHI"]

    spo2_median = float("nan")
    sa2_stats = result["signal_stats"].get("SpO2.1s", {})
    if sa2_stats:
        spo2_median = sa2_stats.get("median", float("nan"))

    leak_median = float("nan")
    leak_stats = result["signal_stats"].get("Leak.2s", {})
    if leak_stats:
        leak_median = leak_stats.get("median", float("nan"))
    # Fallback to STR Leak.50
    if np.isnan(leak_median) and str_row is not None:
        leak_median = str_row.get("Leak.50", float("nan"))

    fl_mean = result["indices"]["fl_mean"]
    pci = result["indices"]["PCI_6bpm"]
    t90 = result["indices"].get("T90", 0.0)
    odi_3 = result["indices"].get("ODI_3", 0.0)

    has_sa2 = not ds_m.get("insufficient_oximetry", True)

    result["ctqi"] = compute_ctqi(
        ahi=ahi_for_ctqi,
        spo2_median=spo2_median,
        leak_median=leak_median,
        fl_mean=fl_mean,
        pci=pci,
        t90=t90 if not np.isnan(t90) else 0.0,
        odi_3=odi_3 if not np.isnan(odi_3) else 0.0,
        has_sa2=has_sa2,
        session_minutes=duration_min,
    )

    return result


# ---------------------------------------------------------------------------
# Longitudinal Statistics (across nights)
# ---------------------------------------------------------------------------

def compute_longitudinal_stats(df_str, signals=None):
    # type: (object, Optional[List[str]]) -> Dict[str, dict]
    """Compute descriptive stats across nights for STR signals.

    Parameters
    ----------
    df_str : pandas.DataFrame
        Output from resmed_edf.read_str(). One row per active night.
    signals : list of str, optional
        Which columns to compute stats for. If None, uses default set.

    Returns
    -------
    Dict keyed by signal name, values are compute_stats() results.
    """
    DEFAULT_SIGNALS = [
        "AHI", "HI", "AI", "OAI", "CAI", "UAI", "RIN", "CSR",
        "Leak.50", "Leak.95", "Leak.70", "Leak.Max",
        "MaskPress.50", "MaskPress.95", "MaskPress.Max",
        "SpO2.50", "SpO2.95", "SpO2.Max", "SpO2Thresh",
        "TidVol.50", "RespRate.50", "MinVent.50",
        "Duration", "S.A.MinPress", "S.A.MaxPress", "S.EPR.Level",
    ]
    if signals is None:
        signals = DEFAULT_SIGNALS

    out = {}
    for sig in signals:
        if sig not in df_str.columns:
            continue
        arr = df_str[sig].values.astype(np.float64)
        out[sig] = compute_stats(arr, sig)
    return out


def build_nightly_summary_row(date_str, session, detection_result,
                              night_indices, str_row=None):
    # type: (str, dict, dict, dict, Optional[dict]) -> dict
    """Build a single row for the nightly summary table.

    Merges key metrics into a flat dict suitable for DataFrame construction.

    Parameters
    ----------
    date_str : str
        Night date as YYYYMMDD.
    session : dict
        Session from resmed_edf.read_session().
    detection_result : dict
        From event_detection.detect_all().
    night_indices : dict
        From compute_night_indices().
    str_row : dict, optional
        STR row for this night.

    Returns
    -------
    Flat dict with one key per metric.
    """
    row = {"date": date_str}

    meta = session.get("meta", {})
    row["duration_min"] = round(meta.get("duration_s", 0) / 60.0, 1)
    row["session_start"] = str(meta.get("start_dt", ""))

    # STR values
    if str_row is not None:
        for k in ["AHI", "OAI", "CAI", "HI", "RIN",
                   "Leak.50", "Leak.95", "MaskPress.50", "MaskPress.95",
                   "SpO2.50", "SpO2.95"]:
            row["str_" + k] = str_row.get(k, float("nan"))

    # Custom counts and indices
    ec = night_indices.get("event_counts", {})
    row["custom_apnea_count"] = ec.get("apnea", 0)
    row["custom_hypopnea_count"] = ec.get("hypopnea", 0)
    row["custom_hypopnea_flow_only_count"] = ec.get("hypopnea_flow_only", 0)
    row["custom_ahi"] = night_indices.get("ahi_custom", float("nan"))

    idx = night_indices.get("indices", {})
    for k in ["ODI_3", "ODI_4", "T90", "spo2_nadir",
              "PCI_6bpm", "PCI_8bpm",
              "fl_mean", "fl_index", "fl_frac_elevated",
              "snore_index", "snore_heavy_index"]:
        row[k] = idx.get(k, float("nan"))

    # CTQI
    ctqi_result = night_indices.get("ctqi", {})
    row["CTQI"] = ctqi_result.get("ctqi", float("nan"))
    row["CTQI_grade"] = ctqi_result.get("grade", "N/A")

    # UARS
    uars = night_indices.get("uars", {})
    row["uars_suggestive"] = uars.get("uars_suggestive", False)
    row["uars_score"] = uars.get("uars_score", 0)

    # Glasgow overall
    glas = night_indices.get("glasgow", {})
    row["glasgow_overall"] = glas.get("overall", float("nan"))
    row["glasgow_n_breaths"] = glas.get("n_breaths", 0)

    # Session timing (for sleep schedule analysis)
    start_dt = meta.get("start_dt")
    end_dt = meta.get("end_dt")
    if start_dt is not None:
        row["session_end"] = str(end_dt) if end_dt else ""
        row["bedtime_hour"] = start_dt.hour + start_dt.minute / 60.0
        # Normalize bedtime: hours after 6 PM (18:00). So 10 PM = 4.0, 1 AM = 7.0
        bh = start_dt.hour + start_dt.minute / 60.0
        row["bedtime_norm"] = bh - 18.0 if bh >= 18.0 else bh + 6.0
        row["weekday"] = start_dt.strftime("%A")
        row["is_weekend"] = start_dt.weekday() >= 4  # Fri/Sat night = weekend sleep
    else:
        row["session_end"] = ""
        row["bedtime_hour"] = float("nan")
        row["bedtime_norm"] = float("nan")
        row["weekday"] = ""
        row["is_weekend"] = False

    # Additional STR pressure stats (IPAP/EPAP for BPAP, MaskPress.Max)
    if str_row is not None:
        for k in ["MaskPress.Max",
                   "TgtIPAP.50", "TgtIPAP.95", "TgtIPAP.Max",
                   "TgtEPAP.50", "TgtEPAP.95", "TgtEPAP.Max",
                   "MaskOn_first", "MaskOff_last", "Duration",
                   "RespRate.50", "TidVol.50", "MinVent.50",
                   "BlowPress.95", "BlowPress.5"]:
            row["str_" + k] = str_row.get(k, float("nan"))

    return row
