"""
event_detection.py — CPAP Event Detection Module
SleepingAtTheTop — NightmaresOfTheBottom / AirSense 11

All detectors operate on numpy arrays with a start_dt and sampling rate.
Returns lists of event dicts matching EVE format:
    {onset_s, duration_s, label, time, confidence}

Dependencies: numpy, scipy.signal (for Gaussian smoothing in breath_segmentation)
No pyedflib, no MNE.

Reference algorithms:
- Apnea/Hypopnea: AASM 2.4 scoring rules
- Flow Limitation: ResMed PLD FlowLim.2s + optional BRP IFS
- Desaturation: AASM 3% ODI definition
- Pulse Surge: Pitson & Stradling (1998)
- Snore: ResMed community thresholds
- UARS Composite: Multi-criteria proxy
- Glasgow Index: DaveSkvn/GlasgowIndex (GPL-3.0) — 9 components
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Utility: sustained run detection
# ---------------------------------------------------------------------------

def _find_sustained_runs(mask, min_samples):
    # type: (np.ndarray, int) -> List[Tuple[int, int]]
    """Find contiguous runs of True in a boolean mask.

    Parameters
    ----------
    mask : np.ndarray
        1-D boolean array.
    min_samples : int
        Minimum run length to keep.

    Returns
    -------
    List of (start_idx, end_idx) tuples. end_idx is exclusive.
    """
    if mask.size == 0:
        return []
    runs = []
    in_run = False
    run_start = 0
    for i in range(mask.size):
        if mask[i] and not in_run:
            in_run = True
            run_start = i
        elif not mask[i] and in_run:
            in_run = False
            if i - run_start >= min_samples:
                runs.append((run_start, i))
    # handle run that extends to end of array
    if in_run and (mask.size - run_start) >= min_samples:
        runs.append((run_start, mask.size))
    return runs


def _merge_close_events(events, gap_s):
    # type: (List[dict], float) -> List[dict]
    """Merge events separated by less than gap_s seconds.

    Keeps the label of the first event in each merged group.
    """
    if len(events) <= 1:
        return events
    merged = [events[0].copy()]
    for ev in events[1:]:
        prev = merged[-1]
        prev_end = prev["onset_s"] + prev["duration_s"]
        if ev["onset_s"] - prev_end < gap_s:
            # extend previous event to cover this one
            new_end = ev["onset_s"] + ev["duration_s"]
            prev["duration_s"] = new_end - prev["onset_s"]
        else:
            merged.append(ev.copy())
    return merged


def _rolling_median_backward(arr, window_samples):
    # type: (np.ndarray, int) -> np.ndarray
    """Compute a backwards-looking rolling median.

    Uses pandas Series.rolling().median() which is C-optimized.
    Falls back to a strided numpy approach if pandas is unavailable.
    NaN-aware: NaN values are skipped in the median computation.
    """
    try:
        import pandas as _pd
        s = _pd.Series(arr, dtype=np.float64)
        out = s.rolling(window=window_samples, min_periods=1).median().values
        return out
    except ImportError:
        pass
    # Fallback: strided approach
    return _rolling_median_strided(arr, window_samples)

def _rolling_mean_centered(arr, window_samples):
    # type: (np.ndarray, int) -> np.ndarray
    """Centered rolling mean for amplitude envelope computation.

    Smooths out breath-cycle zero crossings while preserving the
    amplitude envelope of the flow signal.
    """
    try:
        import pandas as _pd
        s = _pd.Series(arr, dtype=np.float64)
        out = s.rolling(window=window_samples, min_periods=1, center=True).mean().values
        return out
    except ImportError:
        # Fallback: uniform convolution
        kernel = np.ones(window_samples) / window_samples
        return np.convolve(arr, kernel, mode='same')


def _rolling_quantile_backward(arr, window_samples, quantile):
    # type: (np.ndarray, int, float) -> np.ndarray
    """Backward-looking rolling quantile.

    Uses a higher quantile (e.g., 75th percentile) to capture
    'normal breathing amplitude' and resist contamination by
    events that pull the baseline down.
    """
    try:
        import pandas as _pd
        s = _pd.Series(arr, dtype=np.float64)
        out = s.rolling(window=window_samples, min_periods=1).quantile(quantile).values
        return out
    except ImportError:
        # Fallback: use median (less robust but functional)
        return _rolling_median_backward(arr, window_samples)

def _rolling_median_backward_fast(arr, window_samples):
    # type: (np.ndarray, int) -> np.ndarray
    """Fast backwards-looking rolling median for large arrays.

    Same as _rolling_median_backward — pandas handles both sizes
    efficiently via its C implementation.
    """
    return _rolling_median_backward(arr, window_samples)


def _rolling_median_strided(arr, window_samples):
    # type: (np.ndarray, int) -> np.ndarray
    """Fallback strided rolling median when pandas is unavailable."""
    n = arr.size
    stride = max(1, window_samples // 30)
    indices = list(range(0, n, stride))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    medians = np.empty(len(indices), dtype=np.float64)
    for j, i in enumerate(indices):
        start = max(0, i - window_samples)
        seg = arr[start:i] if i > 0 else arr[0:1]
        medians[j] = np.nanmedian(seg)
    out = np.interp(np.arange(n), indices, medians)
    return out


def _make_event(onset_s, duration_s, label, start_dt, confidence="high"):
    # type: (float, float, str, datetime, str) -> dict
    """Create a standard event dict."""
    return {
        "onset_s": round(onset_s, 3),
        "duration_s": round(duration_s, 3),
        "label": label,
        "time": start_dt + timedelta(seconds=onset_s),
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# 5.1  Apnea & Hypopnea Detection
# ---------------------------------------------------------------------------

def detect_apneas(
    flow,           # type: np.ndarray
    fs,             # type: float
    start_dt,       # type: datetime
    press=None,     # type: Optional[np.ndarray]
    baseline_window_s=60.0,   # type: float
    apnea_threshold=0.15,     # type: float
    hypopnea_threshold=0.30,  # type: float
    min_duration_s=8.0,       # type: float
    merge_gap_s=5.0,          # type: float
    pressure_var_threshold=0.02,  # type: float
    envelope_window_s=2.0,    # type: float
    frac_window_s=10.0,       # type: float
    frac_min=0.50,            # type: float
    hypopnea_threshold_ratio=0.70,        # type: float
    hypopnea_min_duration_s=10.0,         # type: float
    hypopnea_envelope_window_s=6.0,       # type: float
    hypopnea_baseline_window_s=120.0,     # type: float
    hypopnea_baseline_quantile=0.65,      # type: float
):
    # type: (...) -> List[dict]
    """Detect apnea and hypopnea events from BRP flow signal.

    Apneas: envelope at or below 15% of baseline (85%+ drop) for 8 s or
    more. The envelope is smoothed, so this approximates the AASM 90% /
    10 s rule.

    Hypopneas (AASM 2.4 flow criterion): a drop of 30% or more from
    baseline lasting at least 10 s. They use their own, longer envelope
    window (6 s, about one slow breath) and baseline so that ordinary
    breath-to-breath variation is not mistaken for a 30% drop. Events
    lasting exactly 10 s sit on the scoring boundary and are partly missed.
    The AASM also requires a 3% desaturation or an arousal; detect_all()
    applies that step (see confirm_hypopneas()).

    Uses a smoothed amplitude envelope with fraction-based detection:
    instead of requiring flow to stay continuously below threshold,
    requires that a sufficient fraction of samples within a sliding
    window are below threshold. This handles the real-world pattern
    where obstructive events contain brief recovery efforts mixed
    with flow cessation.

    Parameters
    ----------
    flow : np.ndarray
        BRP Flow.40ms signal in L/s.
    fs : float
        Sample rate in Hz (typically 25 for AirSense 11 BRP).
    start_dt : datetime
        Session start time (tz-naive, local).
    press : np.ndarray, optional
        BRP Press.40ms signal for OA/CA heuristic.
    baseline_window_s : float
        Rolling baseline window in seconds (default 60).
    apnea_threshold : float
        Envelope/baseline ratio for apnea (0.15 = 85% amplitude drop).
    hypopnea_threshold : float
        Deprecated and ignored; kept so old calls still work.
        Use hypopnea_threshold_ratio.
    min_duration_s : float
        Minimum event duration in seconds.
    merge_gap_s : float
        Merge events separated by less than this (seconds).
    pressure_var_threshold : float
        Press.40ms variance threshold for OA/CA heuristic (cmH2O^2).
    envelope_window_s : float
        Window for amplitude envelope smoothing (seconds).
    frac_window_s : float
        Sliding window size for fraction-based detection (seconds).
    frac_min : float
        Minimum fraction of samples in the window that must be below
        threshold to flag an event (0.50 = 50%).
    hypopnea_threshold_ratio : float
        Envelope/baseline ratio for hypopnea (0.70 = 30% amplitude drop).
    hypopnea_min_duration_s : float
        Minimum hypopnea duration (10 s, AASM).
    hypopnea_envelope_window_s, hypopnea_baseline_window_s,
    hypopnea_baseline_quantile : float
        Envelope smoothing window, baseline window and baseline quantile
        used for hypopnea detection only.

    Returns
    -------
    List of event dicts with keys: onset_s, duration_s, label, time, confidence.
    """
    if flow.size == 0:
        return []

    abs_flow = np.abs(flow).astype(np.float64)
    env_samples = max(1, int(envelope_window_s * fs))
    bl_samples = int(baseline_window_s * fs)
    min_samples = int(min_duration_s * fs)
    frac_win_samples = int(frac_window_s * fs)

    # Step 1: Smooth abs(flow) into amplitude envelope
    envelope = _rolling_mean_centered(abs_flow, env_samples)

    # Step 2: Compute baseline (75th percentile over 60s)
    baseline = _rolling_quantile_backward(envelope, bl_samples, 0.75)
    baseline = np.maximum(baseline, 0.01)

    # Step 3: Compute ratio
    ratio = envelope / baseline

    # Step 4: Fraction-based apnea detection
    # For each sample, compute fraction of nearby samples (within frac_window)
    # where ratio < apnea_threshold
    try:
        import pandas as _pd
        apnea_low = (ratio < apnea_threshold).astype(np.float64)
        apnea_frac = _pd.Series(apnea_low).rolling(
            window=frac_win_samples, min_periods=1, center=True
        ).mean().values
    except ImportError:
        kernel = np.ones(frac_win_samples) / frac_win_samples
        apnea_low = (ratio < apnea_threshold).astype(np.float64)
        apnea_frac = np.convolve(apnea_low, kernel, mode='same')

    apnea_mask = apnea_frac >= frac_min
    apnea_runs = _find_sustained_runs(apnea_mask, min_samples)

    events = []
    apnea_intervals = set()

    for (s, e) in apnea_runs:
        onset = s / fs
        dur = (e - s) / fs
        label = "Apnea"
        conf = "high"

        if press is not None:
            p_seg = press[s:e]
            p_var = np.var(p_seg) if p_seg.size > 0 else 0.0
            if p_var > pressure_var_threshold:
                label = "Obstructive Apnea (est.)"
                conf = "low"
            else:
                label = "Central Apnea (est.)"
                conf = "low"

        events.append(_make_event(onset, dur, label, start_dt, conf))
        for idx in range(s, e):
            apnea_intervals.add(idx)

    # Step 5: Fraction-based hypopnea detection (AASM: 30% drop, 10 s).
    # A separate, longer envelope and baseline are used here so normal
    # breath-to-breath variation does not look like a 30% drop.
    h_env = _rolling_mean_centered(
        abs_flow, max(1, int(hypopnea_envelope_window_s * fs)))
    h_base = _rolling_quantile_backward(
        h_env, int(hypopnea_baseline_window_s * fs), hypopnea_baseline_quantile)
    h_base = np.maximum(h_base, 0.01)
    with np.errstate(invalid="ignore"):
        h_ratio = h_env / h_base
    hypo_low = (h_ratio < hypopnea_threshold_ratio).astype(np.float64)
    try:
        import pandas as _pd
        hypo_frac = _pd.Series(hypo_low).rolling(
            window=frac_win_samples, min_periods=1, center=True
        ).mean().values
    except ImportError:
        kernel = np.ones(frac_win_samples) / frac_win_samples
        hypo_frac = np.convolve(hypo_low, kernel, mode='same')

    hypo_mask = (hypo_frac >= frac_min) & (~apnea_mask)
    for idx in apnea_intervals:
        if idx < hypo_mask.size:
            hypo_mask[idx] = False

    # A reduced-flow stretch that runs into an apnea is part of that apnea,
    # not a separate hypopnea, so drop hypopnea runs that touch an apnea.
    gap = int(merge_gap_s * fs)
    hypo_runs = [
        (s, e) for (s, e) in _find_sustained_runs(
            hypo_mask, int(hypopnea_min_duration_s * fs))
        if not any(s < ae + gap and e > as_ - gap for (as_, ae) in apnea_runs)
    ]
    for (s, e) in hypo_runs:
        onset = s / fs
        dur = (e - s) / fs
        events.append(_make_event(onset, dur, "Hypopnea", start_dt, "high"))

    # Sort and merge
    events.sort(key=lambda ev: ev["onset_s"])
    events = _merge_close_events(events, merge_gap_s)

    return events


def confirm_hypopneas(hypopneas, desat_events, pulse_surges, spo2=None,
                      fs_spo2=1.0, start_spo2=None, desat_window_s=30.0,
                      arousal_window_s=15.0, min_valid_frac=0.5):
    # type: (List[dict], List[dict], List[dict], Optional[np.ndarray], float, Optional[datetime], float, float, float) -> List[dict]
    """Apply the AASM 2.4 association rule to flow-detected hypopneas.

    A hypopnea is kept if a 3% desaturation starts between its onset and
    desat_window_s after it ends, or a pulse surge (used here as an arousal
    proxy, since there is no EEG) starts between its onset and
    arousal_window_s after it ends. Hypopneas with neither are dropped.

    Where there is no usable oximetry (no SA2 data, or SpO2 valid for less
    than min_valid_frac of the event window) a hypopnea cannot be confirmed
    or ruled out. It is returned marked confirmation="flow only" and
    confidence="low". detect_all() keeps these out of custom AHI and
    reports them separately as events_hypopnea_flow_only.

    Each returned event gets a "confirmation" key: "desaturation",
    "arousal proxy", or "flow only".
    """
    out = []
    have_ox = spo2 is not None and spo2.size > 0 and start_spo2 is not None
    for ev in hypopneas:
        t0 = ev["time"]
        t1 = t0 + timedelta(seconds=ev["duration_s"])
        ok_ox = False
        if have_ox:
            a = int((t0 - start_spo2).total_seconds() * fs_spo2)
            b = int((t1 - start_spo2).total_seconds() * fs_spo2
                    + desat_window_s * fs_spo2)
            a, b = max(a, 0), min(b, spo2.size)
            if b > a:
                seg = spo2[a:b]
                valid = np.isfinite(seg) & (seg > 0)
                ok_ox = valid.mean() >= min_valid_frac
        ev = dict(ev)
        if not ok_ox:
            ev["confirmation"] = "flow only"
            ev["confidence"] = "low"
            out.append(ev)
            continue
        d_end = t1 + timedelta(seconds=desat_window_s)
        a_end = t1 + timedelta(seconds=arousal_window_s)
        if any(t0 <= d["time"] <= d_end for d in desat_events):
            ev["confirmation"] = "desaturation"
            out.append(ev)
        elif any(t0 <= p["time"] <= a_end for p in pulse_surges):
            ev["confirmation"] = "arousal proxy"
            out.append(ev)
    return out


# ---------------------------------------------------------------------------
# 5.2  Flow Limitation Detection (PLD-based)
# ---------------------------------------------------------------------------

def detect_flow_limitation(
    flowlim,        # type: np.ndarray
    fs,             # type: float
    start_dt,       # type: datetime
    threshold=0.3,          # type: float
    heavy_threshold=0.5,    # type: float
    min_samples=3,          # type: int
):
    # type: (...) -> Tuple[List[dict], dict]
    """Detect flow limitation events from PLD FlowLim.2s signal.

    Parameters
    ----------
    flowlim : np.ndarray
        PLD FlowLim.2s signal (0–1 scale, 0.5 Hz).
    fs : float
        Sample rate (typically 0.5 Hz for PLD).
    start_dt : datetime
        Session start time.
    threshold : float
        Elevated flow limitation threshold (default 0.3).
    heavy_threshold : float
        Heavy flow limitation threshold (default 0.5).
    min_samples : int
        Minimum consecutive samples (default 3 = 6 seconds at 0.5 Hz).

    Returns
    -------
    Tuple of (events_list, metrics_dict).
    events_list: List of event dicts for elevated FL periods.
    metrics_dict: {fl_mean, fl_frac_elevated, fl_frac_heavy,
                   fl_event_count, fl_heavy_event_count, fl_index, session_hours}
    """
    events = []
    duration_h = flowlim.size / fs / 3600.0 if flowlim.size > 0 else 0.0

    metrics = {
        "fl_mean": float(np.nanmean(flowlim)) if flowlim.size > 0 else float("nan"),
        "fl_frac_elevated": 0.0,
        "fl_frac_heavy": 0.0,
        "fl_event_count": 0,
        "fl_heavy_event_count": 0,
        "fl_index": 0.0,
        "session_hours": duration_h,
    }

    if flowlim.size == 0:
        return events, metrics

    elevated_mask = flowlim > threshold
    heavy_mask = flowlim > heavy_threshold

    metrics["fl_frac_elevated"] = float(np.sum(elevated_mask)) / flowlim.size
    metrics["fl_frac_heavy"] = float(np.sum(heavy_mask)) / flowlim.size

    # Elevated FL events
    el_runs = _find_sustained_runs(elevated_mask, min_samples)
    for (s, e) in el_runs:
        onset = s / fs
        dur = (e - s) / fs
        events.append(_make_event(onset, dur, "Flow Limitation", start_dt))

    # Heavy FL events (subset)
    heavy_runs = _find_sustained_runs(heavy_mask, min_samples)

    metrics["fl_event_count"] = len(el_runs)
    metrics["fl_heavy_event_count"] = len(heavy_runs)
    metrics["fl_index"] = len(el_runs) / duration_h if duration_h > 0 else 0.0

    return events, metrics


# ---------------------------------------------------------------------------
# 5.3  Oxygen Desaturation Detection
# ---------------------------------------------------------------------------

def _filter_spo2_zeros(arr):
    # type: (np.ndarray) -> np.ndarray
    """Replace zeros and -1 sentinels with NaN (signal loss markers)."""
    out = arr.astype(np.float64).copy()
    out[out <= 0] = np.nan
    return out


def detect_desaturations(
    spo2_raw,       # type: np.ndarray
    fs,             # type: float
    start_dt,       # type: datetime
    drop_3pct=3.0,          # type: float
    drop_4pct=4.0,          # type: float
    baseline_window_s=120.0,  # type: float
    min_duration_s=0.0,     # type: float
    min_valid_minutes=10.0, # type: float
    min_consecutive=2,      # type: int
    merge_gap_s=5.0,        # type: float
):
    # type: (...) -> Tuple[List[dict], dict]
    """Detect oxygen desaturation events from SA2 SpO2.1s signal.

    A desaturation is a drop of at least drop_3pct (or drop_4pct) below the
    rolling baseline. Following the AASM definition there is no minimum
    duration: the drop only has to reach the threshold. Two safeguards keep
    noise out: the drop must hold for min_consecutive samples (2 s at 1 Hz,
    which rejects single-sample glitches), and dips separated by less than
    merge_gap_s are counted as one event.

    Parameters
    ----------
    spo2_raw : np.ndarray
        Raw SpO2.1s signal (1 Hz). Zeros = signal loss.
    fs : float
        Sample rate (typically 1 Hz).
    start_dt : datetime
        Session start time.
    drop_3pct : float
        Primary desaturation threshold (3%).
    drop_4pct : float
        Secondary (CMS) threshold (4%).
    baseline_window_s : float
        Rolling baseline window (120s).
    min_duration_s : float
        Minimum event duration (default 0, per AASM).
    min_valid_minutes : float
        Minimum valid signal required to compute ODI.
    min_consecutive : int
        Minimum consecutive samples below threshold before counting event.
    merge_gap_s : float
        Dips separated by less than this are merged into one event.

    Returns
    -------
    Tuple of (events_list, metrics_dict).
    """
    spo2 = _filter_spo2_zeros(spo2_raw)
    valid_count = int(np.sum(~np.isnan(spo2)))
    valid_minutes = valid_count / fs / 60.0
    valid_hours = valid_minutes / 60.0

    nan_metrics = {
        "ODI_3": float("nan"),
        "ODI_4": float("nan"),
        "T90": float("nan"),
        "T88": float("nan"),
        "T85": float("nan"),
        "spo2_nadir": float("nan"),
        "valid_minutes": valid_minutes,
        "insufficient_oximetry": True,
    }

    if valid_minutes < min_valid_minutes:
        return [], nan_metrics

    # Compute rolling baseline (backwards-looking, NaN-aware)
    win_samples = int(baseline_window_s * fs)
    baseline = _rolling_median_backward(spo2, win_samples)

    # Detect 3% desaturation events
    events_3 = _detect_desat_at_threshold(
        spo2, baseline, fs, start_dt, drop_3pct,
        int(min_duration_s * fs), min_consecutive, "Desaturation (3%)",
        int(merge_gap_s * fs)
    )
    # Detect 4% events (subset — for CMS ODI)
    events_4 = _detect_desat_at_threshold(
        spo2, baseline, fs, start_dt, drop_4pct,
        int(min_duration_s * fs), min_consecutive, "Desaturation (4%)",
        int(merge_gap_s * fs)
    )

    # Time below thresholds
    t90 = float(np.nansum(spo2 < 90)) / fs / 60.0  # minutes
    t88 = float(np.nansum(spo2 < 88)) / fs / 60.0
    t85 = float(np.nansum(spo2 < 85)) / fs / 60.0
    spo2_nadir = float(np.nanmin(spo2)) if valid_count > 0 else float("nan")

    metrics = {
        "ODI_3": len(events_3) / valid_hours if valid_hours > 0 else float("nan"),
        "ODI_4": len(events_4) / valid_hours if valid_hours > 0 else float("nan"),
        "T90": round(t90, 2),
        "T88": round(t88, 2),
        "T85": round(t85, 2),
        "spo2_nadir": spo2_nadir,
        "valid_minutes": round(valid_minutes, 1),
        "insufficient_oximetry": False,
    }

    return events_3, metrics


def _detect_desat_at_threshold(spo2, baseline, fs, start_dt, drop,
                               min_samples, min_consecutive, label,
                               merge_gap_samples=0):
    # type: (np.ndarray, np.ndarray, float, datetime, float, int, int, str, int) -> List[dict]
    """Internal: find desaturation events at a specific drop threshold."""
    events = []
    n = spo2.size
    # Boolean mask: sample is below baseline by >= drop AND is not NaN
    below = (~np.isnan(spo2)) & ((baseline - spo2) >= drop)

    # Artifact filter: require min_consecutive samples before starting count
    # Zero out isolated single-sample dips
    if min_consecutive > 1:
        filtered = np.zeros_like(below)
        consec = 0
        for i in range(n):
            if below[i]:
                consec += 1
                if consec >= min_consecutive:
                    # Mark all samples in this consecutive run
                    filtered[i - consec + 1: i + 1] = True
            else:
                consec = 0
        below = filtered

    runs = _find_sustained_runs(below, max(1, min_samples))
    # Merge dips separated by short partial recoveries into one event
    if merge_gap_samples > 0 and len(runs) > 1:
        merged = [list(runs[0])]
        for (s, e) in runs[1:]:
            if s - merged[-1][1] < merge_gap_samples:
                merged[-1][1] = e
            else:
                merged.append([s, e])
        runs = [tuple(r) for r in merged]
    for (s, e) in runs:
        seg = spo2[s:e]
        valid_seg = seg[~np.isnan(seg)]
        nadir = float(np.min(valid_seg)) if valid_seg.size > 0 else float("nan")
        bl_at_start = baseline[s]
        onset = s / fs
        dur = (e - s) / fs
        ev = _make_event(onset, dur, label, start_dt)
        ev["nadir_spo2"] = nadir
        ev["drop_magnitude"] = round(bl_at_start - nadir, 1)
        events.append(ev)

    return events


# ---------------------------------------------------------------------------
# 5.4  Pulse Surge / Arousal Surrogate Detection
# ---------------------------------------------------------------------------

def detect_pulse_surges(
    pulse_raw,      # type: np.ndarray
    fs,             # type: float
    start_dt,       # type: datetime
    surge_threshold_bpm=6.0,    # type: float
    min_duration_s=6.0,         # type: float
    baseline_window_s=30.0,     # type: float
    return_window_s=60.0,       # type: float
    return_margin_bpm=4.0,      # type: float
    min_valid_minutes=10.0,     # type: float
):
    # type: (...) -> Tuple[List[dict], dict]
    """Detect pulse surge events (autonomic arousal surrogate).

    Based on Pitson & Stradling (1998).

    Parameters
    ----------
    pulse_raw : np.ndarray
        SA2 Pulse.1s signal (1 Hz). Zeros = signal loss.
    fs : float
        Sample rate (typically 1 Hz).
    start_dt : datetime
        Session start time.
    surge_threshold_bpm : float
        Minimum pulse rise above baseline (default 6 bpm).
    min_duration_s : float
        Minimum sustained surge duration (default 6s).
    baseline_window_s : float
        Rolling baseline window (default 30s).
    return_window_s : float
        Window within which pulse must return toward baseline (60s).
    return_margin_bpm : float
        Pulse must drop back within this margin of baseline (4 bpm).
    min_valid_minutes : float
        Minimum valid signal required.

    Returns
    -------
    Tuple of (events_list, metrics_dict).
    metrics_dict includes PCI at both 6 and 8 bpm thresholds.
    """
    pulse = _filter_spo2_zeros(pulse_raw)
    valid_count = int(np.sum(~np.isnan(pulse)))
    valid_hours = valid_count / fs / 3600.0

    nan_metrics = {
        "PCI_6bpm": float("nan"),
        "PCI_8bpm": float("nan"),
        "valid_minutes": valid_count / fs / 60.0,
        "insufficient_pulse": True,
    }

    if valid_count / fs / 60.0 < min_valid_minutes:
        return [], nan_metrics

    win_samples = int(baseline_window_s * fs)
    baseline = _rolling_median_backward(pulse, win_samples)

    # Detect at 6 bpm threshold
    events_6 = _detect_surges_at_threshold(
        pulse, baseline, fs, start_dt, 6.0,
        int(min_duration_s * fs), return_window_s, return_margin_bpm
    )
    # Detect at 8 bpm threshold (stricter)
    events_8 = _detect_surges_at_threshold(
        pulse, baseline, fs, start_dt, 8.0,
        int(min_duration_s * fs), return_window_s, return_margin_bpm
    )

    metrics = {
        "PCI_6bpm": len(events_6) / valid_hours if valid_hours > 0 else float("nan"),
        "PCI_8bpm": len(events_8) / valid_hours if valid_hours > 0 else float("nan"),
        "valid_minutes": round(valid_count / fs / 60.0, 1),
        "insufficient_pulse": False,
    }

    return events_6, metrics


def _detect_surges_at_threshold(pulse, baseline, fs, start_dt, threshold,
                                min_samples, return_window_s, return_margin):
    # type: (np.ndarray, np.ndarray, float, datetime, float, int, float, float) -> List[dict]
    """Internal: detect pulse surges at a specific bpm threshold."""
    n = pulse.size
    above = (~np.isnan(pulse)) & ((pulse - baseline) >= threshold)
    runs = _find_sustained_runs(above, min_samples)

    events = []
    return_samples = int(return_window_s * fs)

    for (s, e) in runs:
        # Check return criterion: pulse drops back within margin
        # in the window after the surge ends
        end_check = min(e + return_samples, n)
        returned = False
        bl_at_end = baseline[min(e, n - 1)]
        for j in range(e, end_check):
            if not np.isnan(pulse[j]) and pulse[j] <= bl_at_end + return_margin:
                returned = True
                break
        if not returned:
            continue  # skip surges that don't return (likely sustained HR increase)

        seg = pulse[s:e]
        valid_seg = seg[~np.isnan(seg)]
        peak = float(np.max(valid_seg)) if valid_seg.size > 0 else float("nan")
        rise = peak - baseline[s]

        onset = s / fs
        dur = (e - s) / fs
        ev = _make_event(onset, dur, "Pulse Surge", start_dt)
        ev["peak_bpm"] = round(peak, 1)
        ev["rise_magnitude"] = round(rise, 1)
        events.append(ev)

    return events


# ---------------------------------------------------------------------------
# 5.5  Snore Event Detection
# ---------------------------------------------------------------------------

def detect_snore_events(
    snore,          # type: np.ndarray
    fs,             # type: float
    start_dt,       # type: datetime
    threshold=1.0,          # type: float
    heavy_threshold=2.5,    # type: float
    min_samples=3,          # type: int
):
    # type: (...) -> Tuple[List[dict], dict]
    """Detect snore events from PLD Snore.2s signal.

    Parameters
    ----------
    snore : np.ndarray
        PLD Snore.2s signal (0–5 scale, 0.5 Hz).
    fs : float
        Sample rate (typically 0.5 Hz).
    start_dt : datetime
        Session start time.
    threshold : float
        Elevated snore threshold (default 1.0).
    heavy_threshold : float
        Heavy snore threshold (default 2.5).
    min_samples : int
        Minimum consecutive samples (default 3 = 6s at 0.5 Hz).

    Returns
    -------
    Tuple of (events_list, metrics_dict).
    """
    duration_h = snore.size / fs / 3600.0 if snore.size > 0 else 0.0
    metrics = {
        "snore_mean": float(np.nanmean(snore)) if snore.size > 0 else float("nan"),
        "snore_frac_elevated": 0.0,
        "snore_frac_heavy": 0.0,
        "snore_event_count": 0,
        "snore_heavy_event_count": 0,
        "snore_index": 0.0,
        "snore_heavy_index": 0.0,
    }

    if snore.size == 0:
        return [], metrics

    el_mask = snore > threshold
    heavy_mask = snore > heavy_threshold
    metrics["snore_frac_elevated"] = float(np.sum(el_mask)) / snore.size
    metrics["snore_frac_heavy"] = float(np.sum(heavy_mask)) / snore.size

    events = []
    el_runs = _find_sustained_runs(el_mask, min_samples)
    for (s, e) in el_runs:
        onset = s / fs
        dur = (e - s) / fs
        events.append(_make_event(onset, dur, "Snore", start_dt))

    heavy_runs = _find_sustained_runs(heavy_mask, min_samples)

    metrics["snore_event_count"] = len(el_runs)
    metrics["snore_heavy_event_count"] = len(heavy_runs)
    metrics["snore_index"] = len(el_runs) / duration_h if duration_h > 0 else 0.0
    metrics["snore_heavy_index"] = len(heavy_runs) / duration_h if duration_h > 0 else 0.0

    return events, metrics


# ---------------------------------------------------------------------------
# 5.6  UARS Composite Proxy
# ---------------------------------------------------------------------------

def assess_uars_composite(
    rin,                # type: float
    fl_frac_elevated,   # type: float
    oai,                # type: float
    pressure_slope,     # type: float
    pci,                # type: float
    rin_threshold=5.0,          # type: float
    fl_threshold=0.15,          # type: float  # fraction of session with FlowLim.2s > 0.3
    oai_threshold=5.0,          # type: float
    pressure_slope_threshold=0.5,  # type: float
    pci_threshold=5.0,          # type: float
):
    # type: (...) -> dict
    """Assess UARS-suggestive composite proxy.

   At least three of five criteria must be met for uars_suggestive = True.

    Parameters
    ----------
    rin : float
        RERA index from STR (events/hr).
    fl_frac_elevated : float
        Fraction of session where FlowLim.2s > 0.3.
    oai : float
        Obstructive Apnea Index from STR (events/hr).
    pressure_slope : float
        Linear regression slope of MaskPress.2s over session (cmH2O/hr).
    pci : float
        Pulse Change Index (events/hr).
    """
    criteria = {}
    criteria["rin_elevated"] = (not np.isnan(rin)) and (rin > rin_threshold)
    criteria["fl_elevated"] = (not np.isnan(fl_frac_elevated)) and (fl_frac_elevated > fl_threshold)
    criteria["oai_low"] = (not np.isnan(oai)) and (oai < oai_threshold)
    criteria["pressure_rising"] = (not np.isnan(pressure_slope)) and \
                                  (pressure_slope > pressure_slope_threshold)
    criteria["pci_elevated"] = (not np.isnan(pci)) and (pci > pci_threshold)

    score = sum(1 for v in criteria.values() if v)
    return {
        "uars_suggestive": score >= 3,
        "uars_score": score,
        "uars_criteria": criteria,
    }


# ---------------------------------------------------------------------------
# 5.7  Breath Segmentation
# ---------------------------------------------------------------------------

def breath_segmentation(flow, fs, smooth_ms=100.0, min_breath_s=1.5,
                        max_breath_s=10.0):
    # type: (np.ndarray, float, float, float, float) -> List[dict]
    """Segment BRP flow signal into individual breath cycles.

    Parameters
    ----------
    flow : np.ndarray
        BRP Flow.40ms signal in L/s. Positive = inspiration.
    fs : float
        Sample rate in Hz.
    smooth_ms : float
        Gaussian smoothing sigma in milliseconds.
    min_breath_s : float
        Minimum plausible breath duration (reject shorter).
    max_breath_s : float
        Maximum plausible breath duration (reject longer — likely apnea).

    Returns
    -------
    List of breath dicts with keys:
        start_idx, end_idx, insp_start, insp_end, exp_start, exp_end,
        duration_s, tidal_vol_est, peak_insp_flow
    """
    if flow.size < int(2 * fs):
        return []

    # Smooth with Gaussian filter to reduce noise at zero-crossings
    try:
        from scipy.ndimage import gaussian_filter1d
        sigma_samples = (smooth_ms / 1000.0) * fs
        smoothed = gaussian_filter1d(flow.astype(np.float64), sigma=sigma_samples)
    except ImportError:
        # Fallback: simple moving average
        kernel_size = max(3, int((smooth_ms / 1000.0) * fs) | 1)  # ensure odd
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(flow.astype(np.float64), kernel, mode="same")

    # Find positive zero crossings (inspiration starts)
    # crossing from negative to positive
    sign = np.sign(smoothed)
    # Replace zeros with previous sign to avoid ambiguity
    for i in range(1, sign.size):
        if sign[i] == 0:
            sign[i] = sign[i - 1]

    crossings = []
    for i in range(1, sign.size):
        if sign[i - 1] <= 0 and sign[i] > 0:
            crossings.append(("up", i))    # inspiration start
        elif sign[i - 1] >= 0 and sign[i] < 0:
            crossings.append(("down", i))  # expiration start

    # Filter crossings too close together (< 0.3s apart = noise)
    min_gap = int(0.3 * fs)
    filtered = []
    for i, (ctype, cidx) in enumerate(crossings):
        if i > 0 and cidx - crossings[i - 1][1] < min_gap:
            continue
        filtered.append((ctype, cidx))
    crossings = filtered

    # Build breath cycles: up -> down -> up
    breaths = []
    i = 0
    while i < len(crossings) - 2:
        if crossings[i][0] != "up":
            i += 1
            continue
        # Find next down crossing
        j = i + 1
        while j < len(crossings) and crossings[j][0] != "down":
            j += 1
        if j >= len(crossings):
            break
        # Find next up crossing (start of next breath = end of this breath)
        k = j + 1
        while k < len(crossings) and crossings[k][0] != "up":
            k += 1
        if k >= len(crossings):
            break

        insp_start = crossings[i][1]
        exp_start = crossings[j][1]
        breath_end = crossings[k][1]

        dur_s = (breath_end - insp_start) / fs
        if dur_s < min_breath_s or dur_s > max_breath_s:
            i = k  # skip to next breath
            continue

        # Compute breath metrics on original (not smoothed) flow
        insp_seg = flow[insp_start:exp_start]
        tidal_vol = float(np.sum(np.maximum(insp_seg, 0))) / fs  # L (integral of flow)
        peak_insp = float(np.max(insp_seg)) if insp_seg.size > 0 else 0.0

        breaths.append({
            "start_idx": insp_start,
            "end_idx": breath_end,
            "insp_start": insp_start,
            "insp_end": exp_start,
            "exp_start": exp_start,
            "exp_end": breath_end,
            "duration_s": round(dur_s, 3),
            "tidal_vol_est": round(tidal_vol, 4),
            "peak_insp_flow": round(peak_insp, 4),
        })

        i = k  # advance to next breath

    return breaths


# ---------------------------------------------------------------------------
# Glasgow Index — 9 Components (per-breath)
# ---------------------------------------------------------------------------
# Reference: DaveSkvn/GlasgowIndex (GPL-3.0)
# Thresholds extracted from FlowLimits.js

def compute_glasgow_indices(flow, fs, breaths):
    # type: (np.ndarray, float, List[dict]) -> Tuple[List[dict], dict]
    """Compute Glasgow Index components for each breath.

    Parameters
    ----------
    flow : np.ndarray
        BRP Flow.40ms signal (L/min or L/s — internally consistent).
    fs : float
        Sample rate in Hz.
    breaths : list
        Output from breath_segmentation().

    Returns
    -------
    Tuple of (breaths_with_indices, cumulative_index).
    breaths_with_indices: each breath dict gets an 'indices' sub-dict.
    cumulative_index: fraction of breaths flagged for each component.
    """
    if len(breaths) == 0:
        empty_cum = {k: float("nan") for k in [
            "skew", "spike", "flat_top", "top_heavy", "double_peak",
            "no_pause", "insp_rate", "double_insp", "var_amplitude", "overall"
        ]}
        return breaths, empty_cum

    AMP_WINDOW = 5  # breaths for amplitude variance

    # --- Per-breath computations ---
    # DaveSkvn operates in L/min. Convert once and use his exact thresholds.
    # BRP Flow.40ms is in L/s; multiply by 60 for L/min.
    flow_lmin = flow * 60.0

    GREY_ZONE_UPPER = 5.0     # L/min — DaveSkvn's inspiration boundary
    MIN_PEAK_BUMP = 1.0       # L/min — double peak dip threshold
    FLAT_TOP_VAR = 0.75       # (L/min)^2 — flat top variance threshold
    AMP_VAR_THRESH = 4.0      # (L/min)^2 — variable amplitude threshold
    AMP_WINDOW = 5            # DaveSkvn uses 5 breaths for amplitude variance
    EXTRAP_SAMPLES = 25       # 1 second for no-pause extrapolation

    for i, br in enumerate(breaths):
        # Use L/min flow for the inspiration segment
        insp = flow_lmin[br["insp_start"]:br["insp_end"]].astype(np.float64)

        # Trim to portion above GREY_ZONE_UPPER (DaveSkvn's effective inspiration)
        above_grey = insp > GREY_ZONE_UPPER
        if above_grey.sum() < 8:
            br["indices"] = {k: False for k in [
                "skew", "spike", "flat_top", "top_heavy", "double_peak",
                "no_pause", "insp_rate", "double_insp", "var_amplitude"
            ]}
            br["indices"]["overall"] = 0
            continue

        # Find the contiguous region above grey zone
        first_above = int(np.argmax(above_grey))
        last_above = len(above_grey) - 1 - int(np.argmax(above_grey[::-1]))
        insp_eff = insp[first_above:last_above + 1]

        # DaveSkvn skips inspirations where the peak is not dominant:
        # he walks from the peak toward each edge and abandons if a higher
        # value is found. Approximate this by checking that the max is
        # a single clean peak — the max should be higher than the values
        # at both edges of insp_eff. If the first or last sample is the
        # peak, the inspiration is likely clipped or noisy — skip it.
        # Skip inspirations where peak is at the very edge (clipped)
        peak_idx_eff = int(np.argmax(insp_eff))
        if peak_idx_eff == 0 or peak_idx_eff == insp_eff.size - 1:
            br["indices"] = {k: False for k in [
                "skew", "spike", "flat_top", "top_heavy", "double_peak",
                "no_pause", "insp_rate", "double_insp", "var_amplitude"
            ]}
            br["indices"]["overall"] = 0
            continue

        if insp_eff.size < 8:
            br["indices"] = {k: False for k in [
                "skew", "spike", "flat_top", "top_heavy", "double_peak",
                "no_pause", "insp_rate", "double_insp", "var_amplitude"
            ]}
            br["indices"]["overall"] = 0
            continue

        peak = float(np.max(insp_eff))
        if peak <= GREY_ZONE_UPPER:
            br["indices"] = {k: False for k in [
                "skew", "spike", "flat_top", "top_heavy", "double_peak",
                "no_pause", "insp_rate", "double_insp", "var_amplitude"
            ]}
            br["indices"]["overall"] = 0
            continue

        br["indices"] = {}
        overall = 0
        mid = insp_eff.size // 2

    # 1. SKEW — volume asymmetry around midpoint
        # DaveSkvn: leftPercent < 45 or > 55
        # DaveSkvn excludes the midpoint sample from both sums.
        # DaveSkvn only computes skew for inspirations > 12 samples (>0.5s),
        # setting leftPercent = 50 (benign) for shorter ones.
        mid = insp_eff.size // 2
        if insp_eff.size > 12:
            left_vol = float(np.sum(insp_eff[:mid]))
            right_vol = float(np.sum(insp_eff[mid + 1:]))  # exclude midpoint
            total_vol = left_vol + right_vol
            left_pct = (left_vol / total_vol * 100.0) if total_vol > 0 else 50.0
        else:
            left_pct = 50.0  # benign default for short inspirations
        skew = left_pct < 45 or left_pct > 55
        br["indices"]["skew"] = skew
        if skew:
            overall += 1

        # 2. SPIKE — top90Percent < 20
        threshold_90 = peak * 0.90
        above_90_count = int(np.sum(insp_eff >= threshold_90))
        top90_pct = (above_90_count / insp_eff.size * 100.0)
        spike = top90_pct < 20
        br["indices"]["spike"] = spike
        if spike:
            overall += 1

        # 3. FLAT TOP — midVar < 0.75 (L/min)^2
        q25 = insp_eff.size // 4
        q75 = insp_eff.size - q25
        mid_section = insp_eff[q25:q75]
        if mid_section.size > 1:
            mid_var = float(np.var(mid_section))
            flat_top = mid_var < FLAT_TOP_VAR
        else:
            flat_top = False
        br["indices"]["flat_top"] = flat_top
        if flat_top:
            overall += 1

        # 4. TOP HEAVY — top90Percent > 40
        top_heavy = top90_pct > 40
        br["indices"]["top_heavy"] = top_heavy
        if top_heavy:
            overall += 1

        # 5. DOUBLE PEAK — dip > 1 L/min between two peaks
        first_peak_found = False
        looking_for_next = False
        last_max = 0.0
        lowest_post = 0.0
        double_peak = False
        for val in insp_eff:
            fval = float(val)
            if not first_peak_found:
                if fval > last_max:
                    last_max = fval
                elif fval < last_max:
                    first_peak_found = True
            elif not looking_for_next:
                if (last_max - fval) > MIN_PEAK_BUMP:
                    looking_for_next = True
                    lowest_post = fval
            else:
                if fval < lowest_post:
                    lowest_post = fval
                elif fval > (lowest_post + MIN_PEAK_BUMP):
                    double_peak = True
                    break
        br["indices"]["double_peak"] = double_peak
        if double_peak:
            overall += 1

        # 6. NO PAUSE — extrapolate expiration to find pre-rest duration
        # DaveSkvn: find peak expiration (min below -10 L/min), look 25
        # samples later, extrapolate line to zero crossing. preRest is
        # measured from extrapolated zero to the grey zone crossing
        # (where flow rises above +5 L/min), not the zero crossing.
        no_pause = False
        if i > 0:
            search_from = breaths[i - 1]["exp_start"]
            search_to = br["insp_start"]
            if search_to > search_from:
                exp_seg = flow_lmin[search_from:search_to]
                if exp_seg.size > EXTRAP_SAMPLES:
                    min_idx_local = int(np.argmin(exp_seg))
                    min_idx_global = search_from + min_idx_local
                    min_val = float(flow_lmin[min_idx_global])

                    # DaveSkvn requires min < -10 L/min (GREY_ZONE_LOWER)
                    if min_val < -10.0 and min_idx_global + EXTRAP_SAMPLES < len(flow_lmin):
                        val_plus_1s = float(flow_lmin[min_idx_global + EXTRAP_SAMPLES])
                        if val_plus_1s < 0:
                            denom = min_val - val_plus_1s
                            if abs(denom) > 0.01:
                                intersection = min_idx_global + round(
                                    EXTRAP_SAMPLES * min_val / denom
                                )
                                # Use grey zone crossing as inspiration start
                                # (first_above is relative to insp array, convert to global)
                                grey_zone_start = br["insp_start"] + first_above
                                pre_rest = grey_zone_start - intersection
                                no_pause = pre_rest < 10
                        else:
                            no_pause = True
        br["indices"]["no_pause"] = no_pause
        if no_pause:
            overall += 1

        # 7. INSP RATE — > 20 bpm over last 5 breaths
        if i >= AMP_WINDOW:
            span_samples = br["start_idx"] - breaths[i - AMP_WINDOW]["start_idx"]
            span_s = span_samples / fs if span_samples > 0 else 999
            bpm = (AMP_WINDOW * 60.0 / span_s) if span_s > 0 else 0
            insp_rate = bpm > 20
        else:
            insp_rate = False
        br["indices"]["insp_rate"] = insp_rate
        if insp_rate:
            overall += 1

        # 8. DOUBLE INSP — orphaned inspiration (no matching expiration)
        # DaveSkvn matches expirations to inspirations; orphans are flagged.
        # Approximation: flag when expiration is very short relative to
        # inspiration (suggests two insp cycles sharing one exp).
        insp_dur = (br["insp_end"] - br["insp_start"]) / fs
        exp_dur = (br["exp_end"] - br["exp_start"]) / fs
        double_insp = (exp_dur < insp_dur * 0.4) if insp_dur > 0 else False
        br["indices"]["double_insp"] = double_insp
        if double_insp:
            overall += 1

        # 9. VAR AMPLITUDE — variance of peak amplitude > 4 (L/min)^2
        # DaveSkvn: ampVar over AMP_WINDOW breaths
        if i >= AMP_WINDOW:
            peaks_lmin = [breaths[i - j]["peak_insp_flow"] * 60.0
                          for j in range(AMP_WINDOW)]
            amp_mean = float(np.mean(peaks_lmin))
            amp_var = float(np.mean([(p - amp_mean) ** 2 for p in peaks_lmin]))
            var_amp = amp_var > AMP_VAR_THRESH
        else:
            var_amp = False
        br["indices"]["var_amplitude"] = var_amp
        if var_amp:
            overall += 1

        br["indices"]["overall"] = overall

    # --- Cumulative indices ---
    n_breaths = len(breaths)
    valid_breaths = [b for b in breaths if "indices" in b and "skew" in b["indices"]]
    n_valid = len(valid_breaths)

    if n_valid == 0:
        cum = {k: float("nan") for k in [
            "skew", "spike", "flat_top", "top_heavy", "double_peak",
            "no_pause", "insp_rate", "double_insp", "var_amplitude", "overall"
        ]}
        return breaths, cum

    cum = {}
    for key in ["skew", "spike", "flat_top", "top_heavy", "double_peak",
                "no_pause", "insp_rate", "double_insp", "var_amplitude"]:
        count = sum(1 for b in valid_breaths if b["indices"].get(key, False))
        cum[key] = round(count / n_valid, 4)

    cum["overall"] = round(sum(cum[k] for k in cum if k != "overall"), 4)
    cum["n_breaths"] = n_valid

    return breaths, cum


# ---------------------------------------------------------------------------
# Pressure slope helper (for UARS composite)
# ---------------------------------------------------------------------------

def compute_pressure_slope(mask_press, fs):
    # type: (np.ndarray, float) -> float
    """Compute linear regression slope of mask pressure over session.

    Parameters
    ----------
    mask_press : np.ndarray
        PLD MaskPress.2s signal.
    fs : float
        Sample rate (0.5 Hz).

    Returns
    -------
    Slope in cmH2O/hour.
    """
    if mask_press.size < 10:
        return float("nan")
    valid = ~np.isnan(mask_press)
    if np.sum(valid) < 10:
        return float("nan")
    t_hours = np.arange(mask_press.size)[valid] / fs / 3600.0
    p_valid = mask_press[valid]
    # Simple least-squares
    n = t_hours.size
    t_mean = np.mean(t_hours)
    p_mean = np.mean(p_valid)
    num = np.sum((t_hours - t_mean) * (p_valid - p_mean))
    den = np.sum((t_hours - t_mean) ** 2)
    slope = num / den if den > 0 else 0.0
    return float(round(slope, 4))


# ---------------------------------------------------------------------------
# Orchestrator: detect_all
# ---------------------------------------------------------------------------

def detect_all(session, min_session_minutes=60.0, str_row=None):
    # type: (dict, float) -> dict
    """Run all event detectors on a single-night session.

    Parameters
    ----------
    session : dict
        Output from resmed_edf.read_session(). Expected keys:
        - 'brp': dict with 'Flow.40ms', 'Press.40ms', 'fs', 'start_dt'
        - 'pld': dict with signal arrays and 'fs', 'start_dt'
        - 'sa2': dict with 'SpO2.1s', 'Pulse.1s', 'fs', 'start_dt' (may be None)
        - 'eve': dict with 'events' list (ResMed scored events)
        - 'meta': dict with 'duration_s'
    
    str_row : dict or pandas.Series, optional
        Single row from STR DataFrame for this night. Supplies RIN and OAI
        for the UARS composite. When omitted, those two criteria cannot
        fire (input values are NaN).
    
    Returns
    -------
    dict with keys:
        events_apnea, events_hypopnea (confirmed by desaturation or
        arousal proxy; these count toward custom AHI),
        events_hypopnea_flow_only (no usable oximetry, not in custom AHI),
        events_flow_lim, events_desat,
        events_pulse_surge, events_snore, all_custom_events,
        fl_metrics, desat_metrics, pulse_metrics, snore_metrics,
        uars_assessment, breaths, glasgow_cumulative,
        session_too_short (bool)
    """
    result = {
        "events_apnea": [],
        "events_hypopnea": [],
        "events_hypopnea_flow_only": [],
        "events_flow_lim": [],
        "events_desat": [],
        "events_pulse_surge": [],
        "events_snore": [],
        "all_custom_events": [],
        "fl_metrics": {},
        "desat_metrics": {},
        "pulse_metrics": {},
        "snore_metrics": {},
        "uars_assessment": {},
        "breaths": [],
        "glasgow_cumulative": {},
        "session_too_short": False,
    }

    # Session duration check
    duration_s = session.get("meta", {}).get("duration_s", 0)
    if duration_s < min_session_minutes * 60:
        result["session_too_short"] = True
        return result

    # --- BRP: Apnea/Hypopnea + Breath Segmentation ---
    brp = session.get("brp")
    if brp is not None:
        flow = brp.get("Flow.40ms")
        fs_brp = brp.get("fs", 25.0)
        start_brp = brp.get("start_dt", datetime(2000, 1, 1))
        press = brp.get("Press.40ms")

        if flow is not None and flow.size > 0:
            # Apnea/Hypopnea
            ah_events = detect_apneas(flow, fs_brp, start_brp, press=press)
            result["events_apnea"] = [e for e in ah_events
                                       if "Apnea" in e["label"]]
            result["events_hypopnea"] = [e for e in ah_events
                                          if e["label"] == "Hypopnea"]

            # Breath segmentation + Glasgow
            breaths = breath_segmentation(flow, fs_brp)
            result["breaths"] = breaths
            if len(breaths) > 0:
                breaths, glasgow_cum = compute_glasgow_indices(
                    flow, fs_brp, breaths
                )
                result["breaths"] = breaths
                result["glasgow_cumulative"] = glasgow_cum

    # --- PLD: Flow Limitation + Snore ---
    pld = session.get("pld")
    if pld is not None:
        fs_pld = pld.get("fs", 0.5)
        start_pld = pld.get("start_dt", datetime(2000, 1, 1))

        flowlim = pld.get("FlowLim.2s")
        if flowlim is not None and flowlim.size > 0:
            fl_events, fl_metrics = detect_flow_limitation(
                flowlim, fs_pld, start_pld
            )
            result["events_flow_lim"] = fl_events
            result["fl_metrics"] = fl_metrics

        snore = pld.get("Snore.2s")
        if snore is not None and snore.size > 0:
            sn_events, sn_metrics = detect_snore_events(
                snore, fs_pld, start_pld
            )
            result["events_snore"] = sn_events
            result["snore_metrics"] = sn_metrics

    # --- SA2: Desaturation + Pulse Surge ---
    sa2 = session.get("sa2")
    if sa2 is not None:
        fs_sa2 = sa2.get("fs", 1.0)
        start_sa2 = sa2.get("start_dt", datetime(2000, 1, 1))

        spo2 = sa2.get("SpO2.1s")
        if spo2 is not None and spo2.size > 0:
            desat_events, desat_metrics = detect_desaturations(
                spo2, fs_sa2, start_sa2
            )
            result["events_desat"] = desat_events
            result["desat_metrics"] = desat_metrics

        pulse = sa2.get("Pulse.1s")
        if pulse is not None and pulse.size > 0:
            ps_events, ps_metrics = detect_pulse_surges(
                pulse, fs_sa2, start_sa2
            )
            result["events_pulse_surge"] = ps_events
            result["pulse_metrics"] = ps_metrics

    # --- Hypopnea confirmation (AASM: 3% desat or arousal) ---
    if result["events_hypopnea"]:
        _spo2 = _start = None
        _fs = 1.0
        if sa2 is not None and sa2.get("SpO2.1s") is not None:
            _spo2 = np.asarray(sa2.get("SpO2.1s"), dtype=np.float64)
            _fs = sa2.get("fs", 1.0)
            _start = sa2.get("start_dt")
        result["events_hypopnea_unconfirmed_n"] = len(result["events_hypopnea"])
        _checked = confirm_hypopneas(
            result["events_hypopnea"], result["events_desat"],
            result["events_pulse_surge"], spo2=_spo2, fs_spo2=_fs,
            start_spo2=_start)
        # Hypopneas that could not be checked against oximetry are kept
        # out of events_hypopnea (and so out of custom AHI and the EVE
        # comparison) and reported separately.
        result["events_hypopnea"] = [
            e for e in _checked if e["confirmation"] != "flow only"]
        result["events_hypopnea_flow_only"] = [
            e for e in _checked if e["confirmation"] == "flow only"]

    # --- UARS Composite ---
    # RIN and OAI come from STR (not present in session metadata);
    # they are passed in via str_row. Missing str_row => both NaN =>
    # those two criteria cannot fire.
    def _get(col):
        if str_row is None:
            return float("nan")
        try:
            v = str_row.get(col, float("nan"))
        except AttributeError:
            # pandas Series supports .get; dict does too; plain objects don't
            v = getattr(str_row, col, float("nan"))
        return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else float("nan")

    rin = _get("RIN")
    oai = _get("OAI")
    fl_frac_elevated = result["fl_metrics"].get("fl_frac_elevated", float("nan"))
    pci = result["pulse_metrics"].get("PCI_6bpm", float("nan"))

    pressure_slope = float("nan")
    if pld is not None:
        mp = pld.get("MaskPress.2s")
        if mp is not None and mp.size > 0:
            pressure_slope = compute_pressure_slope(mp, pld.get("fs", 0.5))

    result["uars_assessment"] = assess_uars_composite(
        rin=rin, fl_frac_elevated=fl_frac_elevated, oai=oai,
        pressure_slope=pressure_slope, pci=pci
    )

    # --- Aggregate all custom events ---
    all_events = (
        result["events_apnea"] +
        result["events_hypopnea"] +
        result["events_flow_lim"] +
        result["events_desat"] +
        result["events_pulse_surge"] +
        result["events_snore"]
    )
    all_events.sort(key=lambda e: e["onset_s"])
    result["all_custom_events"] = all_events

    return result


# ---------------------------------------------------------------------------
# Validation helper: compare custom events vs EVE ground truth
# ---------------------------------------------------------------------------

EVE_DEFAULT_DURATION_S = 10.0


def eve_event_window(ev, default_duration_s=EVE_DEFAULT_DURATION_S):
    # type: (dict, float) -> Tuple[datetime, datetime]
    """Return the (start, end) datetimes of a ResMed EVE event.

    ResMed timestamps each event at its END; the recorded duration runs
    back from that time. AirSense 10/11 hypopneas and RERAs carry no
    duration, so default_duration_s (10 s, the scoring minimum) is assumed
    for them.
    """
    end = ev["time"]
    dur = ev.get("duration_s") or 0.0
    if dur <= 0:
        dur = default_duration_s
    return end - timedelta(seconds=dur), end


def compare_with_eve(custom_events, eve_events, tolerance_s=15.0):
    # type: (List[dict], List[dict], float) -> dict
    """Compare custom-detected events against ResMed EVE annotations.

    Uses absolute 'time' datetimes with midpoint comparison to avoid
    reference-frame mismatches between BRP and EVE onset_s values.
    EVE events are placed with eve_event_window(): ResMed's timestamp is
    the END of the event, so its midpoint lies before the timestamp.

    Parameters
    ----------
    custom_events : list
        Events from our detectors (apneas + hypopneas only).
    eve_events : list
        Events from EVE file (ResMed scored).
    tolerance_s : float
        Maximum midpoint time difference to consider a match (seconds).

    Returns
    -------
    dict with: matched, custom_only, eve_only, precision, recall, f1
    """
    resp_labels = {"Apnea", "Hypopnea", "Obstructive Apnea",
                   "Central Apnea", "Unclassified Apnea",
                   "Obstructive Apnea (est.)", "Central Apnea (est.)"}

    custom_resp = [e for e in custom_events if any(
        lbl in e["label"] for lbl in resp_labels
    )]
    eve_resp = [e for e in eve_events if any(
        lbl in e.get("label", "") for lbl in resp_labels
    )]

    matched = 0
    eve_matched = set()

    for ce in custom_resp:
        ce_time = ce.get("time")
        ce_dur = ce.get("duration_s", 0)

        best_j = None
        best_diff = tolerance_s + 1

        for j, ee in enumerate(eve_resp):
            if j in eve_matched:
                continue
            ee_time = ee.get("time")
            ee_dur = ee.get("duration_s") or 0.0
            if ee_dur <= 0:
                ee_dur = EVE_DEFAULT_DURATION_S

            if ce_time is not None and ee_time is not None:
                ce_mid = ce_time + timedelta(seconds=ce_dur / 2.0)
                ee_start, ee_end = eve_event_window(ee)
                ee_mid = ee_start + (ee_end - ee_start) / 2
                diff = abs((ce_mid - ee_mid).total_seconds())
            else:
                ce_mid_s = ce.get("onset_s", 0) + ce_dur / 2.0
                ee_mid_s = ee.get("onset_s", 0) - ee_dur / 2.0
                diff = abs(ce_mid_s - ee_mid_s)

            if diff < best_diff:
                best_diff = diff
                best_j = j

        if best_j is not None and best_diff <= tolerance_s:
            matched += 1
            eve_matched.add(best_j)

    n_custom = len(custom_resp)
    n_eve = len(eve_resp)
    custom_only = n_custom - matched
    eve_only = n_eve - matched

    precision = matched / n_custom if n_custom > 0 else float("nan")
    recall = matched / n_eve if n_eve > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else float("nan"))

    return {
        "matched": matched,
        "custom_total": n_custom,
        "eve_total": n_eve,
        "custom_only": custom_only,
        "eve_only": eve_only,
        "precision": round(precision, 3) if precision == precision else precision,
        "recall": round(recall, 3) if recall == recall else recall,
        "f1": round(f1, 3) if f1 == f1 else f1,
    }