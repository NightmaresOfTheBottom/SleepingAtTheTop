"""
settings_changes.py — CPAP Settings Change Detection & Comparative Analysis
SleepingAtTheTop — NightmaresOfTheBottom / AirSense 11

Detects therapy settings changes from STR.edf longitudinal data,
computes before/after comparisons using Mann-Whitney U, and fits
therapy trajectory regressions.

Dependencies: numpy, pandas, scipy.stats (for Mann-Whitney U)
No pyedflib, no MNE.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None  # degrade gracefully if pandas unavailable

try:
    from scipy.stats import mannwhitneyu
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# 7.1  Settings signals to monitor
# ---------------------------------------------------------------------------

SETTINGS_SIGNALS = [
    "S.A.MinPress",
    "S.A.MaxPress",
    "S.EPR.Level",
    "S.EPR.EPRType",
    "S.Mask",
    "S.RampEnable",
    "S.RampTime",
    "S.C.Press",
]

# Rounding precision per signal to suppress EDF float conversion noise.
# Pressure signals: 1 decimal (ResMed adjusts in 0.2 cmH2O steps).
# Integer-valued signals (EPR level, mask code, ramp enable): round to int.
_ROUND_DIGITS = {
    "S.A.MinPress": 1, "S.A.MaxPress": 1, "S.C.Press": 1,
    "S.RampTime": 0, "S.EPR.Level": 0, "S.EPR.EPRType": 0,
    "S.Mask": 0, "S.RampEnable": 0,
}

# Metrics to compare in before/after analysis
COMPARISON_METRICS = [
    "AHI", "OAI", "CAI", "HI", "RIN",
    "Leak.50", "Leak.95",
    "SpO2.50", "SpO2.95",
    "MaskPress.50", "MaskPress.95",
]

# Metrics where lower is better (for direction labeling)
# MaskPress is intentionally excluded — it is a dependent variable that
# follows settings changes, so "improved"/"worsened" is misleading.
_LOWER_IS_BETTER = {"AHI", "OAI", "CAI", "HI", "RIN",
                    "Leak.50", "Leak.95"}
_HIGHER_IS_BETTER = {"SpO2.50", "SpO2.95"}


# ---------------------------------------------------------------------------
# 7.2  Change Detection
# ---------------------------------------------------------------------------

def detect_settings_changes(df_str):
    # type: (pd.DataFrame) -> pd.DataFrame
    """Detect settings changes from STR DataFrame.

    Parameters
    ----------
    df_str : pd.DataFrame
        One row per day from resmed_edf.read_str().
        Must have a 'Date' or datetime index column.

    Returns
    -------
    pd.DataFrame with columns:
        date, signal, old_value, new_value, change_magnitude
    Sorted by date.
    """
    if pd is None:
        raise ImportError("pandas is required for settings_changes")

    changes = []

    # Always use the DataFrame index (DatetimeIndex from read_str).
    # The "Date" column in STR is a numeric day counter, NOT a real date.
    dates = df_str.index

    for sig in SETTINGS_SIGNALS:
        if sig not in df_str.columns:
            continue

        values = df_str[sig].values.astype(np.float64)
        rnd = _ROUND_DIGITS.get(sig, 2)
        last_valid = None
        last_valid_idx = None

        for i in range(len(values)):
            v = values[i]
            if np.isnan(v):
                continue
            v = round(float(v), rnd)

            if last_valid is not None and v != last_valid:
                changes.append({
                    "date": dates[i],
                    "signal": sig,
                    "old_value": last_valid,
                    "new_value": v,
                    "change_magnitude": round(abs(v - last_valid), 4),
                })

            last_valid = v
            last_valid_idx = i

    df_changes = pd.DataFrame(changes)
    if len(df_changes) == 0:
        return pd.DataFrame(columns=["date", "signal", "old_value",
                                     "new_value", "change_magnitude"])

    df_changes = df_changes.sort_values("date").reset_index(drop=True)
    return df_changes


def group_simultaneous_changes(df_changes):
    # type: (pd.DataFrame) -> List[dict]
    """Group changes on the same date into adjustment sessions.

    Parameters
    ----------
    df_changes : pd.DataFrame
        Output from detect_settings_changes().

    Returns
    -------
    List of dicts, each with:
        date, changes (list of {signal, old_value, new_value, change_magnitude})
    """
    if pd is None or len(df_changes) == 0:
        return []

    groups = []
    for date_val, grp in df_changes.groupby("date"):
        changes_list = []
        for _, row in grp.iterrows():
            changes_list.append({
                "signal": row["signal"],
                "old_value": row["old_value"],
                "new_value": row["new_value"],
                "change_magnitude": row["change_magnitude"],
            })
        groups.append({"date": date_val, "changes": changes_list})

    return groups


# ---------------------------------------------------------------------------
# 7.3  Before/After Comparison
# ---------------------------------------------------------------------------

def _rank_biserial(u, n1, n2):
    # type: (float, int, int) -> float
    """Compute rank-biserial correlation from Mann-Whitney U.

    r = 1 - (2U)/(n1*n2), where U is scipy's U for group 1.
    Range: -1 to +1. Positive means group 2 tends to be larger
    (negative means group 1 tends to be larger).
    """
    denom = n1 * n2
    if denom == 0:
        return float("nan")
    return 1.0 - (2.0 * u) / denom


ALPHA = 0.05  # significance level after Holm correction


def _holm_adjust(p_values):
    # type: (List[float]) -> List[float]
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p_values[i])
        running = max(running, val)
        adj[i] = running
    return adj


def _direction_label(before_median, after_median, metric, effect_size):
    # type: (float, float, str, float) -> str
    """Label change direction as improved/worsened/no change."""
    if abs(effect_size) < 0.1:
        return "no change"

    went_down = after_median < before_median

    if metric in _LOWER_IS_BETTER:
        return "improved" if went_down else "worsened"
    elif metric in _HIGHER_IS_BETTER:
        return "improved" if not went_down else "worsened"
    else:
        return "decreased" if went_down else "increased"


def compare_before_after(df_str, change_date, window=7, min_nights=4,
                         metrics=None, ctqi_series=None):
    # type: (pd.DataFrame, object, int, int, Optional[List[str]], Optional[pd.Series]) -> dict
    """Compute before/after comparison for a settings change.

    Parameters
    ----------
    df_str : pd.DataFrame
        STR data, one row per day.
    change_date : date-like
        The date of the settings change.
    window : int
        Number of nights before/after to include (default 7).
    min_nights : int
        Minimum valid nights required per window (default 4).
    metrics : list of str, optional
        Which metrics to compare. Defaults to COMPARISON_METRICS.
    ctqi_series : pd.Series, optional
        CTQI values indexed by date, to include in comparison.

    Returns
    -------
    Dict with:
        change_date, before_dates, after_dates, confounded,
        insufficient, comparisons (per-metric results)

    Each per-metric result has p_value (raw Mann-Whitney), p_adjusted
    (Holm-corrected across the metrics tested for this change),
    significant (p_adjusted < ALPHA) and direction, which is
    "not significant" unless significant is True.
    """
    if pd is None:
        raise ImportError("pandas required")
    if metrics is None:
        metrics = list(COMPARISON_METRICS)

    # Add CTQI if available
    if ctqi_series is not None and "CTQI" not in metrics:
        metrics = metrics + ["CTQI"]

    result = {
        "change_date": change_date,
        "before_n": 0,
        "after_n": 0,
        "confounded": False,
        "insufficient": False,
        "comparisons": {},
    }

    # Always use index (DatetimeIndex from read_str)
    dates = pd.to_datetime(df_str.index)

    change_dt = pd.to_datetime(change_date)

    # Select before/after windows (exclude change date)
    before_mask = (dates < change_dt) & (dates >= change_dt - pd.Timedelta(days=window))
    after_mask = (dates > change_dt) & (dates <= change_dt + pd.Timedelta(days=window))

    # Filter to active nights (Duration > 0)
    if "Duration" in df_str.columns:
        active = df_str["Duration"].values.astype(float) > 0
        before_mask = before_mask & active
        after_mask = after_mask & active

    before_df = df_str[before_mask]
    after_df = df_str[after_mask]

    result["before_n"] = len(before_df)
    result["after_n"] = len(after_df)

    if len(before_df) < min_nights or len(after_df) < min_nights:
        result["insufficient"] = True
        return result

    # Compare each metric
    for metric in metrics:
        if metric == "CTQI" and ctqi_series is not None:
            # ctqi_series may be indexed by YYYYMMDD strings (from
            # df_nightly_v2) while df_str uses DatetimeIndex.
            # Normalize both sides to date-only Timestamps for matching.
            _ctqi = ctqi_series.copy()
            if not hasattr(_ctqi.index, 'tz'):
                try:
                    _ctqi.index = pd.to_datetime(_ctqi.index)
                except Exception:
                    pass
            _ctqi_idx = pd.to_datetime(_ctqi.index).normalize()
            _ctqi.index = _ctqi_idx

            _before_dates = pd.to_datetime(dates[before_mask]).normalize()
            _after_dates = pd.to_datetime(dates[after_mask]).normalize()
            before_vals = _ctqi.reindex(_before_dates).dropna().values
            after_vals = _ctqi.reindex(_after_dates).dropna().values
        elif metric in df_str.columns:
            before_vals = before_df[metric].dropna().values.astype(float)
            after_vals = after_df[metric].dropna().values.astype(float)
        else:
            continue

        if len(before_vals) < 2 or len(after_vals) < 2:
            result["comparisons"][metric] = {
                "before_median": float("nan"),
                "after_median": float("nan"),
                "insufficient": True,
            }
            continue

        before_med = float(np.median(before_vals))
        after_med = float(np.median(after_vals))
        before_mean = float(np.mean(before_vals))
        after_mean = float(np.mean(after_vals))

        comp = {
            "before_median": round(before_med, 3),
            "after_median": round(after_med, 3),
            "before_mean": round(before_mean, 3),
            "after_mean": round(after_mean, 3),
            "before_n": len(before_vals),
            "after_n": len(after_vals),
            "insufficient": False,
        }

        # Mann-Whitney U test
        if HAS_SCIPY and len(before_vals) >= 2 and len(after_vals) >= 2:
            try:
                u_stat, p_val = mannwhitneyu(before_vals, after_vals,
                                             alternative="two-sided")
                effect = _rank_biserial(u_stat, len(before_vals), len(after_vals))
                comp["U_statistic"] = round(float(u_stat), 2)
                comp["p_value"] = round(float(p_val), 4)
                comp["effect_size"] = round(effect, 4)
                comp["direction"] = _direction_label(
                    before_med, after_med, metric, effect)
            except ValueError:
                comp["U_statistic"] = float("nan")
                comp["p_value"] = float("nan")
                comp["effect_size"] = float("nan")
                comp["direction"] = "error"
        else:
            comp["U_statistic"] = float("nan")
            comp["p_value"] = float("nan")
            comp["effect_size"] = float("nan")
            comp["direction"] = "no scipy"

        result["comparisons"][metric] = comp

    # Many metrics are tested for each change, so correct for multiple
    # comparisons (Holm) and only label a direction when the corrected
    # p-value is below ALPHA. Otherwise small-sample noise would be
    # labelled "improved" or "worsened".
    tested = [m for m, c in result["comparisons"].items()
              if isinstance(c.get("p_value"), float)
              and not np.isnan(c["p_value"])]
    if tested:
        adj = _holm_adjust([result["comparisons"][m]["p_value"] for m in tested])
        for m, pa in zip(tested, adj):
            c = result["comparisons"][m]
            c["p_adjusted"] = round(float(pa), 4)
            c["significant"] = bool(pa < ALPHA)
            if not c["significant"]:
                c["direction"] = "not significant"

    return result


def analyze_all_changes(df_str, df_changes=None, window=7, min_nights=4,
                        ctqi_series=None):
    # type: (pd.DataFrame, Optional[pd.DataFrame], int, int, Optional[pd.Series]) -> List[dict]
    """Run before/after comparison for every detected settings change.

    Also flags confounded windows (two changes < window nights apart).

    Parameters
    ----------
    df_str : pd.DataFrame
        STR data.
    df_changes : pd.DataFrame, optional
        Output from detect_settings_changes(). If None, runs detection.
    window : int
        Before/after window size.
    min_nights : int
        Minimum nights per window.
    ctqi_series : pd.Series, optional
        CTQI values by date.

    Returns
    -------
    List of before/after result dicts, one per change date.
    """
    if df_changes is None:
        df_changes = detect_settings_changes(df_str)

    if len(df_changes) == 0:
        return []

    # Get unique change dates
    change_dates = sorted(df_changes["date"].unique())

    results = []
    for i, cd in enumerate(change_dates):
        ba = compare_before_after(df_str, cd, window=window,
                                  min_nights=min_nights,
                                  ctqi_series=ctqi_series)

        # Check confounded: another change within window nights
        for j, other_cd in enumerate(change_dates):
            if i == j:
                continue
            gap = abs((pd.to_datetime(cd) - pd.to_datetime(other_cd)).days)
            if gap <= window:
                ba["confounded"] = True
                break

        # Attach the changes that happened on this date
        date_changes = df_changes[df_changes["date"] == cd]
        ba["settings_changed"] = date_changes.to_dict("records")

        results.append(ba)

    return results


# ---------------------------------------------------------------------------
# 7.4  Therapy Trajectory Analysis
# ---------------------------------------------------------------------------

def _simple_linreg(x, y):
    # type: (np.ndarray, np.ndarray) -> dict
    """Simple least-squares linear regression.

    Returns dict with slope, intercept, r_squared, n.
    """
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = x.size
    if n < 3:
        return {"slope": float("nan"), "intercept": float("nan"),
                "r_squared": float("nan"), "n": n}

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    ss_xx = np.sum((x - x_mean) ** 2)

    if ss_xx == 0:
        return {"slope": float("nan"), "intercept": float("nan"),
                "r_squared": float("nan"), "n": n}

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "slope": round(float(slope), 6),
        "intercept": round(float(intercept), 4),
        "r_squared": round(float(r_sq), 4),
        "n": n,
    }


def therapy_trajectory(df_str, metric="AHI", change_dates=None):
    # type: (pd.DataFrame, str, Optional[List]) -> dict
    """Fit linear regression to a metric over the full dataset.

    Parameters
    ----------
    df_str : pd.DataFrame
        STR data with Date column or datetime index.
    metric : str
        Column name to fit (e.g., "AHI", "CTQI").
    change_dates : list, optional
        List of change dates for per-segment fitting.

    Returns
    -------
    Dict with:
        overall: regression result (slope in units/month)
        segments: list of per-segment regressions (between changes)
    """
    if pd is None:
        raise ImportError("pandas required")
    if metric not in df_str.columns:
        return {"overall": {"slope": float("nan")}, "segments": []}

    # Build time axis in months from first date
    # Always use index (DatetimeIndex from read_str)
    dates = pd.to_datetime(df_str.index)

    # Filter active nights
    values = df_str[metric].values.astype(np.float64)
    active = ~np.isnan(values)
    if "Duration" in df_str.columns:
        active = active & (df_str["Duration"].values.astype(float) > 0)

    dates_active = dates[active]
    values_active = values[active]

    if len(dates_active) == 0:
        return {"overall": {"slope": float("nan")}, "segments": []}

    day0 = dates_active[0]
    months = np.array([(d - day0).days / 30.44 for d in dates_active])

    # Overall regression
    overall = _simple_linreg(months, values_active)
    overall["slope_per_month"] = overall.pop("slope")
    result = {"overall": overall, "segments": []}

    # Per-segment regression (between change dates)
    if change_dates and len(change_dates) > 0:
        boundaries = [day0] + [pd.to_datetime(d) for d in sorted(change_dates)]
        # Add end date
        boundaries.append(dates_active[-1] + pd.Timedelta(days=1))

        for i in range(len(boundaries) - 1):
            seg_start = boundaries[i]
            seg_end = boundaries[i + 1]
            seg_mask = (dates_active >= seg_start) & (dates_active < seg_end)
            seg_dates = dates_active[seg_mask]
            seg_vals = values_active[np.asarray(seg_mask)]

            if len(seg_dates) < 3:
                result["segments"].append({
                    "start": str(seg_start.date()),
                    "end": str(seg_end.date()),
                    "n": len(seg_dates),
                    "slope_per_month": float("nan"),
                })
                continue

            seg_months = np.array([(d - seg_dates[0]).days / 30.44
                                   for d in seg_dates])
            seg_reg = _simple_linreg(seg_months, seg_vals)
            seg_reg["slope_per_month"] = seg_reg.pop("slope")
            seg_reg["start"] = str(seg_start.date())
            seg_reg["end"] = str(seg_end.date())
            result["segments"].append(seg_reg)

    return result


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def settings_change_summary(df_str, ctqi_series=None):
    # type: (pd.DataFrame, Optional[pd.Series]) -> dict
    """One-call summary: detect changes, compare, fit trajectory.

    Parameters
    ----------
    df_str : pd.DataFrame
        STR data.
    ctqi_series : pd.Series, optional
        CTQI values by date.

    Returns
    -------
    Dict with:
        changes_df: DataFrame of individual signal changes
        grouped_changes: list of adjustment sessions
        before_after: list of comparison results
        trajectory_ahi: AHI trajectory regression
        trajectory_ctqi: CTQI trajectory regression (if ctqi_series given)
    """
    df_changes = detect_settings_changes(df_str)
    grouped = group_simultaneous_changes(df_changes)

    change_dates_list = sorted(df_changes["date"].unique()) if len(df_changes) > 0 else []

    before_after = analyze_all_changes(
        df_str, df_changes=df_changes, ctqi_series=ctqi_series
    )

    traj_ahi = therapy_trajectory(df_str, metric="AHI",
                                 change_dates=change_dates_list)

    traj_ctqi = {}
    if ctqi_series is not None:
        # Build a temporary df with CTQI for trajectory fitting
        ctqi_df = df_str.copy()
        if "CTQI" not in ctqi_df.columns:
            ctqi_df["CTQI"] = ctqi_series
        traj_ctqi = therapy_trajectory(ctqi_df, metric="CTQI",
                                       change_dates=change_dates_list)

    return {
        "changes_df": df_changes,
        "grouped_changes": grouped,
        "before_after": before_after,
        "trajectory_ahi": traj_ahi,
        "trajectory_ctqi": traj_ctqi,
        "n_changes": len(change_dates_list),
        "change_dates": [str(d) for d in change_dates_list],
    }
