"""
report.py — CPAP HTML Report Generator
SleepingAtTheTop — NightmaresOfTheBottom / AirSense 11

Generates:
- Single-night HTML report (report_YYYYMMDD.html)
- Longitudinal HTML report (report_longitudinal.html)

All plots rendered with matplotlib, embedded as base64 PNG.
Self-contained HTML — no external CSS/JS dependencies.

Dependencies: numpy, matplotlib, (pandas for longitudinal)
No pyedflib, no MNE.
"""

from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional
import base64
import sys
import numpy as np

def _in_ipython():
    # type: () -> bool
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


try:
    import matplotlib
    # Reports render figures to PNG and close them, so a non-GUI backend is
    # only needed when running as a plain script. Inside Jupyter, leave the
    # notebook's backend alone so its own plots still display.
    if not _in_ipython() and "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 20px; color: #222; background: #fafafa; }
h1 { color: #1a3a5c; border-bottom: 2px solid #1a3a5c; padding-bottom: 8px; }
h2 { color: #2a5a8c; margin-top: 32px; }
h3 { color: #3a6a9c; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; font-size: 14px; }
th { background: #e8eef4; font-weight: 600; }
tr:nth-child(even) { background: #f4f7fa; }
.grade-excellent { color: #1a8a1a; font-weight: bold; }
.grade-good { color: #2a7a2a; }
.grade-fair { color: #b08a00; }
.grade-poor { color: #c04000; }
.grade-verypoor { color: #a00000; font-weight: bold; }
.ctqi-box { display: inline-block; padding: 8px 16px; border-radius: 8px;
            font-size: 20px; font-weight: bold; margin: 8px 0; }
.ctqi-excellent { background: #d4edda; color: #155724; }
.ctqi-good { background: #d4edda; color: #1a6a2a; }
.ctqi-fair { background: #fff3cd; color: #856404; }
.ctqi-poor { background: #f8d7da; color: #721c24; }
.ctqi-verypoor { background: #f5c6cb; color: #721c24; }
.plot-container { margin: 16px 0; text-align: center; }
.plot-container img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
details { margin: 8px 0; }
summary { cursor: pointer; font-weight: 600; color: #2a5a8c; padding: 4px 0; }
.footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid #ccc;
          font-size: 12px; color: #666; text-align: center; }
.uars-flag { padding: 6px 12px; border-radius: 4px; display: inline-block; }
.uars-yes { background: #f8d7da; color: #721c24; font-weight: bold; }
.uars-no { background: #d4edda; color: #155724; }
.metric-highlight { font-size: 18px; font-weight: bold; }
.stats-combined th { font-size: 12px; white-space: nowrap; }
.stats-combined td { font-size: 12px; text-align: right; padding: 4px 8px; }
.stats-combined td:first-child { text-align: left; font-weight: 600; }
.settings-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0 24px; }
.settings-grid table { width: auto; margin: 4px 0; }
.session-header { color: #555; font-size: 15px; margin-top: 4px; }
"""

_FOOTER = '<div class="footer"></div>'


# ---------------------------------------------------------------------------
# Date / time formatting helpers
# ---------------------------------------------------------------------------

def _format_date_str(date_str):
    # type: (str,) -> str
    """Convert YYYYMMDD to 'Mar 13, 2026' format."""
    try:
        dt = datetime.strptime(date_str, "%Y%m%d")
        return dt.strftime("%b %d, %Y").replace(" 0", " ")
    except (ValueError, TypeError):
        return str(date_str)


def _format_datetime(dt):
    # type: (object,) -> str
    """Format a datetime object to 'Mar 13, 2026 — 9:54 PM' style."""
    if isinstance(dt, datetime):
        return dt.strftime("%b %d, %Y — %I:%M %p").replace(" 0", " ").replace("AM", "AM").replace("PM", "PM")
    return str(dt)


def _format_time(dt):
    # type: (object,) -> str
    """Format a datetime object to '9:54 PM' style."""
    if isinstance(dt, datetime):
        return dt.strftime("%I:%M %p").lstrip("0")
    return str(dt)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig, dpi=100):
    # type: (plt.Figure, int) -> str
    """Render a matplotlib figure to a base64-encoded PNG string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _img_tag(b64, alt="plot"):
    # type: (str, str) -> str
    return (f'<div class="plot-container">'
            f'<img src="data:image/png;base64,{b64}" alt="{alt}"/></div>')


def _make_time_axis(n_samples, fs, start_dt):
    # type: (int, float, datetime) -> np.ndarray
    """Build array of datetime objects for a signal."""
    offsets = np.arange(n_samples) / fs
    return np.array([start_dt + timedelta(seconds=float(s)) for s in offsets])


# ---------------------------------------------------------------------------
# Single-night plots
# ---------------------------------------------------------------------------

def _plot_brp_flow(session, detection_result, max_points=5000):
    # type: (dict, dict, int) -> Optional[str]
    """Full-night downsampled flow waveform with detected events overlaid."""
    if not HAS_MPL:
        return None
    brp = session.get("brp")
    if brp is None:
        return None
    flow = brp.get("Flow.40ms")
    if flow is None or flow.size == 0:
        return None

    fs = brp.get("fs", 25.0)
    start_dt = brp.get("start_dt", datetime(2000, 1, 1))

    # Downsample for plotting
    step = max(1, flow.size // max_points)
    flow_ds = flow[::step]
    t = _make_time_axis(flow_ds.size, fs / step, start_dt)

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, flow_ds, linewidth=0.4, color="#1a3a5c")
    ax.set_ylabel("Flow (L/s)")
    ax.set_title("BRP Flow Waveform (downsampled)")
    ax.axhline(0, color="#999", linewidth=0.5)
    fig.autofmt_xdate()

    # Overlay detected apneas/hypopneas
    colors = {"Apnea": "#d9534f", "Hypopnea": "#f0ad4e",
              "Obstructive Apnea (est.)": "#d9534f",
              "Central Apnea (est.)": "#5bc0de"}
    for ev in (detection_result.get("events_apnea", []) +
               detection_result.get("events_hypopnea", [])):
        c = colors.get(ev["label"], "#d9534f")
        t0 = start_dt + timedelta(seconds=ev["onset_s"])
        t1 = t0 + timedelta(seconds=ev["duration_s"])
        ax.axvspan(t0, t1, alpha=0.25, color=c, linewidth=0)

    ax.set_xlim(t[0], t[-1])
    return _fig_to_base64(fig)


def _plot_brp_detail(session, detection_result, window_min=5):
    # type: (dict, dict, int) -> Optional[str]
    """5-minute detail window of BRP flow showing individual breath morphology.

    Automatically selects the window containing the most detected events.
    Falls back to the middle of the session if no events exist.
    """
    if not HAS_MPL:
        return None
    brp = session.get("brp")
    if brp is None:
        return None
    flow = brp.get("Flow.40ms")
    if flow is None or flow.size == 0:
        return None

    fs = brp.get("fs", 25.0)
    start_dt = brp.get("start_dt", datetime(2000, 1, 1))
    total_s = flow.size / fs
    window_s = window_min * 60

    if total_s <= window_s:
        return None  # session too short for a detail window

    # Pick the most interesting window: find cluster of events
    all_events = (detection_result.get("events_apnea", []) +
                  detection_result.get("events_hypopnea", []))
    if all_events:
        # Use the onset of the first event as center, or densest cluster
        onsets = sorted(ev["onset_s"] for ev in all_events)
        center_s = onsets[len(onsets) // 2]  # median event
    else:
        center_s = total_s / 2.0  # middle of session

    # Clamp window to valid range
    win_start_s = max(0, center_s - window_s / 2)
    win_end_s = min(total_s, win_start_s + window_s)
    win_start_s = max(0, win_end_s - window_s)

    i0 = int(win_start_s * fs)
    i1 = min(int(win_end_s * fs), flow.size)
    flow_win = flow[i0:i1]
    t = _make_time_axis(flow_win.size, fs,
                        start_dt + timedelta(seconds=win_start_s))

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, flow_win, linewidth=0.6, color="#1a3a5c")
    ax.set_ylabel("Flow (L/s)")
    ax.set_title(f"BRP Flow Detail — {window_min}-minute window")
    ax.axhline(0, color="#999", linewidth=0.5)

    # Overlay events within this window
    colors = {"Apnea": "#d9534f", "Hypopnea": "#f0ad4e",
              "Obstructive Apnea (est.)": "#d9534f",
              "Central Apnea (est.)": "#5bc0de"}
    for ev in all_events:
        ev_start = ev["onset_s"]
        ev_end = ev_start + ev["duration_s"]
        if ev_end >= win_start_s and ev_start <= win_end_s:
            c = colors.get(ev["label"], "#d9534f")
            t0 = start_dt + timedelta(seconds=max(ev_start, win_start_s))
            t1 = start_dt + timedelta(seconds=min(ev_end, win_end_s))
            ax.axvspan(t0, t1, alpha=0.25, color=c, linewidth=0)

    fig.autofmt_xdate()
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_pld_panels(session):
    # type: (dict,) -> Optional[str]
    """Multi-panel PLD plot: pressure, leak, resp rate, snore, flow lim."""
    if not HAS_MPL:
        return None
    pld = session.get("pld")
    if pld is None:
        return None

    fs = pld.get("fs", 0.5)
    start_dt = pld.get("start_dt", datetime(2000, 1, 1))

    panels = [
        ("MaskPress.2s", "Pressure (cmH2O)", "#2a5a8c"),
        ("Leak.2s", "Leak (L/s)", "#c04000"),
        ("RespRate.2s", "Resp Rate (bpm)", "#1a6a2a"),
        ("Snore.2s", "Snore", "#8a6a00"),
        ("FlowLim.2s", "Flow Lim", "#6a2a8c"),
    ]
    available = [(sig, label, color) for sig, label, color in panels
                 if pld.get(sig) is not None and pld[sig].size > 0]
    if not available:
        return None

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (sig, label, color) in zip(axes, available):
        arr = pld[sig]
        t = _make_time_axis(arr.size, fs, start_dt)
        ax.plot(t, arr, linewidth=0.6, color=color)
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)

    axes[0].set_title("PLD Trend Signals")
    fig.autofmt_xdate()
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_sa2(session):
    # type: (dict,) -> Optional[str]
    """SpO2 and Pulse dual-panel plot."""
    if not HAS_MPL:
        return None
    sa2 = session.get("sa2")
    if sa2 is None:
        return None

    fs = sa2.get("fs", 1.0)
    start_dt = sa2.get("start_dt", datetime(2000, 1, 1))

    spo2 = sa2.get("SpO2.1s")
    pulse = sa2.get("Pulse.1s")
    if spo2 is None and pulse is None:
        return None

    n_panels = sum(1 for x in [spo2, pulse] if x is not None and x.size > 0)
    if n_panels == 0:
        return None

    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    idx = 0
    if spo2 is not None and spo2.size > 0:
        s = spo2.astype(np.float64).copy()
        s[s <= 0] = np.nan
        t = _make_time_axis(s.size, fs, start_dt)
        axes[idx].plot(t, s, linewidth=0.5, color="#1a6a2a")
        axes[idx].set_ylabel("SpO2 (%)", fontsize=9)
        axes[idx].set_ylim(80, 100)
        idx += 1

    if pulse is not None and pulse.size > 0:
        p = pulse.astype(np.float64).copy()
        p[p <= 0] = np.nan
        t = _make_time_axis(p.size, fs, start_dt)
        axes[idx].plot(t, p, linewidth=0.5, color="#c04000")
        axes[idx].set_ylabel("Pulse (bpm)", fontsize=9)

    axes[0].set_title("SA2 — SpO2 & Pulse")
    fig.autofmt_xdate()
    fig.tight_layout()
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Longitudinal plots
# ---------------------------------------------------------------------------

def _plot_longitudinal_metric(dates, values, label, ylabel,
                              rolling_window=7, color="#1a3a5c",
                              change_dates=None, trend_slope=None,
                              trend_intercept=None):
    # type: (...) -> Optional[str]
    """Generic longitudinal plot with optional rolling median + trend + change markers."""
    if not HAS_MPL or len(dates) == 0:
        return None

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.scatter(dates, values, s=8, alpha=0.5, color=color, zorder=2)

    # Rolling median
    if HAS_PANDAS and len(values) >= rolling_window:
        s = pd.Series(values, index=dates).sort_index()
        roll = s.rolling(rolling_window, min_periods=3).median()
        ax.plot(roll.index, roll.values, linewidth=2, color=color,
                alpha=0.8, label=f"{rolling_window}-day median", zorder=3)

    # Trend line
    if trend_slope is not None and trend_intercept is not None:
        if not np.isnan(trend_slope):
            d0 = min(dates)
            x = np.array([(d - d0).days / 30.44 for d in dates])
            y_hat = trend_slope * x + trend_intercept
            ax.plot(dates, y_hat, "--", color="#999", linewidth=1.5,
                    label=f"trend: {trend_slope:+.3f}/mo", zorder=1)

    # Change markers
    if change_dates:
        for cd in change_dates:
            cd_dt = pd.to_datetime(cd) if HAS_PANDAS else cd
            ax.axvline(cd_dt, color="#d9534f", linewidth=1, linestyle=":",
                       alpha=0.7, zorder=1)

    ax.set_ylabel(ylabel)
    ax.set_title(label)
    ax.legend(fontsize=8, loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_ahi_breakdown(df_str):
    # type: (pd.DataFrame,) -> Optional[str]
    """Stacked bar chart of AHI components over time."""
    if not HAS_MPL or not HAS_PANDAS:
        return None

    cols = ["OAI", "CAI", "HI", "RIN"]
    available = [c for c in cols if c in df_str.columns]
    if not available:
        return None

    # Always use index (DatetimeIndex from read_str)
    dates = pd.to_datetime(df_str.index)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    bottom = np.zeros(len(df_str))
    colors = {"OAI": "#d9534f", "CAI": "#5bc0de", "HI": "#f0ad4e", "RIN": "#6a2a8c"}

    for col in available:
        vals = df_str[col].fillna(0).values.astype(float)
        ax.bar(dates, vals, bottom=bottom, width=1.0,
               color=colors.get(col, "#999"), label=col, alpha=0.8)
        bottom += vals

    ax.set_ylabel("Events/hr")
    ax.set_title("AHI Breakdown by Type")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_correlation_heatmap(df_str):
    # type: (pd.DataFrame,) -> Optional[str]
    """Correlation heatmap of key STR metrics."""
    if not HAS_MPL or not HAS_PANDAS:
        return None

    cols = ["AHI", "OAI", "CAI", "HI", "RIN", "Leak.50", "Leak.95",
            "MaskPress.50", "SpO2.50", "RespRate.50", "TidVol.50"]
    available = [c for c in cols if c in df_str.columns]
    if len(available) < 3:
        return None

    corr = df_str[available].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(available, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(available, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Metric Correlation Heatmap")

    # Annotate cells
    for i in range(len(available)):
        for j in range(len(available)):
            v = corr.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(v) > 0.5 else "black")

    fig.tight_layout()
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# HTML table helpers
# ---------------------------------------------------------------------------

def _html_table(headers, rows, table_id=""):
    # type: (List[str], List[List], str) -> str
    tid = f' id="{table_id}"' if table_id else ""
    h = f"<table{tid}><thead><tr>"
    h += "".join(f"<th>{c}</th>" for c in headers)
    h += "</tr></thead><tbody>"
    for row in rows:
        h += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
    h += "</tbody></table>"
    return h


def _fmt(val, decimals=2):
    # type: (object, int) -> str
    """Format a value for display. Handles NaN, bool, float."""
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float):
        if np.isnan(val):
            return "—"
        return f"{val:.{decimals}f}"
    return str(val)


def _ctqi_css_class(grade):
    # type: (str,) -> str
    mapping = {"Excellent": "ctqi-excellent", "Good": "ctqi-good",
               "Fair": "ctqi-fair", "Poor": "ctqi-poor",
               "Very Poor": "ctqi-verypoor"}
    return mapping.get(grade, "")


def _stats_table_html(stats_dict, title=""):
    # type: (dict, str) -> str
    """Render a compute_stats() result as a collapsible HTML details block."""
    if not stats_dict:
        return ""
    label = stats_dict.get("label", title)
    rows = []
    display_order = [
        ("mean", "Mean"), ("median", "Median"), ("mode", "Mode"),
        ("min", "Min"), ("max", "Max"),
        ("p0_5", "P0.5"), ("p1", "P1"), ("p5", "P5"),
        ("p95", "P95"), ("p99", "P99"), ("p99_5", "P99.5"),
        ("variance", "Variance"), ("std", "Std Dev"), ("cv_pct", "CV %"),
        ("n_valid", "N Valid"), ("n_total", "N Total"), ("pct_valid", "% Valid"),
    ]
    for key, name in display_order:
        if key in stats_dict:
            rows.append([name, _fmt(stats_dict[key], 4)])
    # Extra keys not in standard list
    for k, v in stats_dict.items():
        if k in ("label",) or k in dict(display_order):
            continue
        rows.append([k, _fmt(v)])

    table = _html_table(["Statistic", "Value"], rows)
    return f"<details><summary>{label}</summary>{table}</details>"


# ---------------------------------------------------------------------------
# Single-Night Report
# ---------------------------------------------------------------------------

def _combined_stats_table(sig_stats):
    # type: (dict,) -> str
    """Render all per-signal stats as a single professional combined table.

    Rows = statistics (Mean, Median, Min, Max, P5, P95, Std Dev, N Valid).
    Columns = signal names with human-friendly labels.
    """
    if not sig_stats:
        return ""

    # Human-friendly signal labels
    _SIG_LABELS = {
        "Flow.40ms": "Flow", "Flow.40ms_insp": "Flow (Insp)",
        "Flow.40ms_exp": "Flow (Exp)", "Press.40ms": "Pressure (40ms)",
        "MaskPress.2s": "Mask Press", "Press.2s": "Pressure",
        "EprPress.2s": "EPR Press", "Leak.2s": "Leak",
        "RespRate.2s": "Resp Rate", "TidVol.2s": "Tidal Vol",
        "MinVent.2s": "Min Vent", "Snore.2s": "Snore",
        "FlowLim.2s": "Flow Lim", "SpO2.1s": "SpO2",
        "Pulse.1s": "Pulse",
    }

    # Key stats to show (compact but complete)
    stat_rows = [
        ("mean",    "Mean"),
        ("median",  "Median"),
        ("min",     "Min"),
        ("max",     "Max"),
        ("p5",      "P5"),
        ("p95",     "P95"),
        ("std",     "Std Dev"),
        ("n_valid", "N Valid"),
    ]

    # Order signals: BRP first, then PLD, then SA2
    sig_order = [
        "Flow.40ms", "Flow.40ms_insp", "Flow.40ms_exp", "Press.40ms",
        "MaskPress.2s", "Press.2s", "EprPress.2s", "Leak.2s",
        "RespRate.2s", "TidVol.2s", "MinVent.2s", "Snore.2s",
        "FlowLim.2s", "SpO2.1s", "Pulse.1s",
    ]
    available = [s for s in sig_order if s in sig_stats]
    # Add any signals not in predefined order
    for s in sorted(sig_stats.keys()):
        if s not in available:
            available.append(s)

    if not available:
        return ""

    # Build header
    col_labels = [_SIG_LABELS.get(s, s) for s in available]
    h = '<table class="stats-combined"><thead><tr><th>Statistic</th>'
    h += "".join(f"<th>{c}</th>" for c in col_labels)
    h += "</tr></thead><tbody>"

    for key, label in stat_rows:
        h += f"<tr><td>{label}</td>"
        for sig in available:
            stats = sig_stats[sig]
            val = stats.get(key, float("nan"))
            # Use fewer decimals for large integers
            if key == "n_valid":
                h += f"<td>{int(val):,}</td>" if not (isinstance(val, float) and np.isnan(val)) else "<td>—</td>"
            else:
                h += f"<td>{_fmt(val, 2)}</td>"
        h += "</tr>"

    h += "</tbody></table>"
    return h


# Human-friendly labels for settings keys
_SETTING_LABELS = {
    "S.A.MinPress":  "Min Pressure (cmH₂O)",
    "S.A.MaxPress":  "Max Pressure (cmH₂O)",
    "S.EPR.Level":   "EPR Level",
    "S.EPR.EPRType": "EPR Type",
    "S.Mask":        "Mask Type Code",
    "S.RampEnable":  "Ramp Enabled",
    "S.RampTime":    "Ramp Time (min)",
    "S.C.Press":     "Fixed Pressure (cmH₂O)",
    "Mode":          "Therapy Mode",
}

# Map numeric codes to readable labels where possible
_EPR_TYPE_MAP = {0: "Off", 1: "Ramp Only", 2: "Full Time", 3: "Full Time"}
_MODE_MAP = {0: "CPAP", 1: "APAP", 2: "AutoSet for Her"}
_RAMP_ENABLE_MAP = {0: "Off", 1: "On", 2: "Auto", 3: "On"}


def _format_setting_value(key, val):
    # type: (str, object) -> str
    """Format a setting value with human-readable labels where possible."""
    if isinstance(val, float) and np.isnan(val):
        return "—"
    # Try to convert to int for code-type settings
    try:
        ival = int(round(float(val)))
    except (ValueError, TypeError):
        return str(val)

    if key == "S.EPR.EPRType":
        return _EPR_TYPE_MAP.get(ival, str(ival))
    if key == "Mode":
        return _MODE_MAP.get(ival, str(ival))
    if key == "S.RampEnable":
        return _RAMP_ENABLE_MAP.get(ival, str(ival))
    if key == "S.Mask":
        return str(ival)

    # Pressure / numeric values — show as formatted number
    fval = float(val)
    if fval == int(fval):
        return str(int(fval))
    return _fmt(fval, 1)


def generate_single_night(
    date_str,           # type: str
    session,            # type: dict
    detection_result,   # type: dict
    night_indices,      # type: dict
    eve_comparison=None,  # type: Optional[dict]
    str_row=None,       # type: Optional[dict]
    device_sn="",       # type: str
    output_dir=None,    # type: Optional[Path]
):
    # type: (...) -> str
    """Generate a single-night HTML report.

    Parameters
    ----------
    date_str : str
        Night date as YYYYMMDD.
    session : dict
        From resmed_edf.read_session().
    detection_result : dict
        From event_detection.detect_all().
    night_indices : dict
        From indices.compute_night_indices().
    eve_comparison : dict, optional
        From event_detection.compare_with_eve().
    str_row : dict, optional
        STR row for this night.
    device_sn : str
        Device serial number (accepted for API compat, not displayed).
    output_dir : Path, optional
        If given, write HTML file to this directory.

    Returns
    -------
    HTML string.
    """
    meta = session.get("meta", {})
    duration_min = meta.get("duration_s", 0) / 60.0
    start_dt = meta.get("start_dt", "")
    end_dt = meta.get("end_dt", "")

    # Format the title date
    nice_date = _format_date_str(date_str)

    # Format session times
    if isinstance(start_dt, datetime):
        session_line = (f"{_format_time(start_dt)} → {_format_time(end_dt)}"
                        f" &nbsp;|&nbsp; Duration: {_fmt(duration_min, 0)} min"
                        f" ({_fmt(duration_min / 60.0, 1)} hr)")
    else:
        session_line = (f"{start_dt} → {end_dt} &nbsp;|&nbsp; "
                        f"Duration: {_fmt(duration_min, 1)} min")

    parts = []
    parts.append(f"<!DOCTYPE html><html><head><meta charset='utf-8'>")
    parts.append(f"<title>CPAP Night Report — {nice_date}</title>")
    parts.append(f"<style>{_CSS}</style></head><body>")

    # --- 1. Header ---
    parts.append(f"<h1>CPAP Night Report — {nice_date}</h1>")
    parts.append(f'<p class="session-header">{session_line}</p>')

    # --- 2. Session Summary ---
    ctqi = night_indices.get("ctqi", {})
    ctqi_score = ctqi.get("ctqi", float("nan"))
    ctqi_grade = ctqi.get("grade", "N/A")
    idx = night_indices.get("indices", {})

    parts.append("<h2>Session Summary</h2>")
    parts.append(f'<div class="ctqi-box {_ctqi_css_class(ctqi_grade)}">'
                 f'CTQI: {_fmt(ctqi_score, 1)} — {ctqi_grade}</div>')

    summary_rows = [
        ["AHI (custom)", _fmt(night_indices.get("ahi_custom", float("nan")))],
        ["Leak Median (L/s)", _fmt(idx.get("fl_mean", float("nan")))],
        ["SpO2 Nadir", _fmt(idx.get("spo2_nadir", float("nan")))],
        ["ODI (3%)", _fmt(idx.get("ODI_3", float("nan")))],
        ["PCI (6 bpm)", _fmt(idx.get("PCI_6bpm", float("nan")))],
        ["Flow Lim Mean", _fmt(idx.get("fl_mean", float("nan")))],
        ["Snore Index", _fmt(idx.get("snore_index", float("nan")))],
    ]
    # Add leak median from signal stats
    leak_stats = night_indices.get("signal_stats", {}).get("Leak.2s", {})
    if leak_stats:
        summary_rows[1] = ["Leak Median (L/s)", _fmt(leak_stats.get("median", float("nan")))]

    parts.append(_html_table(["Metric", "Value"], summary_rows))

    # CTQI component breakdown (always visible, not collapsible)
    comps = ctqi.get("components", {})
    if comps:
        comp_rows = [[k.replace("_score", "").upper(), _fmt(v, 1)]
                     for k, v in comps.items()]
        parts.append("<h3>CTQI Component Breakdown</h3>")
        parts.append(_html_table(["Component", "Score (0–100)"], comp_rows))

    # --- 3. UARS Assessment ---
    uars = night_indices.get("uars", {})
    if uars:
        parts.append("<h2>UARS Assessment</h2>")
        flag = uars.get("uars_suggestive", False)
        score = uars.get("uars_score", 0)
        cls = "uars-yes" if flag else "uars-no"
        parts.append(f'<span class="uars-flag {cls}">'
                     f'{"UARS Suggestive" if flag else "Not UARS Suggestive"}'
                     f' — Score: {score}/5</span>')
        criteria = uars.get("uars_criteria", {})
        if criteria:
            cr_rows = [[k.replace("_", " ").title(), _fmt(v)]
                       for k, v in criteria.items()]
            parts.append(_html_table(["Criterion", "Met?"], cr_rows))

    # --- 4. Glasgow Index (moved up — key differentiator) ---
    glas = night_indices.get("glasgow", {})
    if glas and not all(np.isnan(v) if isinstance(v, float) else False
                        for v in glas.values()):
        parts.append("<h2>Glasgow Index (BRP Breath Analysis)</h2>")
        # Separate overall/n_breaths from components
        glas_components = {k: v for k, v in glas.items()
                          if k not in ("overall", "n_breaths")}
        overall = glas.get("overall", float("nan"))
        n_breaths = glas.get("n_breaths", 0)
        parts.append(f"<p><b>Overall Glasgow Index:</b> {_fmt(overall, 3)}"
                     f" &nbsp;|&nbsp; Breaths analyzed: {int(n_breaths):,}</p>")
        g_rows = [[k.replace("_", " ").title(), _fmt(v, 4)]
                   for k, v in glas_components.items()]
        parts.append(_html_table(["Component", "Fraction Flagged"], g_rows))

    # --- 5. Settings in Effect (moved up — important context) ---
    if str_row:
        parts.append("<h2>Settings in Effect</h2>")
        setting_keys = ["Mode", "S.A.MinPress", "S.A.MaxPress", "S.EPR.Level",
                        "S.EPR.EPRType", "S.Mask", "S.RampEnable",
                        "S.RampTime"]
        s_rows = []
        for k in setting_keys:
            if k in str_row:
                label = _SETTING_LABELS.get(k, k)
                value = _format_setting_value(k, str_row[k])
                s_rows.append([label, value])
        if s_rows:
            parts.append(_html_table(["Setting", "Value"], s_rows))

    # --- 6. Event Summary ---
    ec = night_indices.get("event_counts", {})
    parts.append("<h2>Event Summary</h2>")
    ev_rows = [
        ["Apnea", ec.get("apnea", 0)],
        ["Hypopnea", ec.get("hypopnea", 0)],
        ["Hypopnea, flow only (not in AHI)", ec.get("hypopnea_flow_only", 0)],
        ["Desaturation", ec.get("desaturation", 0)],
        ["Pulse Surge", ec.get("pulse_surge", 0)],
        ["Snore", ec.get("snore", 0)],
        ["Flow Limitation", ec.get("flow_limitation", 0)],
        ["Total Respiratory", ec.get("total_respiratory", 0)],
    ]
    parts.append(_html_table(["Event Type", "Count"], ev_rows))

    if eve_comparison:
        parts.append("<h3>Custom Detector vs. ResMed EVE</h3>")
        cmp_rows = [
            ["Custom Total", eve_comparison.get("custom_total", 0)],
            ["EVE Total", eve_comparison.get("eve_total", 0)],
            ["Matched", eve_comparison.get("matched", 0)],
            ["Precision", _fmt(eve_comparison.get("precision", float("nan")), 3)],
            ["Recall", _fmt(eve_comparison.get("recall", float("nan")), 3)],
            ["F1", _fmt(eve_comparison.get("f1", float("nan")), 3)],
        ]
        parts.append(_html_table(["Metric", "Value"], cmp_rows))

    # --- 7. Plots ---
    parts.append("<h2>Signal Plots</h2>")

    b64 = _plot_brp_flow(session, detection_result)
    if b64:
        parts.append(_img_tag(b64, "BRP Flow Waveform"))

    b64 = _plot_brp_detail(session, detection_result)
    if b64:
        parts.append(_img_tag(b64, "BRP Flow Detail (5 min)"))

    b64 = _plot_pld_panels(session)
    if b64:
        parts.append(_img_tag(b64, "PLD Trend Signals"))

    b64 = _plot_sa2(session)
    if b64:
        parts.append(_img_tag(b64, "SA2 SpO2 & Pulse"))

    # --- 8. Per-Signal Descriptive Statistics (combined table) ---
    sig_stats = night_indices.get("signal_stats", {})
    if sig_stats:
        parts.append("<h2>Per-Signal Descriptive Statistics</h2>")
        parts.append(_combined_stats_table(sig_stats))

    # --- Footer ---
    parts.append(_FOOTER)
    parts.append("</body></html>")
    html = "\n".join(parts)

    if output_dir is not None:
        out_path = Path(output_dir) / f"report_{date_str}.html"
        out_path.write_text(html, encoding="utf-8")

    return html


# ---------------------------------------------------------------------------
# Longitudinal Report — helpers
# ---------------------------------------------------------------------------

def _format_change_date(cd):
    # type: (object,) -> str
    """Format a settings change date to 'Oct 11, 2024'."""
    try:
        if HAS_PANDAS:
            ts = pd.Timestamp(cd)
            return ts.strftime("%b %d, %Y").replace(" 0", " ")
        return str(cd)[:10]
    except Exception:
        return str(cd)[:10]


def _pct(n, total):
    # type: (int, int) -> str
    """Format n/total as percentage string."""
    if total <= 0:
        return "—"
    return f"{100.0 * n / total:.1f}%"


def _longitudinal_key_stats_table(longitudinal_stats):
    # type: (dict,) -> str
    """Render key longitudinal signals as a single combined table.

    Shows a subset of the most important signals as columns with
    Mean, Median, Min, Max, P5, P95, Std Dev as rows.
    """
    if not longitudinal_stats:
        return ""

    # Key signals to feature (in display order)
    key_signals = [
        ("AHI", "AHI"), ("Leak.50", "Leak (P50)"),
        ("MaskPress.50", "Pressure (P50)"), ("SpO2.50", "SpO₂ (P50)"),
        ("RespRate.50", "Resp Rate (P50)"), ("Duration", "Duration (min)"),
    ]
    available = [(k, label) for k, label in key_signals
                 if k in longitudinal_stats]
    if not available:
        return ""

    stat_rows = [
        ("mean", "Mean"), ("median", "Median"),
        ("min", "Min"), ("max", "Max"),
        ("p5", "P5"), ("p95", "P95"),
        ("std", "Std Dev"), ("n_valid", "N Nights"),
    ]

    h = '<table class="stats-combined"><thead><tr><th>Statistic</th>'
    h += "".join(f"<th>{label}</th>" for _, label in available)
    h += "</tr></thead><tbody>"

    for key, row_label in stat_rows:
        h += f"<tr><td>{row_label}</td>"
        for sig, _ in available:
            val = longitudinal_stats[sig].get(key, float("nan"))
            if key == "n_valid":
                h += f"<td>{int(val):,}</td>" if not (isinstance(val, float) and np.isnan(val)) else "<td>—</td>"
            else:
                h += f"<td>{_fmt(val, 2)}</td>"
        h += "</tr>"

    h += "</tbody></table>"
    return h


def _longitudinal_all_stats_table(stats_dict):
    # type: (dict,) -> str
    """Render all longitudinal signal stats as a single combined table.

    Signals as columns, stats as rows — same format as the key stats table.
    Handles any number of signals by using compact column headers.
    """
    if not stats_dict:
        return ""

    # Human-friendly signal labels for STR signals
    _STR_LABELS = {
        "OAI": "OAI", "CAI": "CAI", "HI": "HI", "RIN": "RIN",
        "AI": "AI", "UAI": "UAI", "CSR": "CSR",
        "Leak.50": "Leak P50", "Leak.95": "Leak P95",
        "Leak.70": "Leak P70", "Leak.Max": "Leak Max",
        "MaskPress.50": "Press P50", "MaskPress.95": "Press P95",
        "MaskPress.Max": "Press Max",
        "SpO2.50": "SpO₂ P50", "SpO2.95": "SpO₂ P95",
        "SpO2.Max": "SpO₂ Max", "SpO2Thresh": "SpO₂ Thresh",
        "TidVol.50": "TidVol P50", "RespRate.50": "RR P50",
        "MinVent.50": "MinVent P50",
        "FlowLim.2s": "FlowLim", "Snore.2s": "Snore",
        "Duration": "Duration",
        "S.A.MinPress": "Min Press", "S.A.MaxPress": "Max Press",
        "S.EPR.Level": "EPR Level",
    }

    stat_rows = [
        ("mean", "Mean"), ("median", "Median"),
        ("min", "Min"), ("max", "Max"),
        ("p5", "P5"), ("p95", "P95"),
        ("std", "Std Dev"), ("n_valid", "N"),
    ]

    signals = sorted(stats_dict.keys())
    if not signals:
        return ""

    col_labels = [_STR_LABELS.get(s, s) for s in signals]

    h = '<table class="stats-combined"><thead><tr><th>Stat</th>'
    h += "".join(f"<th>{c}</th>" for c in col_labels)
    h += "</tr></thead><tbody>"

    for key, label in stat_rows:
        h += f"<tr><td>{label}</td>"
        for sig in signals:
            stats = stats_dict[sig]
            val = stats.get(key, float("nan"))
            if key == "n_valid":
                if isinstance(val, float) and np.isnan(val):
                    h += "<td>—</td>"
                else:
                    h += f"<td>{int(val):,}</td>"
            else:
                h += f"<td>{_fmt(val, 2)}</td>"
        h += "</tr>"

    h += "</tbody></table>"
    return h


def _classify_change(ba):
    # type: (dict,) -> str
    """Classify a settings change as 'unconfounded', 'confounded', or 'insufficient'."""
    if ba.get("insufficient"):
        return "insufficient"
    if ba.get("confounded"):
        return "confounded"
    return "unconfounded"


def _comparison_is_significant(c, threshold=0.3):
    # type: (dict, float) -> bool
    """True if a per-metric comparison is statistically significant.

    Uses the Holm-corrected result from settings_changes when present.
    Falls back to |effect_size| >= threshold for older result dicts.
    """
    if c.get("insufficient"):
        return False
    if "significant" in c:
        return bool(c["significant"])
    es = c.get("effect_size", 0)
    return isinstance(es, (int, float)) and not np.isnan(es) and abs(es) >= threshold


def _has_significant_effect(ba, threshold=0.3):
    # type: (dict, float) -> bool
    """Check if any comparison metric is statistically significant."""
    comps = ba.get("comparisons", {})
    return any(_comparison_is_significant(c, threshold) for c in comps.values())


def _plot_uars_criterion_analysis(df_nightly, df_uars_diag=None,
                                   change_dates=None, rolling_window=30):
    # type: (pd.DataFrame, Optional[pd.DataFrame], Optional[list], int) -> Optional[str]
    """Three-panel UARS criterion analysis figure.

    Panel 1: uars_score over time (scatter + rolling median)
    Panel 2: per-criterion firing strip (tick per firing night)
    Panel 3: per-criterion firing rate bar chart

    Requires df_uars_diag with crit_* boolean columns (from the §12 batch
    loop's uars_diagnostic output). Falls back to None if not provided.
    """
    if not HAS_MPL or df_uars_diag is None:
        return None

    crit_cols = ["crit_rin_elevated", "crit_fl_elevated", "crit_oai_low",
                 "crit_pressure_rising", "crit_pci_elevated"]
    if not all(c in df_uars_diag.columns for c in crit_cols):
        return None
    if "uars_score" not in df_uars_diag.columns or \
       "date" not in df_uars_diag.columns:
        return None

    df = df_uars_diag.copy()
    df["date_dt"] = pd.to_datetime(df["date"], format="%Y%m%d",
                                    errors="coerce")
    df = df.dropna(subset=["date_dt"]).sort_values("date_dt").reset_index(drop=True)
    if len(df) < 10:
        return None

    crit_labels = [
        ("crit_rin_elevated",    "RIN elevated\n(RIN > 5.0 /hr)"),
        ("crit_fl_elevated",     "FL elevated\n(fl_frac > 0.15)"),
        ("crit_oai_low",         "OAI low\n(OAI < 5.0 /hr)"),
        ("crit_pressure_rising", "Pressure rising\n(slope > 0.5 cmH₂O/hr)"),
        ("crit_pci_elevated",    "PCI elevated\n(PCI > 5.0 /hr)"),
    ]

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 2.5, 2.0], hspace=0.75)

    # --- Panel 1 — score over time -----------------------------------------
    ax1 = fig.add_subplot(gs[0])
    dates = df["date_dt"]
    scores = df["uars_score"].astype(float)

    import numpy as _np
    _np.random.seed(42)
    jitter = _np.random.uniform(-0.12, 0.12, size=len(scores))

    ax1.scatter(dates, scores + jitter, s=10, alpha=0.35, color="#2C6F8F",
                edgecolors="none", label="Nightly score")

    if len(scores) >= rolling_window:
        roll = scores.rolling(rolling_window, min_periods=10).median()
        ax1.plot(dates, roll, linewidth=2, color="#C04000", alpha=0.9,
                 label=f"{rolling_window}-night rolling median")

    ax1.axhline(3, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6,
                zorder=1)
    ax1.text(dates.iloc[-1], 3.05, " ≥3 of 5 threshold", fontsize=8,
             color="#666666", va="bottom", ha="right")

    if change_dates:
        for cd in change_dates:
            cd_dt = pd.to_datetime(cd)
            ax1.axvline(cd_dt, color="#d9534f", linewidth=0.8, linestyle=":",
                        alpha=0.4, zorder=1)

    ax1.set_ylim(-0.5, 5.5)
    ax1.set_yticks(range(0, 6))
    ax1.set_ylabel("UARS score (0–5)")
    ax1.set_title("Panel 1 — UARS composite score over time "
                  "(sum of criteria met per night)",
                  fontsize=10, loc="left")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.2)

    # --- Panel 2 — per-criterion strip -------------------------------------
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    fire_rates = []
    for row_i, (col, _label) in enumerate(crit_labels):
        fired_dates = dates[df[col] == True]
        fr = 100.0 * df[col].mean()
        fire_rates.append(fr)

        if 5 <= fr <= 50:
            color = "#2E7D32"
        else:
            color = "#B0000A"

        if len(fired_dates) > 0:
            ax2.scatter(fired_dates, [row_i] * len(fired_dates),
                        s=10, marker="|", color=color, alpha=0.55,
                        linewidths=1.2)
        ax2.text(1.005, row_i, "{:>5.1f}%".format(fr),
                 transform=ax2.get_yaxis_transform(),
                 va="center", ha="left", fontsize=8, color="#333333",
                 family="monospace")

    ax2.set_yticks(range(len(crit_labels)))
    ax2.set_yticklabels([lbl for _, lbl in crit_labels], fontsize=8)
    ax2.set_ylim(-0.6, len(crit_labels) - 0.4)
    ax2.invert_yaxis()
    ax2.set_title("Panel 2 — Per-criterion firing strip "
                  "(each tick = one night firing that criterion)",
                  fontsize=10, loc="left")
    ax2.grid(True, axis="x", alpha=0.2)

    # --- Panel 3 — firing rate bar chart -----------------------------------
    ax3 = fig.add_subplot(gs[2])
    labels_short = ["rin", "fl", "oai", "pressure", "pci"]

    ax3.axvspan(5, 50, color="#90C695", alpha=0.18, zorder=0)

    bar_colors = ["#2E7D32" if 5 <= fr <= 50 else "#B0000A"
                  for fr in fire_rates]

    y_pos = list(range(len(labels_short)))
    ax3.barh(y_pos, fire_rates, color=bar_colors, alpha=0.8,
             edgecolor="#333", linewidth=0.5)

    for i, fr in enumerate(fire_rates):
        offset = 1.5 if fr < 95 else -1.5
        ha = "left" if fr < 95 else "right"
        text_color = "#000" if fr < 95 else "#FFF"
        ax3.text(fr + offset, i, "{:.1f}%".format(fr),
                 va="center", ha=ha, fontsize=9, color=text_color)

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels_short, fontsize=9)
    ax3.set_xlabel("% of nights firing criterion")
    ax3.set_xlim(0, 105)
    ax3.invert_yaxis()
    ax3.set_title("Panel 3 — Per-criterion firing rate "
                  "(green band = 5%–50% discriminating range)",
                  fontsize=10, loc="left")
    ax3.grid(True, axis="x", alpha=0.2)

    # Shared x-axis formatting
    import matplotlib.dates as _mdates
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(_mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(_mdates.DateFormatter("%Y-%m"))

    ax2.tick_params(axis="x", labelbottom=True)
    for lbl in ax2.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
    ax1.tick_params(axis="x", labelbottom=False)

    fig.suptitle("{} nights analyzed".format(len(df)),
             fontsize=10, fontweight="normal", color="#666",
             y=0.995)

    return _fig_to_base64(fig)


def _plot_glasgow_over_time(df_nightly, change_dates=None):
    # type: (pd.DataFrame, Optional[list]) -> Optional[str]
    """Plot Glasgow Index over time with rolling median."""
    if not HAS_MPL or df_nightly is None:
        return None
    if "glasgow_overall" not in df_nightly.columns or "date" not in df_nightly.columns:
        return None

    df = df_nightly.dropna(subset=["glasgow_overall"]).copy()
    if len(df) < 5:
        return None

    df["date_dt"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_dt")

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.scatter(df["date_dt"], df["glasgow_overall"],
               s=8, alpha=0.4, color="#8e44ad", zorder=2)

    if len(df) >= 14:
        roll = df.set_index("date_dt")["glasgow_overall"].rolling(
            14, min_periods=5).median()
        ax.plot(roll.index, roll.values, linewidth=2, color="#8e44ad",
                alpha=0.8, label="14-day median", zorder=3)

    if change_dates:
        for cd in change_dates:
            cd_dt = pd.to_datetime(cd)
            ax.axvline(cd_dt, color="#d9534f", linewidth=0.8, linestyle=":",
                       alpha=0.5, zorder=1)

    ax.set_ylabel("Glasgow Overall")
    ax.set_title("Glasgow Index Over Time (lower = more eupneic breathing)")
    ax.legend(fontsize=8, loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# Longitudinal Report
# ---------------------------------------------------------------------------

def generate_longitudinal(
    df_str,                 # type: pd.DataFrame
    df_nightly=None,        # type: Optional[pd.DataFrame]
    change_summary=None,    # type: Optional[dict]
    longitudinal_stats=None,  # type: Optional[Dict[str, dict]]
    cooc_summary=None,      # type: Optional[dict]
    device_sn="",           # type: str
    output_dir=None,        # type: Optional[Path]
    df_uars_diag=None,
):
    # type: (...) -> str
    """Generate a longitudinal HTML report.

    Parameters
    ----------
    df_str : pd.DataFrame
        STR data.
    df_nightly : pd.DataFrame, optional
        Batch nightly summary with CTQI, ODI, etc.
        Expected columns: date, CTQI, CTQI_grade, uars_suggestive,
        glasgow_overall (all optional).
    change_summary : dict, optional
        From settings_changes.settings_change_summary().
    longitudinal_stats : dict, optional
        From indices.compute_longitudinal_stats().
    cooc_summary : dict, optional
        Aggregated pulse-flow co-occurrence summary with keys:
        n_nights, total_ps, total_eve, total_ds,
        ps_w_eve, ps_w_fl, ps_w_ds,
        eve_w_ps, eve_w_fl, eve_w_ds,
        ds_w_eve, ds_w_ps, ds_w_fl.
    device_sn : str
        Device serial number (accepted for API compat, not displayed).
    output_dir : Path, optional
        If given, write HTML file to this directory.

    Returns
    -------
    HTML string.
    """
    if not HAS_PANDAS:
        return "<html><body><p>pandas required for longitudinal report</p></body></html>"

    # Date setup — always use index (DatetimeIndex from read_str)
    dates = pd.to_datetime(df_str.index)

    # Filter active nights
    active = df_str["Duration"].values.astype(float) > 0 if "Duration" in df_str.columns else np.ones(len(df_str), dtype=bool)
    n_total = int(np.sum(active))

    # Formatted date range
    d_min = dates.min()
    d_max = dates.max()
    date_range = (f"{d_min.strftime('%b %d, %Y').replace(' 0', ' ')} to "
                  f"{d_max.strftime('%b %d, %Y').replace(' 0', ' ')}")

    change_dates = []
    if change_summary:
        change_dates = change_summary.get("change_dates", [])

    parts = []
    parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    parts.append("<title>CPAP Longitudinal Report</title>")
    parts.append(f"<style>{_CSS}</style></head><body>")

    # =====================================================================
    # 1. Header
    # =====================================================================
    parts.append("<h1>CPAP Longitudinal Report</h1>")
    n_oxi = 0
    if "SpO2.50" in df_str.columns:
        n_oxi = int(np.sum(~np.isnan(df_str["SpO2.50"].values.astype(float)) & active))
    parts.append(f'<p class="session-header">{date_range} &nbsp;|&nbsp; '
                 f"Active Nights: {n_total:,} &nbsp;|&nbsp; "
                 f"Nights with Oximetry: {n_oxi:,}</p>")

    # =====================================================================
    # 2. Key Findings Summary
    # =====================================================================
    parts.append("<h2>Key Findings</h2>")
    kf_rows = []

    # AHI
    if "AHI" in df_str.columns:
        ahi_med = np.nanmedian(df_str["AHI"][active].values.astype(float))
        kf_rows.append(["AHI (median)", _fmt(ahi_med, 2) + " events/hr"])

    # CTQI
    if df_nightly is not None and "CTQI" in df_nightly.columns:
        ctqi_med = df_nightly["CTQI"].dropna().median()
        kf_rows.append(["CTQI (median)", _fmt(ctqi_med, 1)])

    # CTQI grade distribution
    if df_nightly is not None and "CTQI_grade" in df_nightly.columns:
        grades = df_nightly["CTQI_grade"].dropna()
        if len(grades) > 0:
            grade_order = ["Excellent", "Good", "Fair", "Poor", "Very Poor"]
            grade_parts = []
            for g in grade_order:
                n = int((grades == g).sum())
                if n > 0:
                    grade_parts.append(f"{g}: {n} ({_pct(n, len(grades))})")
            if grade_parts:
                kf_rows.append(["CTQI Grades", " &nbsp;|&nbsp; ".join(grade_parts)])

    # Glasgow
    if df_nightly is not None and "glasgow_overall" in df_nightly.columns:
        glas_med = df_nightly["glasgow_overall"].dropna().median()
        if not np.isnan(glas_med):
            kf_rows.append(["Glasgow Index (median)", _fmt(glas_med, 3)])

    # UARS
    if df_nightly is not None and "uars_suggestive" in df_nightly.columns:
        uars_n = int(df_nightly["uars_suggestive"].sum())
        uars_total = int(df_nightly["uars_suggestive"].count())
        kf_rows.append(["UARS-Suggestive Nights",
                        f"{uars_n} of {uars_total} ({_pct(uars_n, uars_total)})"])

    # Co-occurrence headline
    if cooc_summary and cooc_summary.get("total_ps", 0) > 0:
        ps_eve_pct = _pct(cooc_summary["ps_w_eve"], cooc_summary["total_ps"])
        ps_fl_pct = _pct(cooc_summary["ps_w_fl"], cooc_summary["total_ps"])
        kf_rows.append(["Pulse Surge Co-occurrence",
                        f"{ps_eve_pct} with EVE events, "
                        f"{ps_fl_pct} with flow limitation"])

    # Trajectory
    if change_summary:
        traj_ahi = change_summary.get("trajectory_ahi", {}).get("overall", {})
        if traj_ahi.get("slope_per_month") is not None:
            slope = traj_ahi["slope_per_month"]
            r2 = traj_ahi.get("r_squared", float("nan"))
            kf_rows.append(["AHI Trajectory",
                            f"{slope:+.3f} events/hr per month (R²={_fmt(r2, 3)})"])

    if kf_rows:
        parts.append(_html_table(["Metric", "Value"], kf_rows))

    # =====================================================================
    # 3. Longitudinal Plots
    # =====================================================================
    parts.append("<h2>Longitudinal Plots</h2>")

    # AHI over time
    if "AHI" in df_str.columns:
        ahi_vals = df_str["AHI"][active].values.astype(float)
        ahi_dates = dates[active]
        traj = change_summary.get("trajectory_ahi", {}).get("overall", {}) if change_summary else {}
        b64 = _plot_longitudinal_metric(
            ahi_dates, ahi_vals, "AHI Over Time", "AHI (events/hr)",
            color="#d9534f", change_dates=change_dates,
            trend_slope=traj.get("slope_per_month"),
            trend_intercept=traj.get("intercept"))
        if b64:
            parts.append(_img_tag(b64, "AHI Over Time"))

    # CTQI over time (from df_nightly)
    if df_nightly is not None and "CTQI" in df_nightly.columns:
        ctqi_valid = df_nightly.dropna(subset=["CTQI"])
        if len(ctqi_valid) > 0:
            cd = pd.to_datetime(ctqi_valid["date"])
            cv = ctqi_valid["CTQI"].values.astype(float)
            traj = change_summary.get("trajectory_ctqi", {}).get("overall", {}) if change_summary else {}
            b64 = _plot_longitudinal_metric(
                cd, cv, "CTQI Over Time", "CTQI (0–100)",
                color="#1a6a2a", change_dates=change_dates,
                trend_slope=traj.get("slope_per_month"),
                trend_intercept=traj.get("intercept"))
            if b64:
                parts.append(_img_tag(b64, "CTQI Over Time"))

    # Glasgow Index over time (from df_nightly)
    b64 = _plot_glasgow_over_time(df_nightly, change_dates)
    if b64:
        parts.append(_img_tag(b64, "Glasgow Index Over Time"))

    # Leak over time
    if "Leak.50" in df_str.columns:
        vals = df_str["Leak.50"][active].values.astype(float)
        b64 = _plot_longitudinal_metric(
            dates[active], vals, "Median Leak Over Time", "Leak.50 (L/s)",
            color="#c04000", change_dates=change_dates)
        if b64:
            parts.append(_img_tag(b64, "Leak Over Time"))

    # SpO2 over time
    if "SpO2.50" in df_str.columns:
        spo2_vals = df_str["SpO2.50"][active].values.astype(float)
        oxi_mask = ~np.isnan(spo2_vals)
        if np.sum(oxi_mask) > 3:
            b64 = _plot_longitudinal_metric(
                dates[active][oxi_mask], spo2_vals[oxi_mask],
                "Median SpO2 Over Time (oximetry nights)", "SpO2.50 (%)",
                color="#1a6a2a", change_dates=change_dates)
            if b64:
                parts.append(_img_tag(b64, "SpO2 Over Time"))

    # AHI breakdown stacked bar
    b64 = _plot_ahi_breakdown(df_str[active])
    if b64:
        parts.append(_img_tag(b64, "AHI Breakdown"))

    # Correlation heatmap
    b64 = _plot_correlation_heatmap(df_str[active])
    if b64:
        parts.append(_img_tag(b64, "Correlation Heatmap"))

    # =====================================================================
    # 4. UARS Criterion Analysis
    # =====================================================================
    if df_uars_diag is not None and len(df_uars_diag) > 0 and \
       "uars_score" in df_uars_diag.columns:
        parts.append("<h2>UARS Criterion Analysis</h2>")

        n_total = len(df_uars_diag)
        crit_cols = ["crit_rin_elevated", "crit_fl_elevated", "crit_oai_low",
                     "crit_pressure_rising", "crit_pci_elevated"]

        # Build the finding sentence from the data
        if all(c in df_uars_diag.columns for c in crit_cols):
            rates = {c: 100.0 * df_uars_diag[c].mean() for c in crit_cols}
            n_discriminating = sum(1 for r in rates.values() if 5 <= r <= 50)
            max_score = int(df_uars_diag["uars_score"].max())

            # Identify which criteria are discriminating
            disc_names = []
            name_map = {"crit_rin_elevated": "RIN",
                        "crit_fl_elevated": "FL",
                        "crit_oai_low": "OAI",
                        "crit_pressure_rising": "pressure ramp",
                        "crit_pci_elevated": "PCI"}
            for c, r in rates.items():
                if 5 <= r <= 50:
                    disc_names.append(name_map[c])

            if len(disc_names) == 0:
                disc_text = "None of the five criteria operate"
            elif len(disc_names) == 1:
                disc_text = f"Only one of the five criteria ({disc_names[0]}) operates"
            else:
                disc_text = (f"{len(disc_names)} of 5 criteria "
                             f"({', '.join(disc_names)}) operate")

            parts.append(
                f"<p>Per-criterion decomposition across {n_total} nights. "
                f"{disc_text} in a discriminating range (5%–50% firing rate) for "
                f"this dataset; max UARS score observed was {max_score} of 5. "
                f"The composite flag is less informative than the per-criterion "
                f"view for interpreting this user's profile — see the pulse-flow "
                f"co-occurrence section below for the more sensitive UARS-adjacent "
                f"signal.</p>"
            )

        b64 = _plot_uars_criterion_analysis(
            df_nightly, df_uars_diag, change_dates
        )
        if b64:
            parts.append(_img_tag(b64, "UARS Criterion Analysis"))

    # =====================================================================
    # 5. Pulse-Flow Co-Occurrence
    # =====================================================================
    if cooc_summary and cooc_summary.get("total_ps", 0) > 0:
        parts.append("<h2>Pulse-Flow Co-Occurrence Analysis</h2>")
        parts.append(f'<p>Cross-event temporal co-occurrence (±30 s window) across '
                     f'{cooc_summary.get("n_nights", 0):,} nights with oximetry data.</p>')

        t_ps = cooc_summary["total_ps"]
        t_eve = cooc_summary.get("total_eve", 0)
        t_ds = cooc_summary.get("total_ds", 0)

        cooc_rows = [
            ["Pulse Surges", f"{t_ps:,}",
             _pct(cooc_summary.get("ps_w_eve", 0), t_ps),
             _pct(cooc_summary.get("ps_w_fl", 0), t_ps),
             _pct(cooc_summary.get("ps_w_ds", 0), t_ps)],
        ]
        if t_eve > 0:
            cooc_rows.append(
                ["EVE Events", f"{t_eve:,}",
                 "—",
                 _pct(cooc_summary.get("eve_w_fl", 0), t_eve),
                 _pct(cooc_summary.get("eve_w_ds", 0), t_eve)])
        if t_ds > 0:
            cooc_rows.append(
                ["Desaturations", f"{t_ds:,}",
                 _pct(cooc_summary.get("ds_w_eve", 0), t_ds),
                 _pct(cooc_summary.get("ds_w_fl", 0), t_ds),
                 "—"])

        parts.append(_html_table(
            ["Event Type", "Total", "w/ EVE (±30 s)", "w/ FlowLim &gt; 0", "w/ SpO₂ Desat"],
            cooc_rows))

        # Bidirectional: EVE → pulse surge
        if t_eve > 0:
            eve_ps_pct = _pct(cooc_summary.get("eve_w_ps", 0), t_eve)
            parts.append(f"<p>{eve_ps_pct} of EVE events co-occur with a pulse surge (±30 s).</p>")

    # =====================================================================
    # 6. Therapy Trajectory
    # =====================================================================
    if change_summary:
        traj_ahi = change_summary.get("trajectory_ahi", {})
        traj_ctqi = change_summary.get("trajectory_ctqi", {})
        parts.append("<h2>Therapy Trajectory</h2>")

        if traj_ahi:
            ov = traj_ahi.get("overall", {})
            parts.append(f"<p><b>AHI:</b> slope = {_fmt(ov.get('slope_per_month', float('nan')), 4)}/month, "
                         f"R² = {_fmt(ov.get('r_squared', float('nan')), 4)}, "
                         f"n = {ov.get('n', 0)}</p>")

        if traj_ctqi:
            ov = traj_ctqi.get("overall", {})
            parts.append(f"<p><b>CTQI:</b> slope = {_fmt(ov.get('slope_per_month', float('nan')), 4)}/month, "
                         f"R² = {_fmt(ov.get('r_squared', float('nan')), 4)}</p>")

    # =====================================================================
    # 7. Settings Changes (condensed)
    # =====================================================================
    if change_summary and change_summary.get("n_changes", 0) > 0:
        before_after = change_summary.get("before_after", [])
        parts.append("<h2>Settings Changes</h2>")

        # Summary table: one row per change date
        summary_rows = []
        significant_indices = []   # unconfounded with |effect| >= 0.3
        for i, ba in enumerate(before_after):
            cd_str = _format_change_date(ba.get("change_date", ""))
            what_changed = ", ".join(
                f'{_SETTING_LABELS.get(sc["signal"], sc["signal"])}: '
                f'{_format_setting_value(sc["signal"], sc["old_value"])} → '
                f'{_format_setting_value(sc["signal"], sc["new_value"])}'
                for sc in ba.get("settings_changed", []))
            status = _classify_change(ba)
            sig = _has_significant_effect(ba)
            status_label = status.title()
            if sig:
                status_label += " ✦"
            # Only expand unconfounded changes with significant effects
            if sig and status == "unconfounded":
                significant_indices.append(i)
            summary_rows.append([cd_str, what_changed, status_label])

        parts.append(f"<p>{len(before_after)} change date(s) detected. "
                     f"✦ = statistically significant (p &lt; 0.05 after Holm correction). "
                     f"Only unconfounded changes with significant effects are expanded below.</p>")
        parts.append(_html_table(
            ["Date", "Changes", "Status"], summary_rows))

        # Expanded before/after tables for significant unconfounded changes
        shown_any = False
        for i in significant_indices:
            ba = before_after[i]
            cd_str = _format_change_date(ba.get("change_date", ""))
            comps = ba.get("comparisons", {})
            if not comps:
                continue

            if not shown_any:
                parts.append("<h3>Significant Before/After Comparisons</h3>")
                shown_any = True

            what = ", ".join(
                f'{_SETTING_LABELS.get(sc["signal"], sc["signal"])}: '
                f'{_format_setting_value(sc["signal"], sc["old_value"])} → '
                f'{_format_setting_value(sc["signal"], sc["new_value"])}'
                for sc in ba.get("settings_changed", []))
            status_note = ""
            if ba.get("confounded"):
                status_note = " (confounded)"
            parts.append(f"<p><b>{cd_str}</b>{status_note} — {what}</p>")

            ba_rows = []
            for metric, c in comps.items():
                if _comparison_is_significant(c):
                    ba_rows.append([
                        metric,
                        _fmt(c.get("before_median", float("nan")), 3),
                        _fmt(c.get("after_median", float("nan")), 3),
                        _fmt(c.get("p_value", float("nan")), 4),
                        _fmt(c.get("p_adjusted", float("nan")), 4),
                        _fmt(c.get("effect_size", float("nan")), 3),
                        c.get("direction", "—"),
                    ])
            if ba_rows:
                parts.append(_html_table(
                    ["Metric", "Before (med)", "After (med)", "p-value",
                     "p (Holm)", "Effect Size", "Direction"], ba_rows))

        # All remaining changes in a collapsible block
        non_expanded = [i for i in range(len(before_after))
                        if i not in significant_indices]
        if non_expanded:
            parts.append("<details><summary>Other Changes "
                         f"({len(non_expanded)} — confounded, insufficient, "
                         f"or no significant effect)</summary>")
            for i in non_expanded:
                ba = before_after[i]
                cd_str = _format_change_date(ba.get("change_date", ""))
                what = ", ".join(
                    f'{sc["signal"]}: {_fmt(sc["old_value"])} → {_fmt(sc["new_value"])}'
                    for sc in ba.get("settings_changed", []))
                status = _classify_change(ba)
                parts.append(f"<p><b>{cd_str}</b> ({status}) — {what}</p>")
            parts.append("</details>")

    # =====================================================================
    # 8. Longitudinal Summary Statistics (condensed)
    # =====================================================================
    if longitudinal_stats:
        parts.append("<h2>Longitudinal Summary Statistics</h2>")

        # Key signals as combined table
        key_table = _longitudinal_key_stats_table(longitudinal_stats)
        if key_table:
            parts.append(key_table)

        # All signals as a second combined table
        remaining = {k: v for k, v in longitudinal_stats.items()
                     if k not in ("AHI", "Leak.50", "MaskPress.50",
                                  "SpO2.50", "RespRate.50", "Duration")}
        if remaining:
            parts.append(f"<h3>All Signal Statistics ({len(remaining)} signals)</h3>")
            parts.append(_longitudinal_all_stats_table(remaining))

    # =====================================================================
    # Footer
    # =====================================================================
    parts.append(_FOOTER)
    parts.append("</body></html>")
    html = "\n".join(parts)

    if output_dir is not None:
        out_path = Path(output_dir) / "report_longitudinal.html"
        out_path.write_text(html, encoding="utf-8")

    return html
