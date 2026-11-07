r"""
resmed_edf.py
=============
SleepingAtTheTop — NightmaresOfTheBottom / AirSense 11

Zero-dependency EDF reader for ResMed AirSense 11 SD card data.
Handles all file types: BRP, PLD, SA2, EVE, CSL, STR.

Dependencies: Python 3.8+, numpy, pandas (stdlib only for the raw reader)

Usage
-----
    from resmed_edf import read_session, read_str, parse_eve

    # One night
    session = read_session(r"C:/path/to/DATALOG/YYYYMMDD")
    flow   = session["BRP"]["Flow.40ms"]   # numpy array
    leak   = session["PLD"]["Leak.2s"]     # numpy array
    spo2   = session["SA2"]["SpO2.1s"]     # numpy array (filter zeros!)
    events = session["EVE"]                # list of dicts
    meta   = session["meta"]               # start times, fs, durations

    # Longitudinal
    df = read_str(r"C:\ResMed\STR.edf")   # pandas DataFrame, one row per day
"""

from __future__ import annotations
import struct
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal: pure-bytes EDF header parser
# ---------------------------------------------------------------------------

def _parse_header(fp: Path) -> dict:
    """Parse the 256-byte EDF file header + per-signal headers."""
    with open(fp, "rb") as f:
        h = f.read(256)

    ns = int(h[252:256].strip())
    hdr_bytes = int(h[184:192].strip())
    num_records = int(h[236:244].strip())
    record_duration = float(h[244:252].strip())
    startdate = h[168:176].decode("ascii").strip()   # DD.MM.YY
    starttime = h[176:184].decode("ascii").strip()   # HH.MM.SS
    reserved = h[192:236].decode("ascii").strip()    # "EDF+D" or "EDF"

    # Parse start datetime  (ResMed uses 2-digit year, always 20xx)
    d, mo, y2 = startdate.split(".")
    hh, mm, ss = starttime.split(".")
    start_dt = datetime(2000 + int(y2), int(mo), int(d),
                        int(hh), int(mm), int(ss))

    # Per-signal header fields — each is ns * field_width bytes, sequential
    with open(fp, "rb") as f:
        f.seek(256)
        labels      = [f.read(16).decode("ascii", errors="replace").strip() for _ in range(ns)]
        _           = [f.read(80) for _ in range(ns)]   # transducer type (skip)
        dims        = [f.read(8).decode("ascii",  errors="replace").strip() for _ in range(ns)]
        phys_mins   = [float(f.read(8).strip()) for _ in range(ns)]
        phys_maxs   = [float(f.read(8).strip()) for _ in range(ns)]
        dig_mins    = [int(f.read(8).strip())   for _ in range(ns)]
        dig_maxs    = [int(f.read(8).strip())   for _ in range(ns)]
        _           = [f.read(80) for _ in range(ns)]   # prefilter (skip)
        nsamples    = [int(f.read(8).strip())   for _ in range(ns)]

    return dict(
        ns=ns,
        hdr_bytes=hdr_bytes,
        num_records=num_records,
        record_duration=record_duration,
        start_dt=start_dt,
        reserved=reserved,
        is_edfplusd=reserved.startswith("EDF+D"),
        labels=labels,
        dims=dims,
        phys_mins=phys_mins,
        phys_maxs=phys_maxs,
        dig_mins=dig_mins,
        dig_maxs=dig_maxs,
        nsamples=nsamples,
    )


def _scale(raw: np.ndarray, phys_min: float, phys_max: float,
           dig_min: int, dig_max: int) -> np.ndarray:
    """Convert raw int16 digital values to physical units."""
    gain = (phys_max - phys_min) / (dig_max - dig_min)
    offset = phys_min - gain * dig_min
    return raw.astype(np.float32) * gain + offset


# ---------------------------------------------------------------------------
# Internal: read all signals from a standard EDF (BRP / PLD / SA2 / STR)
# ---------------------------------------------------------------------------

def _read_continuous_signals(fp: Path) -> Tuple[dict, dict]:
    """
    Returns:
        header  : the parsed header dict
        signals : {label: np.ndarray}  (Crc16 excluded)
    """
    hdr = _parse_header(fp)
    ns = hdr["ns"]
    total_per_rec = sum(hdr["nsamples"])
    rec_bytes = total_per_rec * 2

    # Pre-allocate per-signal lists
    buffers: Dict[int, list] = {i: [] for i in range(ns)}

    with open(fp, "rb") as f:
        f.seek(hdr["hdr_bytes"])
        for _ in range(hdr["num_records"]):
            block = f.read(rec_bytes)
            if len(block) < rec_bytes:
                break
            vals = struct.unpack_from(f"<{total_per_rec}h", block)
            pos = 0
            for i, n in enumerate(hdr["nsamples"]):
                buffers[i].append(vals[pos:pos + n])
                pos += n

    signals = {}
    for i, label in enumerate(hdr["labels"]):
        if label.lower() == "crc16":
            continue
        raw = np.array([v for chunk in buffers[i] for v in chunk], dtype=np.int16)
        signals[label] = _scale(
            raw,
            hdr["phys_mins"][i], hdr["phys_maxs"][i],
            hdr["dig_mins"][i],  hdr["dig_maxs"][i],
        )

    return hdr, signals


# ---------------------------------------------------------------------------
# Internal: parse EDF+D TAL annotations (EVE / CSL)
# ---------------------------------------------------------------------------

def _parse_tal_bytes(data: bytes, start_dt: datetime) -> List[dict]:
    """
    Parse Time-stamped Annotations List (TAL) bytes from an EDF+D data block.
    TAL format: +onset\x14\x14  or  +onset\x15duration\x14label\x14\x00...
    Bytes 0xEF 0xBF 0xBD (UTF-8 replacement char) are treated as separators/junk.
    """
    # Replace known junk bytes so they don't corrupt splitting
    clean = data.replace(b"\xef\xbf\xbd", b"\x00")
    text = clean.decode("latin-1")

    events = []
    # Split on null bytes to get individual TAL blocks
    for block in text.split("\x00"):
        block = block.strip()
        if not block:
            continue
        # Each block can contain multiple annotations separated by \x14
        parts = block.split("\x14")
        i = 0
        while i < len(parts):
            seg = parts[i].strip()
            if not seg.startswith("+"):
                i += 1
                continue
            # Parse onset and optional duration
            if "\x15" in seg:
                time_part, dur_str = seg.split("\x15", 1)
                onset = _safe_float(time_part[1:])
                dur   = _safe_float(dur_str)
            else:
                onset = _safe_float(seg[1:])
                dur   = 0.0
            # Next non-empty part is the label
            label = ""
            j = i + 1
            while j < len(parts):
                candidate = parts[j].strip().strip("\x00")
                if candidate and not candidate.startswith("+"):
                    label = candidate
                    break
                j += 1
            # Skip empty labels and numeric TAL fragments (unparsed onset/duration strings)
            is_valid = label and not label.replace(".", "").replace("-", "").replace("\x15", "").isdigit()
            if is_valid:
                abs_time = start_dt + timedelta(seconds=onset)
                events.append({
                    "onset_s":    onset,
                    "duration_s": dur,
                    "label":      label,
                    "time":       abs_time,
                })
            i += 1

    return events


def _safe_float(s: str) -> float:
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return 0.0


def _read_annotations(fp: Path) -> Tuple[dict, List[dict]]:
    """Read EVE or CSL EDF+D file; return (header, list of event dicts)."""
    hdr = _parse_header(fp)
    if not hdr["is_edfplusd"]:
        raise ValueError(f"{fp.name} is not EDF+D (reserved='{hdr['reserved']}')")

    ann_idx = next(
        (i for i, l in enumerate(hdr["labels"]) if "annotation" in l.lower()),
        0
    )
    n_ann = hdr["nsamples"][ann_idx]
    total_per_rec = sum(hdr["nsamples"])
    rec_bytes = total_per_rec * 2

    all_data = b""
    with open(fp, "rb") as f:
        f.seek(hdr["hdr_bytes"])
        for _ in range(hdr["num_records"]):
            block = f.read(rec_bytes)
            if len(block) < rec_bytes:
                break
            # Annotation channel occupies first n_ann*2 bytes of each record
            # (per EDF+D spec, annotations channel is always first)
            start_byte = sum(hdr["nsamples"][:ann_idx]) * 2
            end_byte   = start_byte + n_ann * 2
            all_data  += block[start_byte:end_byte]

    events = _parse_tal_bytes(all_data, hdr["start_dt"])
    # Deduplicate (some "Recording starts" entries appear twice)
    seen = set()
    unique = []
    for ev in events:
        key = (ev["onset_s"], ev["label"])
        if key not in seen:
            seen.add(key)
            unique.append(ev)

    return hdr, sorted(unique, key=lambda e: e["onset_s"])


# ---------------------------------------------------------------------------
# Internal: combine several files of one type from the same night
# ---------------------------------------------------------------------------
# A night folder holds one BRP/PLD/SA2 file set per mask-on session, so a
# night with a mask-off break has two or more files of each type. They are
# placed on one time axis starting at the earliest file, with NaN filling the
# gaps between sessions, so sample index / fs still maps to clock time.

def _fs_map(hdr: dict) -> dict:
    return {
        label: (hdr["nsamples"][i] / hdr["record_duration"]
                if hdr["record_duration"] > 0 else 0)
        for i, label in enumerate(hdr["labels"])
        if label.lower() != "crc16"
    }


def _combine_continuous(parts: list) -> Tuple[dict, dict]:
    """Stitch [(path, hdr, signals), ...] onto one time axis.

    Returns (signals, meta) where meta has the same keys read_session()
    always produced, plus end_dt, span_s and sessions.
    """
    parts = sorted(parts, key=lambda p: p[1]["start_dt"])
    first_hdr = parts[0][1]
    base = first_hdr["start_dt"]
    fs = _fs_map(first_hdr)

    sessions = []
    end_dt = base
    for fp, hdr, _ in parts:
        dur = hdr["num_records"] * hdr["record_duration"]
        s_end = hdr["start_dt"] + timedelta(seconds=dur)
        sessions.append({"file": fp.name, "start_dt": hdr["start_dt"],
                         "end_dt": s_end, "duration_s": dur})
        end_dt = max(end_dt, s_end)

    meta = {
        "start_dt":        base,
        "num_records":     sum(h["num_records"] for _, h, _ in parts),
        "record_duration": first_hdr["record_duration"],
        # recorded time only (gaps between sessions excluded)
        "duration_s":      sum(x["duration_s"] for x in sessions),
        "fs":              fs,
        "end_dt":          end_dt,
        "span_s":          (end_dt - base).total_seconds(),
        "sessions":        sessions,
    }

    if len(parts) == 1:
        return parts[0][2], meta

    signals: dict = {}
    labels = []
    for _, _, sig in parts:
        for label in sig:
            if label not in labels:
                labels.append(label)

    for label in labels:
        rate = fs.get(label)
        if rate is None:
            # signal absent from the first file: take its rate from any file
            for _, hdr, sig in parts:
                if label in sig:
                    rate = _fs_map(hdr)[label]
                    break
            fs[label] = rate
        n_total = int(round(meta["span_s"] * rate))
        out = np.full(n_total, np.nan, dtype=np.float32)
        for fp, hdr, sig in parts:
            if label not in sig:
                continue
            if abs(_fs_map(hdr)[label] - rate) > 1e-9:
                print(f"  Warning: {fp.name} {label} has a different sample "
                      f"rate; skipped")
                continue
            arr = sig[label]
            off = int(round((hdr["start_dt"] - base).total_seconds() * rate))
            end = min(off + arr.size, n_total)
            if end > off:
                out[off:end] = arr[:end - off]
        signals[label] = out

    return signals, meta


# ---------------------------------------------------------------------------
# Public: read one night's session folder
# ---------------------------------------------------------------------------

def read_session(folder: str | Path) -> dict:
    """
    Read all EDF files for one night from a DATALOG/YYYYMMDD folder.

    Returns a dict with keys: BRP, PLD, SA2, EVE, CSL, meta
    Each of BRP/PLD/SA2 is a dict {signal_label: np.ndarray}.
    EVE and CSL are lists of event dicts with keys:
        onset_s, duration_s, label, time (datetime)
    meta contains start times, sampling rates, and session duration.

    A night with more than one mask-on session has one BRP/PLD/SA2 file per
    session. All of them are read and placed on one time axis starting at
    the earliest session, with NaN in the gaps between sessions.
    meta[token]["duration_s"] is recorded time (gaps excluded);
    meta[token]["sessions"] lists each file's start, end and duration.

    Signals with all-zero or invalid values are NOT filtered here —
    filter SA2 zeros in downstream code (zeros = signal loss, not true zero).
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Session folder not found: {folder}")

    # Discover files by token.
    # Token regex uses [A-Z0-9] to match names like SA2, SAD (AirSense 10 oximetry).
    token_re = re.compile(r"_([A-Z0-9]{3})\.edf$", re.IGNORECASE)
    by_token: Dict[str, List[Path]] = {}
    for fp in sorted(folder.glob("*.edf")):
        m = token_re.search(fp.name)
        if m:
            by_token.setdefault(m.group(1).upper(), []).append(fp)

    result: dict = {"meta": {}}

    # --- Continuous signal files ---
    for token in ("BRP", "PLD", "SA2"):
        if token not in by_token:
            result[token] = {}
            continue
        parts = []
        for fp in by_token[token]:
            hdr, signals = _read_continuous_signals(fp)
            parts.append((fp, hdr, signals))
        signals, token_meta = _combine_continuous(parts)
        result[token] = signals
        result["meta"][token] = token_meta

    # --- Annotation files ---
    for token in ("EVE", "CSL"):
        if token not in by_token:
            result[token] = []
            continue
        loaded = []
        for fp in by_token[token]:
            try:
                loaded.append(_read_annotations(fp))
            except Exception as e:
                print(f"  Warning: could not parse {fp.name}: {e}")
        if not loaded:
            result[token] = []
            continue
        base = min(h["start_dt"] for h, _ in loaded)
        events = []
        seen = set()
        for _, evs in loaded:
            for ev in evs:
                key = (ev["time"], ev["label"])
                if key in seen:
                    continue
                seen.add(key)
                if len(loaded) > 1:
                    ev = dict(ev)
                    ev["onset_s"] = (ev["time"] - base).total_seconds()
                events.append(ev)
        result[token] = sorted(events, key=lambda e: e["time"])
        result["meta"][token] = {"start_dt": base}

    # Convenience: session start from CSL (earliest), fallback to BRP
    for src in ("CSL", "BRP", "PLD", "SA2"):
        if src in result["meta"]:
            result["meta"]["session_start"] = result["meta"][src]["start_dt"]
            break

    # Session duration in hours (from BRP if available). This is recorded
    # time; with several sessions the gaps between them are not counted.
    if "BRP" in result["meta"]:
        result["meta"]["duration_hr"] = result["meta"]["BRP"]["duration_s"] / 3600
        result["meta"]["session_end"] = result["meta"]["BRP"]["end_dt"]
        result["meta"]["n_sessions"] = len(result["meta"]["BRP"]["sessions"])

    return result


# ---------------------------------------------------------------------------
# Public: read STR.edf longitudinal summary
# ---------------------------------------------------------------------------

def read_str(str_path: str | Path) -> pd.DataFrame:
    """
    Read STR.edf and return a tidy DataFrame with one row per day.

    Columns match signal labels from the EDF header.
    The index is a DatetimeIndex of calendar dates (local time, tz-naive).
    Multi-sample signals (MaskOn, MaskOff) are summarised as:
        MaskOn_first, MaskOn_last  (first/last non-zero value in minutes)

    Invalid/sentinel values (raw int16 outside physical range after scaling,
    or values that decode to the UTF-8 replacement pattern) are replaced with NaN.
    """
    str_path = Path(str_path)
    hdr, signals = _read_continuous_signals(str_path)

    # STR start date
    base_date = hdr["start_dt"].date()
    num_records = hdr["num_records"]

    # Each signal array has length = num_records * nsamples_per_rec
    # For most STR signals nsamples=1, so array length == num_records
    # For MaskOn/MaskOff nsamples=20 (up to 20 mask-on events per night)
    rows = []
    for rec_idx in range(num_records):
        row: dict = {}
        date = base_date + timedelta(days=rec_idx)
        row["date"] = pd.Timestamp(date)

        for label, arr in signals.items():
            n = hdr["nsamples"][hdr["labels"].index(label)]
            chunk = arr[rec_idx * n: rec_idx * n + n]

            if n == 1:
                val = float(chunk[0])
                # Replace out-of-range values with NaN.
                # Use a tight lower bound: physical min - 0.1 (allows for tiny float error).
                # This catches the -1.0 sentinel that ResMed writes when SpO2/pulse
                # oximeter was not worn (phys_min for SpO2 is 0, so -1 is out-of-range).
                pm = hdr["phys_mins"][hdr["labels"].index(label)]
                pM = hdr["phys_maxs"][hdr["labels"].index(label)]
                if val < pm - 0.1 or val > pM + 0.1:
                    val = float("nan")
                row[label] = val
            else:
                # Multi-sample (MaskOn, MaskOff): keep non-zero values
                valid = chunk[chunk > 0]
                row[f"{label}_first"] = float(valid[0])  if len(valid) > 0 else float("nan")
                row[f"{label}_last"]  = float(valid[-1]) if len(valid) > 0 else float("nan")
                row[f"{label}_count"] = int(len(valid))

        rows.append(row)

    df = pd.DataFrame(rows).set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    return df


# ---------------------------------------------------------------------------
# Public: helpers
# ---------------------------------------------------------------------------

def _pld_stat(arr: np.ndarray, stat: str) -> float:
    """Compute a single statistic over a finite-only PLD signal array."""
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return float("nan")
    if stat == "median":
        return float(np.median(valid))
    if stat == "mean":
        return float(np.mean(valid))
    if stat == "p95":
        return float(np.percentile(valid, 95))
    if stat == "max":
        return float(np.max(valid))
    if stat == "min":
        return float(np.min(valid))
    raise ValueError(f"Unknown stat: {stat}")


def session_summary(session: dict) -> dict:
    """
    Return a flat dict of key nightly metrics from a parsed session.
    Covers all signals from BRP, PLD, SA2, and EVE (Crc16 excluded).
    Useful for quick inspection and building per-night summary tables.

    PLD keys use the pattern  <signal_name>_<stat>, e.g. leak_median, leak_p95.
    SA2 keys: spo2_median, spo2_p95, spo2_max, spo2_min,
              pulse_median, pulse_p95, pulse_max, pulse_min.
    BRP keys: flow_mean, flow_std  (gross summary only; use raw array for detection).
    EVE keys: events dict of label->count, ahi_estimate, total_apnea_s.
    """
    meta = session.get("meta", {})
    dur_hr = meta.get("duration_hr", float("nan"))

    session_start = meta.get("session_start")
    summary = {
        "session_start": session_start,
        "session_end":   meta.get("session_end") or (session_start + timedelta(hours=dur_hr) if session_start and dur_hr == dur_hr else None),
        "duration_hr":   round(dur_hr, 3),
    }

    # ------------------------------------------------------------------
    # BRP — high-res flow (gross summary only; waveform stays as numpy)
    # ------------------------------------------------------------------
    brp = session.get("BRP", {})
    if "Flow.40ms" in brp:
        arr = brp["Flow.40ms"]
        valid = arr[np.isfinite(arr)]
        summary["flow_mean"] = float(np.mean(valid))   if len(valid) else float("nan")
        summary["flow_std"]  = float(np.std(valid))    if len(valid) else float("nan")
    if "Press.40ms" in brp:
        arr = brp["Press.40ms"]
        valid = arr[np.isfinite(arr) & (arr > 0)]
        summary["brp_pressure_median"] = float(np.median(valid)) if len(valid) else float("nan")
        summary["brp_pressure_p95"]    = float(np.percentile(valid, 95)) if len(valid) else float("nan")

    # ------------------------------------------------------------------
    # PLD — all 9 signals (Crc16 excluded by reader)
    # ------------------------------------------------------------------
    pld = session.get("PLD", {})

    # (signal_label, summary_key_prefix, [stats_to_compute])
    _pld_specs = [
        ("MaskPress.2s", "mask_pressure", ["median", "p95", "max"]),
        ("Press.2s",     "pressure",      ["median", "p95", "max"]),
        ("EprPress.2s",  "epr_pressure",  ["median", "p95", "max"]),
        ("Leak.2s",      "leak",          ["median", "p95", "max"]),
        ("RespRate.2s",  "resp_rate",     ["median", "p95", "max"]),
        ("TidVol.2s",    "tidal_vol",     ["median", "p95", "max"]),
        ("MinVent.2s",   "min_vent",      ["median", "p95", "max"]),
        ("Snore.2s",     "snore",         ["mean",   "p95", "max"]),
        ("FlowLim.2s",   "flow_lim",      ["mean",   "p95", "max"]),
    ]
    for sig, prefix, stats in _pld_specs:
        if sig in pld:
            arr = pld[sig]
            for stat in stats:
                summary[f"{prefix}_{stat}"] = _pld_stat(arr, stat)

    # ------------------------------------------------------------------
    # SA2 — SpO2 and Pulse (zeros = signal loss, filtered before stats)
    # ------------------------------------------------------------------
    sa2 = session.get("SA2", {})
    if "SpO2.1s" in sa2:
        spo2  = sa2["SpO2.1s"]
        valid = spo2[(spo2 > 0) & np.isfinite(spo2)]
        summary["spo2_median"] = float(np.median(valid))          if len(valid) else float("nan")
        summary["spo2_p95"]    = float(np.percentile(valid, 95))  if len(valid) else float("nan")
        summary["spo2_max"]    = float(np.max(valid))             if len(valid) else float("nan")
        summary["spo2_min"]    = float(np.min(valid))             if len(valid) else float("nan")
    if "Pulse.1s" in sa2:
        pulse = sa2["Pulse.1s"]
        valid = pulse[(pulse > 0) & np.isfinite(pulse)]
        summary["pulse_median"] = float(np.median(valid))         if len(valid) else float("nan")
        summary["pulse_p95"]    = float(np.percentile(valid, 95)) if len(valid) else float("nan")
        summary["pulse_max"]    = float(np.max(valid))            if len(valid) else float("nan")
        summary["pulse_min"]    = float(np.min(valid))            if len(valid) else float("nan")

    # ------------------------------------------------------------------
    # EVE — event counts, AHI estimate, total time in apnea
    # ------------------------------------------------------------------
    _skip_labels = {"Recording starts", "Recording ends"}
    events = session.get("EVE", [])
    counts: dict = {}
    total_apnea_s = 0.0
    _apnea_labels = {"Obstructive Apnea", "Central Apnea", "Hypopnea",
                     "Unclassified Apnea", "RERA"}
    for ev in events:
        label = ev["label"]
        if label in _skip_labels:
            continue
        counts[label] = counts.get(label, 0) + 1
        if label in _apnea_labels:
            total_apnea_s += ev.get("duration_s", 0.0)

    summary["events"]        = counts
    summary["total_apnea_s"] = round(total_apnea_s, 1)

    total_events = sum(counts.values())
    summary["ahi_estimate"]  = round(total_events / dur_hr, 2) if dur_hr > 0 else float("nan")

    # RIN — RERA index (events/hr), key UARS metric
    rera_count = counts.get("RERA", 0)
    summary["rin_index"] = round(rera_count / dur_hr, 2) if dur_hr > 0 else float("nan")

    return summary


def filter_spo2_zeros(arr: np.ndarray) -> np.ndarray:
    """
    Replace invalid SA2 values with NaN.
    ResMed writes 0 for signal loss and -1 as a sentinel when the oximeter
    was not worn or not detected.  Both are invalid and must be excluded
    before any stats or plotting.
    Valid SpO2 range : 1 – 100 %
    Valid Pulse range: 1 – 300 bpm
    """
    out = arr.copy().astype(np.float32)
    out[out <= 0] = float("nan")   # catches 0 (signal loss) and -1 (not worn)
    return out


# ---------------------------------------------------------------------------
# CLI: quick sanity check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, pprint

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python resmed_edf.py <DATALOG/YYYYMMDD>   # one night")
        print("  python resmed_edf.py <STR.edf>            # longitudinal")
        sys.exit(0)

    target = Path(sys.argv[1])

    if target.is_dir():
        print(f"Reading session: {target}")
        session = read_session(target)
        summary = session_summary(session)
        print("\n=== Session Summary ===")
        pprint.pprint(summary)
        print("\n=== Signal shapes ===")
        for token in ("BRP", "PLD", "SA2"):
            data = session.get(token, {})
            for label, arr in data.items():
                fs = session["meta"].get(token, {}).get("fs", {}).get(label, "?")
                print(f"  {token}/{label:20s}: {len(arr):>8,} samples  fs={fs} Hz")
        print("\n=== EVE Events ===")
        for ev in session.get("EVE", []):
            print(f"  {ev['time'].strftime('%H:%M:%S')}  {ev['duration_s']:5.1f}s  {ev['label']}")

    elif target.suffix.lower() == ".edf":
        print(f"Reading STR: {target}")
        df = read_str(target)
        print(f"\nShape: {df.shape}  ({df.index[0].date()} → {df.index[-1].date()})")
        print("\nLast 5 nights:")
        cols = ["Duration", "AHI", "OAI", "CAI", "HI", "Leak.50", "SpO2.50", "MaskPress.50"]
        available = [c for c in cols if c in df.columns]
        print(df[available].tail(5).to_string())

    else:
        print(f"Pass a session folder (DATALOG/YYYYMMDD) or STR.edf path.")