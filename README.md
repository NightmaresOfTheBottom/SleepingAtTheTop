# SleepingAtTheTop

> "Sleeping at the top, nightmares of the bottom..."
> - Lil Wayne

An exploratory, open-source look at everything your ResMed **AirSense 11**
records on its SD card, and what you can do with it.

Most CPAP users only ever see a handful of numbers: AHI, leak, usage hours.
The SD card holds far more than that. Your machine logs your full breathing
waveform 25 times a second, a dozen trend signals every 2 seconds, oximetry
if you use it, every event it scored, and a daily summary going back as long
as you've owned it. This project reads all of it with plain Python and shows
some of what becomes possible once you have it.

A big motivation was looking for proxies for arousals, since AHI can be near
zero while sleep is still fragmented.

This is a personal exploration, not a finished product. Some analyses are
solid, some are experiments, and some are half-formed ideas I wanted to test.
The goal is to show what is possible and to encourage you to explore your own
data and build your own tools.

**AI-assisted code.** I'm not a professional software developer. I designed
the analyses, chose the methods, and checked results against OSCAR, ResMed's
own scoring, and my in-lab sleep studies, but much of the code was written
with AI coding tools. Expect rough edges, and read the code before trusting
any number it produces.

**This is not OSCAR, not a medical device, and not a diagnosis.** It is meant
as a companion to OSCAR, not a replacement. Compare anything here against
OSCAR and your clinician's report; numbers will differ.

**Tested only on AirSense 11 AutoSet data.** It has not been tested on the
AirCurve 11, AirSense 10, or any other model. If you try it on another
machine, please report what worked and what broke.

Author: **NightmaresOfTheBottom** (ApneaBoard)

## What's on your SD card

| File | Rate | What it holds |
|---|---|---|
| `BRP` | 25 Hz | Flow and pressure waveform, every breath |
| `PLD` | every 2 s | Mask pressure, EPR pressure, leak, respiratory rate, tidal volume, minute ventilation, snore, flow limitation |
| `SA2` | 1 Hz | SpO2 and pulse (only with a compatible oximeter) |
| `EVE` | per event | Every apnea, hypopnea, RERA, and central event ResMed scored, with timestamps |
| `STR` | 1 row per day | Nightly AHI, OAI, CAI, HI, RERA index, leak and pressure percentiles, mask on/off times, and your settings history |

## What this project explores

- Arousal proxies: pulse surges from oximetry (Pulse Change Index) and an
  experimental UARS-suggestive composite
- Scoring apneas and hypopneas from the raw waveform and comparing the
  results against ResMed's own events
- Oxygen desaturation detection (3% and 4% ODI)
- Breath-by-breath Glasgow Index
- CTQI, an experimental 0-100 therapy quality score
- Before/after statistics for every pressure, EPR, or mask change you made
- Sleep schedule patterns, cross-night correlations, and long-term trends
- Self-contained HTML reports for single nights and full history
- CSV exports of nightly summaries, event lists, settings changes, and raw
  per-night BRP, PLD, and SA2 signals, so you can work with the numbers in
  Excel or anything else

Treat all of these as starting points, not validated metrics.

## What you need

- A copy of your AirSense 11 SD card on disk. The folder you select must
  contain `DATALOG/` (night folders named `YYYYMMDD`) and usually `STR.edf`.
  `Identification.json` and `SETTINGS/CurrentSettings.json` are optional.
- Windows, macOS, or Linux with a desktop session (the notebook uses a folder
  picker).
- Python 3.8+ and Jupyter.

```text
YourSDCopy/
  DATALOG/
    20240115/
      *_BRP.edf   flow waveform
      *_PLD.edf   pressure / leak / flow limitation
      *_SA2.edf   SpO2 / pulse (if you used an oximeter)
      *_EVE.edf   ResMed-scored events
    ...
  STR.edf
  Identification.json          (optional)
  SETTINGS/CurrentSettings.json  (optional)
```

**Do not upload your SD card dump, CSV exports, or HTML reports to GitHub.**
Those files are health data (flow, SpO2, sleep times, device serial).

## Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook SleepingAtTheTop.ipynb
```

When the first cell runs, choose the **SD card root** (the folder that contains
`DATALOG` and `STR.edf`), then pick a night to inspect.

`tkinter` (folder dialog) ships with official Python on Windows. On some Mac
or remote Jupyter setups the dialog fails. Run the notebook locally on the
machine that has the SD copy.

## What's in this repo

| File | Role |
|---|---|
| `SleepingAtTheTop.ipynb` | Main pipeline (run this) |
| `resmed_edf.py` | EDF reader: BRP, PLD, SA2, EVE, STR |
| `event_detection.py` | Apnea/hypopnea, flow limitation, desats, pulse surges, UARS composite, Glasgow Index |
| `indices.py` | Nightly stats and CTQI (CPAP Therapy Quality Index) |
| `settings_changes.py` | Detect pressure/EPR/mask changes from STR |
| `report.py` | Single-night and longitudinal HTML reports |

ResMed writes its files as EDF+D with a few quirks that general-purpose EDF
libraries don't always handle cleanly. `resmed_edf.py` reads them directly, so
there is no `pyedflib` or MNE dependency to fight with.

**It is slow.** Batch-processing many nights takes a long time.

## Make it your own

Fork it, strip it down, or pull out one module and build something new.
`resmed_edf.py` works on its own: give it a night folder and you get numpy
arrays for every signal. Everything after that is up to you.

If you build something, post it on ApneaBoard so others can learn from it.

## Privacy

This published copy has:

- No SD-card data, exports, or HTML reports
- Notebook **outputs cleared** (executed notebooks leak paths, serials, and plots)
- Device serial read from *your* `Identification.json` at runtime, not hardcoded

Before you commit after running the notebook:

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace SleepingAtTheTop.ipynb
```

Keep `.gitignore` as-is so `*.edf`, `DATALOG/`, `exports/`, and `report_*.html`
never get added.

## Limitations

- Written against **AirSense 11** EDF layout (`EDF+D`). Other ResMed models are
  untested and may not parse.
- Custom AHI / ODI / UARS / CTQI are research metrics, not AASM-certified
  scoring.
- Hypopneas follow the AASM 2.4 rule: a 30% flow drop for at least 10 s with
  a 3% desaturation or an arousal. There is no EEG, so a pulse surge stands
  in for an arousal. Without oximetry, hypopneas are scored on flow alone and
  marked low confidence. Events lasting barely 10 s are partly missed.
- Settings-change comparisons use 7 nights before and after each change. A
  change is only called improved or worsened when the Holm-corrected
  p-value is below 0.05, so with this little data many real changes will
  show as not significant.
- Glasgow Index thresholds come from [DaveSkvn/GlasgowIndex](https://github.com/DaveSkvn/GlasgowIndex) (GPL-3.0); see `NOTICE.md`.
- ResMed is a trademark of ResMed. This project is unofficial.

## Related projects

Other open-source tools worth knowing about. I have no affiliation with any
of them.

- **OSCAR**: the standard open-source viewer for CPAP data. Start here.
- **AirwayLab**: a comprehensive PAP data analysis tool.
- **Glasgow Index Multi-Night Analyzer**: Glasgow Index across all of your
  nights, with trends.
- **EDFbrowser**: a general EDF viewer that can export selected signals to CSV.

## License

[GPL-3.0-or-later](LICENSE). You may use and fork it; keep the license and
Glasgow Index attribution.

## Forum / sharing

If you post this on ApneaBoard (or similar):

1. Link the GitHub repo, not a zip of your SD card.
2. Say it is an extra research tool, not an OSCAR replacement.
3. Remind people that **their** data stays on **their** machine.
