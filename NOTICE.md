# Third-party notices

## Glasgow Index

`event_detection.compute_glasgow_indices()` reimplements the nine-component
Glasgow Index using thresholds documented in
[DaveSkvn/GlasgowIndex](https://github.com/DaveSkvn/GlasgowIndex)
(`FlowLimits.js`, GPL-3.0).

That origin is why this project is licensed under GPL-3.0-or-later rather than
a permissive license. If you fork, keep this attribution.

## Sleep scoring references

Custom apnea/hypopnea and desaturation detectors follow published AASM-style
rules as implemented by the author. They are **not** an official AASM scorer
and will not match a sleep lab 1:1.

Pulse-surge detection follows the approach described by Pitson & Stradling
(1998).
