"""Repair operations the agent can apply. Each works on one trace (or on a gather's headers)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from .defects import shift_samples
from .qc import spike_scores

FIXES: tuple[str, ...] = (
    "kill",
    "flip_polarity",
    "shift",
    "despike",
    "lowcut",
    "notch",
    "rescale",
    "offset_shift",
)

FIX_DESCRIPTIONS: dict[str, str] = {
    "kill": "Zero the trace and mark it dead (for dead or noisy traces). No value.",
    "flip_polarity": "Multiply the trace by -1. No value.",
    "shift": "Move the trace in time by `value` ms (positive = later). To undo a trace that is late by L ms, use -L.",
    "despike": "Replace isolated single-sample glitches by interpolation. Optional `value`: score threshold (default 15).",
    "lowcut": "Zero-phase low-cut (high-pass) filter. Optional `value`: cutoff in Hz (default 8).",
    "notch": "Zero-phase notch filter. Optional `value`: frequency in Hz (default 50).",
    "rescale": "Multiply the trace by `value`; if omitted, scale it to the median RMS of its 8 nearest traces.",
    "offset_shift": "Add `value` metres to the header offset of every listed trace (use all traces for a gather geometry error).",
}


def despike(x: np.ndarray, threshold: float = 15.0) -> tuple[np.ndarray, int]:
    """Replace isolated glitches. A glitch also raises the score of the two samples beside
    it (by about half), so only local score maxima above ``threshold`` are replaced, each by
    the mean of its two original neighbours."""
    x = np.asarray(x, dtype=np.float64)
    score = spike_scores(x)
    centre = np.zeros_like(score, dtype=bool)
    centre[1:-1] = (score[1:-1] > threshold) & (score[1:-1] >= score[:-2]) & (score[1:-1] >= score[2:])
    centre[:3] = centre[-3:] = False  # filter edge transients, not glitches
    y = x.copy()
    idx = np.nonzero(centre)[0]
    y[idx] = 0.5 * (x[idx - 1] + x[idx + 1])
    return y, int(idx.size)


def lowcut(x: np.ndarray, dt: float, cutoff_hz: float = 8.0) -> np.ndarray:
    sos = butter(4, cutoff_hz, btype="high", fs=1.0 / dt, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def notch(x: np.ndarray, dt: float, freq_hz: float = 50.0, q: float = 30.0) -> np.ndarray:
    b, a = iirnotch(freq_hz, q, fs=1.0 / dt)
    return filtfilt(b, a, np.asarray(x, dtype=np.float64))


def shift_ms(x: np.ndarray, dt: float, ms: float) -> np.ndarray:
    return shift_samples(np.asarray(x, dtype=np.float64), int(round(ms / 1000.0 / dt)))
