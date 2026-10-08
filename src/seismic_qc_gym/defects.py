"""Defect catalogue and seeded injection with exact ground truth.

The catalogue follows the trace-editing and preprocessing problems described in
Yilmaz, *Seismic Data Analysis* (SEG, 2001), chapter 1: noisy traces and transient
glitches are deleted, polarity reversals are corrected, swell noise is removed by
low-cut filtering, monofrequency (power-line) noise is notched, and field-geometry
errors must be fixed before stacking. Dead traces, timing errors and bad gain are
common additions to that list in field QC.

Every injected defect is recorded as a ``DefectInstance`` with the parameters the
grader needs to score a repair, and the clean survey is kept as the reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey

from .synth import Survey

DEFECT_TYPES: tuple[str, ...] = (
    "dead_trace",
    "noisy_trace",
    "spike",
    "reversed_polarity",
    "time_shift",
    "swell_noise",
    "powerline_noise",
    "gain_error",
    "geometry_error",
)
TRACE_DEFECTS: tuple[str, ...] = DEFECT_TYPES[:-1]
GATHER_DEFECTS: tuple[str, ...] = ("geometry_error",)
RUN_DEFECTS: tuple[str, ...] = ("swell_noise", "powerline_noise")

RECOMMENDED_FIX: dict[str, str] = {
    "dead_trace": "kill",
    "noisy_trace": "kill",
    "spike": "despike",
    "reversed_polarity": "flip_polarity",
    "time_shift": "shift",
    "swell_noise": "lowcut",
    "powerline_noise": "notch",
    "gain_error": "rescale",
    "geometry_error": "offset_shift",
}

DEFECT_DESCRIPTIONS: dict[str, str] = {
    "dead_trace": "No usable signal (zero or near-zero amplitude). Cannot be recovered: kill it.",
    "noisy_trace": "Strong broadband noise over the whole trace; signal no longer coherent with neighbours. Kill it.",
    "spike": "A transient glitch: one or a few samples with extreme amplitude. Remove with despike.",
    "reversed_polarity": "Trace recorded with inverted polarity; anti-correlates with its neighbours. Flip it.",
    "time_shift": "Timing error: the whole trace is delayed or advanced relative to its neighbours. Shift it back.",
    "swell_noise": "Very low-frequency, high-amplitude noise on a run of adjacent traces over part of the record. Low-cut filter.",
    "powerline_noise": "Monofrequency 50 Hz contamination on a run of adjacent traces. Notch filter at 50 Hz.",
    "gain_error": "Trace amplitude scaled up or down relative to its neighbours, waveform unchanged. Rescale it.",
    "geometry_error": (
        "Gather-level header error: every offset in the gather is shifted by the same amount, so the direct-wave "
        "first breaks no longer line up with the header offsets. Correct the offsets with offset_shift."
    ),
}

DIFFICULTY: dict[str, dict] = {
    "easy": dict(
        n_per_gather=(1, 3), gap=2, dead_level=0.0, noisy=(6.0, 10.0), spike=(30.0, 60.0),
        shift_ms=(40.0, 80.0), gain_hi=(5.0, 10.0), gain_lo=(0.08, 0.15), swell=(6.0, 10.0),
        powerline=(1.5, 3.0), geom_k=(6, 10), compound=0.0,
    ),
    "medium": dict(
        n_per_gather=(2, 5), gap=2, dead_level=0.0, noisy=(3.5, 6.0), spike=(12.0, 30.0),
        shift_ms=(20.0, 40.0), gain_hi=(2.8, 5.0), gain_lo=(0.18, 0.33), swell=(3.0, 6.0),
        powerline=(0.7, 1.5), geom_k=(3, 6), compound=0.0,
    ),
    "hard": dict(
        n_per_gather=(3, 7), gap=1, dead_level=0.01, noisy=(2.0, 3.5), spike=(6.0, 12.0),
        shift_ms=(12.0, 20.0), gain_hi=(1.9, 2.8), gain_lo=(0.33, 0.5), swell=(1.5, 3.0),
        powerline=(0.3, 0.7), geom_k=(2, 3), compound=0.3,
    ),
}

POWERLINE_HZ = 50.0

# Single-trace defects that can co-occur on one trace in "hard" surveys (compound defects).
COMPOUNDABLE: tuple[str, ...] = ("reversed_polarity", "time_shift", "gain_error", "spike")


@dataclass
class DefectInstance:
    """One planted defect. Gather-level defects list every trace of the gather."""

    defect_type: str
    gather_id: int
    traces: tuple[int, ...]
    params: dict = field(default_factory=dict)

    @property
    def gather_level(self) -> bool:
        return self.defect_type in GATHER_DEFECTS

    def to_dict(self) -> dict:
        return {
            "defect_type": self.defect_type,
            "gather_id": self.gather_id,
            "traces": list(self.traces),
            "params": dict(self.params),
        }


def shift_samples(x: np.ndarray, k: int) -> np.ndarray:
    """Shift a trace by ``k`` samples (positive = later), filling with zeros."""
    out = np.zeros_like(x)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        out[:k] = x[-k:]
    else:
        out[:] = x
    return out


def _lowfreq_noise(rng: np.random.Generator, n: int, dt: float, cutoff_hz: float = 5.0) -> np.ndarray:
    sos = butter(4, cutoff_hz, btype="low", fs=1.0 / dt, output="sos")
    x = sosfiltfilt(sos, rng.standard_normal(n))
    return x / (np.sqrt(np.mean(x**2)) + 1e-12)


def _apply_single(g, i: int, dtype: str, cfg: dict, rng: np.random.Generator, ref_rms: float) -> dict:
    """Apply one single-trace defect to trace ``i`` in place and return its parameters."""
    ns, dt = g.n_samples, g.dt
    params: dict = {}
    if dtype == "dead_trace":
        g.data[i] = (cfg["dead_level"] * ref_rms * rng.standard_normal(ns)).astype(np.float32)
        params["level"] = cfg["dead_level"]
    elif dtype == "noisy_trace":
        mult = rng.uniform(*cfg["noisy"])
        g.data[i] += (mult * ref_rms * rng.standard_normal(ns)).astype(np.float32)
        params["noise_mult"] = float(mult)
    elif dtype == "spike":
        n_spk = int(rng.integers(1, 4))
        pos = rng.choice(np.arange(int(0.05 / dt), ns - 5), size=n_spk, replace=False)
        amps = rng.uniform(*cfg["spike"], size=n_spk) * ref_rms * rng.choice([-1.0, 1.0], n_spk)
        g.data[i, pos] += amps.astype(np.float32)
        params["samples"] = sorted(int(p) for p in pos)
        params["amp_mult"] = [float(a / ref_rms) for a in amps]
    elif dtype == "reversed_polarity":
        g.data[i] *= -1.0
    elif dtype == "time_shift":
        ms = rng.uniform(*cfg["shift_ms"]) * (1.0 if rng.random() < 0.5 else -1.0)
        k = int(round(ms / 1000.0 / dt))
        k = k if k != 0 else 1
        g.data[i] = shift_samples(g.data[i], k)
        params["shift_samples"] = k
        params["shift_ms"] = k * dt * 1000.0
    elif dtype == "gain_error":
        scale = rng.uniform(*cfg["gain_hi"]) if rng.random() < 0.5 else rng.uniform(*cfg["gain_lo"])
        g.data[i] *= np.float32(scale)
        params["scale"] = float(scale)
    else:
        raise ValueError(dtype)
    return params


def _place(used: np.ndarray, rng: np.random.Generator, length: int, gap: int, max_tries: int = 200) -> int | None:
    """Find a start index for a run of ``length`` traces not within ``gap`` of used traces."""
    n = used.size
    lo, hi = 1, n - 1 - length  # keep defects off the first and last trace
    if hi < lo:
        return None
    for _ in range(max_tries):
        s = int(rng.integers(lo, hi + 1))
        a, b = max(0, s - gap), min(n, s + length + gap)
        if not used[a:b].any():
            return s
    return None


def inject_defects(
    clean: Survey,
    rng: np.random.Generator,
    difficulty: str = "medium",
    defect_types: Sequence[str] = DEFECT_TYPES,
    geometry_prob: float = 0.25,
) -> tuple[Survey, list[DefectInstance]]:
    """Return a defective copy of ``clean`` and the list of planted defects."""
    if difficulty not in DIFFICULTY:
        raise ValueError(f"unknown difficulty {difficulty!r}")
    unknown = set(defect_types) - set(DEFECT_TYPES)
    if unknown:
        raise ValueError(f"unknown defect types {sorted(unknown)}")
    cfg = DIFFICULTY[difficulty]
    survey = clean.copy()
    truth: list[DefectInstance] = []
    trace_types = [t for t in defect_types if t in TRACE_DEFECTS]
    allow_geometry = "geometry_error" in defect_types
    geometry_forced = allow_geometry and not trace_types

    for g in survey.gathers:
        n, ns, dt = g.n_traces, g.n_samples, g.dt
        t = np.arange(ns) * dt
        clean_rms = np.sqrt(np.mean(g.data.astype(np.float64) ** 2, axis=1))

        if allow_geometry and (geometry_forced or rng.random() < geometry_prob):
            k = int(rng.integers(cfg["geom_k"][0], cfg["geom_k"][1] + 1))
            sign = 1.0 if rng.random() < 0.5 else -1.0
            if g.offsets[0] - k * g.group_interval < 0.5 * g.group_interval:
                sign = 1.0
            delta = sign * k * g.group_interval
            g.offsets = g.offsets + delta
            truth.append(
                DefectInstance("geometry_error", g.gather_id, tuple(range(n)), {"delta_m": float(delta)})
            )

        if not trace_types:
            continue
        used = np.zeros(n, dtype=bool)
        n_def = int(rng.integers(cfg["n_per_gather"][0], cfg["n_per_gather"][1] + 1))
        for _ in range(n_def):
            dtype = str(rng.choice(trace_types))
            if dtype == "swell_noise":
                length = int(rng.integers(3, 9))
            elif dtype == "powerline_noise":
                length = int(rng.integers(4, 13))
            else:
                length = 1
            start = _place(used, rng, length, cfg["gap"])
            if start is None:
                continue
            idx = tuple(range(start, start + length))
            used[start : start + length] = True
            params: dict = {}

            if dtype not in RUN_DEFECTS:
                i = idx[0]
                params = _apply_single(g, i, dtype, cfg, rng, float(clean_rms[i]))
                truth.append(DefectInstance(dtype, g.gather_id, idx, params))
                partners = [d for d in COMPOUNDABLE if d != dtype and d in trace_types]
                if dtype in COMPOUNDABLE and partners and rng.random() < cfg.get("compound", 0.0):
                    second = str(rng.choice(partners))
                    params2 = _apply_single(g, i, second, cfg, rng, float(clean_rms[i]))
                    truth.append(DefectInstance(second, g.gather_id, idx, params2))
                continue
            if dtype == "swell_noise":
                mult = rng.uniform(*cfg["swell"])
                t1 = rng.uniform(0.15, 0.55) * t[-1]
                width = rng.uniform(0.3, 0.6) * t[-1]
                a, b = int(t1 / dt), min(ns, int((t1 + width) / dt))
                taper = np.zeros(ns)
                taper[a:b] = tukey(b - a, alpha=0.4)
                base = _lowfreq_noise(rng, ns, dt)
                level = mult * float(np.median(clean_rms[list(idx)]))
                for i in idx:
                    own = _lowfreq_noise(rng, ns, dt)
                    noise = 0.75 * base + 0.25 * own
                    noise /= np.sqrt(np.mean(noise[a:b] ** 2)) + 1e-12
                    g.data[i] += (level * noise * taper).astype(np.float32)
                params.update({"noise_mult": float(mult), "window_ms": [a * dt * 1000.0, b * dt * 1000.0]})
            elif dtype == "powerline_noise":
                mult = rng.uniform(*cfg["powerline"])
                phase0 = rng.uniform(0, 2 * np.pi)
                for j, i in enumerate(idx):
                    phase = phase0 + 0.15 * j + rng.normal(0.0, 0.05)
                    g.data[i] += (mult * clean_rms[i] * np.sqrt(2.0) * np.sin(2 * np.pi * POWERLINE_HZ * t + phase)).astype(
                        np.float32
                    )
                params.update({"amp_mult": float(mult), "freq_hz": POWERLINE_HZ})
            truth.append(DefectInstance(dtype, g.gather_id, idx, params))

    return survey, truth
