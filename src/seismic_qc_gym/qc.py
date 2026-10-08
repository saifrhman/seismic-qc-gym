"""QC measurements on a gather: the numbers the inspection tools return.

All functions are pure: they read a ``Gather`` and return plain Python data.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.signal import butter, correlate, filtfilt, hilbert, iirnotch, sosfiltfilt
from scipy.stats import theilslopes

from .synth import Gather

_EPS = 1e-12


def _rnd(v: float, sig: int = 3):
    """Round to ``sig`` significant figures for compact tool output."""
    if v is None or not np.isfinite(v):
        return None
    if v == 0:
        return 0.0
    return float(f"{v:.{sig}g}")


def _select(n: int, traces: Sequence[int] | None) -> list[int]:
    if traces is None:
        return list(range(n))
    out = sorted({int(t) for t in traces})
    bad = [t for t in out if t < 0 or t >= n]
    if bad:
        raise ValueError(f"trace indices out of range 0..{n - 1}: {bad[:5]}")
    return out


def neighbour_indices(i: int, n: int, k: int) -> list[int]:
    return [j for j in range(i - k, i + k + 1) if j != i and 0 <= j < n]


def trace_rms(data: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(data, dtype=np.float64) ** 2, axis=-1))


def neighbour_rms_ratio(data: np.ndarray, k: int = 6) -> np.ndarray:
    rms = trace_rms(data)
    n = rms.size
    ratio = np.empty(n)
    for i in range(n):
        med = np.median(rms[neighbour_indices(i, n, k)])
        ratio[i] = rms[i] / med if med > _EPS else np.inf
    return ratio


def spike_scores(x: np.ndarray, half_window: int = 6) -> np.ndarray:
    """Per-sample spike score: deviation from linear interpolation of the two adjacent
    samples, relative to the median of that deviation in the surrounding window.
    A wavelet peak raises the deviation over several samples; a glitch raises it at one."""
    x = np.asarray(x, dtype=np.float64)
    d = np.zeros_like(x)
    d[1:-1] = np.abs(x[1:-1] - 0.5 * (x[:-2] + x[2:]))
    n = x.size
    pad = np.pad(d, half_window, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(pad, 2 * half_window + 1)
    # median of the window excluding the centre sample
    left = windows[:, :half_window]
    right = windows[:, half_window + 1 :]
    local = np.median(np.concatenate([left, right], axis=1), axis=1)
    floor = 1.4826 * np.median(d) + _EPS
    return d / np.maximum(local, floor)[:n]


def trace_attributes(g: Gather, traces: Sequence[int] | None = None) -> dict:
    """Amplitude attributes per trace, with gather medians for comparison."""
    idx = _select(g.n_traces, traces)
    x = g.data.astype(np.float64)
    rms = trace_rms(x)
    adjacent = neighbour_rms_ratio(x, k=2)
    local = neighbour_rms_ratio(x, k=6)
    scale = float(np.median(rms)) + _EPS
    zero_frac = np.mean(np.abs(x) <= 1e-6 * scale, axis=1)
    # Ignore the first and last few samples, where filter edge transients live.
    spike = np.array([spike_scores(x[i])[3:-3].max() for i in range(g.n_traces)])
    rows = [
        [i, _rnd(rms[i] / scale), _rnd(adjacent[i]), _rnd(local[i]), _rnd(spike[i]), _rnd(zero_frac[i], 2)]
        for i in idx
    ]
    return {
        "gather_id": g.gather_id,
        "columns": ["trace", "rms_rel", "rms_ratio_adjacent", "rms_ratio_local", "spike_index", "zero_fraction"],
        "rows": rows,
        "gather_median": {
            "rms_ratio_local": _rnd(float(np.median(local))),
            "spike_index": _rnd(float(np.median(spike))),
        },
        "notes": (
            "rms_rel: trace RMS / gather median RMS. rms_ratio_adjacent: trace RMS / median RMS of traces i-2..i+2; "
            "rms_ratio_local: the same over i-6..i+6. A single mis-scaled trace deviates on both; a smooth "
            "amplitude bump over several traces deviates on local only; a loud run nearby pulls healthy traces' "
            "ratios down. spike_index: largest single-sample glitch score on the trace (glitches score far above "
            "the gather median). zero_fraction: share of samples that are exactly zero."
        ),
    }


def pair_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int | None, float | None]:
    """Lag in samples by which trace ``a`` is later than trace ``b`` (at the correlation peak of
    largest magnitude) and the signed normalised correlation there."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= _EPS:
        return None, None
    ns = a.size
    full = correlate(a, b, mode="full", method="fft")  # full[k + ns - 1] = sum_n a[n + k] b[n]
    mid = ns - 1
    window = full[mid - max_lag : mid + max_lag + 1]
    best = int(np.argmax(np.abs(window)))
    return best - max_lag, float(window[best] / denom)


def _consistency_one(x: np.ndarray, i: int, max_lag: int) -> tuple[float | None, float | None, float | None, int]:
    """Signed correlations with the left and right neighbours, and the trace's time shift
    relative to them (samples). Local moveout cancels by symmetry: a healthy trace is
    later than its left neighbour by about as much as it is earlier than its right one."""
    n = x.shape[0]
    cands = []
    for k in (1, 2):
        lft, rgt = i - k, i + k
        if lft < 0 or rgt >= n:
            continue
        lag_l, c_l = pair_lag(x[i], x[lft], max_lag)
        lag_r, c_r = pair_lag(x[i], x[rgt], max_lag)
        if lag_l is None or lag_r is None:
            continue
        quality = min(abs(c_l), abs(c_r))
        if quality < 0.2:
            continue  # one side is incoherent (dead or noisy neighbour, or this trace is noise)
        cands.append((c_l, c_r, 0.5 * (lag_l + lag_r), k, quality))
    if not cands:
        # Fall back to whatever pair exists, even if incoherent (e.g. the trace itself is noise).
        for k in (1, 2):
            lft, rgt = i - k, i + k
            if lft >= 0 and rgt < n:
                lag_l, c_l = pair_lag(x[i], x[lft], max_lag)
                lag_r, c_r = pair_lag(x[i], x[rgt], max_lag)
                if lag_l is not None and lag_r is not None:
                    cands.append((c_l, c_r, 0.5 * (lag_l + lag_r), k, min(abs(c_l), abs(c_r))))
                    break
    # A shifted neighbour biases the nearest pair by half its shift; the farther pair is
    # unaffected. A truly shifted trace shows the same shift on both pairs, so keep the
    # pair with the smaller shift (ties go to the more coherent pair).
    best = min(cands, key=lambda c: (abs(c[2]), -c[4])) if cands else None
    if best is None:
        # Edge trace: predict the moveout to the next trace from the following pair.
        step = 1 if i == 0 else -1
        j1, j2 = i + step, i + 2 * step
        if 0 <= j2 < n:
            lag_1, c_1 = pair_lag(x[i], x[j1], max_lag)
            _, c_2 = pair_lag(x[i], x[j2], max_lag)
            lag_m, c_m = pair_lag(x[j1], x[j2], max_lag)
            if lag_1 is not None and lag_m is not None and c_2 is not None:
                # corr with the nearest and second-nearest trace (both on the same side)
                if abs(c_m) < 0.2:  # the moveout reference pair is incoherent: no timing estimate
                    return (c_1, c_2, None, 1)
                return (c_1, c_2, float(lag_1 - lag_m), 1)
        return (None, None, None, 0)
    c_l, c_r, shift, k, _ = best
    return c_l, c_r, shift, k


def neighbour_consistency(g: Gather, traces: Sequence[int] | None = None, max_lag_ms: float = 120.0) -> dict:
    """Cross-correlate each trace with its neighbours on either side.

    corr_left / corr_right are the signed normalised correlations at the best lag (both
    clearly negative = the trace is inverted relative to its neighbours). shift_ms is how much
    later the trace is than its neighbours after local moveout cancels (positive = late)."""
    idx = _select(g.n_traces, traces)
    x = g.data.astype(np.float64)
    max_lag = max(1, int(round(max_lag_ms / 1000.0 / g.dt)))
    rows = []
    corrs = []
    for i in idx:
        c_l, c_r, shift, k = _consistency_one(x, i, max_lag)
        if c_l is None:
            rows.append([i, None, None, None])
            continue
        rows.append([i, _rnd(c_l), _rnd(c_r), _rnd(shift * g.dt * 1000.0) if shift is not None else None])
        corrs.append(0.5 * (c_l + c_r))
    return {
        "gather_id": g.gather_id,
        "columns": ["trace", "corr_left", "corr_right", "shift_ms"],
        "rows": rows,
        "gather_median": {"corr": _rnd(float(np.median(corrs))) if corrs else None},
        "notes": (
            "Each trace is cross-correlated with its nearest healthy-looking neighbours on both sides (lags up to "
            f"+/-{max_lag_ms:g} ms). corr_left/corr_right are signed: both strongly negative means the trace is "
            "inverted. shift_ms > 0 means the trace is late relative to its neighbours after local moveout "
            "cancels; to undo it, shift by -shift_ms. Low |corr| on both sides means the trace is not coherent "
            "with its neighbours (noise). Dead traces return null."
        ),
    }


def pick_first_breaks(g: Gather, sta_ms: float = 12.0, lta_ms: float = 200.0, ratio: float = 8.0) -> np.ndarray:
    """First-break times (ms): an energy-ratio trigger finds the first strong arrival, then the
    pick moves to the peak of its envelope, which for a zero-phase wavelet is the arrival time.
    NaN where no arrival is found (dead traces)."""
    dt = g.dt
    ns = g.n_samples
    # Pre-filter so swell and power-line noise cannot trigger or hide the arrival.
    x = sosfiltfilt(butter(4, 5.0, btype="high", fs=1.0 / dt, output="sos"), g.data.astype(np.float64), axis=1)
    if 0.5 / dt > 55.0:
        b, a = iirnotch(50.0, 30.0, fs=1.0 / dt)
        x = filtfilt(b, a, x, axis=1)
    sta = max(2, int(round(sta_ms / 1000.0 / dt)))
    lta = max(sta + 1, int(round(lta_ms / 1000.0 / dt)))
    look_back = max(2, int(round(0.03 / dt)))
    refine = max(3, int(round(0.06 / dt)))
    env = np.abs(hilbert(x, axis=1))
    picks = np.full(g.n_traces, np.nan)
    for i in range(g.n_traces):
        e = x[i] ** 2
        c = np.concatenate([[0.0], np.cumsum(e)])
        if c[-1] <= _EPS:
            continue
        t = np.arange(ns - sta)
        post = (c[t + sta] - c[t]) / sta
        lo = np.maximum(0, t - lta)
        pre = (c[t] - c[lo]) / np.maximum(1, t - lo)
        floor = 1e-3 * np.max(post)
        r = post / np.maximum(pre, floor)
        hits = np.nonzero((r > ratio) & (post > 0.02 * np.max(post)))[0]
        if hits.size == 0:
            continue
        onset = int(hits[0])
        a, b = max(0, onset - look_back), min(ns, onset + refine)
        seg = env[i, a:b]
        # First envelope peak reaching 60% of the window maximum: the first arrival itself,
        # not a stronger event arriving just after it.
        thresh = 0.6 * seg.max()
        k = int(np.argmax(seg))
        for j in range(seg.size):
            left = seg[j - 1] if j > 0 else -np.inf
            right = seg[j + 1] if j + 1 < seg.size else -np.inf
            if seg[j] >= thresh and seg[j] >= left and seg[j] >= right:
                k = j
                break
        frac = 0.0
        if 0 < k < seg.size - 1:  # parabolic interpolation of the envelope peak
            den = seg[k - 1] - 2.0 * seg[k] + seg[k + 1]
            if den < 0:
                frac = float(np.clip(0.5 * (seg[k - 1] - seg[k + 1]) / den, -0.5, 0.5))
        picks[i] = (a + k + frac) * dt * 1000.0
    return picks


def first_break_check(g: Gather, max_fit_offset_m: float = 900.0) -> dict:
    """Fit near-offset first-break times against header offsets with a robust (Theil-Sen) line.

    Near the source the first arrival is the direct wave, so picks should follow
    t = offset / v with an intercept near zero; at far offsets refracted and wide-angle
    energy can arrive first, so those traces are left out of the fit. A clear non-zero
    intercept means the header offsets of the whole gather are shifted; the offset
    correction that removes it is reported."""
    picks = pick_first_breaks(g)
    true_rank = np.arange(g.n_traces)
    # Use the channel order (not the possibly wrong header values) to choose near traces.
    n_near = int(np.searchsorted(np.sort(g.offsets - g.offsets.min()), max_fit_offset_m - g.offsets.min(), side="right"))
    n_near = max(8, min(g.n_traces, n_near))
    in_fit = (true_rank < n_near) & np.isfinite(picks)
    if in_fit.sum() < 5:
        return {"gather_id": g.gather_id, "error": "too few first-break picks"}
    slope, intercept, _, _ = theilslopes(picks[in_fit], g.offsets[in_fit], method="joint")
    velocity = 1000.0 / slope if slope > 0 else float("nan")
    resid = picks - (intercept + slope * g.offsets)
    rows = []
    for i in range(g.n_traces):
        r = _rnd(resid[i]) if (true_rank[i] < n_near) else None
        rows.append([i, _rnd(g.offsets[i], 5), _rnd(picks[i], 4), r])
    correction = intercept / 1000.0 * velocity
    return {
        "gather_id": g.gather_id,
        "fit": {
            "velocity_mps": _rnd(velocity, 4),
            "intercept_ms": _rnd(intercept),
            "suggested_offset_correction_m": _rnd(correction),
            "traces_in_fit": f"0-{n_near - 1}",
            "n_picks": int(in_fit.sum()),
            "residual_mad_ms": _rnd(float(np.median(np.abs(resid[in_fit])))),
        },
        "columns": ["trace", "header_offset_m", "first_break_ms", "residual_ms"],
        "rows": rows,
        "notes": (
            "Near-offset direct-wave picks should fit t = offset / v through the origin (intercept within a few "
            "ms); far traces are left out of the fit (residual null) because other arrivals can come first there. "
            "A large intercept means every header offset in the gather is off by the same amount; adding "
            "suggested_offset_correction_m to all offsets removes it. One trace with a large residual is a timing "
            "or picking problem on that trace, not a gather geometry error."
        ),
    }


def spectrum_check(g: Gather, traces: Sequence[int] | None = None, line_hz: float = 50.0) -> dict:
    """Spectral attributes per trace: dominant frequency, low-frequency energy ratio, and a
    narrow-band line ratio at ``line_hz`` measured on the last 40% of the record, where
    reflections are weak and a constant power-line tone stands out."""
    idx = _select(g.n_traces, traces)
    x = g.data.astype(np.float64)
    w = np.hanning(g.n_samples)
    p = np.abs(np.fft.rfft(x * w, axis=1)) ** 2
    f = np.fft.rfftfreq(g.n_samples, g.dt)
    nyq = 0.5 / g.dt
    sig_band = (f >= 6.0) & (f <= min(100.0, nyq))
    low_band = f < 6.0
    low = p[:, low_band].sum(axis=1) / (p[:, sig_band].sum(axis=1) + _EPS)
    above2 = f > 2.0
    dom = f[above2][np.argmax(p[:, above2], axis=1)]

    late = x[:, int(0.6 * g.n_samples) :]
    wl = np.hanning(late.shape[1])
    pl = np.abs(np.fft.rfft(late * wl, axis=1)) ** 2
    fl = np.fft.rfftfreq(late.shape[1], g.dt)
    line_in = np.abs(fl - line_hz) <= 1.0
    line_ref = (np.abs(fl - line_hz) <= 10.0) & (np.abs(fl - line_hz) > 2.0)
    line = pl[:, line_in].mean(axis=1) / (np.median(pl[:, line_ref], axis=1) + _EPS)

    dead = p.sum(axis=1) <= _EPS
    rows = []
    for i in idx:
        if dead[i]:
            rows.append([i, None, None, None])
        else:
            rows.append([i, _rnd(dom[i]), _rnd(low[i]), _rnd(line[i])])
    live = ~dead
    return {
        "gather_id": g.gather_id,
        "columns": ["trace", "dominant_hz", "low_freq_ratio", f"line_{line_hz:g}hz_ratio"],
        "rows": rows,
        "gather_median": {
            "low_freq_ratio": _rnd(float(np.median(low[live]))) if live.any() else None,
            f"line_{line_hz:g}hz_ratio": _rnd(float(np.median(line[live]))) if live.any() else None,
        },
        "notes": (
            "low_freq_ratio: energy below 6 Hz / energy in 6-100 Hz over the whole trace (swell noise raises it on "
            f"a run of adjacent traces). line_{line_hz:g}hz_ratio: power within 1 Hz of {line_hz:g} Hz / median power "
            "2-10 Hz either side, measured on the last 40% of the record (a power-line tone raises it far above "
            "the gather median)."
        ),
    }


def geometry_survey_check(gathers: Sequence[Gather]) -> dict:
    """Run the near-offset first-break fit on every gather and compare intercepts across the
    survey. A gather whose intercept stands out from the others has shifted header offsets."""
    rows = []
    intercepts = []
    for g in gathers:
        fb = first_break_check(g)
        if "error" in fb:
            rows.append([g.gather_id, None, None, None])
            continue
        fit = fb["fit"]
        rows.append([g.gather_id, fit["velocity_mps"], fit["intercept_ms"], fit["suggested_offset_correction_m"]])
        intercepts.append(fit["intercept_ms"])
    med = float(np.median(intercepts)) if intercepts else float("nan")
    for r in rows:
        r.append(_rnd(r[2] - med) if r[2] is not None else None)
    return {
        "columns": ["gather_id", "velocity_mps", "intercept_ms", "suggested_offset_correction_m", "intercept_minus_survey_median_ms"],
        "rows": rows,
        "survey_median_intercept_ms": _rnd(med),
        "notes": (
            "Near-offset direct-wave fit per gather. Correct geometry gives an intercept within a few ms of zero; "
            "an intercept of tens of ms means every header offset in that gather is shifted by the same amount, "
            "and suggested_offset_correction_m is the fix (offset errors are usually whole multiples of the group "
            "interval). The survey median is only a secondary check: if several gathers are shifted it moves too."
        ),
    }
