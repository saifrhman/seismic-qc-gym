"""Synthetic marine pre-stack shot gathers.

Each gather is a 2D streamer record: a direct wave travelling at water velocity
(linear moveout through the origin), a water-bottom reflection and its first
multiple, and a set of primary reflections with hyperbolic moveout, convolved
with a zero-phase Ricker wavelet, with mild trace-to-trace coupling variation
and band-limited random noise.

The model is deliberately simple. It is not meant to reproduce real field data,
only to give every planted defect an exact ground truth and a clean reference
for scoring repairs. Real gathers can be loaded instead (see ``segy.py``) and
the same defects planted into them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class AcquisitionRegime:
    """Acquisition and earth-model parameters that define a family of surveys."""

    name: str
    n_traces: int = 96
    dt: float = 0.004  # seconds
    n_samples: int = 750  # 3.0 s at 4 ms
    near_offset: float = 150.0  # metres
    group_interval: float = 25.0  # metres
    shot_interval: float = 50.0  # metres
    peak_freq: tuple[float, float] = (20.0, 30.0)  # Ricker peak frequency, Hz
    snr: tuple[float, float] = (8.0, 16.0)  # reflection RMS / noise RMS
    water_velocity: tuple[float, float] = (1480.0, 1520.0)  # m/s

    @property
    def record_length(self) -> float:
        return self.n_samples * self.dt


# Regime A is used for training and validation; regime B is held out for testing
# (different wavelet band, sampling, group interval and noise level).
REGIMES: dict[str, AcquisitionRegime] = {
    "A": AcquisitionRegime(name="A"),
    "B": AcquisitionRegime(
        name="B",
        n_traces=120,
        dt=0.002,
        n_samples=1250,  # 2.5 s at 2 ms
        near_offset=100.0,
        group_interval=12.5,
        shot_interval=25.0,
        peak_freq=(32.0, 45.0),
        snr=(5.0, 10.0),
    ),
}


@dataclass
class Gather:
    """One shot gather: trace data plus the trace headers the QC tools use."""

    gather_id: int
    data: np.ndarray  # (n_traces, n_samples), float32
    offsets: np.ndarray  # header offsets in metres, float64
    channels: np.ndarray  # 1-based channel numbers
    source_x: float
    dt: float
    group_interval: float

    @property
    def n_traces(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    def times_ms(self) -> np.ndarray:
        return np.arange(self.n_samples) * self.dt * 1000.0

    def copy(self) -> "Gather":
        return Gather(
            gather_id=self.gather_id,
            data=self.data.copy(),
            offsets=self.offsets.copy(),
            channels=self.channels.copy(),
            source_x=self.source_x,
            dt=self.dt,
            group_interval=self.group_interval,
        )


@dataclass
class Survey:
    """A line of shot gathers sharing one acquisition regime."""

    gathers: list[Gather]
    dt: float
    regime: str
    meta: dict = field(default_factory=dict)

    def gather(self, gather_id: int) -> Gather:
        for g in self.gathers:
            if g.gather_id == gather_id:
                return g
        raise KeyError(f"no gather with id {gather_id}")

    @property
    def gather_ids(self) -> list[int]:
        return [g.gather_id for g in self.gathers]

    def copy(self) -> "Survey":
        return Survey(
            gathers=[g.copy() for g in self.gathers],
            dt=self.dt,
            regime=self.regime,
            meta=dict(self.meta),
        )


def ricker(peak_freq: float, dt: float, length: float = 0.2) -> np.ndarray:
    """Zero-phase Ricker wavelet sampled at ``dt``, centred on the middle sample."""
    half = int(round(length / 2 / dt))
    t = np.arange(-half, half + 1) * dt
    a = (np.pi * peak_freq * t) ** 2
    return ((1.0 - 2.0 * a) * np.exp(-a)).astype(np.float64)


def _add_event(spikes: np.ndarray, times: np.ndarray, amps: np.ndarray, dt: float) -> None:
    """Place one event per trace at fractional sample times (linear interpolation)."""
    n_traces, n_samples = spikes.shape
    s = times / dt
    k = np.floor(s).astype(int)
    w = s - k
    for i in range(n_traces):
        if 0 <= k[i] < n_samples - 1:
            spikes[i, k[i]] += amps[i] * (1.0 - w[i])
            spikes[i, k[i] + 1] += amps[i] * w[i]


def make_survey(regime: AcquisitionRegime, n_gathers: int, rng: np.random.Generator) -> Survey:
    """Generate a clean survey of ``n_gathers`` shot gathers along one line."""
    dt = regime.dt
    n, ns = regime.n_traces, regime.n_samples
    t_max = regime.record_length

    f_peak = rng.uniform(*regime.peak_freq)
    v_water = rng.uniform(*regime.water_velocity)
    snr = rng.uniform(*regime.snr)
    wavelet = ricker(f_peak, dt)

    # Layered earth model shared by the line, with gentle dips along it.
    wb_t0 = rng.uniform(0.12, 0.30)
    n_ref = int(rng.integers(6, 11))
    t0s = np.sort(rng.uniform(wb_t0 + 0.15, t_max - 0.15, n_ref))
    v_grad = rng.uniform(500.0, 900.0)  # RMS velocity increase per second of two-way time
    vrms = np.maximum(v_water + 50.0, v_water + v_grad * (t0s - wb_t0) + rng.normal(0.0, 30.0, n_ref))
    refl = rng.uniform(0.05, 0.2, n_ref) * rng.choice([-1.0, 1.0], n_ref)
    dips = rng.normal(0.0, 4e-5, n_ref)  # seconds of t0 change per metre of shot position
    wb_dip = rng.normal(0.0, 2e-5)

    offsets = regime.near_offset + regime.group_interval * np.arange(n)
    x_max = offsets[-1]
    gathers: list[Gather] = []
    for gid in range(n_gathers):
        source_x = gid * regime.shot_interval
        direct = np.zeros((n, ns))
        refl_part = np.zeros((n, ns))

        # Direct wave: linear moveout through the origin, amplitude decaying with offset.
        _add_event(direct, offsets / v_water, 0.8 * np.sqrt(regime.near_offset / offsets), dt)

        # Water-bottom reflection and its first (polarity-reversed) multiple.
        t0_wb = wb_t0 + wb_dip * source_x
        _add_event(refl_part, np.sqrt(t0_wb**2 + (offsets / v_water) ** 2), np.full(n, 0.35), dt)
        _add_event(refl_part, np.sqrt((2 * t0_wb) ** 2 + (offsets / v_water) ** 2), np.full(n, -0.12), dt)

        # Primary reflections: hyperbolic moveout, mild AVO and attenuation.
        for j in range(n_ref):
            t0 = t0s[j] + dips[j] * source_x
            t = np.sqrt(t0**2 + (offsets / vrms[j]) ** 2)
            avo = 1.0 - 0.25 * (offsets / x_max) ** 2
            amp = refl[j] * avo * np.exp(-0.35 * t0)
            _add_event(refl_part, t, amp, dt)

        direct = fftconvolve(direct, wavelet[None, :], mode="same", axes=1)
        refl_part = fftconvolve(refl_part, wavelet[None, :], mode="same", axes=1)

        # Band-limited noise scaled against the reflection energy.
        noise = fftconvolve(rng.standard_normal((n, ns)), wavelet[None, :], mode="same", axes=1)
        noise *= (np.sqrt(np.mean(refl_part**2)) / snr) / (np.sqrt(np.mean(noise**2)) + 1e-12)

        coupling = 1.0 + rng.normal(0.0, 0.06, n)
        data = (direct + refl_part + noise) * coupling[:, None]
        gathers.append(
            Gather(
                gather_id=gid,
                data=data.astype(np.float32),
                offsets=offsets.astype(np.float64).copy(),
                channels=np.arange(1, n + 1),
                source_x=float(source_x),
                dt=dt,
                group_interval=regime.group_interval,
            )
        )

    meta = {
        "peak_freq_hz": float(f_peak),
        "water_velocity": float(v_water),
        "snr": float(snr),
        "n_reflectors": n_ref,
    }
    return Survey(gathers=gathers, dt=dt, regime=regime.name, meta=meta)
