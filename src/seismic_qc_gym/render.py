"""Render gathers as PNG images (for vision-language models and for figures)."""

from __future__ import annotations

import io
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .synth import Gather  # noqa: E402


def gather_clip(data: np.ndarray, percentile: float = 98.0) -> float:
    """Display clip from the median trace's amplitude distribution, so one bad trace cannot set it."""
    per_trace = np.percentile(np.abs(data), percentile, axis=1)
    clip = float(np.median(per_trace))
    return clip if clip > 0 else 1.0


def draw_gather(
    ax,
    gather: Gather,
    traces: Sequence[int] | None = None,
    t_range_ms: tuple[float, float] | None = None,
    clip: float | None = None,
    title: str | None = None,
) -> None:
    data = gather.data
    idx = np.arange(gather.n_traces) if traces is None else np.asarray(sorted(traces))
    times = gather.times_ms()
    if t_range_ms is not None:
        mask = (times >= t_range_ms[0]) & (times <= t_range_ms[1])
    else:
        mask = np.ones_like(times, dtype=bool)
    img = data[idx][:, mask].T
    c = clip if clip is not None else gather_clip(data)
    t0, t1 = times[mask][0], times[mask][-1]
    ax.imshow(
        img,
        aspect="auto",
        cmap="gray_r",
        vmin=-c,
        vmax=c,
        extent=(idx[0] - 0.5, idx[-1] + 0.5, t1, t0),
        interpolation="nearest",
    )
    ax.set_xlabel("trace index")
    ax.set_ylabel("time (ms)")
    if title:
        ax.set_title(title, fontsize=10)


def render_gather_png(
    gather: Gather,
    traces: Sequence[int] | None = None,
    t_range_ms: tuple[float, float] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (8.0, 6.0),
    dpi: int = 100,
) -> bytes:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    draw_gather(ax, gather, traces, t_range_ms, title=title or f"gather {gather.gather_id}")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
