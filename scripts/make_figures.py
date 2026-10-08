"""Figures for the README: a gallery of the planted defects and one episode before/after QC.

    python scripts/make_figures.py   (writes docs/figures/*.png)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from seismic_qc_gym import qc  # noqa: E402
from seismic_qc_gym.agents import HeuristicAgent  # noqa: E402
from seismic_qc_gym.defects import inject_defects  # noqa: E402
from seismic_qc_gym.env import QCEnv  # noqa: E402
from seismic_qc_gym.render import draw_gather, gather_clip  # noqa: E402
from seismic_qc_gym.synth import REGIMES, make_survey  # noqa: E402
from seismic_qc_gym.tasks import make_tasks  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"


def gallery() -> None:
    clean = make_survey(REGIMES["A"], 1, np.random.default_rng(42))
    g0 = clean.gathers[0]
    clip = gather_clip(g0.data)
    panels = [
        ("dead_trace", 0), ("noisy_trace", 0), ("spike", 0),
        ("reversed_polarity", 0), ("time_shift", 0), ("swell_noise", 0),
        ("powerline_noise", 0), ("gain_error", 0), ("geometry_error", 0),
    ]
    fig, axs = plt.subplots(3, 3, figsize=(13, 11))
    for ax, (dtype, _) in zip(axs.ravel(), panels):
        bad, truth = inject_defects(clean, np.random.default_rng(7), "easy", defect_types=(dtype,), geometry_prob=1.0)
        d = truth[0]
        g = bad.gather(d.gather_id)
        if dtype == "geometry_error":
            picks = qc.pick_first_breaks(g)
            near = np.arange(g.n_traces) < 40
            fit = qc.first_break_check(g)["fit"]
            ax.plot(g.offsets[near], picks[near], "o", ms=3, color="black", label="picks vs header offset")
            xs = np.linspace(0, g.offsets[near].max(), 10)
            v = fit["velocity_mps"]
            ax.plot(xs, fit["intercept_ms"] + xs / v * 1000.0, "-", color="#c0392b", label=f"fit: intercept {fit['intercept_ms']:.0f} ms")
            ax.plot(xs, xs / v * 1000.0, "--", color="grey", label="through origin")
            ax.set_xlabel("header offset (m)")
            ax.set_ylabel("first break (ms)")
            ax.legend(fontsize=7, loc="upper left")
            ax.set_title(f"geometry_error (offsets shifted {d.params['delta_m']:+.0f} m)", fontsize=10)
            continue
        lo, hi = max(0, d.traces[0] - 12), min(g.n_traces - 1, d.traces[-1] + 12)
        draw_gather(ax, g, range(lo, hi + 1), (0, 2000), clip=clip, title=f"{dtype}  (traces {d.traces[0]}-{d.traces[-1]})")
        for other in truth:
            if other.gather_id == d.gather_id and other.traces[-1] >= lo and other.traces[0] <= hi:
                ax.axvspan(other.traces[0] - 0.5, other.traces[-1] + 0.5, color="#c0392b", alpha=0.12)
    fig.suptitle("Planted defects (easy difficulty, regime A); the shaded traces are defective", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "defect_gallery.png", dpi=110)
    plt.close(fig)


def episode() -> None:
    task = make_tasks("val", 2, difficulty="medium", n_gathers=2)[1]
    env = QCEnv(task)
    gid = max(env.ep.state.gather_ids, key=lambda g: sum(len(d.traces) for d in env.ep.truth if d.gather_id == g and d.defect_type != "geometry_error"))
    before = env.ep.initial.gather(gid)
    result = HeuristicAgent().run(env)
    after = env.ep.state.gather(gid)
    clean = env.ep.clean.gather(gid)
    clip = gather_clip(clean.data)
    fig, axs = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    draw_gather(axs[0], before, clip=clip, title="as delivered (planted defects)")
    draw_gather(axs[1], after, clip=clip, title=f"after heuristic QC (episode reward {result['reward']:.2f})")
    draw_gather(axs[2], clean, clip=clip, title="clean reference (hidden from the agent)")
    for d in env.ep.truth:
        if d.gather_id == gid and d.defect_type != "geometry_error":
            axs[0].axvspan(d.traces[0] - 0.5, d.traces[-1] + 0.5, color="#c0392b", alpha=0.12)
    fig.suptitle(f"{task.task_id}, gather {gid}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "episode_before_after.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    gallery()
    episode()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
