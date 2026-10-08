"""Bar chart of mean episode reward per agent and split, from results/*.json.

    python scripts/plot_results.py   (writes docs/figures/results.png)

Agents found in results/ are plotted in a fixed order; an LLM run saved with
`--tag <name>` appears as its own row.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "oracle": "oracle (reads the truth)",
    "heuristic": "rule-based heuristic",
    "noop": "no-op",
    "flag_all": "flag everything",
    "kill_all": "kill everything",
}
ORDER = ["oracle", "heuristic", "llm", "noop", "flag_all", "kill_all"]
SPLITS = {"val": ("validation (regime A)", "#2a78d6"), "test": ("held-out test (regime B)", "#eb6834")}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load() -> dict[str, dict[str, float]]:
    rewards: dict[str, dict[str, float]] = {}
    for f in sorted((ROOT / "results").glob("*.json")):
        agent, split = f.stem.rsplit("_", 1)
        if split not in SPLITS:
            continue
        rewards.setdefault(agent, {})[split] = json.loads(f.read_text())["summary"]["reward"]
    return rewards


def sort_key(agent: str) -> tuple[int, str]:
    base = agent.split("_")[0] if agent.split("_")[0] in ORDER else agent
    return (ORDER.index(base) if base in ORDER else len(ORDER), agent)


def main() -> None:
    rewards = load()
    agents = sorted(rewards, key=sort_key)
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(agents) + 1.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    height = 0.36
    gap = 0.02
    for k, (split, (label, color)) in enumerate(SPLITS.items()):
        ys = [i + (k - 0.5) * (height + gap) for i in range(len(agents))]
        vals = [rewards[a].get(split) for a in agents]
        ax.barh([y for y, v in zip(ys, vals) if v is not None], [v for v in vals if v is not None],
                height=height, color=color, label=label, zorder=3)
        for y, v in zip(ys, vals):
            if v is None:
                continue
            ax.text(v + (0.02 if v >= 0 else -0.02), y, f"{v:+.3f}".replace("-", "−"), va="center",
                    ha="left" if v >= 0 else "right", fontsize=8, color=INK_2, zorder=4)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([LABELS.get(a, a.replace("_", " ")) for a in agents], color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color=INK_2, linewidth=1, zorder=2)
    ax.set_xlim(-1.0, 1.15)
    ax.set_xlabel("mean episode reward  (0.5 × detection F1 + 0.5 × repair − damage)", color=INK_2, fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    ax.set_title("Reward by agent, 60 episodes per split", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    out = ROOT / "docs" / "figures" / "results.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
