"""Seeded task specifications and splits.

train and val use acquisition regime A. test uses regime B, which is held out from all
development: different wavelet band, sample rate, group interval, spread length and noise
level. Thresholds and prompts should be tuned on train/val only; test is for reporting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .defects import DEFECT_TYPES

DIFFICULTIES = ("easy", "medium", "hard")

SPLITS: dict[str, dict] = {
    "train": {"seed_base": 0, "regime": "A"},
    "val": {"seed_base": 100_000, "regime": "A"},
    "test": {"seed_base": 200_000, "regime": "B"},
}


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    seed: int
    regime: str = "A"
    difficulty: str = "medium"
    n_gathers: int = 4
    defect_types: tuple[str, ...] = DEFECT_TYPES
    geometry_prob: float = 0.25
    budget: int = 120

    def to_dict(self) -> dict:
        d = asdict(self)
        d["defect_types"] = list(self.defect_types)
        return d


def make_tasks(
    split: str,
    n: int,
    difficulty: str | None = None,
    n_gathers: int = 4,
    budget: int | None = None,
    start: int = 0,
) -> list[TaskSpec]:
    """``n`` tasks from ``split``. With ``difficulty=None`` the tasks cycle easy/medium/hard."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {list(SPLITS)}")
    if difficulty is not None and difficulty not in DIFFICULTIES:
        raise ValueError(f"unknown difficulty {difficulty!r}")
    cfg = SPLITS[split]
    tasks = []
    for i in range(start, start + n):
        diff = difficulty or DIFFICULTIES[i % len(DIFFICULTIES)]
        tasks.append(
            TaskSpec(
                task_id=f"{split}-{i:05d}",
                seed=cfg["seed_base"] + i,
                regime=cfg["regime"],
                difficulty=diff,
                n_gathers=n_gathers,
                budget=budget if budget is not None else 30 * n_gathers,
            )
        )
    return tasks
