"""Programmatic grader: detection, repair, damage and a scalar reward.

Detection is scored on items: one item per (gather, trace, defect type) for trace-level
defects and one per (gather, defect type) for gather-level defects, so a gather-wide
geometry error counts once rather than once per trace.

Repair measures how much of each planted error the agent removed, against the clean
survey. Damage measures harm done to traces and headers that had no defect, normalised
by the number of true defect items so that one wrong fix costs about as much as a right
fix earns. The reward combines the three; each component is reported separately.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .defects import DEFECT_TYPES, GATHER_DEFECTS, DefectInstance
from .synth import Survey

GATHER = -1  # trace index used for gather-level items

KILL_ONLY = ("dead_trace", "noisy_trace")


@dataclass(frozen=True)
class RewardWeights:
    detection: float = 0.5
    repair: float = 0.5
    damage: float = 1.0


def truth_items(truth: list[DefectInstance]) -> set[tuple[int, int, str]]:
    items: set[tuple[int, int, str]] = set()
    for d in truth:
        if d.defect_type in GATHER_DEFECTS:
            items.add((d.gather_id, GATHER, d.defect_type))
        else:
            for t in d.traces:
                items.add((d.gather_id, t, d.defect_type))
    return items


def _f1(tp: int, n_true: int, n_pred: int) -> tuple[float, float, float]:
    precision = tp / n_pred if n_pred else (1.0 if n_true == 0 else 0.0)
    recall = tp / n_true if n_true else 1.0
    f1 = 2 * tp / (n_true + n_pred) if (n_true + n_pred) else 1.0
    return precision, recall, f1


def grade(
    truth: list[DefectInstance],
    clean: Survey,
    initial: Survey,
    final: Survey,
    flags: set[tuple[int, int, str]],
    killed: set[tuple[int, int]],
    touched: set[tuple[int, int]],
    n_calls: int = 0,
    n_invalid: int = 0,
    weights: RewardWeights = RewardWeights(),
) -> dict:
    """Score one episode. ``flags`` holds (gather, trace or GATHER, type); ``killed`` and
    ``touched`` hold (gather, trace) pairs for killed traces and traces any data fix touched."""
    t_items = truth_items(truth)

    # --- detection (typed items) -------------------------------------------------------
    tp = len(t_items & flags)
    precision, recall, f1 = _f1(tp, len(t_items), len(flags))
    per_type = {}
    for dtype in DEFECT_TYPES:
        tt = {i for i in t_items if i[2] == dtype}
        ff = {i for i in flags if i[2] == dtype}
        if not tt and not ff:
            continue
        p, r, f = _f1(len(tt & ff), len(tt), len(ff))
        per_type[dtype] = {"n_true": len(tt), "n_flagged": len(ff), "precision": p, "recall": r, "f1": f}

    # --- trace-level localisation, ignoring type -------------------------------------
    true_traces = {(g, t) for (g, t, d) in t_items if t != GATHER}
    found_traces = {(g, t) for (g, t, d) in flags if t != GATHER} | set(touched)
    lp, lr, lf = _f1(len(true_traces & found_traces), len(true_traces), len(found_traces))

    # --- repair ------------------------------------------------------------------------
    repair_items: list[tuple[str, float]] = []
    for d in truth:
        if d.defect_type in GATHER_DEFECTS:
            cg, fg = clean.gather(d.gather_id), final.gather(d.gather_id)
            ok = np.max(np.abs(fg.offsets - cg.offsets)) <= 0.5 * cg.group_interval
            repair_items.append((d.defect_type, 1.0 if ok else 0.0))
            continue
        for t in d.traces:
            if d.defect_type in KILL_ONLY:
                repair_items.append((d.defect_type, 1.0 if (d.gather_id, t) in killed else 0.0))
                continue
            x0 = clean.gather(d.gather_id).data[t].astype(np.float64)
            xi = initial.gather(d.gather_id).data[t].astype(np.float64)
            xf = final.gather(d.gather_id).data[t].astype(np.float64)
            e0 = np.linalg.norm(xi - x0)
            e1 = np.linalg.norm(xf - x0)
            if not np.isfinite(e1):
                q = 0.0
            else:
                q = 1.0 if e0 <= 1e-12 else float(np.clip(1.0 - e1 / e0, 0.0, 1.0))
            repair_items.append((d.defect_type, q))
    repair = float(np.mean([q for _, q in repair_items])) if repair_items else 1.0
    repair_by_type: dict[str, list[float]] = defaultdict(list)
    for dtype, q in repair_items:
        repair_by_type[dtype].append(q)

    # --- damage to clean traces and headers -----------------------------------------------
    defective = {(g, t) for (g, t, d) in t_items if t != GATHER}
    geo_gathers = {g for (g, t, d) in t_items if t == GATHER}
    data_damage = 0.0
    n_clean_damaged = 0
    clean_killed = 0
    for cg in clean.gathers:
        fg = final.gather(cg.gather_id)
        x0 = cg.data.astype(np.float64)
        xf = fg.data.astype(np.float64)
        for t in range(cg.n_traces):
            if (cg.gather_id, t) in defective:
                continue
            if (cg.gather_id, t) in killed:
                dmg = 1.0
                clean_killed += 1
            elif not np.all(np.isfinite(xf[t])):
                dmg = 1.0
            else:
                denom = np.linalg.norm(x0[t]) + 1e-12
                dmg = float(min(1.0, np.linalg.norm(xf[t] - x0[t]) / denom))
            if dmg > 1e-6:
                n_clean_damaged += 1
            data_damage += dmg
    header_damage = 0
    for cg in clean.gathers:
        if cg.gather_id in geo_gathers:
            continue
        fg = final.gather(cg.gather_id)
        diff = np.abs(fg.offsets - cg.offsets)
        if not np.all(np.isfinite(diff)) or np.max(diff) > 0.5 * cg.group_interval:
            header_damage += 1
    n_items = max(1, len(t_items))
    damage = float(min(1.0, (data_damage + header_damage) / n_items))

    reward = weights.detection * f1 + weights.repair * repair - weights.damage * damage
    return {
        "reward": float(reward),
        "detection": {"precision": precision, "recall": recall, "f1": f1, "n_true": len(t_items), "n_flagged": len(flags)},
        "localisation": {"precision": lp, "recall": lr, "f1": lf},
        "repair": repair,
        "damage": damage,
        "damage_detail": {
            "data_damage_sum": float(data_damage),
            "clean_traces_damaged": n_clean_damaged,
            "clean_traces_killed": clean_killed,
            "clean_gathers_with_header_damage": header_damage,
        },
        "per_type": {
            k: {**v, "repair": float(np.mean(repair_by_type[k])) if repair_by_type.get(k) else None}
            for k, v in per_type.items()
        },
        "n_calls": n_calls,
        "n_invalid_calls": n_invalid,
        "weights": {"detection": weights.detection, "repair": weights.repair, "damage": weights.damage},
    }
