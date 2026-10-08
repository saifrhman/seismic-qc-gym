"""A classical rule-based QC baseline that uses the same tools as any other agent.

Per survey it runs the geometry check; per gather it runs three inspection tools,
classifies every trace with fixed thresholds in a priority order (dead, spike, power-line,
swell, noisy, gain, polarity), applies those fixes, and then resolves timing errors one at
a time, re-measuring after each fix, because a shifted trace biases its neighbours'
timing estimates. Thresholds were set from feature distributions on regime A surveys
only; regime B (the test split) was not used to set them.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..env import QCEnv

FIX_FOR = {
    "dead_trace": "kill",
    "noisy_trace": "kill",
    "spike": "despike",
    "reversed_polarity": "flip_polarity",
    "time_shift": "shift",
    "swell_noise": "lowcut",
    "powerline_noise": "notch",
    "gain_error": "rescale",
}
RUN_TYPES = ("swell_noise", "powerline_noise")


@dataclass(frozen=True)
class Thresholds:
    dead_rms_ratio: float = 0.05
    dead_zero_fraction: float = 0.5
    spike_index: float = 20.0
    line_ratio: float = 20.0
    low_freq_ratio: float = 0.2
    noisy_rms_ratio: float = 1.8
    noisy_max_corr: float = 0.48
    gain_hi: float = 1.6
    gain_lo: float = 0.62
    gain_min_corr: float = 0.5
    polarity_max_corr: float = -0.3
    shift_ms: float = 8.0
    shift_min_corr: float = 0.3
    geometry_intercept_ms: float = 12.0
    max_timing_rounds: int = 6


def classify(attr: list, cons: list, spec: list, th: Thresholds) -> str | None:
    """Defect type for one trace from its tool rows (timing errors are handled separately)."""
    _, _, adj, local, spike, zero = attr
    _, c_l, c_r, _shift = cons
    _, _, low, line = spec
    if (local is not None and local < th.dead_rms_ratio) or (zero is not None and zero > th.dead_zero_fraction):
        return "dead_trace"
    if spike is not None and spike > th.spike_index:
        return "spike"
    if line is not None and line > th.line_ratio:
        return "powerline_noise"
    if low is not None and low > th.low_freq_ratio:
        return "swell_noise"
    cmax = max(c_l, c_r) if c_l is not None and c_r is not None else None
    cabs = max(abs(c_l), abs(c_r)) if c_l is not None and c_r is not None else None
    # Noise is incoherent with both neighbours; an inverted trace is still strongly (negatively) coherent.
    if local is not None and local > th.noisy_rms_ratio and (cabs is None or cabs < th.noisy_max_corr):
        return "noisy_trace"
    if cmax is None:
        return None

    def off(r):
        return r is not None and (r > th.gain_hi or r < th.gain_lo)

    if off(adj) and off(local) and cmax > th.gain_min_corr:
        return "gain_error"
    if cmax < th.polarity_max_corr:
        return "reversed_polarity"
    return None


def _runs(indices: list[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for i in sorted(indices):
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


class HeuristicAgent:
    name = "heuristic"

    def __init__(self, thresholds: Thresholds = Thresholds()):
        self.th = thresholds

    def run(self, env: QCEnv) -> dict:
        th = self.th
        summary = env.call("survey_summary").content
        gi = float(summary["group_interval_m"])

        geo = env.call("geometry_survey_check").content
        for gid, _v, intercept, correction, _dev in geo.get("rows", []):
            if intercept is not None and abs(intercept) > th.geometry_intercept_ms:
                k = round(correction / gi)
                if k != 0:
                    env.call("flag_defect", {"gather_id": gid, "traces": "all", "defect_type": "geometry_error"})
                    env.call("apply_fix", {"gather_id": gid, "traces": "all", "fix": "offset_shift", "value": k * gi})

        for gid in summary["gather_ids"]:
            if env.done:
                break
            self._gather(env, gid)
        if not env.done:
            env.call("submit_report", {"summary": "heuristic QC pass"})
        return env.result

    def _gather(self, env: QCEnv, gid: int) -> None:
        """Two passes per gather: a second look after the first fixes catches a defect that
        was hidden behind another one on the same trace."""
        flagged: dict[int, set[str]] = {}
        killed: set[int] = set()
        for _ in range(2):
            found = self._pass(env, gid, flagged, killed)
            if env.done or not found:
                return

    def _pass(self, env: QCEnv, gid: int, flagged: dict[int, set[str]], killed: set[int]) -> int:
        th = self.th
        attr = env.call("trace_attributes", {"gather_id": gid}).content["rows"]
        cons = env.call("neighbour_consistency", {"gather_id": gid}).content["rows"]
        spec = env.call("spectrum_check", {"gather_id": gid}).content["rows"]
        n = len(attr)
        labels: dict[int, str] = {}
        for a, c, s in zip(attr, cons, spec):
            i = a[0]
            if i in killed:
                continue
            d = classify(a, c, s, th)
            if d and d not in flagged.get(i, set()):
                labels[i] = d

        # Re-check gain candidates against the nearest healthy traces only.
        rms_rel = {row[0]: row[1] for row in attr}
        busy = {i for i, d in labels.items() if d != "gain_error"} | killed
        busy |= {i for i, ds in flagged.items() if ds & set(RUN_TYPES)}
        gain_factor: dict[int, float] = {}
        for i in [i for i, d in labels.items() if d == "gain_error"]:
            healthy = [j for j in sorted(range(n), key=lambda j: abs(j - i)) if j != i and j not in busy and rms_rel.get(j)]
            nb = healthy[:4]
            if len(nb) < 2 or not rms_rel.get(i):
                labels.pop(i)
                continue
            ref = sorted(rms_rel[j] for j in nb)[len(nb) // 2]
            ratio = rms_rel[i] / ref
            if th.gain_lo <= ratio <= th.gain_hi:
                labels.pop(i)
            else:
                gain_factor[i] = 1.0 / ratio

        found = 0
        by_type: dict[str, list[int]] = {}
        for i, d in labels.items():
            by_type.setdefault(d, []).append(i)
        for dtype, idx in by_type.items():
            groups = _runs(idx) if dtype in RUN_TYPES else [[i] for i in sorted(idx)]
            for grp in groups:
                if env.done:
                    return found
                env.call("flag_defect", {"gather_id": gid, "traces": grp, "defect_type": dtype})
                args = {"gather_id": gid, "traces": grp, "fix": FIX_FOR[dtype]}
                if dtype == "gain_error":
                    args["value"] = gain_factor[grp[0]]
                env.call("apply_fix", args)
                for i in grp:
                    flagged.setdefault(i, set()).add(dtype)
                    if FIX_FOR[dtype] == "kill":
                        killed.add(i)
                found += 1

        # Timing errors: fix the largest remaining shift, then re-measure.
        skip = killed | {i for i, ds in flagged.items() if ds & (set(RUN_TYPES) | {"time_shift"})}
        for _ in range(th.max_timing_rounds):
            if env.done:
                return found
            rows = env.call("neighbour_consistency", {"gather_id": gid}).content.get("rows", [])
            best = None
            for i, c_l, c_r, shift in rows:
                if i in skip or shift is None or c_l is None:
                    continue
                need = 2 * th.shift_ms if i in (0, n - 1) else th.shift_ms
                if abs(shift) >= need and min(c_l, c_r) > th.shift_min_corr:
                    if best is None or abs(shift) > abs(best[1]):
                        best = (i, shift)
            if best is None:
                break
            i, shift = best
            skip.add(i)
            flagged.setdefault(i, set()).add("time_shift")
            env.call("flag_defect", {"gather_id": gid, "traces": [i], "defect_type": "time_shift"})
            env.call("apply_fix", {"gather_id": gid, "traces": [i], "fix": "shift", "value": -float(shift)})
            found += 1
        return found
