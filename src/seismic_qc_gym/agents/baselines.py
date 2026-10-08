"""Reference policies for validating the grader.

OracleAgent reads the ground truth and applies the recommended fixes, which gives the
ceiling the repair operators allow. NoOp, FlagAll and KillAll are degenerate policies
that a sound reward must not pay for.
"""

from __future__ import annotations

from ..defects import DEFECT_TYPES, GATHER_DEFECTS, RECOMMENDED_FIX
from ..env import QCEnv


class NoOpAgent:
    name = "noop"

    def run(self, env: QCEnv) -> dict:
        env.call("submit_report", {"summary": "no action"})
        return env.result


class FlagAllAgent:
    """Flags every trace with every defect type and fixes nothing."""

    name = "flag_all"

    def run(self, env: QCEnv) -> dict:
        for gid in env.ep.state.gather_ids:
            for dtype in DEFECT_TYPES:
                env.call("flag_defect", {"gather_id": gid, "traces": "all", "defect_type": dtype})
        env.call("submit_report", {"summary": "flagged everything"})
        return env.result


class KillAllAgent:
    """Kills every trace and flags each as noisy."""

    name = "kill_all"

    def run(self, env: QCEnv) -> dict:
        for gid in env.ep.state.gather_ids:
            env.call("flag_defect", {"gather_id": gid, "traces": "all", "defect_type": "noisy_trace"})
            env.call("apply_fix", {"gather_id": gid, "traces": "all", "fix": "kill"})
        env.call("submit_report", {"summary": "killed everything"})
        return env.result


class OracleAgent:
    """Uses the planted ground truth: flags each defect and applies the recommended fix
    with the true parameters. Its score is the ceiling of the repair operators."""

    name = "oracle"

    def run(self, env: QCEnv) -> dict:
        for d in env.ep.truth:
            if d.defect_type in GATHER_DEFECTS:
                env.call("flag_defect", {"gather_id": d.gather_id, "traces": "all", "defect_type": d.defect_type})
                env.call(
                    "apply_fix",
                    {"gather_id": d.gather_id, "traces": "all", "fix": "offset_shift", "value": -d.params["delta_m"]},
                )
                continue
            traces = list(d.traces)
            env.call("flag_defect", {"gather_id": d.gather_id, "traces": traces, "defect_type": d.defect_type})
            fix = RECOMMENDED_FIX[d.defect_type]
            args = {"gather_id": d.gather_id, "traces": traces, "fix": fix}
            if d.defect_type == "time_shift":
                args["value"] = -d.params["shift_ms"]
            elif d.defect_type == "gain_error":
                args["value"] = 1.0 / d.params["scale"]
            env.call("apply_fix", args)
        env.call("submit_report", {"summary": "oracle"})
        return env.result
