"""The QC environment: one episode = one seeded survey with planted defects.

Agents interact only through tools (``QCEnv.call``): inspection tools return measurements
of the current data, action tools flag defects and apply fixes, and ``submit_report`` ends
the episode and grades it. The same tool definitions back the LLM agent and the MCP server.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import qc
from .defects import DEFECT_DESCRIPTIONS, DEFECT_TYPES, GATHER_DEFECTS, RECOMMENDED_FIX, inject_defects
from .fixes import FIX_DESCRIPTIONS, FIXES, despike, lowcut, notch, shift_ms
from .grading import GATHER, RewardWeights, grade
from .render import render_gather_png
from .synth import REGIMES, Survey, make_survey
from .tasks import TaskSpec

_TRACES_SCHEMA = {
    "description": 'Trace indices: a list of ints, a single int, a range string like "10-14", or "all".',
    "anyOf": [
        {"type": "array", "items": {"type": "integer"}},
        {"type": "integer"},
        {"type": "string"},
    ],
}


def _spec(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties or {},
            "required": required or [],
            "additionalProperties": False,
        },
    }


TOOL_SPECS: list[dict] = [
    _spec(
        "survey_summary",
        "Survey layout (gathers, traces, sampling, offsets), the defect catalogue with recommended fixes, and the call budget.",
    ),
    _spec(
        "trace_headers",
        "Trace headers for one gather: trace index, channel number and header offset in metres.",
        {"gather_id": {"type": "integer"}, "traces": _TRACES_SCHEMA},
        ["gather_id"],
    ),
    _spec(
        "trace_attributes",
        "Per-trace amplitude attributes for one gather: RMS relative to the gather and to neighbouring traces, "
        "a spike index for single-sample glitches, and the fraction of zero samples. Includes gather medians.",
        {"gather_id": {"type": "integer"}, "traces": _TRACES_SCHEMA},
        ["gather_id"],
    ),
    _spec(
        "neighbour_consistency",
        "Cross-correlates each trace with its neighbours on both sides: signed correlations (negative on both "
        "sides = inverted polarity) and the trace's time shift relative to its neighbours in ms (positive = late).",
        {"gather_id": {"type": "integer"}, "traces": _TRACES_SCHEMA},
        ["gather_id"],
    ),
    _spec(
        "spectrum_check",
        "Per-trace spectral attributes: dominant frequency, low-frequency (<6 Hz) energy ratio for swell noise, and "
        "a narrow-band 50 Hz line ratio for power-line noise. Includes gather medians.",
        {"gather_id": {"type": "integer"}, "traces": _TRACES_SCHEMA},
        ["gather_id"],
    ),
    _spec(
        "first_break_check",
        "Picks direct-wave first breaks in one gather and fits them against header offsets (near traces only): "
        "velocity, intercept, a suggested header offset correction, and per-trace residuals.",
        {"gather_id": {"type": "integer"}},
        ["gather_id"],
    ),
    _spec(
        "geometry_survey_check",
        "Runs the first-break fit on every gather and compares intercepts across the survey to find gathers whose "
        "header offsets are shifted.",
    ),
    _spec(
        "render_gather",
        "Renders a gather (or a subset of traces and a time window) as a PNG image, for visual inspection.",
        {
            "gather_id": {"type": "integer"},
            "traces": _TRACES_SCHEMA,
            "t_start_ms": {"type": "number"},
            "t_end_ms": {"type": "number"},
        },
        ["gather_id"],
    ),
    _spec(
        "flag_defect",
        "Record a defect. Use the exact defect type. For geometry_error (a whole-gather header problem) pass "
        'traces="all"; it counts once per gather.',
        {
            "gather_id": {"type": "integer"},
            "traces": _TRACES_SCHEMA,
            "defect_type": {"type": "string", "enum": list(DEFECT_TYPES)},
            "note": {"type": "string"},
        },
        ["gather_id", "traces", "defect_type"],
    ),
    _spec(
        "remove_flags",
        "Remove flags you recorded earlier on these traces (all types, or one type).",
        {
            "gather_id": {"type": "integer"},
            "traces": _TRACES_SCHEMA,
            "defect_type": {"type": "string", "enum": list(DEFECT_TYPES)},
        },
        ["gather_id", "traces"],
    ),
    _spec(
        "apply_fix",
        "Apply a repair to traces in one gather. Fixes: "
        + "; ".join(f"{k}: {v}" for k, v in FIX_DESCRIPTIONS.items()),
        {
            "gather_id": {"type": "integer"},
            "traces": _TRACES_SCHEMA,
            "fix": {"type": "string", "enum": list(FIXES)},
            "value": {"type": "number", "description": "Fix parameter where needed (see fix descriptions)."},
        },
        ["gather_id", "traces", "fix"],
    ),
    _spec(
        "submit_report",
        "End the episode. Call once, after flagging and fixing everything you found.",
        {"summary": {"type": "string", "description": "Short QC summary of what you found and fixed."}},
    ),
]
TOOL_NAMES = [t["name"] for t in TOOL_SPECS]


def instructions_text(n_gathers: int, n_traces: int, dt_ms: float, record_ms: float, budget: int) -> str:
    lines = [
        f"You are quality-controlling a marine 2D seismic line: {n_gathers} shot gathers of {n_traces} traces, "
        f"{dt_ms:g} ms sampling, {record_ms:g} ms records.",
        "Some traces and gathers contain planted defects. A defect can affect a single trace, a run of adjacent "
        "traces, or the header offsets of a whole gather. Possible defect types and their fixes:",
    ]
    for d in DEFECT_TYPES:
        lines.append(f"- {d} (fix: {RECOMMENDED_FIX[d]}): {DEFECT_DESCRIPTIONS[d]}")
    lines += [
        "For every defect you find: record it with flag_defect using the exact type, then repair it with "
        "apply_fix. Changing healthy traces or headers is penalised, so act only on evidence.",
        f"Call submit_report when you are done. You have {budget} tool calls in total; the episode is "
        "submitted automatically when they run out.",
    ]
    return "\n".join(lines)


@dataclass
class ToolResult:
    name: str
    ok: bool
    content: dict
    image_png: bytes | None = None

    def to_text(self) -> str:
        return json.dumps(self.content, separators=(",", ":"), default=str)


@dataclass
class Episode:
    clean: Survey
    initial: Survey
    state: Survey
    truth: list
    flags: set = field(default_factory=set)
    killed: set = field(default_factory=set)
    touched: set = field(default_factory=set)
    fix_log: list = field(default_factory=list)


def parse_traces(traces: Any, n: int) -> list[int]:
    """Accept a list of ints, an int, "all", or a range/comma string such as "10-14" or "3,7,9"."""
    if traces is None:
        return list(range(n))
    if isinstance(traces, (bool, float)):
        raise ValueError("traces must be an int, a list of ints, a range string, or 'all'")
    if isinstance(traces, int):
        out = [traces]
    elif isinstance(traces, str):
        s = traces.strip().lower()
        if s == "all":
            return list(range(n))
        out = []
        for part in s.replace(" ", "").split(","):
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
    else:
        out = [int(t) for t in traces]
    out = sorted(set(out))
    bad = [t for t in out if t < 0 or t >= n]
    if bad:
        raise ValueError(f"trace indices out of range 0..{n - 1}: {bad[:5]}")
    if not out:
        raise ValueError("no traces given")
    return out


class QCEnv:
    """One seeded QC episode with a tool budget. Call ``reset`` (done in __init__) to start."""

    def __init__(self, task: TaskSpec, weights: RewardWeights = RewardWeights()):
        self.task = task
        self.weights = weights
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> dict:
        regime = REGIMES[self.task.regime]
        clean = make_survey(regime, self.task.n_gathers, np.random.default_rng([self.task.seed, 0]))
        initial, truth = inject_defects(
            clean,
            np.random.default_rng([self.task.seed, 1]),
            self.task.difficulty,
            self.task.defect_types,
            self.task.geometry_prob,
        )
        self.ep = Episode(clean=clean, initial=initial, state=initial.copy(), truth=truth)
        self.n_calls = 0
        self.n_invalid = 0
        self.trajectory: list[dict] = []
        self.done = False
        self.result: dict | None = None
        self.report = ""
        return self.observation()

    @property
    def budget(self) -> int:
        return self.task.budget

    def observation(self) -> dict:
        g0 = self.ep.state.gathers[0]
        return {
            "task_id": self.task.task_id,
            "instructions": instructions_text(
                len(self.ep.state.gathers), g0.n_traces, g0.dt * 1000.0, g0.n_samples * g0.dt * 1000.0, self.budget
            ),
            "gather_ids": self.ep.state.gather_ids,
        }

    # ------------------------------------------------------------------ dispatch
    def call(self, name: str, args: dict | None = None) -> ToolResult:
        args = dict(args or {})
        if self.done:
            return ToolResult(name, False, {"error": "episode already submitted"})
        self.n_calls += 1
        handler = getattr(self, f"_t_{name}", None)
        image = None
        if handler is None or name not in TOOL_NAMES:
            self.n_invalid += 1
            content: dict = {"error": f"unknown tool {name!r}; available: {TOOL_NAMES}"}
            ok = False
        else:
            try:
                out = handler(**args)
                if isinstance(out, tuple):
                    content, image = out
                else:
                    content = out
                ok = "error" not in content
                if not ok:
                    self.n_invalid += 1
            except (ValueError, KeyError, TypeError) as exc:
                self.n_invalid += 1
                content, ok = {"error": f"{type(exc).__name__}: {exc}"}, False

        if not self.done and self.n_calls >= self.budget:
            self._submit("(budget exhausted)")
            content = dict(content)
            content["budget_exhausted"] = True
        content = dict(content)
        content["calls_left"] = max(0, self.budget - self.n_calls)
        self.trajectory.append(
            {"step": self.n_calls, "tool": name, "args": args, "ok": ok, "result": content, "image": image is not None}
        )
        return ToolResult(name, ok, content, image)

    def invalid_call(self, name: str, message: str) -> ToolResult:
        """Record a call the agent could not even express (e.g. malformed JSON arguments).
        It costs a call like any other and counts as invalid."""
        if self.done:
            return ToolResult(name, False, {"error": "episode already submitted"})
        self.n_calls += 1
        self.n_invalid += 1
        content: dict = {"error": message}
        if self.n_calls >= self.budget:
            self._submit("(budget exhausted)")
            content["budget_exhausted"] = True
        content["calls_left"] = max(0, self.budget - self.n_calls)
        self.trajectory.append({"step": self.n_calls, "tool": name, "args": None, "ok": False, "result": content, "image": False})
        return ToolResult(name, False, content)

    def submit(self, summary: str = "") -> dict:
        """Grade the episode (agents normally call the submit_report tool instead)."""
        if not self.done:
            self._submit(summary)
        return self.result

    def _submit(self, summary: str) -> None:
        self.report = summary
        ep = self.ep
        self.result = grade(
            ep.truth,
            ep.clean,
            ep.initial,
            ep.state,
            ep.flags,
            ep.killed,
            ep.touched,
            n_calls=self.n_calls,
            n_invalid=self.n_invalid,
            weights=self.weights,
        )
        self.done = True

    # ------------------------------------------------------------------ helpers
    def _gather(self, gather_id: Any):
        try:
            return self.ep.state.gather(int(gather_id))
        except (KeyError, ValueError, TypeError):
            raise ValueError(f"unknown gather_id {gather_id!r}; valid ids: {self.ep.state.gather_ids}")

    # ------------------------------------------------------------------ inspection tools
    def _t_survey_summary(self) -> dict:
        g0 = self.ep.state.gathers[0]
        return {
            "n_gathers": len(self.ep.state.gathers),
            "gather_ids": self.ep.state.gather_ids,
            "traces_per_gather": g0.n_traces,
            "dt_ms": g0.dt * 1000.0,
            "n_samples": g0.n_samples,
            "record_length_ms": g0.n_samples * g0.dt * 1000.0,
            "group_interval_m": g0.group_interval,
            "header_offset_range_m": {
                g.gather_id: [float(g.offsets.min()), float(g.offsets.max())] for g in self.ep.state.gathers
            },
            "defect_types": {d: {"fix": RECOMMENDED_FIX[d], "description": DEFECT_DESCRIPTIONS[d]} for d in DEFECT_TYPES},
            "fixes": FIX_DESCRIPTIONS,
            "budget": self.budget,
        }

    def _t_trace_headers(self, gather_id, traces=None) -> dict:
        g = self._gather(gather_id)
        idx = parse_traces(traces, g.n_traces)
        return {
            "gather_id": g.gather_id,
            "columns": ["trace", "channel", "offset_m"],
            "rows": [[i, int(g.channels[i]), float(g.offsets[i])] for i in idx],
        }

    def _t_trace_attributes(self, gather_id, traces=None) -> dict:
        g = self._gather(gather_id)
        return qc.trace_attributes(g, parse_traces(traces, g.n_traces) if traces is not None else None)

    def _t_neighbour_consistency(self, gather_id, traces=None) -> dict:
        g = self._gather(gather_id)
        return qc.neighbour_consistency(g, parse_traces(traces, g.n_traces) if traces is not None else None)

    def _t_spectrum_check(self, gather_id, traces=None) -> dict:
        g = self._gather(gather_id)
        return qc.spectrum_check(g, parse_traces(traces, g.n_traces) if traces is not None else None)

    def _t_first_break_check(self, gather_id) -> dict:
        return qc.first_break_check(self._gather(gather_id))

    def _t_geometry_survey_check(self) -> dict:
        return qc.geometry_survey_check(self.ep.state.gathers)

    def _t_render_gather(self, gather_id, traces=None, t_start_ms=None, t_end_ms=None):
        g = self._gather(gather_id)
        idx = parse_traces(traces, g.n_traces) if traces is not None else None
        t_range = None
        if t_start_ms is not None or t_end_ms is not None:
            t_range = (float(t_start_ms or 0.0), float(t_end_ms if t_end_ms is not None else g.n_samples * g.dt * 1000.0))
        png = render_gather_png(g, idx, t_range)
        return {"gather_id": g.gather_id, "image": "png", "bytes": len(png)}, png

    # ------------------------------------------------------------------ action tools
    def _t_flag_defect(self, gather_id, traces, defect_type, note: str = "") -> dict:
        g = self._gather(gather_id)
        if defect_type not in DEFECT_TYPES:
            raise ValueError(f"unknown defect_type {defect_type!r}; valid: {list(DEFECT_TYPES)}")
        if defect_type in GATHER_DEFECTS:
            self.ep.flags.add((g.gather_id, GATHER, defect_type))
            added = 1
        else:
            idx = parse_traces(traces, g.n_traces)
            before = len(self.ep.flags)
            self.ep.flags.update((g.gather_id, t, defect_type) for t in idx)
            added = len(self.ep.flags) - before
        return {"flagged": defect_type, "gather_id": g.gather_id, "new_flags": added, "total_flags": len(self.ep.flags)}

    def _t_remove_flags(self, gather_id, traces, defect_type: str | None = None) -> dict:
        g = self._gather(gather_id)
        idx = set(parse_traces(traces, g.n_traces))
        # A gather-level flag goes only when asked for by type, or when every trace is cleared.
        if defect_type in GATHER_DEFECTS or (isinstance(traces, str) and traces.strip().lower() == "all"):
            idx.add(GATHER)
        before = len(self.ep.flags)
        self.ep.flags = {
            f
            for f in self.ep.flags
            if not (f[0] == g.gather_id and f[1] in idx and (defect_type is None or f[2] == defect_type))
        }
        return {"removed": before - len(self.ep.flags), "total_flags": len(self.ep.flags)}

    def _t_apply_fix(self, gather_id, traces, fix, value: float | None = None) -> dict:
        g = self._gather(gather_id)
        if fix not in FIXES:
            raise ValueError(f"unknown fix {fix!r}; valid: {list(FIXES)}")
        idx = parse_traces(traces, g.n_traces)
        details: dict = {}
        if fix in ("shift", "offset_shift") and value is None:
            raise ValueError(f"fix {fix!r} needs a value")
        if value is not None and (isinstance(value, bool) or not np.isfinite(float(value))):
            raise ValueError("value must be a finite number")
        if fix == "offset_shift":
            g.offsets[idx] = g.offsets[idx] + float(value)
            details["offset_change_m"] = float(value)
        else:
            rms_all = qc.trace_rms(g.data)
            for t in idx:
                x = g.data[t].astype(np.float64)
                if fix == "kill":
                    y = np.zeros_like(x)
                    self.ep.killed.add((g.gather_id, t))
                elif fix == "flip_polarity":
                    y = -x
                elif fix == "shift":
                    y = shift_ms(x, g.dt, float(value))
                elif fix == "despike":
                    y, n_fixed = despike(x, float(value) if value is not None else 15.0)
                    details.setdefault("samples_replaced", {})[t] = n_fixed
                elif fix == "lowcut":
                    y = lowcut(x, g.dt, float(value) if value is not None else 8.0)
                elif fix == "notch":
                    f0 = float(value) if value is not None else 50.0
                    if not 0 < f0 < 0.5 / g.dt:
                        raise ValueError(f"notch frequency must be between 0 and {0.5 / g.dt:g} Hz")
                    y = notch(x, g.dt, f0)
                elif fix == "rescale":
                    if value is None:
                        nb = qc.neighbour_indices(t, g.n_traces, 4)
                        target = float(np.median(rms_all[nb]))
                        factor = target / rms_all[t] if rms_all[t] > 0 else 1.0
                    else:
                        factor = float(value)
                    y = x * factor
                    details.setdefault("factor", {})[t] = round(float(factor), 4)
                else:  # pragma: no cover - guarded above
                    raise ValueError(fix)
                g.data[t] = y.astype(np.float32)
                self.ep.touched.add((g.gather_id, t))
        self.ep.fix_log.append({"gather_id": g.gather_id, "traces": idx, "fix": fix, "value": value})
        return {"applied": fix, "gather_id": g.gather_id, "traces": idx, **details}

    def _t_submit_report(self, summary: str = "") -> dict:
        self._submit(summary)
        return {"submitted": True}
