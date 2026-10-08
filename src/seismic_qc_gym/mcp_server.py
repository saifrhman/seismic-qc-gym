"""MCP server exposing the QC environment's tools (Model Context Protocol, Python SDK 2.x).

Run over stdio:  python -m seismic_qc_gym.mcp_server   (or the `qcgym-mcp` script)

Any MCP client (a desktop assistant, an agent framework, the MCP Inspector) can then start an
episode with new_episode, inspect gathers, flag and fix defects, and submit. Over MCP the
grade is returned by submit_report, because the client is usually a person or an external
harness evaluating an agent; in-process training reads ``QCEnv.result`` instead.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import Image, MCPServer

from .env import QCEnv, instructions_text  # noqa: F401  (instructions_text re-exported for clients)
from .tasks import make_tasks

INSTRUCTIONS = (
    "Seismic QC gym. Call new_episode first, then follow its instructions: inspect gathers with the QC tools, "
    "record each defect with flag_defect, repair it with apply_fix, and finish with submit_report."
)

server = MCPServer("seismic-qc-gym", instructions=INSTRUCTIONS)
_state: dict[str, QCEnv | None] = {"env": None}

Traces = list[int] | int | str | None


def _env() -> QCEnv:
    env = _state["env"]
    if env is None:
        raise ValueError("no episode running: call new_episode first")
    return env


def _call(name: str, args: dict[str, Any]) -> dict:
    args = {k: v for k, v in args.items() if v is not None}
    return _env().call(name, args).content


@server.tool()
def new_episode(split: str = "val", index: int = 0, difficulty: str | None = None, n_gathers: int = 4) -> dict:
    """Start a new seeded QC episode from a task split (train, val or test) and return its instructions."""
    task = make_tasks(split, 1, difficulty, n_gathers=n_gathers, start=index)[0]
    _state["env"] = QCEnv(task)
    return _state["env"].observation()


@server.tool()
def survey_summary() -> dict:
    """Survey layout, the defect catalogue with recommended fixes, and the call budget."""
    return _call("survey_summary", {})


@server.tool()
def trace_headers(gather_id: int, traces: Traces = None) -> dict:
    """Trace headers for one gather: trace index, channel number and header offset (m)."""
    return _call("trace_headers", {"gather_id": gather_id, "traces": traces})


@server.tool()
def trace_attributes(gather_id: int, traces: Traces = None) -> dict:
    """Per-trace RMS ratios, spike index and zero fraction for one gather, with gather medians."""
    return _call("trace_attributes", {"gather_id": gather_id, "traces": traces})


@server.tool()
def neighbour_consistency(gather_id: int, traces: Traces = None) -> dict:
    """Signed correlations with neighbouring traces and each trace's time shift relative to them (ms)."""
    return _call("neighbour_consistency", {"gather_id": gather_id, "traces": traces})


@server.tool()
def spectrum_check(gather_id: int, traces: Traces = None) -> dict:
    """Dominant frequency, low-frequency (<6 Hz) ratio and 50 Hz line ratio per trace, with gather medians."""
    return _call("spectrum_check", {"gather_id": gather_id, "traces": traces})


@server.tool()
def first_break_check(gather_id: int) -> dict:
    """Near-offset first-break fit against header offsets: velocity, intercept, suggested offset correction."""
    return _call("first_break_check", {"gather_id": gather_id})


@server.tool()
def geometry_survey_check() -> dict:
    """First-break fit for every gather, to find gathers whose header offsets are shifted."""
    return _call("geometry_survey_check", {})


@server.tool()
def render_gather(
    gather_id: int, traces: Traces = None, t_start_ms: float | None = None, t_end_ms: float | None = None
) -> Image:
    """Render a gather (optionally a subset of traces and a time window) as a PNG image."""
    args = {"gather_id": gather_id, "traces": traces, "t_start_ms": t_start_ms, "t_end_ms": t_end_ms}
    result = _env().call("render_gather", {k: v for k, v in args.items() if v is not None})
    if result.image_png is None:
        raise ValueError(result.content.get("error", "render failed"))
    return Image(data=result.image_png, format="png")


@server.tool()
def flag_defect(gather_id: int, traces: list[int] | int | str, defect_type: str, note: str = "") -> dict:
    """Record a defect with its exact type (geometry_error: traces="all", counted once per gather)."""
    return _call("flag_defect", {"gather_id": gather_id, "traces": traces, "defect_type": defect_type, "note": note})


@server.tool()
def remove_flags(gather_id: int, traces: list[int] | int | str, defect_type: str | None = None) -> dict:
    """Remove flags recorded earlier on these traces (all types, or one type)."""
    return _call("remove_flags", {"gather_id": gather_id, "traces": traces, "defect_type": defect_type})


@server.tool()
def apply_fix(gather_id: int, traces: list[int] | int | str, fix: str, value: float | None = None) -> dict:
    """Apply a repair: kill, flip_polarity, shift (value ms), despike, lowcut (Hz), notch (Hz), rescale, offset_shift (value m)."""
    return _call("apply_fix", {"gather_id": gather_id, "traces": traces, "fix": fix, "value": value})


@server.tool()
def submit_report(summary: str = "") -> dict:
    """End the episode and return its grade."""
    env = _env()
    content = env.call("submit_report", {"summary": summary}).content
    return {**content, "result": env.result}


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
