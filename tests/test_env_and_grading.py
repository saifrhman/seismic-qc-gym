import numpy as np
import pytest

from seismic_qc_gym.agents import FlagAllAgent, HeuristicAgent, KillAllAgent, NoOpAgent, OracleAgent
from seismic_qc_gym.env import TOOL_NAMES, TOOL_SPECS, QCEnv, parse_traces
from seismic_qc_gym.grading import GATHER, truth_items
from seismic_qc_gym.tasks import TaskSpec, make_tasks


def _task(**kw):
    base = dict(task_id="t", seed=4, regime="A", difficulty="medium", n_gathers=2, budget=60)
    base.update(kw)
    return TaskSpec(**base)


def test_every_tool_spec_has_a_handler():
    for spec in TOOL_SPECS:
        assert hasattr(QCEnv, f"_t_{spec['name']}")
        assert spec["parameters"]["type"] == "object"
    assert len(TOOL_NAMES) == len(set(TOOL_NAMES))


def test_parse_traces_forms():
    assert parse_traces("all", 5) == [0, 1, 2, 3, 4]
    assert parse_traces("1-3", 10) == [1, 2, 3]
    assert parse_traces("4, 2,2", 10) == [2, 4]
    assert parse_traces(7, 10) == [7]
    assert parse_traces([3, 1], 10) == [1, 3]
    for bad in ("12", [-1], "x", 1.5):
        with pytest.raises(ValueError):
            parse_traces(bad, 10)


def test_same_task_same_episode():
    a, b = QCEnv(_task()), QCEnv(_task())
    assert [d.to_dict() for d in a.ep.truth] == [d.to_dict() for d in b.ep.truth]
    assert np.array_equal(a.ep.state.gathers[0].data, b.ep.state.gathers[0].data)


def test_bad_calls_cost_budget_and_are_counted():
    env = QCEnv(_task(budget=4))
    assert "error" in env.call("no_such_tool").content
    assert "error" in env.call("trace_attributes", {"gather_id": 99}).content
    assert "error" in env.call("apply_fix", {"gather_id": 0, "traces": [1], "fix": "shift"}).content  # no value
    assert env.n_invalid == 3 and not env.done
    out = env.call("survey_summary").content
    assert out["budget_exhausted"] and env.done and env.result is not None
    assert "error" in env.call("survey_summary").content


def test_render_returns_png():
    env = QCEnv(_task())
    res = env.call("render_gather", {"gather_id": 0, "traces": "0-20", "t_start_ms": 0, "t_end_ms": 1000})
    assert res.ok and res.image_png[:8] == b"\x89PNG\r\n\x1a\n"


def test_geometry_flag_counts_once_per_gather():
    env = QCEnv(_task(defect_types=("geometry_error",)))
    env.call("flag_defect", {"gather_id": 0, "traces": [1, 2, 3], "defect_type": "geometry_error"})
    env.call("flag_defect", {"gather_id": 0, "traces": "all", "defect_type": "geometry_error"})
    assert env.ep.flags == {(0, GATHER, "geometry_error")}


def test_oracle_reaches_the_ceiling_and_degenerate_policies_do_not_pay():
    tasks = make_tasks("train", 6)
    scores = {}
    for agent in (OracleAgent(), NoOpAgent(), FlagAllAgent(), KillAllAgent()):
        scores[agent.name] = [agent.run(QCEnv(t)) for t in tasks]
    oracle = scores["oracle"]
    assert all(r["detection"]["f1"] == 1.0 and r["damage"] == 0.0 for r in oracle)
    assert np.mean([r["reward"] for r in oracle]) > 0.9
    assert all(r["reward"] == 0.0 for r in scores["noop"])
    assert np.mean([r["reward"] for r in scores["flag_all"]]) < 0.05
    assert np.mean([r["reward"] for r in scores["kill_all"]]) < -0.5


def test_damaging_a_healthy_trace_is_penalised():
    env = QCEnv(_task())
    defective = {(g, t) for g, t, _ in truth_items(env.ep.truth)}
    healthy = next(t for t in range(10, 90) if (0, t) not in defective)
    env.call("apply_fix", {"gather_id": 0, "traces": [healthy], "fix": "flip_polarity"})
    result = env.submit()
    assert result["damage"] > 0 and result["reward"] < 0


def test_heuristic_baseline_on_train():
    results = [HeuristicAgent().run(QCEnv(t)) for t in make_tasks("train", 6)]
    assert np.mean([r["reward"] for r in results]) > 0.85
    assert all(r["n_invalid_calls"] == 0 for r in results)


def test_non_finite_values_are_rejected():
    env = QCEnv(_task())
    for fix in ("offset_shift", "rescale", "shift"):
        out = env.call("apply_fix", {"gather_id": 0, "traces": [5], "fix": fix, "value": float("nan")}).content
        assert "error" in out
    assert env.n_invalid == 3


def test_removing_trace_flags_keeps_the_gather_flag():
    env = QCEnv(_task())
    env.call("flag_defect", {"gather_id": 0, "traces": "all", "defect_type": "geometry_error"})
    env.call("flag_defect", {"gather_id": 0, "traces": [3, 4], "defect_type": "spike"})
    env.call("remove_flags", {"gather_id": 0, "traces": [3, 4]})
    assert env.ep.flags == {(0, GATHER, "geometry_error")}
    env.call("remove_flags", {"gather_id": 0, "traces": "all"})
    assert env.ep.flags == set()
