import asyncio
import json

import pytest

from seismic_qc_gym.agents.llm import LLMAgent, ScriptedClient
from seismic_qc_gym.defects import GATHER_DEFECTS, RECOMMENDED_FIX
from seismic_qc_gym.env import QCEnv
from seismic_qc_gym.tasks import TaskSpec


def _task():
    return TaskSpec(task_id="llm-test", seed=9, regime="A", difficulty="easy", n_gathers=2, budget=80)


def _oracle_script(env):
    """Tool calls an ideal model would make, built from the planted truth."""
    turns = [[("survey_summary", {}), ("geometry_survey_check", {})]]
    for d in env.ep.truth:
        if d.defect_type in GATHER_DEFECTS:
            turns.append([
                ("flag_defect", {"gather_id": d.gather_id, "traces": "all", "defect_type": d.defect_type}),
                ("apply_fix", {"gather_id": d.gather_id, "traces": "all", "fix": "offset_shift", "value": -d.params["delta_m"]}),
            ])
            continue
        args = {"gather_id": d.gather_id, "traces": list(d.traces), "fix": RECOMMENDED_FIX[d.defect_type]}
        if d.defect_type == "time_shift":
            args["value"] = -d.params["shift_ms"]
        if d.defect_type == "gain_error":
            args["value"] = 1 / d.params["scale"]
        turns.append([("flag_defect", {"gather_id": d.gather_id, "traces": list(d.traces), "defect_type": d.defect_type}), ("apply_fix", args)])
    turns.append([("submit_report", {"summary": "done"})])
    return turns


def test_llm_loop_executes_tool_calls_and_submits():
    env = QCEnv(_task())
    client = ScriptedClient(_oracle_script(env))
    agent = LLMAgent(client)
    result = agent.run(env)
    assert env.done and result["detection"]["f1"] == 1.0 and result["reward"] > 0.85
    # every assistant tool call is answered by exactly one tool message
    calls = [tc["id"] for m in agent.messages if m["role"] == "assistant" for tc in m.get("tool_calls", [])]
    answers = [m["tool_call_id"] for m in agent.messages if m["role"] == "tool"]
    assert calls == answers
    assert agent.messages[0]["role"] == "system"
    assert "render_gather" not in [t["name"] for t in agent.tools()]
    assert "render_gather" in [t["name"] for t in LLMAgent(client, vision=True).tools()]


def test_llm_bad_json_and_silence_are_handled():
    env = QCEnv(_task())

    class Client(ScriptedClient):
        def complete(self, messages, tools):
            turn = super().complete(messages, tools)
            if turn.tool_calls:
                turn.tool_calls[0].arguments = "{not json"
            return turn

    agent = LLMAgent(Client([[("survey_summary", {})], [], [], []]), max_nudges=2)
    result = agent.run(env)
    assert env.done and result["n_invalid_calls"] == 1
    tool_msgs = [json.loads(m["content"]) for m in agent.messages if m["role"] == "tool"]
    assert "invalid arguments" in tool_msgs[0]["error"]


def test_vision_attaches_images():
    env = QCEnv(_task())
    agent = LLMAgent(ScriptedClient([[("render_gather", {"gather_id": 0, "traces": "0-10"})], [("submit_report", {})]]), vision=True)
    agent.run(env)
    image_msgs = [m for m in agent.messages if m["role"] == "user" and isinstance(m["content"], list)]
    assert image_msgs and image_msgs[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_mcp_server_round_trip():
    pytest.importorskip("mcp")
    from seismic_qc_gym.mcp_server import server

    async def go():
        names = {t.name for t in await server.list_tools()}
        assert {"new_episode", "trace_attributes", "apply_fix", "render_gather", "submit_report"} <= names
        r = await server.call_tool("new_episode", {"split": "train", "index": 2})
        assert "instructions" in r.content[0].text
        r = await server.call_tool("render_gather", {"gather_id": 0, "traces": [0, 1, 2, 3]})
        assert r.content[0].mime_type == "image/png"
        r = await server.call_tool("submit_report", {"summary": "nothing"})
        assert json.loads(r.content[0].text)["result"]["reward"] == 0.0

    asyncio.run(go())


def test_eval_cli_writes_results(tmp_path, capsys):
    from seismic_qc_gym.eval import main

    main(["--agent", "oracle", "--split", "train", "--n", "2", "--out", str(tmp_path), "--save-trajectories"])
    data = json.loads((tmp_path / "oracle_train.json").read_text())
    assert data["summary"]["n_episodes"] == 2 and data["summary"]["detection_f1"] == 1.0
    assert len(list((tmp_path / "trajectories" / "oracle_train").glob("*.json"))) == 2
    assert "| episodes |" in capsys.readouterr().out
