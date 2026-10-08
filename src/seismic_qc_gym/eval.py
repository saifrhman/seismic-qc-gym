"""Run an agent over a task split and report aggregate scores.

    python -m seismic_qc_gym.eval --agent heuristic --split val --n 30
    python -m seismic_qc_gym.eval --agent llm --model gpt-4o-mini --split val --n 9
    python -m seismic_qc_gym.eval --agent llm --model qwen/qwen3-32b --base-url https://openrouter.ai/api/v1 --vision

Writes results/<agent>_<split>.json (every episode plus the summary) and, with
--save-trajectories, one JSON file per episode with every tool call and result.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .agents import FlagAllAgent, HeuristicAgent, KillAllAgent, NoOpAgent, OracleAgent
from .defects import DEFECT_TYPES
from .env import QCEnv
from .tasks import TaskSpec, make_tasks

SIMPLE_AGENTS = {
    "heuristic": HeuristicAgent,
    "oracle": OracleAgent,
    "noop": NoOpAgent,
    "flag_all": FlagAllAgent,
    "kill_all": KillAllAgent,
}


def run_episode(agent, task: TaskSpec) -> tuple[dict, QCEnv]:
    env = QCEnv(task)
    t0 = time.time()
    result = agent.run(env)
    result = dict(result)
    result["task"] = task.to_dict()
    result["seconds"] = round(time.time() - t0, 3)
    return result, env


def summarise(results: list[dict]) -> dict:
    def mean(key_fn):
        vals = [key_fn(r) for r in results]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    out = {
        "n_episodes": len(results),
        "reward": mean(lambda r: r["reward"]),
        "detection_f1": mean(lambda r: r["detection"]["f1"]),
        "detection_precision": mean(lambda r: r["detection"]["precision"]),
        "detection_recall": mean(lambda r: r["detection"]["recall"]),
        "repair": mean(lambda r: r["repair"]),
        "damage": mean(lambda r: r["damage"]),
        "tool_calls": mean(lambda r: r["n_calls"]),
        "invalid_calls": mean(lambda r: r["n_invalid_calls"]),
    }
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_diff[r["task"]["difficulty"]].append(r)
    out["by_difficulty"] = {
        d: {"n": len(rs), "reward": float(np.mean([r["reward"] for r in rs])), "f1": float(np.mean([r["detection"]["f1"] for r in rs]))}
        for d, rs in sorted(by_diff.items())
    }
    per_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        for dtype, v in r["per_type"].items():
            if v["n_true"] == 0:
                continue  # type not present in this episode (only false flags)
            per_type[dtype]["recall"].append(v["recall"])
            per_type[dtype]["precision"].append(v["precision"])
            if v["repair"] is not None:
                per_type[dtype]["repair"].append(v["repair"])
    out["per_type"] = {
        d: {k: float(np.mean(v)) for k, v in per_type[d].items()} for d in DEFECT_TYPES if d in per_type
    }
    llm = [r["llm"] for r in results if "llm" in r]
    if llm:
        out["llm"] = {k: float(np.mean([x[k] for x in llm])) for k in llm[0]}
    return out


def markdown(summary: dict, title: str) -> str:
    f = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
    lines = [
        f"### {title}",
        "",
        "| episodes | reward | detection F1 | precision | recall | repair | damage | tool calls |",
        "|---|---|---|---|---|---|---|---|",
        f"| {summary['n_episodes']} | {f(summary['reward'])} | {f(summary['detection_f1'])} | "
        f"{f(summary['detection_precision'])} | {f(summary['detection_recall'])} | {f(summary['repair'])} | "
        f"{f(summary['damage'])} | {summary['tool_calls']:.1f} |",
        "",
        "| defect type | recall | precision | repair |",
        "|---|---|---|---|",
    ]
    for d, v in summary["per_type"].items():
        lines.append(f"| {d} | {f(v.get('recall'))} | {f(v.get('precision'))} | {f(v.get('repair'))} |")
    return "\n".join(lines)


def api_hint(exc: Exception) -> str:
    text = str(exc).lower()
    status = getattr(exc, "status_code", None)
    if status == 401:
        return "Check OPENAI_API_KEY (and OPENAI_BASE_URL for OpenRouter or a local server)."
    if status == 429 or "rate limit" in text:
        return ("Rate limit reached. Free OpenRouter models allow 20 requests/minute and 50 or 1,000 requests/day; "
                "add --min-interval 3.5, run fewer episodes, or try again after the daily reset.")
    if "tool" in text and ("support" in text or "endpoint" in text):
        return "This model or provider does not support tool calling; pick a model whose page lists tool support."
    if status == 404:
        return "Model not found: check the --model id."
    return "See docs/RUNNING.md, section Troubleshooting."


def build_agent(args):
    if args.agent in SIMPLE_AGENTS:
        return SIMPLE_AGENTS[args.agent]()
    if args.agent == "llm":
        from .agents.llm import LLMAgent, OpenAICompatibleClient

        if not args.model:
            raise SystemExit("--model is required for --agent llm")
        client = OpenAICompatibleClient(
            args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_retries=args.max_retries,
            min_interval=args.min_interval,
        )
        return LLMAgent(client, max_turns=args.max_turns, vision=args.vision)
    raise SystemExit(f"unknown agent {args.agent!r}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", default="heuristic", choices=[*SIMPLE_AGENTS, "llm"])
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    p.add_argument("--n-gathers", type=int, default=4)
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-turns", type=int, default=80)
    p.add_argument("--vision", action="store_true", help="attach render_gather images (vision-language models)")
    p.add_argument("--max-retries", type=int, default=5, help="retries per model request on rate limits and server errors")
    p.add_argument("--min-interval", type=float, default=0.0,
                   help="minimum seconds between model requests (3.5 keeps under 20 requests/minute)")
    p.add_argument("--out", default="results")
    p.add_argument("--tag", default="", help="suffix for output file names")
    p.add_argument("--save-trajectories", action="store_true")
    args = p.parse_args(argv)

    tasks = make_tasks(args.split, args.n, args.difficulty, n_gathers=args.n_gathers, start=args.start)
    agent = build_agent(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.agent}{'_' + args.tag if args.tag else ''}_{args.split}"
    results = []
    for k, task in enumerate(tasks, 1):
        try:
            result, env = run_episode(agent, task)
        except Exception as exc:  # model API errors: report clearly and keep what finished
            if type(exc).__module__.split(".")[0] != "openai":
                raise
            print(f"\nModel API error on {task.task_id}: {type(exc).__name__}: {exc}")
            print(api_hint(exc))
            break
        results.append(result)
        print(
            f"[{k}/{len(tasks)}] {task.task_id} {task.difficulty:6s} reward {result['reward']:+.3f} "
            f"F1 {result['detection']['f1']:.3f} repair {result['repair']:.3f} damage {result['damage']:.3f} "
            f"calls {result['n_calls']}",
            flush=True,
        )
        if args.save_trajectories:
            tdir = out / "trajectories" / name
            tdir.mkdir(parents=True, exist_ok=True)
            record = {"task": task.to_dict(), "truth": [d.to_dict() for d in env.ep.truth], "trajectory": env.trajectory, "result": result}
            if hasattr(agent, "messages"):
                record["messages"] = [m for m in agent.messages if m.get("role") != "user" or isinstance(m.get("content"), str)]
            (tdir / f"{task.task_id}.json").write_text(json.dumps(record, default=str, indent=1))
    if not results:
        raise SystemExit(1)
    summary = summarise(results)
    (out / f"{name}.json").write_text(json.dumps({"summary": summary, "episodes": results}, default=str, indent=1))
    print()
    print(markdown(summary, f"{args.agent} on {args.split} ({len(tasks)} episodes)"))


if __name__ == "__main__":
    main()
