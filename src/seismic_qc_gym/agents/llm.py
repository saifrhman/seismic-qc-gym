"""Tool-calling LLM agent over any OpenAI-compatible chat-completions endpoint.

The agent sees the task instructions and the tool definitions, and runs a plain
think-act loop: the model requests tool calls, the environment executes them, and the
results go back as tool messages until the model calls submit_report or runs out of
turns. With ``vision=True`` gather images from render_gather are attached as image
content, for vision-language models.

The model client is a small protocol so the loop can be tested without a network
(``ScriptedClient``) and pointed at OpenAI, OpenRouter, vLLM or any compatible server.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from typing import Protocol

from ..env import TOOL_SPECS, QCEnv

SYSTEM_PROMPT = """You are an experienced seismic processing geophysicist doing pre-stack quality control.
You work only through the tools. Be systematic and evidence-driven:
1. Start with survey_summary and geometry_survey_check.
2. For each gather, run trace_attributes, neighbour_consistency and spectrum_check. Compare each trace with
   the gather medians and with its neighbours; real defects stand far outside the healthy range.
3. Record each defect with flag_defect (exact type), then repair it with apply_fix using the recommended fix.
   Group adjacent traces of a run defect (swell, power-line) in one call.
4. One trace can hide a second defect. After fixing, re-run the relevant check on that gather. A time-shifted
   trace makes its immediate neighbours look about half as shifted in the opposite direction: fix the largest
   shift first, then re-measure.
5. Changing healthy traces or headers is penalised as much as missing a defect, so act only on clear evidence.
6. Call submit_report with a short summary when every gather is done. Watch calls_left in each tool result."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class AssistantTurn:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict | None = None


class ChatClient(Protocol):
    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn: ...


def openai_tools(specs: list[dict]) -> list[dict]:
    return [{"type": "function", "function": spec} for spec in specs]


class OpenAICompatibleClient:
    """Chat-completions client. Reads OPENAI_API_KEY / OPENAI_BASE_URL unless given."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        max_retries: int = 5,
        min_interval: float = 0.0,
    ):
        from openai import OpenAI  # optional dependency

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_interval = min_interval  # seconds between requests, e.g. 3.5 for a 20-requests/minute limit
        self._last = 0.0
        self.n_requests = 0
        # The SDK retries rate-limit (429) and server errors with exponential backoff.
        self.client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            max_retries=max_retries,
        )

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        kwargs = dict(model=self.model, messages=messages, tools=openai_tools(tools), temperature=self.temperature)
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        self._last = time.monotonic()
        self.n_requests += 1
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = [
            ToolCall(tc.id, tc.function.name, tc.function.arguments or "{}")
            for tc in (msg.tool_calls or [])
            if getattr(tc, "function", None) is not None
        ]
        usage = None
        if getattr(resp, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
            }
        return AssistantTurn(msg.content, calls, usage)


class ScriptedClient:
    """Replays a fixed sequence of turns; each turn is a list of (tool_name, args) pairs.
    An empty list produces a text-only reply. Used for tests and dry runs."""

    def __init__(self, turns: list[list[tuple[str, dict]]]):
        self.turns = list(turns)
        self.seen: list[list[dict]] = []
        self._n = 0

    def complete(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        self.seen.append(list(messages))
        if not self.turns:
            return AssistantTurn("done", [])
        turn = self.turns.pop(0)
        calls = []
        for name, args in turn:
            self._n += 1
            calls.append(ToolCall(f"call_{self._n}", name, json.dumps(args)))
        return AssistantTurn(None if calls else "thinking", calls)


class LLMAgent:
    name = "llm"

    def __init__(
        self,
        client: ChatClient,
        max_turns: int = 80,
        vision: bool = False,
        system_prompt: str = SYSTEM_PROMPT,
        max_nudges: int = 2,
    ):
        self.client = client
        self.max_turns = max_turns
        self.vision = vision
        self.system_prompt = system_prompt
        self.max_nudges = max_nudges
        self.messages: list[dict] = []
        self.stats: dict = {}

    def tools(self) -> list[dict]:
        # Text-only models cannot read images, so they do not get the render tool.
        return [t for t in TOOL_SPECS if self.vision or t["name"] != "render_gather"]

    def run(self, env: QCEnv) -> dict:
        obs = env.observation()
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": obs["instructions"] + f"\nGather ids: {obs['gather_ids']}."},
        ]
        tools = self.tools()
        nudges = turns = prompt_tokens = completion_tokens = 0
        while not env.done and turns < self.max_turns:
            turns += 1
            reply = self.client.complete(self.messages, tools)
            if reply.usage:
                prompt_tokens += reply.usage.get("prompt_tokens", 0)
                completion_tokens += reply.usage.get("completion_tokens", 0)
            msg: dict = {"role": "assistant", "content": reply.content or ""}
            if reply.tool_calls:
                msg["tool_calls"] = [
                    {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments}}
                    for tc in reply.tool_calls
                ]
            self.messages.append(msg)

            if not reply.tool_calls:
                nudges += 1
                if nudges > self.max_nudges:
                    break
                self.messages.append(
                    {
                        "role": "user",
                        "content": "Continue with the tools: inspect and fix any remaining gathers, or call "
                        "submit_report if you have finished.",
                    }
                )
                continue

            images = []
            for tc in reply.tool_calls:
                if env.done:
                    content = json.dumps({"error": "episode already submitted"})
                else:
                    try:
                        args = json.loads(tc.arguments or "{}")
                        if not isinstance(args, dict):
                            raise ValueError("arguments must be a JSON object")
                        result = env.call(tc.name, args)
                        content = result.to_text()
                        if result.image_png is not None and self.vision:
                            images.append((tc.id, result.image_png))
                    except (json.JSONDecodeError, ValueError) as exc:
                        content = env.invalid_call(tc.name, f"invalid arguments: {exc}").to_text()
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
            for call_id, png in images:
                b64 = base64.b64encode(png).decode("ascii")
                self.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Image returned by render_gather ({call_id})."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }
                )

        if not env.done:
            env.call("submit_report", {"summary": "(agent stopped without submitting)"})
        self.stats = {"turns": turns, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        result = dict(env.result)
        result["llm"] = self.stats
        return result
