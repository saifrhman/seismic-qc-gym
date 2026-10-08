# Running Seismic QC Gym on your own machine

This guide covers installing the project, reproducing the baseline results, running an LLM agent through OpenRouter (or any OpenAI-compatible endpoint), and connecting the tools to an MCP client.

## 1. Requirements

- Python 3.10 to 3.13. 3.12 is recommended.
- Git.
- About 200 MB of disk space for the Python dependencies.
- No GPU is needed: the environment, the tools and the baselines run on a laptop CPU.
- Optional:
  - an OpenRouter or OpenAI API key, for the LLM agent;
  - Node.js, for the MCP Inspector.

## 2. Install

macOS and Linux:

```bash
git clone https://github.com/saifrhman/seismic-qc-gym.git
cd seismic-qc-gym
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

Windows (PowerShell):

```powershell
git clone https://github.com/saifrhman/seismic-qc-gym.git
cd seismic-qc-gym
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

If PowerShell refuses to run `Activate.ps1`, allow local scripts once with `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again.

`.[all]` installs the core dependencies plus the optional extras:

| Extra | Installs | Needed for |
|---|---|---|
| (core) | numpy, scipy, matplotlib | the environment and baselines |
| `llm` | the `openai` SDK | the LLM agent |
| `mcp` | the MCP Python SDK 2.x | the MCP server |
| `segy` | segyio | future real-data loading |
| `dev` | pytest | the tests |

Pick only what you need, for example `pip install -e ".[llm,dev]"`.

## 3. Check the install

```bash
python -m pytest
```

All 35 tests should pass in about 15 seconds. They cover the generator, every defect type, the QC tools, the grader (including the degenerate policies), the environment, the LLM loop (with a scripted model, so no network is needed), the MCP server and the evaluation CLI.

## 4. Reproduce the baselines

```bash
for agent in oracle heuristic noop flag_all kill_all; do
  for split in val test; do
    python -m seismic_qc_gym.eval --agent $agent --split $split --n 60
  done
done
python scripts/summarise_results.py   # writes results/baselines.md
python scripts/plot_results.py        # writes docs/figures/results.png
python scripts/make_figures.py        # writes the defect gallery and the before/after figure
```

Each run prints one line per episode and a summary table, and writes `results/<agent>_<split>.json`. Episodes are seeded, so the numbers match the README exactly. The full set of baseline runs takes a few minutes.

Useful options:

| Option | Meaning |
|---|---|
| `--n 30` | Number of episodes |
| `--start 100` | First episode index (to continue a split) |
| `--difficulty hard` | Only one difficulty level (default: cycle easy/medium/hard) |
| `--n-gathers 2` | Gathers per episode (default 4) |
| `--save-trajectories` | Write every tool call, result, the planted truth and the grade to `results/trajectories/` |
| `--tag name` | Suffix for the output file names, so runs do not overwrite each other |

## 5. Run an LLM agent

The agent talks to any OpenAI-compatible chat-completions endpoint that supports tool (function) calling.

### 5.1 Set your key

Use environment variables, and never put a key in a file inside the repository.

```bash
export OPENAI_API_KEY="sk-or-..."                       # your OpenRouter key
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

```powershell
$env:OPENAI_API_KEY = "sk-or-..."
$env:OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
```

### 5.2 Know your limits

Free OpenRouter models (ids ending in `:free`) are limited to 20 requests per minute. The daily limit is 50 requests until you have bought at least 10 credits, then 1,000. Check where you stand:

```bash
curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENAI_API_KEY"
```

The `free_model_daily_requests` field shows `used`, `limit` and `remaining` for the current UTC day.

Every agent turn is one request. As a rough guide:

- an episode with 1 gather takes about 10 to 25 requests;
- an episode with 4 gathers takes about 30 to 80.

With 50 requests a day, run 1-gather episodes. With 1,000 a day, a 9-episode validation run with 4 gathers fits comfortably.

### 5.3 Smoke test

Run one small episode first to check that the model can call tools at all:

```bash
python -m seismic_qc_gym.eval --agent llm --model nvidia/nemotron-3.5-lightning:free \
  --split val --n 1 --n-gathers 1 --min-interval 3.5 --save-trajectories --tag nemotron-smoke
```

`--min-interval 3.5` spaces requests at least 3.5 s apart, to stay under 20 per minute. The SDK also retries rate-limit errors with backoff, five times by default (`--max-retries`).

### 5.4 Full run

With the 1,000-request daily limit:

```bash
python -m seismic_qc_gym.eval --agent llm --model nvidia/nemotron-3.5-lightning:free \
  --split val --n 9 --min-interval 3.5 --save-trajectories --tag nemotron
python -m seismic_qc_gym.eval --agent llm --model nvidia/nemotron-3.5-lightning:free \
  --split test --n 9 --min-interval 3.5 --save-trajectories --tag nemotron
```

With 50 requests a day, add `--n-gathers 1 --n 2`, and use `--start` on later days to continue where you stopped (for example `--start 2`).

Results land in `results/llm_nemotron_val.json` and `results/llm_nemotron_test.json`. Then refresh the tables and the chart:

```bash
python scripts/summarise_results.py
python scripts/plot_results.py
```

The LLM rows appear next to the baselines. To compare fairly, compare against baselines run with the same `--n`, `--n-gathers` and `--start`.

### 5.5 Other endpoints

| Endpoint | Settings |
|---|---|
| OpenAI | Unset `OPENAI_BASE_URL`, set `OPENAI_API_KEY`, and pass an OpenAI model id |
| Local server (vLLM, llama.cpp server, Ollama's OpenAI-compatible API) | `--base-url http://localhost:8000/v1` (or your server's address), any non-empty `OPENAI_API_KEY`, and the model name the server exposes. The model must support tool calling. |

### 5.6 Vision mode

`--vision` adds the `render_gather` tool and attaches the gather images to the conversation, for vision-language models. Use a model that accepts image input.

## 6. Use the tools from an MCP client

Start the server over stdio:

```bash
python -m seismic_qc_gym.mcp_server      # or: qcgym-mcp
```

Most MCP clients take a JSON entry like this. Use the full path to the Python inside `.venv` if the client does not run in your activated environment:

```json
{
  "mcpServers": {
    "seismic-qc-gym": {
      "command": "/path/to/seismic-qc-gym/.venv/bin/python",
      "args": ["-m", "seismic_qc_gym.mcp_server"]
    }
  }
}
```

A session goes like this:

1. `new_episode` (split, index, difficulty);
2. `survey_summary` and `geometry_survey_check`;
3. the per-gather checks;
4. `flag_defect` and `apply_fix`;
5. `submit_report`, which returns the grade.

To browse the tools interactively, use the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector python -m seismic_qc_gym.mcp_server
```

## 7. Use the environment from Python

```python
from seismic_qc_gym import QCEnv, make_tasks

env = QCEnv(make_tasks("val", 1)[0])
print(env.observation()["instructions"])
attrs = env.call("trace_attributes", {"gather_id": 0}).content
env.call("flag_defect", {"gather_id": 0, "traces": [12], "defect_type": "reversed_polarity"})
env.call("apply_fix", {"gather_id": 0, "traces": [12], "fix": "flip_polarity"})
env.call("submit_report", {"summary": "done"})
print(env.result["reward"], env.result["detection"], env.result["repair"], env.result["damage"])
```

`env.trajectory` holds every call and result. `env.ep.truth` holds the planted defects; read it only for analysis after submitting.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `401` / authentication error | Check `OPENAI_API_KEY`, and `OPENAI_BASE_URL` for OpenRouter. |
| `429` / rate limit | Use `--min-interval 3.5`. Check the daily allowance with the `curl` command above; run fewer or 1-gather episodes; continue the next day with `--start`. |
| "No endpoints found that support tool use", or the model never calls a tool | The model or provider does not support function calling. Choose a model whose OpenRouter page lists tool support. |
| `404` model not found | Check the model id; free variants end in `:free`. |
| Episodes end with "agent stopped without submitting" | The model stopped calling tools. Inspect the saved trajectory; raise `--max-turns`, or try a stronger model. |
| Slow runs | Free models can be slow. The smoke test shows the time per episode. |
| `ModuleNotFoundError: mcp` or `openai` | Install the extra: `pip install -e ".[mcp]"` or `".[llm]"`. |
| MCP import error mentioning FastMCP | The server targets MCP SDK 2.x (`MCPServer`); upgrade with `pip install -U "mcp>=2"`. |
