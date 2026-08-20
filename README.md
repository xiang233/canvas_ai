# Canvas Student Agent

[![CI](https://github.com/xiang233/canvas_ai/actions/workflows/ci.yml/badge.svg?branch=samantha)](https://github.com/xiang233/canvas_ai/actions/workflows/ci.yml)

A ReAct agent for Canvas LMS, tailored for student accounts: ask about
grades, assignments, and deadlines in natural language, or ask what your
lecture slides and course papers actually say. Content questions are
answered by retrieval over OpenAI Vector Stores built from your own
course files.

Runner-Up (top 5 teams) at the WashU AI Hackathon, October 2025.
Development since then has focused on making the agent measurable and
trustworthy: a trajectory-level evaluation harness, a RAG ablation study
with an independent judge, multi-turn memory with bounded context
growth, and correct pagination and retry behavior pinned by tests.

**Cloudflare version**: [GitHub](https://github.com/xiang233/cf_ai_canvas_agent/)

---

## Key capabilities

- 23 read-only Canvas/RAG tools: courses, assignments, grades, files,
  discussions, announcements, quizzes, calendar, and semantic search
  over uploaded course materials. Write operations are excluded at
  build time by a side-effect whitelist, with a CI assertion guarding
  the invariant.
- Conversational CLI (`canvas_chat.py`) with persistent multi-turn
  memory, so follow-ups like "what about the second one?" resolve
  against earlier turns.
- Bulk file sync (`file_index_downloader.py`) with optional OpenAI
  Vector Store upload, which is what powers content-question RAG.
- HTTP + WebSocket API (`api_server.py`) with per-session agents, so
  concurrent users get isolated conversations; interactive docs at `/docs`.
- WebSocket bridge (`ws_server.py`) for machine-to-machine use (see
  the status note in that section).
- MCP server (`mcp_server.py`) exposing the same 23 read-only tools to
  Claude Desktop and other MCP clients, behind the same side-effect
  whitelist.

## Engineering highlights

Every claim below is backed by a measurement or a test in this repo.

- **Trajectory-level evaluation** (`evals/`): cases assert on what the
  agent did (tools called, step counts, hallucination probes), not just
  the final text. Ground truth is computed live from the Canvas API at
  run time, so nothing about real enrollment is committed to the repo.
- **RAG ablation study** (`evals/rag_*`): factorial design over
  knowledge-base access and question wording, judged by a different
  model family (Claude judging GPT output, to avoid self-preference)
  with a 5-class attribution rubric that separates fabrication from
  false refusal. Without the knowledge base the agent got 0/18 content
  questions right (17 were refusals); with it, 17/35 fully correct
  under a strict judge and 25/35 answered from actual course material,
  with 3 fabrication cases. That asymmetry is why fabrication and
  refusal are tracked as separate metrics instead of one score.
- **Bounded multi-turn memory**: `AGENT_MEMORY_KEEP_RECENT=N` keeps the
  most recent N steps verbatim and compacts older ones. On an 8-turn
  session this cut total input tokens from 438K to 166K (a 62%
  reduction) with cross-turn recall verified intact.
- **Real pagination**: all 13 list endpoints follow Canvas `Link`
  headers to the end instead of silently truncating at one page. A
  runaway guard appends an explicit `_truncated` marker rather than
  truncating silently again.
- **Retry with backoff**: exponential backoff plus jitter,
  `Retry-After` support, and a retry whitelist (429/5xx retry, 4xx
  fail fast), pinned by tests against a scripted local HTTP server.
- **Measured non-features**: periodic re-planning is built in but off
  by default, because on short queries it cost roughly +56% latency and
  +29% input tokens with no correctness change (see
  `configs/canvas_agent_config.py`).

## Repository layout

    canvas_ai/
    ├── canvas_chat.py            # Interactive console entry point
    ├── file_index_downloader.py  # Bulk download + vector store upload
    ├── ws_server.py              # WebSocket bridge (see status note)
    ├── configs/
    │   └── canvas_agent_config.py
    ├── src/
    │   ├── agent/                # Agent builder and prompts
    │   ├── tools/                # Canvas tools, retry, pagination
    │   ├── models/               # Azure/OpenAI provider with auto fallback
    │   └── mcp/                  # Dormant tool library (see its README)
    ├── evals/                    # Trajectory harness + RAG ablation
    ├── tests/                    # Assertable tests, no API keys needed
    └── examples/                 # Guided demos

## Quickstart

Requires Python 3.11+, a Canvas student API token, and an OpenAI API key.

    pip install -r requirements.txt

Create a `.env` (template in `env_example.txt`):

    LLM_PROVIDER=auto            # azure | openai | auto (fall back by availability)
    OPENAI_API_KEY=sk-...
    CANVAS_URL=https://yourschool.instructure.com/
    CANVAS_ACCESS_TOKEN=...

> **Generating a Canvas access token**: Canvas → Account → Settings →
> Approved Integrations → **+ New Access Token**.

Then:

    python canvas_chat.py

In-CLI commands: `help`, `status`, `examples`, `clear` (also clears
conversation memory), `exit`.

To enable content questions ("what does lecture 3 say about..."), build
the knowledge base first:

    python file_index_downloader.py            # download + vector store upload
    python file_index_downloader.py --upload-only

## HTTP API

    python api_server.py        # 127.0.0.1:8000, docs at /docs

Each session owns its own agent instance, so concurrent users never
share conversation state; sessions carry multi-turn context (first
message resets, later ones keep it) and are evicted after
`API_SESSION_TTL` seconds of inactivity, capped at `API_MAX_SESSIONS`.
`POST /api/chat` returns a `session_id` to pass back on follow-ups;
`/ws/chat` speaks the same session model. `ALLOWED_ORIGINS` configures
CORS (no wildcard).

This is a different concurrency model from `ws_server.py`, which shares
one process-wide agent and serializes with a single-connection lock.

## MCP server

The same tool set is available over the Model Context Protocol, so
Claude Desktop or Claude Code can query your Canvas directly:

    python mcp_server.py    # stdio transport

Claude Desktop config (`claude_desktop_config.json`):

    {
      "mcpServers": {
        "canvas": {
          "command": "python",
          "args": ["/absolute/path/to/mcp_server.py"],
          "env": {"CANVAS_URL": "...", "CANVAS_ACCESS_TOKEN": "...", "OPENAI_API_KEY": "..."}
        }
      }
    }

The server registers exactly the tools the agent uses, after the
read-only filter; a CI assertion keeps the two in sync.

## Configuration knobs

| Env var | Default | Effect |
|---|---|---|
| `LLM_PROVIDER` | `auto` | `azure` / `openai`, or auto-fallback |
| `AGENT_MEMORY_KEEP_RECENT` | off | Keep last N steps verbatim, compact older ones |
| `AGENT_PLANNING_INTERVAL` | off | Re-plan every N steps (measured not worth it for short queries) |
| `CANVAS_MAX_RETRIES` | 3 | Retry budget for 429/5xx |
| `CANVAS_TIMEOUT` | 30 | Per-request timeout (s) |

## Tests and evals

    python -m tests.test_canvas_retry          # 6 cases: retry/backoff against scripted server
    python -m tests.test_canvas_pagination     # 4 cases: Link-header paging, partial failure
    python -m tests.test_eval_checks           # 16 cases: assertion-primitive boundaries
    python -m tests.test_prompt_consistency    # 8 cases: prompt/config/MCP consistency
    python -m tests.test_api_sessions          # 9 cases: per-session agent isolation

None of these need API keys or network. They run as a hard gate on
every pull request (see `.github/workflows/ci.yml`, Python 3.11 and
3.13). LLM behavior evals run via a manual-dispatch workflow
(`.github/workflows/eval.yml`); the reasoning for manual-only triggering
is documented at the top of that file. The eval suites (`evals/`) hit
real Canvas and LLM APIs and cost money; see the docstrings in
`evals/harness.py` and `evals/rag_experiment.py`.

Two eval-design decisions worth reading in source: ground truth is
computed at runtime so real enrollment never lands in git, and the
`NoError` / `NoAbstention` split exists because a single "no failure
phrases" check was rewarding the agent for inventing answers instead of
honestly abstaining, the opposite of what the RAG judge's
false-refusal metric measures.

## WebSocket service: status note

`ws_server.py` serves chat and download over WebSocket (WSS, Origin
allowlist, single-connection lock). **Known issue inherited from the
original codebase: the password/TOTP handshake described in early docs
was never wired into the message loop.** `authenticate_payload()`
exists but is never called, so connections are effectively
unauthenticated. Fine for localhost use; do not expose this to the
internet as-is. The session-store functions (`create_session` etc.)
are likewise dead code.

    python ws_server.py    # then send {"type": "chat", "query": "..."}

## Project history

The project began as a team entry at the WashU AI Hackathon in October
2025, where it placed Runner-Up (top 5 teams), built on a
smolagents-derived agent framework (`src/base/`, `src/agent/`) with the
original Canvas tool set, CLI and WebSocket entry points, and the file
downloader. `src/mcp/` (51 GAIA benchmark tools) is dormant heritage
from that framework and is not imported by the runtime; see
`src/mcp/README.md`.

Post-hackathon development in this repository: the evaluation harness
and RAG ablation, retry and pagination in the tool base class,
multi-turn memory and its compression, the planning-interval
measurement, the prompt cleanup (few-shot examples now use real
tools), and all of `tests/`.

## Troubleshooting

- **Canvas 401**: regenerate the access token, confirm `CANVAS_URL`
  points at your school's subdomain.
- **Vector store upload errors**: `openai>=1.20.0` and Assistants v2
  access.
- **Dependencies**: `pip install -r requirements.txt`.
