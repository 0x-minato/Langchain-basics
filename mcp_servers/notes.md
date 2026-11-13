# MCP (Model Context Protocol)

CampusX MCP playlist, then wired into the LangGraph chatbot. Previous notes:

- `tools-calling/` — `@tool` **inside** the same Python process as the model
- `agents/` — ReAct loop that **calls those functions**
- `langgraph/chatbot/` — `ToolNode` + DuckDuckGo + RAG; it **loads** this folder’s math server

This folder is not LangGraph. An MCP server is a **separate program** that **exposes tools** over a standard protocol. Claude Desktop, Cursor, Inspector, and `MultiServerMCPClient` are all **clients**. They speak MCP. They do not import `add()` as a Python function.

```text
old way (this repo, tools-calling/):
    chatbot.py  ──imports──►  @tool def multiply(...)   same process

MCP way (this folder):
    Claude / Inspector / LangGraph
            │  stdio (JSON-RPC)
            ▼
    python main.py   ← FastMCP process with @mcp.tool() functions
```

**One line:** MCP = USB-C for tools. Write the server once; any host that speaks MCP can plug in.

---

## How to restudy this folder

| Order | File | What it proves |
| --- | --- | --- |
| 1 | this `notes.md` | why MCP exists; what jobs ask for vs what these files are |
| 2 | `dice.py` | smallest FastMCP server (`roll_dice`, `add_numbers`) |
| 3 | `expenses.py` | tools that touch SQLite; `cur.description` is cursor metadata |
| 4 | `main.py` | math tools the Streamlit chatbot actually calls |
| 5 | `langgraph/chatbot/chatbot.py` (`MultiServerMCPClient`) | LangGraph as an MCP **client** |

Use **venv312**. Inspector is pinned to **0.16.8** (Node 18). Do **not** `uv run` — it used to pull FastMCP 4 and break `tools/call`. Do **not** name this project `mcp` or a file `random.py` (Python stdlib).

```powershell
D:\projects\AI\langchain_models\venv312\Scripts\Activate.ps1
cd D:\projects\AI\langchain_models\mcp_servers
fastmcp dev main.py
```

Or `npm run dev` (runs Inspector **0.16.8** against `venv312` + `main.py`).

---

## Why MCP is needed

### The problem jobs are actually hiring to solve

By 2025–2026, companies do not want “a chatbot that can search.” They want **agents that can use the company’s systems**: tickets, CRM, warehouse SQL, Slack, GitHub, internal HTTP APIs.

If every agent framework ships its **own** tool format, you rewrite the same “list invoices” wrapper for:

- Claude Desktop
- Cursor / VS Code Copilot
- ChatGPT / custom GPTs
- LangGraph / LangChain
- Some intern’s FastAPI prototype

That is the pre-MCP world. Your `@tool` in `tools-calling/currency_converter.py` **only exists inside that script**. Claude cannot see it. A second LangGraph app cannot see it unless you copy-paste.

MCP (Anthropic, late 2024; now an industry protocol) is the **shared plug**:

```text
one MCP server  (tools + schemas + auth to YOUR data)
        ▲
        │  same protocol
   ┌────┼────────────┬──────────────┐
Claude  Cursor   LangGraph     Inspector
Desktop          (this repo)
```

So the job is not “know FastMCP syntax.” The job is **make enterprise capabilities available to any agent host, safely, once**.

### What current AI / MCP job posts actually list

Titles you will see: *AI Integration Engineer*, *MCP & Connectors*, *Agentic Integration Engineer*, *AI Engineer – MCP Server Development*. Typical must-haves (2025–26 listings, condensed):

| They ask for | What it means in this folder | What it means at work |
| --- | --- | --- |
| MCP servers in **Python or TypeScript** | `FastMCP` + `@mcp.tool()` | production service, versioned, owned |
| **Tools, resources, prompts**, JSON schemas | you only have **tools** | resources = read-only context (files, DB rows); prompts = reusable templates |
| Connect agents to **APIs / DBs / SaaS** | SQLite `expenses.py`; HTTP is the same idea | Salesforce, Workday, GitHub, warehouse SQL |
| Typed **input schemas** | type hints on `main.py` (`a: float`) | Pydantic, validation, refuse garbage args |
| **Auth**: OAuth, secrets, least privilege | **not here** (local stdio, no login) | tokens, scoped tools, never ship keys in URLs |
| Read vs **write**, approval for side effects | `list_expenses` vs `add_expense` | HITL / `interrupt` before “buy / delete / email” |
| **stdio** and/or HTTP/SSE transports | these servers are **stdio** | remote MCP over HTTP for many users |
| LangChain / **LangGraph** / tool calling | `MultiServerMCPClient` in `chatbot.py` | same tools inside an agent graph |
| Tests, CI, observability | **not here** | pytest the tools; log every tool call |

This tutorial is the **left column**. Interviews will probe the **right column**. Do not pretend `dice.py` is a production connector.

### Why “just put `@tool` in the chatbot” is not enough for a job

| In-process `@tool` (LangChain) | MCP server |
| --- | --- |
| Lives in `chatbot.py` | Lives in **another process** (`python main.py`) |
| Only that app can call it | **Any** MCP client can call it |
| Ship a new bot → copy the functions | Ship a new bot → point config at the same server |
| Secrets sit next to the LLM code | Secrets sit in the **server**; the model host never sees the DB password |
| Hard to give to a non-Python host (Claude Desktop) | Claude Desktop is a first-class client |

LangGraph still **uses** MCP. It does not replace it. `chatbot.py` pulls math tools from this folder and feeds them to the same `ToolNode` as DuckDuckGo and `rag_tool`.

---

## Advantages (the ones that show up in interviews)

**1. Write once, consume many times.**  
`main.py` is usable from Inspector, Claude Desktop, and the Streamlit graph without rewriting `add` / `divide`.

**2. Process isolation.**  
A crash or `random.py` import bug in the server does not have to take down the whole IDE. The client sees “connection closed,” not a dead Kernel.

**3. Clear contract (JSON schema).**  
`@mcp.tool()` + type hints → the host knows the args. Same reason Cohere needed `query: str` on `rag_tool`. Jobs call this “tool interface design.”

**4. Least privilege is possible.**  
A host only gets the tools you registered. You can ship a **read-only** server (list expenses) and a separate **write** server (add expense) with different auth. In-process `@tool` lists tend to grow into “the bot can do everything.”

**5. Swap the model without swapping integrations.**  
MCP does not care if the host is Cohere, Claude, or GPT. The server is not a Cohere feature.

**6. Standard, so vendors ship servers for you.**  
GitHub, DBs, browser, Slack already have MCP servers. Job: **compose** them + build the 2–3 custom ones for your company. You do not reimplement GitHub.

**7. Fits agent graphs.**  
`langchain-mcp-adapters` turns MCP tools into LangChain `BaseTool`s. Your ReAct graph does not change. That is the line on LangGraph job posts: “MCP + LangGraph.”

What MCP is **not**:

- Not a better LLM
- Not RAG (RAG is retrieve-then-generate; MCP is **how tools are exposed**)
- Not a replacement for HITL (dangerous writes still need approval)
- Not magic auth — you still implement OAuth on the server

---

## The protocol, in this repo’s terms

### Host, client, server

```text
Host     = the app the human uses (Claude Desktop, Cursor, your Streamlit UI)
Client   = the MCP library inside that host (or MultiServerMCPClient)
Server   = dice.py / expenses.py / main.py  (FastMCP)
```

You talk to the **host**. The host asks the **server** to run a tool. The model never executes Python on the server; it **requests** `add(a=2, b=3)` the same way as `bind_tools`.

### Transports

| Transport | How it runs | This folder |
| --- | --- | --- |
| **stdio** | host **spawns** `python main.py`, talks on stdin/stdout | yes |
| SSE / HTTP | server already running, host connects to a URL | not used here |

stdio is why Windows paths and `sys.executable` matter. The host must launch **your** venv’s Python, not `python3` from PATH (Mac leftover) and not a different interpreter that lacks FastMCP.

### FastMCP vs the official `mcp` SDK

`pyproject.toml` pins `fastmcp>=2.3,<3` and `mcp>=1.6,<2`. FastMCP is a **thin wrapper**: `FastMCP("Math")` + `@mcp.tool()` instead of hand-writing JSON-RPC handlers. Jobs may say “MCP SDK” — same protocol, more boilerplate. Inspector and Claude speak the protocol, not FastMCP specifically.

`mcp.run()` under `if __name__ == "__main__"` starts the stdio loop. **Never** name the file `mcp.py` (shadows the package) or `random.py` (shadows stdlib; FastMCP import then dies).

---

## File: `dice.py` — first server

```python
mcp = FastMCP(name="Demo Server")

@mcp.tool()
def roll_dice(n_dice: int = 1) -> list[int]:
    ...
@mcp.tool()
def add_numbers(a: float, b: float) -> float:
    ...
```

This is the Claude Desktop “demo-server” from the old README: `roll_dice` + `add_numbers`. Same shape as LangChain `@tool`, different decorator, **different process**.

Proof you understand it: Inspector lists two tools; calling `add_numbers` with `{a: 1, b: 2}` returns `3` without that function living in the Inspector process.

---

## File: `expenses.py` — tools that own data

Server name: `ExpensesTracker`. SQLite file next to the script (`expenses.db`). `init_db()` runs `CREATE TABLE IF NOT EXISTS` on import so the first tool call does not crash.

| Tool | Side effect |
| --- | --- |
| `add_expense(...)` | **write** — `INSERT`, returns `{"status": "ok", "id": ...}` |
| `list_expenses()` | **read** — `SELECT`, returns a list of dicts |

This split (read vs write) is what job posts mean by “don’t give the agent a god tool.”

### `cur.description` is not a column

```python
cols = [d[0] for d in cur.description]
return [dict(zip(cols, r)) for r in cur.fetchall()]
```

SQLite cursors have **metadata** after a `SELECT`. `cur.description` is a sequence of tuples; `d[0]` is the **column name** (`id`, `date`, `amount`, …). `fetchall()` is rows as tuples. `zip(cols, r)` → `{"id": 1, "date": "...", ...}` so the model / Inspector sees JSON, not `(1, "2026-01-01", 12.5, ...)`.

There is no `description` column on `expenses`.

---

## File: `main.py` — what the chatbot actually calls

Server name: `Math`. Tools: `add`, `subtract`, `multiply`, `divide`, `power`, `square_root`, `percentage`. Typed `float` args. `divide` / `square_root` / `percentage` raise `ValueError` on bad input — that error becomes the tool result, not a silent `None`.

This is the file `langgraph/chatbot/chatbot.py` points at:

```python
MATH_SERVER = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "mcp_servers", "main.py")
)

client = MultiServerMCPClient({
    "arith": {
        "transport": "stdio",
        "command": sys.executable,   # this venv’s python.exe, never "python3"
        "args": [MATH_SERVER],
    }
})

mcp_tools = client.get_tools()   # LangChain BaseTool list
tools = [search_tool, *mcp_tools, rag_tool]
```

If MCP fails to start, `load_mcp_tools()` catches and returns `[]` — the bot still has search + RAG, just no math server.

**Wrong** (connection closed on Windows): `"command": "python3"` and a Mac path. **Right:** `sys.executable` + absolute `main.py`.

---

## How to run / inspect (keep these; they ate hours)

### Inspector (local)

From `mcp_servers`, with venv312:

```powershell
fastmcp dev main.py
```

Inspector **0.16.8** (the version uses a **dot**, not a comma). `package.json` pins it. `npm run dev` launches Inspector with `..\venv312\Scripts\python.exe main.py`.

Do **not** `uv run fastmcp dev` here — it pulled FastMCP 4 and `tools/call` returned `-32602`.

### Claude Desktop on this Windows machine (MSIX)

`uv run fastmcp install claude-desktop` can report success and still write the **wrong** config. Claude 4 (Store/MSIX) reads:

`%LOCALAPPDATA%\Claude-Data\claude_pzs8sxrjxfjjc\claude_desktop_config.json`

not `%APPDATA%\Claude\claude_desktop_config.json`.

Closing the window is not enough. Quit from the **tray** (or Task Manager: every `claude.exe`), then reopen. New chat → composer **+** → **Connectors**. You want the server name (`Demo Server` / `Math`) and its tools.

PowerShell has no `which`. Use `Get-Command uv` or `where.exe uv`.

Config `command` must be the **full path** to `venv312\Scripts\python.exe`, `args` the **full path** to `dice.py` or `main.py`.

---

## Map: MCP vs everything else you already learned

| Thing | Job | Lives in |
| --- | --- | --- |
| `@tool` / `bind_tools` | model **asks** for a function | same process (`tools-calling/`) |
| Agent / `ToolNode` | **loop** until no more calls | LangGraph / AgentExecutor |
| RAG | **read chunks**, not call APIs | FAISS + `rag_tool` |
| HITL `interrupt` | **human gates** a dangerous tool | `langgraph/HITL/` |
| **MCP** | **publish** tools so many hosts can call them | this folder |

A production agent often stacks all four: MCP server (capabilities) → LangGraph (orchestration) → RAG (docs) → HITL (writes).

---

## What to watch / traps

| Symptom | Likely cause |
| --- | --- |
| `circular import` / FastMCP dies on import | file named `random.py` or `mcp.py` |
| Inspector `tools/call` `-32602` | FastMCP 4 via `uv run` |
| Claude does not list the server | wrong JSON path (Roaming vs `Claude-Data\...`); didn’t fully quit Claude |
| LangGraph `MCP Connection closed` | `python3` / bad path; use `sys.executable` + abs `main.py` |
| Chatbot has no math tools | `load_mcp_tools()` swallowed the exception — read the terminal |
| `expenses.db` “lesson” | skip it — runtime data, not a script |

Skip: `mcp_servers/.venv/`, `node_modules/`, `uv.lock` as study material. `pyproject.toml` only matters as “FastMCP 2.x, not 4.”

---

## One-line map

```text
dice.py        = hello-world FastMCP (stdio tools)
expenses.py    = tools that own a SQLite DB (read vs write)
main.py        = math server the Streamlit graph actually loads
chatbot.py     = MCP client (MultiServerMCPClient → ToolNode)
```

**One line:** MCP is how you **expose** tools as a product; LangGraph is how you **orchestrate** them; jobs want both, plus auth and not demo-quality servers.
)
