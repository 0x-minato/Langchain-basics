# MCP servers

Tutorial folder. **Do not name this project `mcp`**, and **do not name a file `random.py`** — those names belong to Python’s stdlib. The dice demo is `dice.py`.

Use **venv312**. Inspector is pinned to **0.16.8** (Node 18). The version uses a **dot**: `0.16.8`, not a comma.

```powershell
D:\projects\AI\langchain_models\venv312\Scripts\Activate.ps1
cd D:\projects\AI\langchain_models\mcp_servers
fastmcp dev main.py
```

Or:

```powershell
npm run dev
```

Do **not** use `uv run` (that used to pull FastMCP 4 and break `tools/call` with `-32602`).

## Claude Desktop (Windows MSIX)

`uv run fastmcp install claude-desktop` reports success but writes the **wrong** file. Claude 4 reads:

`%LOCALAPPDATA%\Claude-Data\claude_pzs8sxrjxfjjc\claude_desktop_config.json`

Closing the window is not enough. Quit from the tray (or Task Manager: end every `claude.exe`), then reopen.

In Claude, open a **new chat**, click **+** on the composer, then **Connectors**. You should see **demo-server** with `roll_dice` and `add_numbers`. Older UI: Settings → Developer.

PowerShell has no `which`. Use `Get-Command uv` or `where.exe uv`.
