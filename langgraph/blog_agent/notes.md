# Blog agent

CampusX LangGraph — **orchestrator–workers**, with a **router**, optional **web research**, and a **reducer subgraph** that merges sections and may add images.

Previous notes:

- `workflows/` — sequential / parallel / conditional / iterative **shapes**
- `core_concepts/` — `Send` / map-reduce, `operator.add`, Pregel barrier
- `sub-graphs/` — compiled graph as a node (`add_node("reducer", subgraph)`)
- `tools-calling/` — the model emits a schema (`with_structured_output`)

This is not a chatbot. One `invoke` in, one markdown file out. The graph **plans** a post, **fans out** one worker per section, **reduces** the pieces, then optionally **draws diagrams**.

```text
START → router ─┬─ (needs research) → research → orchestrator
                └─ (closed book) ──────────────► orchestrator
                                                      │
                                              Send() × N tasks
                                                      ▼
                                                   worker  (parallel)
                                                      │
                                                   reducer subgraph
                                     merge → decide_images → generate_and_place
                                                      │
                                                     END
```

**One line:** router picks closed/hybrid/open book → orchestrator writes a `Plan` → `Send` one worker per task → reducer concatenates + maybe Gemini images.

---

## How to restudy this folder

One file. Read it top to bottom: schemas → router → research → plan → `Send` → workers → subgraph.

| Order | Piece | What it proves |
| --- | --- | --- |
| 1 | `State` + Pydantic models | shared notebook + structured LLM output |
| 2 | `router_node` / `route_next` | conditional: research or skip |
| 3 | `research_node` | Tavily → `EvidenceItem`s, recency filter |
| 4 | `orchestrator_node` | one `Plan` with 5–9 `Task`s |
| 5 | `fanout` + `Send` | **map**: one worker payload per task |
| 6 | `worker_node` | writes one `##` section; `operator.add` joins |
| 7 | `reducer_subgraph` | **reduce**: merge → image plan → write `.md` |

Needs `OPENAI_API_KEY` (`gpt-4.1-mini`). Optional: `TAVILY_API_KEY` (research), `GOOGLE_API_KEY` (Gemini images). Run from **repo root** with `venv312`. This file is a script that **compiles** `app`; it does not `invoke` until you do.

```python
from datetime import date
from langgraph.blog_agent.blog_agent import app   # or paste/run in a notebook

out = app.invoke({
    "topic": "Attention in transformers, for beginners",
    "as_of": date.today().isoformat(),
})
print(out["final"][:500])
```

Unlike `langgraph/chatbot/`, this graph has **no checkpointer**. One shot. Restart = start over.

This file still uses **ChatOpenAI**, not Cohere. Structured plans + long sections were written against `gpt-4.1-mini`. Do not silently swap models unless you re-test the schemas.

---

## The one idea: map–reduce over a blog outline

`workflows/parallel.py` fans out **fixed** nodes (`notes` + `quiz`) known at compile time. Here the number of sections is **unknown until the orchestrator returns**. That is what `Send` is for.

```text
orchestrator returns Plan(tasks=[t1, t2, ..., tN])
fanout emits Send("worker", {task: t1, ...}), Send("worker", {task: t2, ...}), ...
Pregel runs N copies of worker in one superstep
each returns {"sections": [(id, markdown)]}
operator.add concatenates those lists
barrier: all workers done → reducer
```

Interview line: *orchestrator–workers = planner LLM + `Send` + a reducer.* `RunnableParallel` cannot do this; the keys are not known when you `compile()`.

The image pipeline is a **shared-state subgraph** (same `State` as the parent), like `subgraph_shares.ipynb`, not the mapper style in `sub_graph.ipynb`.

---

## State (the notebook)

```python
class State(TypedDict):
    topic: str
    mode: str                          # closed_book | hybrid | open_book
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    as_of: str                         # ISO date you pass in
    recency_days: int                  # 7 / 45 / 3650 from the router
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str
```

| Field | Who writes it | Why |
| --- | --- | --- |
| `topic`, `as_of` | you, on `invoke` | the only required inputs |
| `mode`, `queries`, `recency_days` | router | whether to search, how stale is OK |
| `evidence` | research | URLs workers may cite |
| `plan` | orchestrator | outline + per-section flags |
| `sections` | **every** worker | must have a **reducer** (`operator.add`) or parallel writes clobber |
| `merged_md` → `final` | reducer subgraph | one markdown document |

`sections` is a list of `(task_id, markdown)` so `merge_content` can `sorted(..., key=lambda x: x[0])` and restore outline order. Workers finishing in random order is expected.

---

## Schemas (what the LLMs are allowed to emit)

All routing / planning / image decisions use `llm.with_structured_output(...)`. Same idea as `chatModels/structured_bot.py` and LTM `MemoryDecision`.

| Model | Job |
| --- | --- |
| `RouterDecision` | `needs_research`, `mode`, `queries` |
| `EvidencePack` / `EvidenceItem` | clean search hits (url, title, date, snippet) |
| `Plan` / `Task` | blog outline; each task has bullets, `target_words`, flags |
| `GlobalImagePlan` / `ImageSpec` | `[[IMAGE_1]]` placeholders + prompts |

`Task` flags the workers actually obey:

| Flag | Meaning |
| --- | --- |
| `requires_research` | this section should use evidence |
| `requires_citations` | hybrid: attach `[Source](url)` |
| `requires_code` | include a minimal snippet |

---

## Router — closed / hybrid / open book

`router_node` does **not** write the blog. It only answers: “do we need the web?”

| Mode | `needs_research` | `recency_days` | Typical topic |
| --- | --- | --- | --- |
| `closed_book` | false | 3650 (ignore dates) | “what is a transformer?” |
| `hybrid` | true | 45 | “LangGraph checkpointing with current APIs” |
| `open_book` | true | 7 | “this week in LLM news” |

```python
def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"
```

Conditional edge map: `{"research": "research", "orchestrator": "orchestrator"}`. Closed-book skips Tavily entirely (cheaper, no `TAVILY_API_KEY` needed).

---

## Research — Tavily, then a second LLM pass

```text
queries from router (max 10)
    → TavilySearchResults per query
    → llm.with_structured_output(EvidencePack)
    → dedupe by URL
    → if open_book: drop items older than as_of - recency_days
```

`_tavily_search` returns `[]` if the key is missing or the import fails. Then `research_node` returns `{"evidence": []}`. The graph **still continues**. Open-book with empty evidence should produce a plan that admits “we don’t have sources,” not invented headlines. That is in `ORCH_SYSTEM`.

`published_at` is normalized to `YYYY-MM-DD` or left null. The recency filter **drops** items with no parseable date in open-book mode (`d := _iso_to_date(...)` must be truthy). Hybrid does not apply that cutoff.

---

## Orchestrator — one `Plan`

Structured `Plan`: title, audience, tone, `blog_kind`, constraints, `tasks`.

Open-book **forces** `blog_kind="news_roundup"` after the LLM returns, so workers get the “do not turn this into a tutorial” guard even if the model forgot.

Evidence is truncated to 16 items in the prompt. The orchestrator may mark some tasks `requires_research=True` / `requires_citations=True` in hybrid mode.

---

## Fanout — `Send` is not `add_edge`

```python
def fanout(state: State):
    return [
        Send("worker", { "task": task.model_dump(), "topic": ..., "evidence": ... })
        for task in state["plan"].tasks
    ]
```

`Send(node_name, payload)` schedules **that** node with **that** dict as its input. The worker does **not** receive the full parent `State`. It receives a **slice**: one task + copies of topic/mode/plan/evidence.

```python
g.add_conditional_edges("orchestrator", fanout, ["worker"])
```

The third argument `["worker"]` is the list of nodes `fanout` is allowed to target. After the superstep, `g.add_edge("worker", "reducer")` runs **once all workers have returned** (Pregel barrier), not after the first worker.

If you used a normal edge `orchestrator → worker`, you would run **one** worker on the whole state. `Send` is the difference between “write section 3” and “write the entire blog in one call.”

---

## Worker — one section, then `operator.add`

Input is the `Send` payload, not `State`. It rebuilds `Task` / `Plan` / `EvidenceItem` from dicts, calls the LLM as plain markdown (not structured), returns:

```python
return {"sections": [(task.id, section_md)]}
```

Because `sections` is `Annotated[..., operator.add]`, N workers → N tuples appended. Without the annotation, the last worker **wins** and you lose the other sections (the parallel-graph bug from `workflows/`).

Open-book workers are told: no event/company/funding claim unless it is in the provided URLs; otherwise write “Not found in provided sources.” Citations are markdown links, not a vector store.

---

## Reducer subgraph (shared `State`)

```text
START → merge_content → decide_images → generate_and_place_images → END
```

Compiled as `reducer_subgraph` and attached with `g.add_node("reducer", reducer_subgraph)` — same pattern as `subgraph_shares.ipynb`. Keys (`merged_md`, `image_specs`, `final`) already live on parent `State`, so there is no mapper.

### `merge_content`

Sort `sections` by `task.id`, join with blank lines, prefix `# {plan.blog_title}`.

### `decide_images`

Structured `GlobalImagePlan`: rewrite markdown with `[[IMAGE_1]]` … (max 3) plus `ImageSpec` (filename, alt, caption, prompt, size). If no diagrams help, `images=[]` and placeholders equal the input.

### `generate_and_place_images`

- No specs → write `{slug}.md` from merged markdown, `final = md`.
- Else `images/` directory; Gemini `gemini-2.5-flash-image` for each spec if the file does not already exist.
- Success: replace placeholder with `![alt](images/file)\n*caption*`.
- Failure (no `GOOGLE_API_KEY`, safety, quota): replace with a blockquote that still contains the prompt so the `.md` is usable.

`_safe_slug` turns the title into a filename (`attention_in_transformers.md`). Files are written **cwd-relative**, so run from a folder you are willing to clutter, or `cd` into `langgraph/blog_agent/`.

---

## vs everything else in this repo

| | This graph | You already have |
| --- | --- | --- |
| Shape | router + optional retrieve + **map–reduce** + subgraph | `iterative.ipynb` is a **cycle**; this is a **tree that fans out once** |
| Parallel | N workers chosen at **runtime** (`Send`) | `parallel.py` / essay judges: N fixed at compile time |
| Subgraph | reducer is a 3-node graph on **shared** State | `subgraph_shares.ipynb` |
| Tools | Tavily + Gemini as **Python in a node**, not `bind_tools` | chatbot `ToolNode` + MCP |
| Memory | none | `memory/` STM/LTM |
| HITL | none — images generate without approval | `HITL/` `interrupt` before a side effect |

Generating images is a **side effect** (writes bytes, may cost money). HITL would sit before `generate_and_place_images`. Not wired here.

---

## What to watch

| Symptom | Likely cause |
| --- | --- |
| `plan` is None in merge | orchestrator failed schema / you invoked without going through it |
| Only one section in the blog | forgot `operator.add` on `sections`, or `Send` not used |
| Sections in the wrong order | merge must `sort` by `task.id`; workers are unordered |
| Empty evidence / weak news post | no `TAVILY_API_KEY`, or open-book dates all `null` so the cutoff drops them |
| `.md` has `IMAGE GENERATION FAILED` | missing `GOOGLE_API_KEY` or Gemini blocked; doc still written |
| OpenAI errors | this file is **not** Cohere; needs `OPENAI_API_KEY` |
| `app` at the bottom does nothing | it compiles; you still `app.invoke({...})` |
| Files appear in repo root | `Path(filename).write_text` uses the process cwd |

There is no `if __name__ == "__main__"` invoke. Treat `blog_agent.py` like a notebook’s last `app` cell.

---

## One-line map

```text
router         = do we search?
research       = Tavily → EvidenceItem list
orchestrator   = Plan with N Task
fanout / Send  = N parallel workers
worker         = one ## section → sections += [(id, md)]
reducer        = merge markdown → maybe [[IMAGE_k]] → Gemini → .md on disk
```

**One line:** blog agent = route → (research) → plan → `Send` workers → reducer subgraph; `operator.add` is what makes the fan-in work.
)
