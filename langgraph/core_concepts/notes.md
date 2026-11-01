# LangGraph core concepts

CampusX LangGraph playlist **video 4**. Previous note (`langchain_vs_langgraph/`) was *why* a graph. This one is the **vocabulary**: workflow styles, graph / node / edge, **state**, **reducers**, and the **Pregel** execution model.

You already built four of the five workflow styles in LCEL — without calling them that. LangGraph is the same shapes, plus **cycles** and a **shared state object**.

```text
LCEL:     each step’s output is the next step’s input (a baton)
LangGraph: every step reads/writes a shared notebook (state)
           edges say who runs next
           reducers say how two writes to the same field combine
```

---

## LLM workflow styles

These names come from how people **compose** LLM calls (Anthropic “building effective agents,” LangGraph tutorials). They are **patterns**, not classes. You pick one (or nest them) per product.

### 1. Prompt chaining (sequential)

Fixed pipe. Output of step 1 is the input of step 2. **You** decide the order at code time.

```text
A → B → C → answer
```

You have this as `sequencial-chain.py`: report → summarise. RAG naive path is also a chain: retrieve → stuff → generate.

| When it fits | When it breaks |
| --- | --- |
| Translate, then extract, then format | Step 2 should sometimes **skip** or **repeat** |
| One RAG retrieve-then-answer | Chunks are junk and you need to retrieve **again** |

LangGraph: `add_edge("a", "b")` then `add_edge("b", "c")`. A straight line is a valid graph. Overkill unless you later add a loop.

---

### 2. Routing

One **classifier** (LLM or rules) picks **which** chain runs.

```text
          ┌─ branch A (positive thanks)
input → classify
          └─ branch B (negative apology)
```

You have this as `conditional-chain.py` (`RunnableBranch` on sentiment). RAG version: “chit-chat → no retrieve; doc question → retrieve.”

LangGraph: **conditional edge**. A function reads state and returns the **name** of the next node (`"retrieve"` vs `"chat"` vs `END`).

---

### 3. Parallelization

Same input, several LLM (or tool) calls **at once**, then merge.

```text
        ┌─ notes ─┐
topic ──┤         ├─ merge
        └─ quiz ──┘
```

You have this as `parallel-chain.py` and the RAG `RunnableParallel({question, context})`.

LangGraph: from one node, **two edges** to two nodes with **no** dependency on each other. They run in the **same superstep** (see Pregel below). Merge is a downstream node that reads both fields from state.

Watch: if both nodes write the **same** state key, you **must** set a reducer or one write **clobbers** the other.

---

### 4. Orchestrator–workers

A **planner** LLM breaks the job into subtasks, **fans out** to workers (often the same node with different payloads), then a synthesizer merges.

```text
question → orchestrator (plan: [q1, q2, q3])
                │
                ├─ worker(q1)
                ├─ worker(q2)     ← parallel map
                └─ worker(q3)
                │
                ▼
           synthesizer → answer
```

Not a fixed parallel dict — the **number** of workers depends on the plan. Multi-query RAG and “research these 4 sub-questions” are this pattern.

LangGraph: `Send()` / map-reduce APIs. The orchestrator node **emits** one Send per worker. Workers are the same node function, different state slice.

You do **not** have this in LCEL. `RunnableParallel` needs keys known at graph-build time.

---

### 5. Evaluator–optimizer (generate / critique / loop)

Generate a draft, a **second** call grades it, if it fails **go back** and generate again (maybe with feedback in state).

```text
          ┌──────────────────────────┐
          ▼                          │
     generator → evaluator ─ fail ───┘
                      │
                    pass → END
```

This **is** Self-RAG / a writing loop / “supported by the docs?” CRAG is a cousin: retrieve → **grade docs** → (fail) rewrite query → retrieve.

LCEL cannot cycle. You faked it with `for _ in range(6)` in `currency_converter.py`. LangGraph: edge from evaluator **back** to generator, plus `recursion_limit` so it cannot spin forever (`max_iterations` on `AgentExecutor` was the same idea).

---

### Pattern cheat sheet

| Style | LCEL you already used | LangGraph primitive |
| --- | --- | --- |
| Chaining | `a \| b \| c` | normal edges, line |
| Routing | `RunnableBranch` | `add_conditional_edges` |
| Parallel | `RunnableParallel` | several nodes, same superstep |
| Orchestrator | (not really) | `Send` / map-reduce |
| Evaluator–optimizer | Python `while` | **cycle** + grader node |

Interview line: *chains are DAGs; agents and CRAG are cyclic graphs with a stop rule.*

---

## Why a graph?

A **graph** is vertices + edges. Here:

- **Node** = a unit of work (function): LLM call, retriever, tool, grader, rewriter, “format docs”
- **Edge** = control flow: “after this node, run that node”
- **State** = the only data nodes share (the notebook)

```text
  [retrieve] ──edge──► [grade] ──conditional──► [generate]
                            │
                            └── (docs bad) ──► [rewrite] ──► [retrieve]  ← cycle
```

**Why not a Python script with if/else?** You can. You already did. The graph gives you:

| Without a graph | With LangGraph |
| --- | --- |
| `messages.append` by hand | state + reducers |
| `if` / `while` mixed with LLM calls | **visible** topology (draw it) |
| crash → start over | checkpoint after each node |
| hard to run two tools in parallel | same superstep |
| HITL = you `input()` in the terminal | `interrupt` / resume `thread_id` |

**Importance of the graph:** it is the **spec** of the product. “Our RAG is retrieve → grade → generate, with a rewrite loop” is a picture, not a 200-line `for` loop.

---

## Nodes

A node is a **function** (or Runnable):

```python
def retrieve(state: State) -> dict:
    docs = retriever.invoke(state["question"])
    return {"documents": docs}   # partial update, not the whole State
```

Rules:

- **Input:** current `state` (or a subset, depending on API)
- **Output:** a **dict of fields to update**, not “the next node’s argument”
- Side effects (HTTP, FAISS) happen **inside** the node
- Keep nodes small: one job (retrieve **or** generate **or** grade)

`ToolNode`, `call_model`, your `get_context_text` lambda — those all become nodes.

If a node returns `{}` or `None`, it still ran; it just did not change state. Edges still fire.

---

## Edges

Edges answer **who is allowed to run after this node**.

| Kind | Meaning | Example |
| --- | --- | --- |
| **Normal** | always go to B | `retrieve → generate` |
| **Conditional** | function `(state) → next_node_name` | `tool_calls? tools : END` |
| **START / END** | entry and halt | `set_entry_point`, `END` |
| **Cycle** | edge to an earlier node | evaluator → generator |
| **Fan-out** | one node, several outgoing | parallel notes + quiz |
| **Fan-in** | several nodes, one next | merge after parallel |

Conditional edge functions should return **node names** (strings), not booleans. LangGraph looks up that name in the graph.

`START` is a virtual node. Your first real node is whatever you attach to START.

**Importance of edges:** they are the **only** way control moves. A node cannot `goto "generate"` by returning a string in the update dict (unless you use the Command/`Send` APIs on purpose). Default model: **return state updates; edges pick the successor.**

---

## State — the shared notebook

**State** is a typed snapshot of “everything this run knows right now.”

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class State(TypedDict):
    question: str                          # replace
    documents: Annotated[list, operator.add]  # append
    messages: Annotated[list[AnyMessage], add_messages]
    conversion_rate: float | None          # replace
```

Every node **reads** this object and **returns patches**. LangGraph **applies** patches with reducers, then the next node sees the new snapshot.

### Why state matters

1. **Decouples nodes.** Generate does not receive retriever output as a function argument. It reads `state["documents"]`. You can insert `grade` in between without changing `generate`’s signature.

2. **This is the memory of the graph.** LCEL’s “memory” is whatever you stuffed into the next prompt. Here `messages`, `documents`, `conversion_rate` **survive** across nodes and supersteps. Checkpointers **persist** this object so a chat can resume.

3. **Conditional edges inspect state.** `should_continue(state)` looks at `state["messages"][-1].tool_calls`. No global `state = {}` like `currency_converter.py`.

4. **Parallel writes need a contract.** Two nodes both returning `{"documents": [...]}` without a reducer = last writer wins (or an error, depending on version). The reducer **is** that contract.

5. **Debugging.** Print state after each node. That is the LangGraph equivalent of printing `tool_calls`.

**Not** the vector store. FAISS still holds chunks. State holds **this request’s** question, retrieved docs, messages, flags.

**Not** the model’s weights. If you drop `messages` from state, the graph “forgets” the conversation even if Cohere could have used it.

CampusX line: *state is the single source of truth for one graph execution (one thread).*

---

## Reducers — how an update becomes the new state

When a node returns `{"messages": [ai_msg]}`, LangGraph does **not** always do `state["messages"] = [ai_msg]` (that would **wipe** history).

Each field has a **reducer**: a function

```text
new_value = reducer(old_value, update)
```

If you **omit** a reducer, the default is **replace**:

```text
new_value = update
```

You declare reducers with `Annotated`:

```python
from typing import Annotated
import operator

class State(TypedDict):
    topic: str                              # replace
    score: Annotated[int, operator.add]     # add
    docs: Annotated[list, operator.add]     # concat lists
    messages: Annotated[list, add_messages] # smart append
    meta: Annotated[dict, merge_dicts]      # merge keys
```

### 1. Replace (default)

Last write wins. Use for **scalars** that should be overwritten: `question`, `conversion_rate`, `next_action`, `grade` (`"yes"` / `"no"`).

```text
state.question = "USD to INR"
node returns {"question": "capital of Antarctica"}
→ state.question is now "capital of Antarctica"
```

Two parallel nodes both replacing `question` → **undefined / last one**. Don’t do that. Give them different keys, or don’t write that field in parallel.

### 2. Add / append (`operator.add`)

`reducer(old, new)` is `old + new`.

| Type | Effect |
| --- | --- |
| `int` / `float` | **sum** (`score: 1` then `+ 2` → `3`) |
| `list` | **concatenate** (`[a]` + `[b]` → `[a, b]`) |
| `str` | concatenate strings (rarely what you want) |

```python
documents: Annotated[list, operator.add]
```

Retrieve node returns `{"documents": [doc1, doc2]}`. Later rewrite-retrieve returns `{"documents": [doc3]}`. State has `[doc1, doc2, doc3]` — **accumulated**. If you wanted **only** the second retrieval, use **replace** (no annotation) or return a command that overwrites.

Parallel retrieve A and retrieve B both `operator.add` on `documents` → **merged list**. That is the correct reducer for fan-in.

### 3. Merge (dicts / custom)

`operator.add` does not deep-merge dicts the way you think (`{a:1} + {b:2}` is a TypeError). You write:

```python
def merge_dicts(left: dict, right: dict) -> dict:
    return {**left, **right}   # right keys overwrite

class State(TypedDict):
    flags: Annotated[dict, merge_dicts]
```

Node A: `{"flags": {"retrieved": True}}`  
Node B (parallel): `{"flags": {"graded": True}}`  
→ `{"retrieved": True, "graded": True}`

If both set the **same** key, right-hand update wins inside your merge function. You can write a smarter merge (lists inside dicts, etc.).

### 4. `add_messages` (special list reducer)

Not the same as `operator.add` on a list.

- Appends new `HumanMessage` / `AIMessage` / `ToolMessage`
- If an incoming message has an **id** that already exists, it **replaces** that message (useful for updates)
- Knows LangChain message objects

```python
from langgraph.graph.message import add_messages

messages: Annotated[list, add_messages]
```

This is why the agent loop **stops wiping chat history**. Returning `{"messages": [ai_msg]}` **adds** a turn; it does not set `messages = [ai_msg]`.

Use `add_messages` for chat. Use `operator.add` for bags of docs/strings. Use replace for “the current draft.”

### Mental picture

```text
state.messages = [Human]
node: return {"messages": [AI]}

replace:     messages = [AI]           # history gone — bug
add list:    messages = [Human, AI]    # ok for raw lists
add_messages: same, plus id-aware upsert
```

```text
state.conversion_rate = None
node: return {"conversion_rate": 95.3}

replace:  95.3                         # what you want
add:      error or None+95.3 nonsense  # don’t use add on a rate
```

**Rule:** ask “if two nodes write this key, should we **keep both**, **sum**, **merge keys**, or **keep the latest**?” That answer **is** the reducer.

---

## Execution model (Pregel)

LangGraph’s runtime is inspired by **Google Pregel** (2010): *Pregel: A System for Large-Scale Graph Processing*. Pregel runs huge graphs (PageRank, shortest path) with a simple idea: **think like a vertex**, in **lock-step rounds**.

### Pregel in one paragraph

Time is split into **supersteps**.

1. Every **active vertex** runs a function: read incoming messages, update its value, send messages to neighbours.
2. A **barrier**: no vertex starts the next round until **all** in this round finish (Bulk Synchronous Parallel).
3. Vertices can **vote to halt**. The job ends when all have halted **and** no messages are in flight.
4. A message to a halted vertex **wakes** it.

You do not write a global `for each edge`. You write **per vertex**. Parallelism is “all ready vertices in this superstep.”

### How LangGraph copies that

| Pregel | LangGraph |
| --- | --- |
| Vertex | **Node** |
| Vertex value | slice of **state** (channels) |
| Messages to neighbours | **state updates** flowing along **edges** |
| Superstep | one round: run all **scheduled** nodes (can be parallel) |
| Barrier | apply **reducers**, then schedule the next set |
| Vote to halt | no remaining nodes → hit **END** |
| Wake | an edge (or `Send`) schedules a node again |

```text
superstep 0:  START → schedule [retrieve]
superstep 1:  run retrieve   → updates documents
              edges → schedule [grade]
superstep 2:  run grade      → updates grade="bad"
              conditional → schedule [rewrite]
superstep 3:  run rewrite    → updates question
              edge → schedule [retrieve]     ← cycle, like waking a vertex
superstep 4:  run retrieve   → ...
...
superstep n:  generate → END  (no one scheduled)
```

**Parallel superstep** (notes + quiz):

```text
superstep 1:  run [notes_node, quiz_node] together
              barrier: reducers merge both writes
superstep 2:  run merge_node
```

That is why two nodes writing `documents` without a reducer explode: at the **barrier**, Pregel must **combine** messages to the same channel. The reducer **is** that combine function.

**Orchestrator `Send`:** Pregel lets a vertex send to many others. `Send("worker", {"subq": q1})` schedules extra worker instances in the **next** superstep — dynamic fan-out, not a static edge list.

### Why this model (not a thread soup)

- **Deterministic rounds** — easier to checkpoint: save state at the barrier, resume later (`thread_id`).
- **Parallel where safe** — independent nodes in one superstep.
- **Cycles are normal** — a node running again is a vertex waking, not a stack overflow of recursive Python (until `recursion_limit`).
- **Same API for 2 nodes or 50** — you still only write node functions.

LangGraph is **not** Google-scale PageRank. It borrows the **mental model**: nodes are vertices; time is supersteps; reducers merge messages to the same state key.

`recursion_limit` ≈ max supersteps. `currency_converter` `range(6)` was a DIY superstep cap.

---

## Tiny walk (currency, as a graph)

State:

```text
messages          add_messages
conversion_rate   replace
```

```text
START → agent → (tool_calls?)
              ├─ yes → tools → agent   ← cycle (Pregel wake)
              └─ no  → END
```

- Agent node: LLM with bound tools; returns `{"messages": [AIMessage]}`.
- Tools node: runs `get_conversion_factor` or `convert`; returns `{"messages": [ToolMessage], "conversion_rate": 95.3}` on the first tool.
- Reducer **appends** messages; **replaces** rate.
- Next agent turn sees both the tool result **and** the rate on state — no Python global.

---

## Mental model

```text
Graph   = map of the kitchen
Node    = a station (grill, expo)
Edge    = ticket routing
State   = the ticket (shared)
Reducer = how two cooks write on the same line
          replace = overwrite the special
          add     = add another side dish
          merge   = fill more boxes on the ticket
Pregel  = service in waves: all current tickets move,
          then the pass, then the next wave
```

Workflow styles are **shapes** you draw with those primitives:

| Style | Draw |
| --- | --- |
| Chain | line |
| Route | diamond |
| Parallel | fork then join |
| Orchestrator | fork with **N decided at runtime** |
| Evaluator–optimizer | loop with a grade |

---

## Cheat sheet

| Term | Meaning |
| --- | --- |
| State | typed notebook for **one** run / thread |
| Node | `state → {field: update}` |
| Edge | next node(s); may be conditional or a cycle |
| Replace | default reducer; latest value |
| `operator.add` | sum numbers or concat lists |
| `add_messages` | append chat messages (id-aware) |
| Merge | custom reducer, usually dict `{**a, **b}` |
| Superstep | Pregel round: run ready nodes, reduce, reschedule |
| `Send` | dynamic “message to a node” (orchestrator) |
| END | no vertices left to run |

---

## Map onto this repo

| Pattern | You already | LangGraph next |
| --- | --- | --- |
| Chain | `sequencial-chain.py`, `yt-chatbot` | line of nodes |
| Route | `conditional-chain.py` | conditional edge |
| Parallel | `parallel-chain.py` | two nodes, one superstep |
| Loop / eval–opt | `currency_converter.py` `for` | cycle + replace/add_messages |
| Orchestrator | not yet | `Send` after LangGraph basics |
| State + reducers | globals / `messages.append` | `TypedDict` + `Annotated` |
