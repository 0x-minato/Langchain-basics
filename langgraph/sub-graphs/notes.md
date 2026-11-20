# Subgraphs

CampusX LangGraph playlist — **nested graphs**. Previous notes:

- `workflows/` — one graph, one `State`, nodes and edges
- `persistence/` — checkpoints on **one** compiled graph
- `HITL/` — pause **inside** a node
- `mcp_servers/` — tools that live in **another process**

This folder is still a **line** (`answer → translate`). What is new: the translate step is **its own compiled graph**, not only a function.

```text
a node is a function:     state → dict of updates
a subgraph is a graph:    START → nodes → END
                          that you either invoke() or add_node()
```

**One line:** a subgraph is a reusable graph you nest inside a parent. Same four pieces (State → nodes → edges → compile), just **inside** another graph.

---

## How to restudy this folder

Both notebooks do the **same product**: English answer, then Hindi translation. They differ in **how the inner graph is attached**.

| Order | File | What it proves |
| --- | --- | --- |
| 1 | `sub_graph.ipynb` | **Different** state schemas. Parent node calls `subgraph.invoke(...)` and **maps** keys by hand. |
| 2 | `subgraph_shares.ipynb` | **Same** `ParentState`. Compiled subgraph is `add_node("translate", subgraph)`. Keys overlap; no mapper. |

Needs `CO_API_KEY` / `COHERE_API_KEY`. Run with `venv312`. After compile, look at the mermaid: first file’s `translate` is a **black-box function**; second file’s `translate` **is** the inner graph.

---

## Why nest a graph at all?

You already know sequential nodes (`blog.py`: outline → write → polish). You do **not** need a subgraph for two LLM calls.

Use a subgraph when the inner piece is a **real pipeline** you want to:

| Reason | Example |
| --- | --- |
| **Reuse** | Hindi translator used by a chatbot, a blog graph, and a PDF graph |
| **Hide internals** | Parent only cares about “English in, Hindi out,” not the inner prompt |
| **Own it as a unit** | Translation team compiles `subgraph`; parent team only `invoke`s or `add_node`s it |
| **Keep parent state clean** | Inner graph can have keys (`input_text`, `translated_text`) the parent never stores |

If the inner thing is **one function with no inner edges**, a node is enough. Subgraphs pay off when the inner graph later grows (detect language → translate → proofread) without rewriting every parent.

This is the same instinct as MCP: **publish a capability once**. MCP publishes **tools** to other processes. A subgraph publishes a **graph** to other graphs **in the same Python app**.

---

## Two ways to nest (the whole folder)

LangGraph has two composition styles. You have **one notebook each**.

```text
sub_graph.ipynb                         subgraph_shares.ipynb
────────────────                        ─────────────────────
ParentState ≠ SubState                  both use ParentState

def translate_answer(state):            parent.add_node("translate", subgraph)
    result = subgraph.invoke({
        "input_text": state["answer_eng"]
    })
    return {"answer_hin": result["translated_text"]}

mapper node is a normal function        compiled graph IS the node
you pick which keys cross the           overlapping keys are shared
boundary                                automatically
```

| | `sub_graph.ipynb` | `subgraph_shares.ipynb` |
| --- | --- | --- |
| How it runs | Python `subgraph.invoke` **inside** a parent node | Parent Pregel **steps into** the compiled subgraph |
| Schemas | **must** differ or you still map | **must** share the keys the inner graph reads/writes |
| Adapter | you write it | none |
| Parent mermaid | `translate` looks like any other node | `translate` can show the inner `translate_text` |
| When the inner graph grows | mapper stays; inner edges stay private | parent still has one node named `translate`; inner edges live in the subgraph |
| Checkpoints / streaming of inner nodes | parent sees **one** node (`translate_answer`). Inner steps are a nested `invoke` | parent can checkpoint **inner** nodes (`checkpoint_ns`) |

Pick **invoke + map** when names do not match (`answer_eng` vs `input_text`).  
Pick **add_node(subgraph)** when you are happy to share keys and want the inner graph to be a first-class step.

---

## File: `sub_graph.ipynb` — call `invoke` from a node

### Inner graph (its own world)

```text
SubState = {input_text, translated_text}

START → translate_text → END
```

```python
def translate_text(state: SubState):
    ...
    return {"translated_text": subgraph_llm.invoke(prompt).content}
```

This graph **does not know** what a `question` is. It only translates `input_text`. That is the point.

### Parent graph

```text
ParentState = {question, answer_eng, answer_hin}

START → answer (generate_answer) → translate (translate_answer) → END
```

`generate_answer` writes `answer_eng`.  
`translate_answer` is **not** the subgraph. It is a **wrapper**:

```python
def translate_answer(state: ParentState):
    result = subgraph.invoke({"input_text": state["answer_eng"]})
    return {"answer_hin": result["translated_text"]}
```

Two mappings:

```text
parent.answer_eng      ──►  subgraph.input_text
subgraph.translated_text ──► parent.answer_hin
```

If you forget the map and pass `ParentState` into `subgraph.invoke`, LangGraph will complain (or ignore) keys the subgraph did not declare. The wrapper is the contract.

### What `invoke` returned (your run)

```text
question     = "What is quantum physics"
answer_eng   = long English explanation (Cohere)
answer_hin   = Hindi translation of that explanation
```

Parent `invoke` is the only call you make. The inner `subgraph.invoke` is an implementation detail of the `translate` node.

### Mental model

```text
parent.invoke({question})
        │
        ├─ node "answer"     → writes answer_eng
        └─ node "translate"  → Python function
                 │
                 └─ subgraph.invoke({input_text: answer_eng})
                          └─ node "translate_text" → translated_text
                 │
                 └─ return {answer_hin: ...}
```

The subgraph is a **function that happens to be a graph**. Same idea as calling `retriever.invoke(query)` inside `rag_tool`.

---

## File: `subgraph_shares.ipynb` — add the compiled graph as a node

No wrapper. No second TypedDict.

```python
class ParentState(TypedDict):
    question: str
    answer_eng: str
    answer_hin: str
```

Inner node reads **`answer_eng`** (parent’s key) and writes **`answer_hin`** (parent’s key):

```python
def translate_text(state: ParentState):
    prompt = f"""... Text:\n{state["answer_eng"]}"""
    return {"answer_hin": subgraph_llm.invoke(prompt).content}

subgraph = StateGraph(ParentState) ... compile()
```

Parent:

```python
parent_builder.add_node("answer", generate_answer)
parent_builder.add_node("translate", subgraph)   # compiled graph, not a function

parent_builder.add_edge(START, "answer")
parent_builder.add_edge("answer", "translate")
parent_builder.add_edge("translate", END)
```

`add_node` accepts either a function **or** a compiled `StateGraph`. That is the whole trick.

### Shared keys

LangGraph copies overlapping fields into the subgraph and copies the subgraph’s updates back.

```text
parent state:  question, answer_eng, answer_hin
subgraph reads:           answer_eng
subgraph writes:                      answer_hin
question is unused by the subgraph (and that is fine)
```

If the subgraph used `input_text` here, it would **break** — that key is not on `ParentState`. Shared-schema subgraphs only work when the inner graph’s keys are a **subset** of (or equal to) the parent’s.

### Mental model

```text
parent.invoke({question})
        │
        ├─ node "answer"      → generate_answer writes answer_eng
        └─ node "translate"   → THIS IS THE SUBGRAPH
                 START → translate_text → END
                 reads answer_eng, writes answer_hin
```

No Python mapper. Pregel **enters** the subgraph the same way it enters a function node, then returns.

Your run is the same shape of dict as the first notebook: `question`, `answer_eng`, `answer_hin`. The Hindi string is the proof the inner node ran on shared state.

---

## Same job, two wiring diagrams

```text
sub_graph.ipynb
═══════════════
ParentState                          SubState
question                             input_text
answer_eng  ──map──► invoke ──map──► translated_text
answer_hin  ◄──────────────────────┘


subgraph_shares.ipynb
═════════════════════
ParentState  <══ same TypedDict ══>  subgraph
question
answer_eng  ──────────────────────►  translate_text
answer_hin  ◄──────────────────────┘
```

If you later add `tone` to the translator:

- **invoke style:** add `tone` to `SubState`, pass it in the mapper. Parent can stay without `tone`.
- **shared style:** `tone` must exist on `ParentState` (or you go back to a mapper). Parent state gets noisier.

That is the real tradeoff: **isolation vs convenience**.

---

## Compile twice

Both files `compile()` the subgraph **before** the parent uses it.

```python
subgraph = subgraph_builder.compile()
# then either:
#   subgraph.invoke(...)          inside a node
#   parent.add_node(..., subgraph)
graph = parent_builder.compile()
```

You invoke the **parent**. You almost never `subgraph.invoke` from the notebook’s last cell in the shared style — the parent does that for you.

Neither notebook passes a **checkpointer**. That is fine for learning. If you later `parent.compile(checkpointer=...)`:

- invoke-style: the inner `subgraph.invoke` is a nested run. Give the subgraph its **own** checkpointer if you need inner memory, or treat it as a one-shot function.
- add_node-style: LangGraph namespaces inner checkpoints (`checkpoint_ns` — the persistence notes said ignore this until subgraphs). Inner node names live under the parent node `"translate"`.

---

## Subgraph vs node vs MCP vs HITL

| Thing | What you nest | Boundary |
| --- | --- | --- |
| Node | a **function** | state dict in / patch out |
| **Subgraph** | a **graph** | schema map **or** shared keys |
| MCP server | **tools** in another process | JSON-RPC stdio |
| HITL | a **pause** inside a node | `interrupt` / `Command` |

Do not nest a subgraph to call Alpha Vantage. That is a tool (or MCP).  
Do not nest a subgraph to wait for a human. That is `interrupt()`.  
Do nest a subgraph when the inner work is **several nodes** you want to reuse.

---

## What to watch

| Symptom | Likely cause |
| --- | --- |
| `KeyError: 'input_text'` in the shared notebook | you copied `translate_text` from the first file; shared style must read `answer_eng` |
| `KeyError: 'answer_eng'` in the first notebook’s subgraph | you passed parent state into `subgraph.invoke` without mapping |
| Hindi missing, English present | `translate` node never wired (`add_edge("answer", "translate")`) |
| Mermaid shows no inner `translate_text` on the parent | first notebook — expected; inner graph is behind `invoke` |
| Subgraph `compile()` then you `add_node` a **builder** | must pass the **compiled** graph (`subgraph`), not `subgraph_builder` |
| Extra keys on the subgraph in shared mode | keep subgraph state a **subset** of parent, or switch to mapper |

`question` is unused in both inner graphs. That is OK. Unused parent keys are not an error.

---

## One-line map

```text
sub_graph.ipynb          = two schemas. you are the adapter (invoke + map).
subgraph_shares.ipynb    = one schema. LangGraph is the adapter (add_node).
```

**One line:** subgraph = a compiled graph used as a step; map keys yourself, or share the TypedDict and `add_node`.
)
