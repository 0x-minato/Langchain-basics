# Human in the loop (HITL)

CampusX LangGraph playlist — **interrupt / Command**. Previous notes:

- `persistence/` — checkpointer, `thread_id`, `invoke(None)` (crash resume)
- `chatbot/` — ReAct: `chat_node` ⇄ `ToolNode`
- `tools-calling/` — the model **asks** for a function; Python **runs** it

This folder is not a new graph shape. The graphs are still a line or a ReAct cycle. What is new: **one `invoke` can stop in the middle of a node**, wait for a person, then **continue from that same `interrupt()` line**.

```text
outer Python input()     = wait BETWEEN invokes (chatbot while-loop)
interrupt()              = wait INSIDE one invoke (this folder)
```

The persistence notes called graph-native HITL “not in a file yet.” These two files **are** that missing piece.

```text
without a checkpointer:  interrupt has nowhere to sit. the pause is lost.
with a checkpointer:     invoke returns, human thinks, Command(resume=...)
                         wakes the same thread_id
```

---

## How to restudy this folder

| Order | File | What it proves |
| --- | --- | --- |
| 1 | `hitl.ipynb` | `interrupt()` **before** the model answers. Resume payload is a **dict**. |
| 2 | `chat_bot_hitl.py` | `interrupt()` **inside a tool**. Resume payload is a **string**. ReAct + `ToolNode`. |

Both need `CO_API_KEY` / `COHERE_API_KEY`. The stock script also hits Alpha Vantage (live HTTP). Run from the **repo root** with `venv312`.

---

## HITL — the one idea

### `interrupt(payload)` is not `input()`

```python
from langgraph.types import interrupt, Command

decision = interrupt(payload)   # first time: pause the graph
                                # after Command(resume=x): this line RETURNS x
```

| | `input("yes/no")` | `interrupt(payload)` |
| --- | --- | --- |
| Who blocks | the **Python process** (Jupyter cell / CLI) | nobody. `invoke` **returns** |
| Where you wait | outside the graph | inside a node / tool |
| How you continue | type in the same process | `invoke(Command(resume=...), config)` |
| Survives closing the notebook | no | **yes**, if the checkpointer is on disk (`SqliteSaver`). `MemorySaver` dies with the kernel |

`interrupt` **requires** `compile(checkpointer=...)` and the same `thread_id` on resume. Otherwise LangGraph has no snapshot that says “we were sitting on this line.”

### What `invoke` returns when paused

Not a finished state. A dict that still has `messages` **plus** a special key:

```python
result["messages"]       # whatever was already in state
result["__interrupt__"]  # list of Interrupt objects
result["__interrupt__"][0].value   # the payload you passed to interrupt(...)
```

`__interrupt__` is a **list** because two parallel nodes could pause at once. These files only ever have one.

### Resume is not a new HumanMessage

| Call | Means |
| --- | --- |
| `invoke({"messages": [HumanMessage(...)]}, config)` | new user turn, graph starts at `START` |
| `invoke(None, config)` | fault tolerance: “keep going, no new data” |
| `invoke(Command(resume=x), config)` | HITL: “the `interrupt()` line should now **return `x`**” |

`Command(resume=...)` does **not** append a chat message. It injects a value into the paused node.

### On resume, the node restarts from the top

This is the gotcha. LangGraph does **not** freeze Python like a generator.

```text
first invoke:
    chat_node starts
    hits interrupt(payload)  →  no resume value yet  →  PAUSE
    (the rest of chat_node does not run)

Command(resume=x):
    chat_node starts AGAIN from line 1
    hits interrupt(payload)  →  resume value exists  →  returns x
    rest of chat_node runs
```

So:

- Put **side effects after** `interrupt()`, or they run twice (once before pause, once on replay).
- The notebook is correct: interrupt **first**, then maybe `model.invoke`.
- `purchase_stock` is correct: interrupt **first**, then “place the order.”

`interrupt()` is replay-safe: the second time it just returns `x`.

---

## Two places you can interrupt

Same API. Different *when*.

```text
hitl.ipynb                         chat_bot_hitl.py
─────────                          ────────────────
START → chat_node → END            START → chat_node ⇄ tools → (back)

interrupt sits in chat_node        interrupt sits in purchase_stock
BEFORE the LLM call                AFTER the model already chose the tool
                                   (ToolNode is executing it)
```

| File | Gate | Typical product meaning |
| --- | --- | --- |
| `hitl.ipynb` | “May the **model even answer** this question?” | content policy, “don’t send yet” |
| `chat_bot_hitl.py` | “May this **side effect** run?” | buy stock, send email, delete a row |

You interrupt the **dangerous** step, not the whole graph. `get_stock_price` has no `interrupt` — looking up a quote is cheap. Buying is not.

---

## File: `hitl.ipynb`

### Graph

```text
START → chat_node → END
```

No tools. One node. Checkpointer = `MemorySaver()`. `thread_id = "test_thread"`.

### The node

```python
def chat_node(state: ChatState):
    decision = interrupt({
        "type": "approval",
        "reason": "Model is about to answer a question",
        "question": state["messages"][-1].content,
        "instruction": "Approve this message yes/no",
    })

    if decision["approved"] == "no":
        return {"messages": [AIMessage(content="Not Approved")]}
    else:
        response = model.invoke(state["messages"])
        return {"messages": [response]}
```

`payload` is a **dict**. That dict is what the UI (here: `print` + `input`) shows the human. It is not sent to Cohere.

`decision["approved"]` means resume **must** be a dict with an `"approved"` key. The notebook does:

```python
Command(resume={"approved": user_input})   # user_input is "yes" or "no"
```

If you resume with the bare string `"no"`, `decision["approved"]` blows up (`str` has no that key). Match the shape you read.

### What you actually ran

**1. First `invoke`** — user question as a tuple `("user", "Explain gradient descent...")`. LangGraph turns that into a `HumanMessage`.

Returns:

```text
messages:
  [HumanMessage("Explain gradient descent in simple terms")]
  # no AIMessage — the model never ran

__interrupt__:
  Interrupt(value={type, reason, question, instruction}, id=...)
```

**2. Read the prompt**

```python
message = result["__interrupt__"][0].value
message["instruction"]   # "Approve this message yes/no"
```

**3. Human types `no`** (your cell output).

**4. Resume**

```python
workflow.invoke(Command(resume={"approved": "no"}), config=config)
```

`chat_node` replays, `interrupt()` returns `{"approved": "no"}`, branch writes `AIMessage("Not Approved")`.

Final state:

```text
[HumanMessage("Explain gradient descent..."),
 AIMessage("Not Approved")]
```

If you had typed `yes`, the `else` branch would call `ChatCohere` and you would get a real explanation.

### What this is *not*

- Not a tool-approval graph. There are no tools.
- Not `interrupt_before=["chat_node"]` (compile-time breakpoint). This is **dynamic** `interrupt()` inside the node, with a payload you designed.
- Not `update_state`. You are not editing history; you are answering a pause.

---

## File: `chat_bot_hitl.py`

Same `ChatState` + `add_messages` + `MemorySaver` as the persistence chatbot. New pieces: tools, `tools_condition`, `interrupt` **inside** `purchase_stock`.

### Graph (ReAct)

```text
START → chat_node → tools_condition
                       │
                       ├─ no tool_calls → END
                       └─ tool_calls    → tools (ToolNode)
                                              │
                                              └─ back to chat_node
```

`chat_node` only calls the model. It never talks to Alpha Vantage. `ToolNode(tools)` runs whatever the model requested.

### Two tools, one gated

| Tool | HITL? | Why |
| --- | --- | --- |
| `get_stock_price(symbol)` | no | HTTP GET quote. Safe-ish. |
| `purchase_stock(symbol, quantity)` | **yes** | fake buy — this is the side effect you do not want the model to fire alone |

```python
@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    decision = interrupt(
        f"Approve buying {quantity} shares of {symbol}? (yes/no)"
    )
    if isinstance(decision, str) and decision.lower() == "yes":
        return {"status": "success", ...}
    return {"status": "cancelled", ...}
```

Payload is a **string**, not a dict. So resume is also a string:

```python
Command(resume=decision)   # "yes" / "no" / whatever they typed
```

`get_stock_price` interpolates `symbol` into the Alpha Vantage URL and `return r.json()`. No pause.

### CLI loop — two `invoke`s per dangerous turn

```text
You: buy 10 AAPL
        │
        ▼
invoke({HumanMessage}, thread_id="demo-thread")
        │
        ├─ chat_node → AIMessage(tool_calls=[purchase_stock])
        ├─ tools → purchase_stock → interrupt("Approve buying 10...")
        └─ invoke RETURNS  (graph not finished)

result["__interrupt__"] is non-empty
        │
print HITL prompt
Your decision: yes
        │
invoke(Command(resume="yes"), SAME thread_id)
        │
        ├─ ToolNode replays purchase_stock
        ├─ interrupt() returns "yes"
        ├─ tool returns success dict → ToolMessage
        ├─ chat_node → model writes a sentence
        └─ print Bot: ...
```

If they type `no`, the tool still **returns** (cancelled dict). That is a successful tool result, not an error. The model then tells the user the purchase was declined.

`get_stock_price` never takes this branch. `__interrupt__` is empty; you print the bot line immediately.

### Why `thread_id` is fixed (`"demo-thread"`)

Resume **must** hit the same checkpoint. A new thread would start a new graph with no paused `interrupt`. Same rule as persistence chat.

### `Command` vs sending another HumanMessage

If after the HITL prompt you did:

```python
chatbot.invoke(
    {"messages": [HumanMessage(content="yes")]},
    config=config,
)
```

that would be a **new turn**, not an answer to `interrupt()`. The purchase tool would stay paused (or you would stack a confusing extra message). Resume is `Command(resume=...)`.

---

## Payload shape — keep it consistent

Whoever **reads** `interrupt()`’s return value decides what you must `resume`.

| File | `interrupt(...)` | Code then does | So `Command(resume=...)` must be |
| --- | --- | --- | --- |
| `hitl.ipynb` | a **dict** (shown to the human) | `decision["approved"] == "no"` | `{"approved": "yes"}` or `{"approved": "no"}` |
| `chat_bot_hitl.py` | a **string** (shown to the human) | `decision.lower() == "yes"` | `"yes"` / `"no"` |

The payload you pass **into** `interrupt` (what `__interrupt__[0].value` is) and the value you pass **into** `Command(resume=)` are **two different objects**:

```text
interrupt(PAYLOAD)     →  shown in result["__interrupt__"][0].value
Command(resume=ANSWER) →  becomes the return value of interrupt()
```

The notebook’s PAYLOAD is a dict of UI fields. The ANSWER is `{"approved": "no"}`. They are not the same dict.

---

## HITL vs the other persistence jobs

Same knob: checkpointer + `thread_id`. Different second call.

| Job | Graph stopped because | You resume with |
| --- | --- | --- |
| Short-term memory | it **finished** (`next=()`) | new `HumanMessage` |
| Fault tolerance | a node **crashed** | `invoke(None, config)` |
| Time travel | you picked an old `checkpoint_id` | `invoke(None, snapshot.config)` |
| **HITL** | a node called **`interrupt()`** | `Command(resume=...)` |

A crash is not HITL. `input()` in a `while True` is not graph-native HITL. Both wait for a human in English; only `interrupt()` pauses **inside** `invoke`.

`chat_bot_hitl.py` uses **both**:

- outer `while True: input("You:")` — normal chat turns
- inner `interrupt` + `input("Your decision:")` — only when a purchase is pending

---

## Compile-time breakpoints (not used here)

LangGraph can also pause **without** calling `interrupt()`:

```python
graph.compile(checkpointer=memory, interrupt_before=["tools"])
```

That stops **before** `ToolNode` runs, every time. No payload. You inspect `get_state`, maybe `update_state` the tool args, then `invoke(None)` or `Command(resume=True)`.

These files use **dynamic** `interrupt()` instead: pause only when `purchase_stock` runs, and send a human-readable prompt.

---

## What to watch while you rerun

### Notebook

1. After the first `invoke`, `result["messages"]` has **only** the Human. If you already see an AI explanation, the interrupt did not fire (missing checkpointer, or you resumed already).
2. `result["__interrupt__"][0].value["question"]` should be the user text.
3. Resume with a **dict**, not `"no"`.
4. Kernel restart wipes `MemorySaver` — the paused thread is gone; start from the first `invoke` again.

### Script

1. Ask something that **does not** buy (`what's AAPL trading at?`) → no `HITL:` line.
2. Ask to buy → `HITL: Approve buying ...` then `Your decision:`.
3. `yes` vs `no` should change the tool’s `status` and the bot’s last sentence.
4. `exit` / `quit` leave the loop. That does not resume a pending interrupt; if you quit while paused, that `MemorySaver` thread dies with the process.

### Shared traps

| Symptom | Likely cause |
| --- | --- |
| `interrupt` errors / pause ignored | forgot `checkpointer=` on `compile` |
| Resume starts a new chat | different `thread_id`, or you sent a `HumanMessage` instead of `Command` |
| `TypeError` / `decision["approved"]` fails | resume shape ≠ what the node reads |
| Email / HTTP / buy happens twice | side effect **before** `interrupt()`; node replayed |
| `__interrupt__` KeyError | use `.get("__interrupt__", [])` like the script; finished runs have no key |

---

## One-line map

```text
hitl.ipynb         = pause before thinking. human yes/no. then maybe call Cohere.
chat_bot_hitl.py   = model may call purchase_stock. ToolNode hits interrupt.
                     human yes/no. then tool returns. then model talks.
```

**One line:** HITL = `interrupt(payload)` + checkpointer + same `thread_id` + `Command(resume=answer)`.
)
