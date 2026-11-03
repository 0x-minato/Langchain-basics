# LangGraph persistence

CampusX playlist **videos 10–11-ish** (chatbot + persistence / time travel). Previous notes:

- `langchain_vs_langgraph/` — why a graph
- `core_concepts/` — state, reducers, Pregel
- `workflows/` — sequential / parallel / conditional / iterative **shapes**

This folder is not a new shape. The graphs are still a **line**:

```text
chatbot:           START → chat_node → END
fault_tolerance:   START → step1 → step2 → step3 → END
```

What is new: **the run is saved**. After each finished node, LangGraph writes a **checkpoint**. You can come back later with the same `thread_id` (and optionally a `checkpoint_id`) and the graph remembers state.

```text
without a checkpointer:  invoke is a goldfish. next invoke starts empty.
with a checkpointer:     invoke is a chapter. next invoke on the same thread continues the book.
```

The four names people use — **short-term memory**, **fault tolerance**, **time travel**, **human in the loop** — are **the same mechanism** looked at from four jobs. The mechanism is: `compile(checkpointer=...)` + `config = {"configurable": {"thread_id": "..."}}`.

---

## How to restudy this folder

| Order | File | What it proves |
| --- | --- | --- |
| 1 | `chatbot.ipynb` | short-term memory (`thread_id`) + time travel (`checkpoint_id`, `update_state`) |
| 2 | `fault_tolerance.ipynb` | crash mid-node, `get_state`, resume with `invoke(None, config)` |

You do **not** have a dedicated HITL notebook yet. HITL still belongs in these notes because it uses the **same** checkpointer. The chatbot `while True: input()` is a **Python** loop around `invoke`. Graph-native HITL is `interrupt()` inside a node. Both wait for a human; only `interrupt()` pauses **inside** one `invoke`.

---

## Persistence — the one idea

### Checkpointer

A **checkpointer** is a store for snapshots of state. You plug it in at compile time:

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
chat_bot = graph.compile(checkpointer=checkpointer)
```

`MemorySaver` lives in **RAM**. Kernel restart → history gone. That is why `get_state_history` is empty after Restart. For a process crash that kills Python, you want **SQLite / Postgres** (`SqliteSaver`, `PostgresSaver`). Same API, durable disk.

You already use `MemorySaver` in both files. Fine for learning. Not fine for production.

### What gets saved

A checkpoint is taken **after a node returns**, at the Pregel barrier. Not in the middle of `time.sleep(30)`. Not in the middle of `model.invoke`.

```text
step1 returns {step1: True}  →  checkpoint  (step1=True, next=step2)
step2 still sleeping         →  no new checkpoint
step2 raises                 →  still the step1 checkpoint
```

That is the whole of fault tolerance.

### `thread_id` vs `checkpoint_id`

```python
config = {
    "configurable": {
        "thread_id": "2",           # which conversation / run
        "checkpoint_id": "1f1d..."  # optional: which moment inside that conversation
    }
}
```

| Key | Means | Like |
| --- | --- | --- |
| `thread_id` | one logical session | a chat room, a ticket, a user |
| `checkpoint_id` | one snapshot on that thread | a git commit |

Omit `checkpoint_id` → **latest** snapshot on that thread.  
Pass an old `checkpoint_id` → time travel to that commit (and **fork** a new branch when you invoke again).

`checkpoint_ns` shows up in snapshots as `''`. Ignore it until subgraphs.

### `invoke` without config

```python
chat_bot.invoke({"messages": [HumanMessage("what is the capital of india")]})
```

No `thread_id` → LangGraph still runs the graph, but **does not persist** it as a named thread you can resume. Your capital-of-India cell is a one-shot. The `while` loop is the persistent chat because it passes `config`.

### Reading checkpoints

```python
chat_bot.get_state(config)            # latest snapshot for this thread
list(chat_bot.get_state_history(config))  # all snapshots, newest first
```

A `StateSnapshot` has the fields you actually used:

| Field | Meaning |
| --- | --- |
| `values` | the state dict at that moment |
| `next` | which node(s) will run if you continue. `()` means this run already hit END |
| `config` | includes `thread_id` + `checkpoint_id` |
| `parent_config` | previous checkpoint (the git parent) |
| `metadata["step"]` | 0 = just received input, 1 = after first node, … |
| `tasks` | what was scheduled at that snapshot |

**`next` is the resume cursor.**  
`next=('chat_node',)` → `invoke(None)` will run `chat_node`.  
`next=()` → this invoke already finished; `invoke(None)` will not call the model again unless you fork from an earlier snapshot.

### Resume with `None`

```python
workflow.invoke(None, config=config)
```

`None` = “no new user input, continue from the checkpoint in `config`.”  
A new dict = “this is additional input merged into state, then run.”

The chatbot loop always sends a **new** `HumanMessage`. Fault-tolerance resume sends `None` because the human is not adding a turn; the machine is finishing unfinished work.

---

## Short-term memory

**Short-term** = memory of **this thread**, for **this session**. The chatbot remembering “your name is Aryan” on the next `invoke`.

It is **not** the model’s weights. It is **not** a vector store. It is the `messages` list on state, kept by the checkpointer, grown by the `add_messages` reducer.

```text
Long-term memory (later videos)  = facts across threads / users / days  (store, search)
Short-term memory (this folder)  = the live conversation on one thread_id
```

### Two layers that people mix up

| Layer | Lives in | Job |
| --- | --- | --- |
| **State + reducer** | one `invoke` | `add_messages` appends the AI reply onto `[Human]` so you return `[Human, AI]` |
| **Checkpointer** | across `invoke`s | next `invoke` on the same `thread_id` **loads** `[Human, AI]` before adding the new Human |

Without `add_messages`, `return {"messages": [res]}` would **replace** history with only the AI message (your first chatbot bug was a cousin of this).  
Without a checkpointer, the next `invoke` would start with only the new HumanMessage — the model would forget the name.

### Why the graph is still `START → chat_node → END`

Each user turn is **one** invoke, one trip through the line. Multi-turn is **not** a cycle in the graph. It is:

```text
Python while True:
    human types
    invoke({messages: [HumanMessage]}, thread_id=2)   # START → chat_node → END
    print last AI
```

The checkpointer stitches turns together. That is why you do **not** draw `chat_node → START`. `START` is the beginning of **this** invoke, not the beginning of the friendship.

### `[-1]` vs `[1]`

`response["messages"][-1].content` is the **latest** message (the new AI). `[1]` only works when the list is exactly two items. After three turns you have 6 messages.

---

## Fault tolerance

If a node **dies**, you should not redo work that already **returned**.

```text
step1 done     → checkpoint (step1=True, next=step2)
step2 crashes  → that checkpoint is still there
resume         → run step2 again, then step3
                 step1 does not print again
```

Limits (important):

1. Checkpoints are **per finished node**, not per line of Python. A 30s sleep that dies at second 3 **restarts from 0** on resume.
2. Side effects inside a node (charged API call, sent email) may run twice if the node crashes after the side effect but before `return`. Keep nodes small; make side effects idempotent if you can.
3. `MemorySaver` does not survive killing the kernel. Fault tolerance in this notebook is “exception in step 2, same process.” Real crashes need SQLite.

Resume API is the same as time travel to **latest**: `invoke(None, config)` with only `thread_id`.

---

## Time travel

Time travel = pick an **old** `checkpoint_id` and either:

1. **Replay** — run again from that snapshot (new AI answer to the same human text)
2. **Edit then replay** — `update_state` patches values, then `invoke(None)` continues from the **new** checkpoint (a fork)

History is a linked list of commits:

```text
newest  step=1  next=()              messages=[Human("helo"), AI("Hello!...")]
        step=0  next=(chat_node,)    messages=[Human("helo")]
oldest  next=(__start__,)            messages=[]
```

Replay the **step=0** snapshot, not the newest. The newest already finished (`next=()`). Replaying it does nothing useful.

```python
to_replay = next(s for s in history if s.next == ("chat_node",))
chat_bot.invoke(None, config=to_replay.config)
```

That **forks**. The original finished conversation is still in history. The new invoke adds a sibling commit. Same `thread_id`, new `checkpoint_id`.

### `update_state`

```python
updated_config = chat_bot.update_state(
    to_replay.config,
    {"messages": [edited_human]},
)
chat_bot.invoke(None, config=updated_config)
```

`update_state` writes a checkpoint **as if a node had returned that patch**. It does not call Cohere. `invoke(None)` then runs `next`.

**`add_messages` upserts by `id`.** If you pass a new `HumanMessage` with a **new** id, it **appends**. If you reuse the old message’s `id` and change `content`, it **replaces** that turn. That is how you change `"helo "` into `"what is the capital of france"` without stacking two humans.

```python
old = to_replay.values["messages"][0]
edited = HumanMessage(content="what is the capital of france", id=old.id)
```

Your run: after update, `get_state` showed only France; `next` was still `('chat_node',)`; the model answered Paris.

### Time travel is not Git reset

You do not delete the future. You **branch**. `get_state_history` will show both the original `"helo "` timeline and the France fork.

---

## Human in the loop (HITL)

HITL = the graph **stops and waits for a person**, then continues **with that person’s input in state**.

Two implementations. Do not confuse them.

### 1. What you already have — outer Python loop

```python
while True:
    user_input = input()
    if user_input in ["exit", "bye", "quit"]:
        break
    chat_bot.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
```

Each wait happens **between** invokes. The graph always runs start-to-end. The human is the driver of the `while`. Persistence is what makes those invokes one conversation.

Use this for a chatbot. It is simple. It cannot pause **in the middle** of a node (e.g. “approve this tool call before I execute it”).

### 2. Graph-native — `interrupt()` (not in a file yet)

```python
from langgraph.types import interrupt, Command

def approve_node(state):
    decision = interrupt({"question": "send this email?", "draft": state["draft"]})
    return {"approved": decision}

# first invoke runs until interrupt, then returns with an interrupt payload
# later:
graph.invoke(Command(resume="yes"), config=config)
```

Here **one** `invoke` pauses. The checkpointer stores “we are sitting on `approve_node`, waiting.” Another process, another day, `Command(resume=...)` wakes that thread. This is HITL as a first-class Pregel halt, not `input()` in Jupyter.

That is why HITL is filed under **persistence**. Without a checkpointer, there is nowhere to sit while the human thinks.

Typical HITL moments:

- approve / reject a tool call
- edit a draft before send
- pick among retrieved docs
- type a missing slot (“what city?”)

`get_state(config).tasks` / `.interrupts` is how you see a paused HITL snapshot. Your `fault_tolerance` crash is **not** HITL: it is an exception, not a planned pause.

---

## Map: four jobs, one knob

```text
                    compile(checkpointer=...)
                    config.thread_id
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
 short-term memory      fault tolerance         time travel
 next invoke sees       crash → resume          old checkpoint_id
 yesterday's messages   from next node          replay / edit fork
        │
        └──────────► HITL
                     pause this thread until a human resumes it
```

| Job | You pass | You call |
| --- | --- | --- |
| Remember the chat | `thread_id` | `invoke({new human}, config)` |
| Survive a crash | same `thread_id` | `invoke(None, config)` after the exception |
| Replay a moment | `thread_id` + old `checkpoint_id` | `invoke(None, snapshot.config)` |
| Change the past | same + `update_state` | then `invoke(None, updated_config)` |
| HITL (native) | `thread_id` | `interrupt()` then `Command(resume=...)` |

---

## File: `chatbot.ipynb`

### Graph

```text
START → chat_node → END
```

One LLM node. Topology is the same as `basic_seq.ipynb`. Persistence is the only upgrade.

### State

```python
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

`add_messages` is the chat reducer: append new messages; if an incoming message has an **id that already exists**, replace that message (needed for time-travel edits).

### `chat_node`

```python
def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    res = model.invoke(messages)
    return {"messages": [res]}
```

The model sees the **full** list. You return **only the new AI message**. The reducer appends it.

Bug you already hit: `return {"messages": [messages]}` wrapped the whole list in another list. `add_messages` tried to parse that inner list as a message → `ValueError: not enough values to unpack`. Fix: `[res]`.

### Compile

```python
checkpointer = MemorySaver()
chat_bot = graph.compile(checkpointer=checkpointer)
```

Without this, `thread_id` does nothing useful.

### One-shot cell (no config)

Capital of India. Works, but is **not** on a thread. You cannot `get_state` it later.

### The chat loop (`thread_id = '2'`)

```python
config = {"configurable": {"thread_id": thread_id}}
chat_bot.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
```

Each turn: new HumanMessage in, checkpointer loads prior messages, `chat_node` sees history, AI appended, checkpoint written.

Your recorded turn:

```text
user: helo
AI:   Hello! How can I assist you today?
```

`exit` / `bye` / `quit` break the `while`. The kernel stays busy on `input()` until then — that is a Jupyter stall, not LangGraph.

### `get_state` / history

`thread_id='1'` was never written → empty snapshot, `values={}`, history `[]`.  
`thread_id='2'` has the conversation.

History you actually printed (newest first):

```text
checkpoint ...8001-d89e425cff7e   next=()              last = "Hello! How can I assist you today?"
checkpoint ...8000-4d3b0dbfda4a   next=(chat_node,)    last = "helo "
checkpoint ...bfff-d4a28073e9ff   next=(__start__,)    last = None
```

Read that bottom-up: empty → human arrived (`next` is the node that will answer) → node finished (`next` empty).

### Time travel cells

**Replay.** Pick `next == ("chat_node",)`. `invoke(None, to_replay.config)` runs the node again on `"helo "`. You got another greeting. Same question, new (or similar) completion.

**Edit.** Copy the human’s `id`, change content to `"what is the capital of france"`, `update_state`, then `invoke(None)`.

Your output:

```text
state now: ['what is the capital of france']
next: ('chat_node',)
HumanMessage : what is the capital of france
AIMessage    : The capital of France is **Paris**. ...
```

The original `"helo "` timeline is still in history. This is a fork.

---

## File: `fault_tolerance.ipynb`

No LLM. Pure proof that checkpoints are **after** a node returns.

### Graph

```text
START → step1 (sleep 1s) → step2 (sleep 30s, can crash) → step3 (sleep 1s) → END
```

### State

```python
class FaultState(TypedDict):
    step1: bool
    step2: bool
    step3: bool
```

All **replace** booleans. Invoke starts them `False`.

### Programmatic crash

`crash_step2` is a flag. Step 2 sleeps in 0.2s slices and raises `RuntimeError` if the flag is True.

A `threading.Timer(3, ...)` sets the flag 3s after invoke starts. Step 1 is 1s, so you are ~2s into step 2 when it dies.

Your output:

```text
step 1 running...
step 1 done
step 2 running — sleeping 30s
timer fired: interrupting step 2
caught: simulated crash during step 2
```

### After the crash

```text
values: {'step1': True, 'step2': False, 'step3': False}
next node: ('step2',)
```

Step 1’s return was checkpointed. Step 2 never returned, so `step2` is still `False` and it is still next.

### Resume

```python
crash_step2 = False   # otherwise step 2 dies again immediately
workflow.invoke(None, config=config)
```

Your output:

```text
step 2 running — sleeping 30s
step 2 done
step 3 running...
step 3 done
{'step1': True, 'step2': True, 'step3': True}
```

`next node: ()` — finished. Step 1 did **not** print a second time.

If you forget `crash_step2 = False`, resume raises in step 2 again. The flag is a demo tool, not part of LangGraph.

### Why step 2 sleeps 30s **again**

The first attempt died at ~2s. There is no checkpoint at “18s remaining.” Resume starts the node from the top. That is correct Pregel, not a bug.

---

## Side-by-side

| | `chatbot.ipynb` | `fault_tolerance.ipynb` |
| --- | --- | --- |
| Nodes | 1 LLM | 3 sleeps |
| Why persist | remember chat turns | skip finished steps after a crash |
| `invoke` input each time | new `HumanMessage` | first: initial bools; later: `None` |
| `next` you care about | `('chat_node',)` to replay a turn | `('step2',)` after the crash |
| Extra API | `get_state_history`, `update_state` | timer + exception |
| HITL | Python `input()` between turns | none (crash is not a human pause) |

---

## Bugs / traps in this folder

1. **`return {"messages": [messages]}`** — nested list, `add_messages` explodes. Return `[res]`.
2. **`invoke` without `config`** — no thread, cannot resume or inspect.
3. **`MemorySaver` + Restart Kernel** — history vanishes. Re-run compile + the chat loop before time travel cells.
4. **Replay the newest snapshot** (`next=()`) — nothing runs. Replay the snapshot whose `next` is the node you want.
5. **`update_state` with a new message id** — appends a second Human. Reuse `id` to edit.
6. **Resume with `crash_step2` still True** — step 2 crashes forever.
7. **`input()` loop left running** — kernel is blocked; type `exit` or interrupt. Not a LangGraph hang.
8. **Wrong `thread_id`** — you queried `'1'` while chat wrote `'2'` → empty state.
9. **Thinking a cycle back to `START` is how chat works** — no. Outer `while` + checkpointer.
10. **Expecting mid-sleep checkpoints** — they do not exist.

---

## Cheat sheet

```text
MemorySaver / SqliteSaver     plug into compile()
thread_id                     which session
checkpoint_id                 which snapshot (omit = latest)
get_state(config)             latest snapshot
get_state_history(config)     all snapshots, newest first
snapshot.next                 nodes that will run on resume
invoke(None, config)          continue / replay, no new input
invoke({...}, config)         merge new input, then run
update_state(config, patch)   write a fork without running nodes
add_messages                  append, or replace if same message id
interrupt() + Command(resume) graph-native HITL (not coded here yet)
```

---

## 30-second recap

```text
Persistence     = save state after every finished node
Short-term      = that save, used as the chat’s memory on one thread
Fault tolerance = crash → invoke(None) from next
Time travel     = invoke(None) from an old checkpoint_id; update_state to edit
HITL            = planned pause (interrupt) or a Python while around invoke
```

If you can say what `next` is after a crash, and which history row to replay so `chat_node` runs again, you understand this folder.
)
