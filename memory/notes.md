# Memory in GenAI (overview)

CampusX memory lecture — **why** models forget, then the notebooks that actually build STM and LTM.

The first half of this file is the lecture (stateless `y = f_θ(x)`, why STM is not enough). The second half is the **code deep dives** for this folder’s notebooks.

You have already *used* short-term memory without naming it:

| You did this | That was |
| --- | --- |
| `chatModels/messages.py` append `AIMessage` | history in the next prompt |
| `chatModels/chatbot.py` `history[-MAX_MESSAGES:]` | sliding-window short-term memory |
| `langgraph/persistence/` `thread_id` + checkpointer | same idea, saved between `invoke`s |
| `yt-chatbot/` / FAISS RAG | a cousin of **semantic** long-term memory (over docs, not “who the user is”) |

```text
the model        = a stateless function  (tokens in → tokens out)
memory           = EVERYTHING you persist and re-inject around that function
short-term (STM) = this conversation, stuffed into the context window
long-term (LTM)  = facts / episodes / skills that survive after the window is gone
```

**One line:** LLMs do not remember. Apps remember, then **show** the model the relevant past on the next call.

---

## How to restudy this folder

Needs `CO_API_KEY` / `COHERE_API_KEY`. Postgres notebooks need `memory/docker-compose.yml` up (`localhost:5442`) and `psycopg[binary]` in `venv312`.

| Order | File | What it proves |
| --- | --- | --- |
| — | this `notes.md` (lecture half) | `y = f_θ(x)`, STM vs LTM as ideas |
| 1 | `stm.ipynb` | STM **storage**: `PostgresSaver` + `thread_id`. Thread-1 knows the name; thread-2 does not. |
| 2 | `stm_trimming.ipynb` | STM **policy**: `trim_messages` — shrink `x` by a token budget, checkpointer still holds the full log |
| 3 | `4_stm_deletion.ipynb` | STM **policy**: `RemoveMessage` — actually drop old turns from state |
| 4 | `5_stm_summarization.ipynb` | Compress old turns into `summary`, keep last 2 verbatim (`[:-2]`) |
| 5 | `6_ltm_basics.ipynb` | `InMemoryStore` + namespace. `put` / `get` / `search`. Optional semantic `query=` |
| 6 | `7_ltm_implementation.ipynb` | Read LTM into the system prompt → extract facts → skip duplicates → remember then chat |
| 7 | `8_ltm_postgres.ipynb` | Same graph, `PostgresStore` — LTM survives kernel restart |

`checkpointer` = STM (this **thread**). `store` = LTM (this **user**). Do not mix `thread_id` and `user_id`.

---

## Introduction: why memory matters in GenAI

A useful assistant is not a single clever reply. It is a **relationship over time**:

- “My name is Aryan” → two turns later it still uses the name
- yesterday’s bug report is still the bug report
- the agent already knows you prefer Cohere, Hindi, and MiniLM embeddings
- a support bot does not re-ask for the order id every message

Without memory, every `invoke` is a stranger. Products that feel “smart” (ChatGPT, Cursor, your Streamlit threads) are mostly **memory systems with an LLM in the middle**.

Three jobs memory has to do in GenAI:

| Job | If it fails |
| --- | --- |
| Continuity | the bot contradicts itself / asks again |
| Personalization | same generic answer for every user |
| Grounding | it “remembers” a fact that was never said, or forgets one that was |

RAG is memory of **documents**. Chat history is memory of **this talk**. User profile is memory of **the person**. Same problem, different stores.

---

## How LLMs work at inference (stateless by default)

The lecture writes the model as:

```text
y = f_θ(x)
```

That is the whole of inference. Read it as English: **output = the network, with frozen weights, applied to this input.**

| Symbol | Name | What it is in a chat API |
| --- | --- | --- |
| **x** | input | everything in **this** request: system + history + user text (+ RAG / memories if you stuffed them). Tokenized prompt. |
| **θ** (theta) | parameters | the billions of weights from **training**. Cohere `command-a-03-2025` is one frozen θ. |
| **f** | the function | the architecture + one forward pass (attention, MLPs, next-token softmax). |
| **y** | output | the tokens the API returns (`AIMessage.content`). |

Training vs inference, in the same symbols:

```text
TRAINING (once, expensive)
    many (x, y*) pairs  →  update θ  →  a new θ
    you are changing the function itself

INFERENCE (every chat message)
    y = f_θ(x)     θ is locked
    you are only allowed to change x
```

### What the equation is saying

1. **It is a function, not a person.** Same `f` and same `θ` and same `x` → same `y` (temperature 0). There is no inner diary. `invoke` is `f_θ(x)`, then the process is done.

2. **θ does not move during chat.** “Remember my name” does **not** write into θ. Fine-tuning would. A normal Cohere/OpenAI call will not. If θ changed every turn, every user would overwrite the base model.

3. **The only knob you have at inference is x.** Memory, RAG, system prompts, few-shot examples — they are all **edits to x**. That is why the assembler in the next section exists: it builds a better `x` before calling `f_θ`.

4. **y is not stored inside f.** If you want the reply in the next turn, **you** copy `y` back into the next `x` (append `AIMessage`). If you throw `y` away, it is gone.

```text
turn 1:  x1 = [Human("my name is Aryan")]
         y1 = f_θ(x1)                        → "Hi Aryan"

turn 2, no memory:
         x2 = [Human("what is my name?")]    → θ never saw "Aryan" in x2
         y2 = f_θ(x2)                        → goldfish

turn 2, with STM:
         x2 = [Human("my name is Aryan"), AI("Hi Aryan"), Human("what is my name?")]
         y2 = f_θ(x2)                        → "Aryan"
```

Same `f`, same `θ`. Different `x` → different `y`. **That** is memory, written in the equation.

5. **x has a hard length cap.** That cap is the **context window**. `f_θ` is not defined for an arbitrarily long `x`. If `|x|` (plus room for `y`) exceeds the window, the API truncates, errors, or you must shrink `x` (sliding window, summary, retrieve-only). A 128k window means: `x` may be huge, not: θ started storing chats.

```text
x  =  [ tokens the model is allowed to attend to ]
      |←—————— context window (max |x| + |y|) ————→|

y  =  f_θ(x)     only those tokens participate
```

Tokens **not** in `x` are invisible to this call, even if they lived in yesterday’s `x`. The window is a property of **one** evaluation of `f_θ`, not a memory chip.

- No hidden notebook inside the API
- No “I stored your name in my neurons” during the chat
- Closing the HTTP request drops everything that was only in that `x`

Longer context windows do **not** make the model stateful. They only let `x` be longer. Still `y = f_θ(x)`, still one shot, still stateless.

This is why `LLMs/llm_demo.py` and `chatModels/chat_model.py` feel like goldfish: each `invoke` is a new `y = f_θ(x)` with only the `x` you built for that call.

---

## The core problem: LLMs have no memory

```text
Turn 1:  Human("my name is Aryan")     →  AI("Hi Aryan")
Turn 2:  Human("what is my name?")     →  ???
```

If turn 2’s payload is only `Human("what is my name?")`, the model has **never seen** “Aryan” in this forward pass. It will guess, refuse, or hallucinate.

The bug is not Cohere. The bug is: **you did not put the previous messages in the prompt.**

```text
what you think happens:   model.remember("Aryan")
what actually happens:    model.invoke([only the latest HumanMessage])
```

Same problem across sessions: even a perfect in-chat history dies when the process restarts (`MemorySaver`) or the user opens a **new** thread. Weights still did not learn the user.

---

## Building memory around LLMs (first principles)

Do not wait for a “memory-enabled” model. Treat the LLM as a CPU and **you** build RAM + disk:

```text
                    ┌────────────┐
     stores  ──read─►│  assembler │──prompt──► LLM ──► reply
     (RAM/DB)        │  (app)     │◄──write──┘
                    └────────────┘
```

The assembler (your Python) always does four moves:

1. **Load** relevant past (history, facts, retrieved chunks)
2. **Compose** the prompt (system + memory + new user text)
3. **Call** the stateless LLM
4. **Write back** whatever should exist next time (append messages, upsert a fact)

That loop **is** memory. LangChain `MessagesPlaceholder`, LangGraph `add_messages` + checkpointer, RAG retrievers, future LTM stores — all four are step 1 and step 4 with different backends.

First-principles test: if you cannot point to **where the bytes live** between turns, you do not have memory. You have a vibe.

---

## Context window and in-context learning

In `y = f_θ(x)`, the **context window** is the maximum size of `x` (usually counted together with `y`). It is not RAM inside the model. It is the **legal length of the input** for one forward pass.

Attention in a transformer looks at pairs of tokens **inside this `x`**. A token that did not make it into `x` cannot be attended to. So the window is the answer to: “how much of the past can I even put into `x`?”

```text
x = | system | retrieved memories | chat history | new user turn |
y = | …generation…                                               |
    |<-------------- context window: |x| + |y| ≤ N --------------->|
```

Everything you call “memory” at inference has to **fit in x** or be **summarized / retrieved** into `x`. If it is not in `x`, `f_θ` cannot use it.

**In-context learning** = the model adapts from tokens **in this window**, not from a weight update.

| In the window | The model can |
| --- | --- |
| Few-shot examples | copy the format (`chat_history.txt` style lines) |
| Prior turns | continue the conversation |
| RAG chunks | answer from the PDF |
| “User name = Aryan” | use the name |

If it is not in the window (and not in the frozen weights), it does not exist for this `invoke`.

Overflow policy is a product choice: drop oldest turns, summarize, or retrieve only the relevant slice. That is already the sliding window in `chatModels/chatbot.py`.

---

## Short-term memory using conversation history

**STM (this lecture’s meaning)** = the live thread: recent `HumanMessage` / `AIMessage` pairs sent back every turn.

Not weights. Not a vector store. Not “memory of last month.” Just **this conversation’s buffer**.

```text
[]  →  [H1, A1]  →  [H1, A1, H2, A2]  →  …  →  window of last N
```

Enough for “what did I just say?” and “your name is Aryan” **until** the window slides off or the session dies.

Code: `stm.ipynb` → trimming → deletion → summarization, below. Here: STM **is** history-in-the-prompt.

---

## How chatbots implement short-term memory

Every chatbot you have written is the same loop:

```text
history = []                    # or load from SQLite / checkpointer
while True:
    user = input()
    prompt = system + history + user
    ai = model.invoke(prompt)
    history.append(user, ai)
    history = history[-N:]      # stay inside the window
```

| Layer in this repo | Mechanism |
| --- | --- |
| `chatModels/chatbot.py` | Python list + `MessagesPlaceholder("history")` + `[-10:]` |
| LangGraph chat graphs | `messages: Annotated[list[BaseMessage], add_messages]` |
| Persistence | `MemorySaver` / `AsyncSqliteSaver` + `thread_id` so the list survives `invoke` |
| Streamlit UI | sidebar thread = which `thread_id` to load |

The **model** never appends. Your app does. The checkpointer is STM **storage**; the window is STM **policy**.

---

## Limitations of short-term memory

STM is necessary and **not sufficient**.

| Limit | What it means |
| --- | --- |
| **Context size / cost** | every turn resends the whole window; tokens = money and latency |
| **Forgetting by sliding** | turn 1 falls off at `MAX_MESSAGES`; the name can vanish |
| **Session-scoped** | new chat / new `thread_id` / kernel restart (`MemorySaver`) → blank user |
| **Noise** | 40 turns of chit-chat bury the one fact that matters |
| **No structure** | “prefers MiniLM” is just another sentence, not a typed profile field |
| **No sharing** | thread A does not tell thread B |

When the product need is “remember me next week” or “remember this across all my chats,” you have left STM.

---

## Why we need long-term memory

LTM is for facts that must outlive the window **and** the session.

```text
STM:  this thread, this context window
LTM:  this user (or this org), days/weeks, many threads
```

You want LTM when:

- the user should not re-introduce themselves
- preferences / constraints must stick (“never use OpenAI”, “I’m learning LangGraph”)
- an agent should recall **what it did** last Tuesday (episodes), not only last 10 messages
- knowledge is too large to paste (then you retrieve, like RAG)

RAG on ISLR is LTM of a **book**. Chat LTM is usually LTM of a **person / agent**. Same retrieve-and-stuff pattern; different writers.

Code: `6_ltm_basics.ipynb` → `7_ltm_implementation.ipynb` → `8_ltm_postgres.ipynb`, below. Here: only the map.

---

## Types of long-term memory (names only)

Cognitive-science labels used in MemGPT / Letta / a lot of agent papers. Three buckets. Not three Python classes you must write today.

| Type | Stores | GenAI example |
| --- | --- | --- |
| **Episodic** | episodes / events in time | “On 14 Nov we debuged Cohere `TOOL_CALLS_MISSING_RESULTS`” |
| **Semantic** | facts, beliefs, profile | “User is Aryan; uses venv312; Cohere `command-a-03-2025`” |
| **Procedural** | how to do a job | “For PDFs: MiniLM + FAISS; for math: MCP `main.py`; don’t embed ISLR with Cohere” |

```text
episodic   = what happened
semantic   = what is true
procedural = how we do things
```

STM chat logs are a **raw stream**. LTM is that stream **distilled** into one of these three so you can retrieve a fact without replaying 200 messages.

RAG chunks ≈ semantic memory of documents. A playbook / tool policy ≈ procedural. Checkpoint history ≈ a crude episodic log (usually too fat to use as LTM).

---

## How long-term memory works (high-level architecture)

Timestamp ~00:47. Architecture only — no store APIs here.

```text
turn N
  user message
       │
       ▼
  retrieve from LTM          ← semantic search / key lookup / graph walk
       │
       ▼
  build prompt = system + retrieved memories + STM window + user
       │
       ▼
  LLM (stateless)
       │
       ▼
  reply to user
       │
       ▼
  extract / decide what to save   ← new fact? new episode? skip?
       │
       ▼
  upsert LTM store
```

Four boxes, always:

| Box | Job |
| --- | --- |
| **Store** | vector DB, KV, SQL, graph — bytes that survive the process |
| **Write path** | when to save (every turn vs “that looks like a fact”) |
| **Read path** | what to pull for *this* user message (top-k, filters, recency) |
| **Prompt assembler** | pack retrieved items into the context window without blowing the budget |

STM still sits in the prompt (recent turns). LTM sits in the store and only **enters** the prompt when retrieved. If you dump the whole LTM into every call, you have reinvented a worse context window.

Optional fifth box: **forgetting / conflict** (new fact overwrites old, decay unused episodes). That is already a challenge, not a beginner feature.

You already own a baby version of this for **docs**: `ingest_pdf` writes FAISS, `rag_tool` reads it. User-memory is the same diagram with a different extractor.

---

## Challenges and tools for memory systems

Timestamp ~00:53.

### Challenges (why this is a product, not a list.append)

| Challenge | Why it hurts |
| --- | --- |
| **What to write** | save everything → junk; save too little → still goldfish |
| **When to write** | every turn is slow/expensive; never is amnesia |
| **What to retrieve** | wrong memories in the prompt cause confident errors |
| **Conflicts** | “name is Aryan” then “call me Ari” — which is true? |
| **Staleness** | last month’s “I’m a beginner” vs today |
| **Privacy** | LTM is PII; checkpointers and vector stores need the same care as a DB |
| **Eval** | hard to unit-test “it remembered the right thing” |
| **Cost** | extra LLM calls to extract memories + extra tokens every turn |

STM challenges are mostly **window and cost**. LTM challenges are **correctness and governance**.

### Tools (map, not a shopping list)

You do not need all of these. Know which layer they sit on.

| Tool / idea | Layer |
| --- | --- |
| Your Python `history` list / sliding window | STM, in-process |
| LangGraph **checkpointer** (`MemorySaver`, SQLite) | STM (and resume), per `thread_id` |
| LangGraph **Store** (cross-thread) | LTM hook in the same library |
| Vector DB (Chroma, FAISS, …) | semantic retrieve (docs **or** memories) |
| RAG pipeline you already have | semantic LTM over **files** |
| Mem0, Zep, Letta (MemGPT) | opinionated **user/agent memory** services |
| ChatGPT / product “memories” | host-side LTM, not your app’s DB |

LangChain’s old `ConversationBufferMemory` / `ConversationSummaryMemory` classes are the lecture’s historical names for STM. In this repo you already replaced that idea with an explicit list or a checkpointer — keep doing that.

---

## Future of memory in LLMs

Timestamp ~00:56. Direction of travel, not APIs to memorize.

```text
now:     stateless model + app-built STM/LTM (this whole note)
next:    longer windows  (less sliding; still stateless; still costly)
         native product memory  (OpenAI/Claude “remember this”)
         memory as a platform layer  (store + extract + retrieve as infra)
later:   models that write durable state on purpose (research / products)
         agents with a stable identity across tools, graphs, MCP hosts
```

What does **not** go away:

- Someone still decides **what** is stored and **what** is injected
- Context windows stay finite, so retrieval stays
- HITL still belongs in front of dangerous writes (including “save this as a fact about the user”)

What to watch in this folder: checkpointer = STM; `Store` = user LTM; RAG = document LTM.

---

# Short-term memory (deep dive)

STM in LangGraph = the `messages` list on **one** `thread_id`, kept by a **checkpointer**.

```text
compile(checkpointer=...)
config = {"configurable": {"thread_id": "thread-1"}}

invoke({new HumanMessage}, config)   →  checkpointer loads prior messages,
                                         add_messages appends, model sees the thread
```

Two jobs people mix up:

| Job | What | This folder |
| --- | --- | --- |
| **Storage** | survive the next `invoke` (and the process, if Postgres) | `stm.ipynb` `PostgresSaver` |
| **Policy** | `x` is too long — trim, delete, or summarize before `f_θ` | trimming / deletion / summarization |

`MessagesState` already has `messages: Annotated[list[BaseMessage], add_messages]`. You do not re-write the reducer. Nodes return `{"messages": [new AI]}` and it **appends**.

`InMemorySaver` dies with the kernel. `PostgresSaver` does not — that is the point of `stm.ipynb` vs the later STM notebooks (those use `InMemorySaver` because they are teaching **policy**, not durability).

---

## File: `stm.ipynb` — durable STM (Postgres checkpointer)

Graph is a line: `START → call_model`. No tools. `call_model` is `model.invoke(state["messages"])`.

```text
DB_URI = postgresql://postgres:postgres@localhost:5442/postgres
docker compose  (memory/docker-compose.yml)  maps 5442 → 5432
```

```python
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()   # once: create checkpoint tables
    graph = builder.compile(checkpointer=checkpointer)
    graph.invoke(..., {"configurable": {"thread_id": "thread-1"}})
```

Proof:

| Thread | Turns | Model |
| --- | --- | --- |
| `thread-1` | “Hi, my name is kashyap” then “What is my name?” | knows Kashyap |
| `thread-2` | only “What is my name?” | does not |

Same weights θ. Different **checkpointed `x`**. `get_state(t1)` reads Postgres, not RAM.

Windows: `psycopg` without `psycopg-binary` → `no pq wrapper available`. Install the binary extra. Restart the Jupyter kernel after.

`with from_conn_string` **closes** the connection on exit. Keep `invoke` / `get_state` **inside** the `with`, or open a new one in the next cell (cell 8 does that).

This is the same idea as `langgraph/persistence/chatbot.ipynb` (`MemorySaver`) and `langgraph/chatbot/chatbot.py` (`AsyncSqliteSaver`). Different disk.

---

## File: `stm_trimming.ipynb` — shrink `x`, keep the log

Checkpointer = `InMemorySaver` (full history still in state). Before the LLM call:

```python
messages = trim_messages(
    state["messages"],
    strategy="last",
    token_counter=count_tokens_approximately,
    max_tokens=MAX_TOKENS,   # 150 in the notebook
)
response = model.invoke(messages)   # not the full state["messages"]
```

```text
state["messages"]     = everything the checkpointer has  (can grow)
trim_messages(...)    = the slice that actually becomes x
f_θ(trimmed)          = the call
return {messages: [AI]}  →  checkpointer still appends; trim did NOT delete
```

Trimming is a **prompt** policy. `get_state` after many turns still shows the old Human/AI pairs. Tokens in the **API call** stayed under 150; tokens **on disk** did not.

`strategy="last"` = keep the newest messages that fit. Oldest fall out of `x` first (same instinct as `history[-MAX_MESSAGES:]` in `chatModels/chatbot.py`, but counted in **tokens**, not message count).

---

## File: `4_stm_deletion.ipynb` — actually drop turns

`RemoveMessage(id=m.id)` is an instruction to `add_messages`: **delete this id**. Not a chat turn.

```text
START → chat → cleanup → END

cleanup:
    if len(messages) > 10:
        to_remove = msgs[:6]     # earliest six
        return {messages: [RemoveMessage(id=...) for those]}
```

After cleanup, `get_state` **no longer has** those six. Unlike trimming, the past is gone from STM. You cannot time-travel to them unless you forked an older checkpoint first.

`msgs[:6]` = first six (oldest, if the list is chronological). Opposite of `[:-2]`.

---

## File: `5_stm_summarization.ipynb` — compress, then delete

State is `MessagesState` plus `summary: str`.

```text
START → chat → (len(messages) > 6?) → summarize → END
                              └ no → END
```

`chat_node` prepends the summary as a system message, then the remaining verbatim turns, then calls the model.

`summarize_conversation`:

1. Ask the model to summarize (or extend) using the current messages
2. Write `summary`
3. Delete everything **except the last two** messages

```python
messages_to_delete = state["messages"][:-2]   # all except last 2
return {
    "summary": response.content,
    "messages": [RemoveMessage(id=m.id) for m in messages_to_delete],
}
```

`[:-2]` is **not** “the last two.” It is “drop the tail of length 2 from the slice,” i.e. keep `[-2:]` in state.

```text
before:  [m0, m1, m2, m3, m4, m5]  + maybe a summary
delete:  [m0, m1, m2, m3]
keep:    [m4, m5]   plus the new summary string
next x:  system(summary) + m4 + m5 + new user
```

So: old facts live in `summary` (cheap, lossy). The last exchange stays word-for-word (so the model does not immediately forget what it just said).

---

## STM policies next to each other

| Policy | What happens to state | What the LLM sees | Can you get the old text back? |
| --- | --- | --- | --- |
| Nothing (`stm.ipynb`) | full log in checkpointer | full log (until the window explodes) | yes, `get_state` |
| Trim | full log stays | last N **tokens** | yes in state, no in that `invoke`’s `x` |
| Delete | ids gone | whatever remains | no (this thread) |
| Summarize | old turns gone; `summary` field | summary + last 2 | only as the summary’s paraphrase |

All four still need a checkpointer or the next `invoke` starts empty.

---

# Long-term memory (deep dive)

LTM in this folder = LangGraph **Store**, keyed by **user**, not thread.

```text
checkpointer  thread_id="chat-1"   messages for THIS conversation
store         user_id="u1"         facts about THIS person, any conversation
```

```python
namespace = ("user", user_id, "details")   # a folder in the store
store.put(namespace, key, {"data": "User likes pizza"})
store.get(namespace, key)
store.search(namespace)                    # all items in that folder
store.search(namespace, query="...", limit=3)   # semantic, if indexed
```

A new `thread_id` with the same `user_id` should still see “likes pizza.” That is the whole point of LTM vs STM.

`compile(store=store)` injects the store into nodes as `store: BaseStore`. You also pass `user_id` in `config["configurable"]`.

---

## File: `6_ltm_basics.ipynb` — store API, no chatbot

1. `InMemoryStore()` — RAM. Kernel restart → empty.
2. Namespaces isolate users: `("user", "u1")` vs `("user", "u2")`. Searching u2 never returns u1’s pizza.
3. `put(ns, key, value)` / `get(ns, key)` / `search(ns)` (list everything).
4. Optional: `InMemoryStore(index={"embed": embedding_model, "dims": 1536})` then `search(ns, query="what is the user learning", limit=1)`.

The semantic half of this notebook still uses `OpenAIEmbeddings`. The chat LTM notebooks use Cohere. List-all `search(ns)` does **not** need embeddings.

Value shape in this folder is always a small dict `{"data": "..."}`. The store does not care; your nodes must read `.value["data"]`.

---

## File: `7_ltm_implementation.ipynb` — four teaching graphs

Same store idea, four times, each adding one job. All Cohere `command-a-03-2025`.

### 1. Read-only (seeded profile)

You `put` Nitish’s profile **before** the graph. One node `chat`:

- `store.search(("user", user_id, "details"))`
- stuff the blob into `SYSTEM_PROMPT_TEMPLATE` as `{user_details_content}`
- `llm.invoke([system] + state["messages"])`

The model **uses** memory. It never **writes** memory.

### 2. Write-only (`remember_only_node`)

Structured output:

```python
class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[str]
```

`extractor_llm.with_structured_output(MemoryDecision)` reads the last user message and `put`s atomic sentences (`uuid` keys). Returns `"Noted."` — **no** personalized answer. Proves the write path alone.

### 3. Write without duplicates

Each candidate is `MemoryItem(text, is_new)`. The prompt includes **current** store contents. Only `is_new=True` is `put`. Saying “my name is Nitish” twice does not create two rows.

### 4. Merged: remember then chat

```text
START → remember → chat → END
```

`remember_node` extracts + dedupes + `put`. Returns `{}` (does not add a chat message).  
`chat_node` reads the store (including what remember just wrote) and answers.

Config is `user_id`, not `thread_id`. There is **no** checkpointer in this notebook — STM of the live `messages` list exists only **inside one** `invoke`. Across invokes you only have LTM. (A product graph usually has **both**: checkpointer + store.)

---

## File: `8_ltm_postgres.ipynb` — LTM that survives restart

Same merged graph as §4. Backend:

```python
PostgresStore.from_conn_string(
    "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
)
store.setup()                 # once: store tables, not checkpoint tables
graph = builder.compile(store=store)
```

`PostgresSaver` (STM, `stm.ipynb`) and `PostgresStore` (LTM, this file) can share the same Docker Postgres. They are **different tables / APIs**.

Last cell: new Python process, only `store.search(("user", "u1", "details"))` — Nitish is still there. That is the persistence check. `InMemoryStore` cannot pass it.

---

## Checkpointer vs store (do not swap them)

| | Checkpointer | Store |
| --- | --- | --- |
| LangGraph object | `PostgresSaver` / `InMemorySaver` / `SqliteSaver` | `PostgresStore` / `InMemoryStore` |
| Key | `thread_id` | namespace + key (you use `user_id`) |
| Typical payload | full `messages` (+ other state) | small facts `{"data": "..."}` |
| Compile | `checkpointer=...` | `store=...` |
| Notebook | `stm.ipynb` | `6` / `7` / `8` |

A Streamlit sidebar “chat” is a `thread_id`. A logged-in account is a `user_id`. You want both: this chat’s wording (STM) and “this human likes Python” (LTM).

---

## Map onto this repo

```text
stateless invoke          LLMs/llm_demo.py
append history            chatModels/messages.py
STM window (list)         chatModels/chatbot.py  [-MAX_MESSAGES:]
STM + persistence         langgraph/persistence/  +  stm.ipynb (Postgres)
STM trim / delete / sum   stm_trimming / 4_stm_deletion / 5_stm_summarization
semantic LTM (docs)       yt-chatbot / rag_tool / FAISS
user LTM (RAM)            6_ltm_basics / 7_ltm_implementation
user LTM (Postgres)       8_ltm_postgres
```

---

## One-line map

```text
y = f_θ(x)     = inference: frozen weights θ, only x may change
x              = this prompt (must fit the context window)
y              = this reply (copy into the next x if you want STM)
STM            = this thread’s messages (checkpointer) ± trim/delete/summary
LTM            = store[user] facts; retrieve a slice back into x
your job       = build x, call f_θ, write y (and facts) back
```

**One line:** memory is an **application** problem; `f_θ` never changes — you only change `x`.

)
