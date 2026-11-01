# LangChain vs LangGraph

This folder is CampusX LangGraph playlist **video 3**: same models and tools you already have, different **control flow**.

You did not “finish LangChain and start a rival library.” LangGraph is the **runtime for loops, branches, and durable state**. LangChain is still how you talk to Cohere, write prompts, parse output, and define tools.

```text
LangChain   = LEGO bricks (ChatCohere, PromptTemplate, @tool, retriever, LCEL |)
LangGraph   = the circuit board those bricks plug into (state, nodes, edges, loops)
```

In this venv (LangChain 1.x), `from langchain.agents import create_agent` **is a LangGraph**. Classic `AgentExecutor` in `agents/agent.py` is the old loop. Your `currency_converter.py` `for _ in range(6)` is a LangGraph you wrote by hand.

---

## One-line difference

| | LangChain (what you used) | LangGraph (what you learn next) |
| --- | --- | --- |
| Job | **Compose** LLM calls | **Orchestrate** a **stateful** process |
| Shape | Pipe / DAG: A → B → C | **Graph** that can **cycle**: A → B → A |
| Memory of the run | each `invoke` is mostly stateless (unless you pass `messages`) | a **State** object lives across nodes |
| Who picks the next step | **you**, in the `|` order | **you** (edges) **or** the model (conditional edge) |
| Native loops | you write `for` / `AgentExecutor` | **edges back** to an earlier node |
| Persist / resume / HITL | not the point | checkpointers, interrupts |

**LangChain answers:** “call this model with this prompt and these tools.”  
**LangGraph answers:** “what is the **state**, which **node** runs next, when do we **stop**, can we **pause** and come back?”

---

## What LangChain is (you already built this)

LangChain is a **toolkit around LLM APIs**.

| Brick | Your file |
| --- | --- |
| Chat models | `chatModels/`, `ChatCohere` |
| Prompts + parsers | `jsonoutputparser.py`, Pydantic parsers |
| LCEL chains | `masti-with-chains/`, `runnables/` |
| Loaders / split / embed / store | `doc_loaders/` → `vector_store/` |
| Retrievers + RAG | `retrievers/`, `yt-chatbot/chatbot.py` |
| Tools + bind_tools | `tools-calling/` |
| Classic agent loop | `agents/agent.py` (`langchain_classic`) |

LCEL is a **Runnable graph**, but it is meant to be **acyclic**: data flows forward.

```text
question → retriever → format_docs ─┐
                                    ├─ prompt → model → parser → string
question → passthrough ─────────────┘
```

That is `yt-chatbot`. One shot. No “if chunks are junk, search the web and try again” **as a first-class cycle**. You *can* fake a cycle with a Python `while` (`currency_converter.py`). That `while` is the pain LangGraph removes.

CampusX line: LangChain is great at **components**. It is awkward at **long-running agents**.

---

## What LangGraph is

LangGraph is a **state machine** for LLM apps.

```text
State  = TypedDict / dataclass  (messages, conversion_rate, next_step, …)
Node   = a function:  state → partial update
Edge   = who runs after whom
         - always:     retrieve → generate
         - conditional: if tool_calls else END
         - cycle:      grade docs → (bad) → rewrite query → retrieve again
```

You declare the graph **once**. The library walks it, keeps state, can **checkpoint** after every node (SQLite / Postgres), **time-travel**, and **interrupt** for a human (“approve this refund?”).

```text
                 START
                   │
                   ▼
            ┌────────────┐
       ┌───►│ chat node  │
       │    └─────┬──────┘
       │          │
       │     tool_calls?
       │      yes │  no
       │          ▼     ▼
       │     ┌───────┐ ┌─────┐
       └─────│ tools │ │ END │
             └───────┘ └─────┘
```

That **is** `currency_converter.py` and `create_agent`. LangGraph draws it instead of you appending `AIMessage` / `ToolMessage` and hoping the order is right.

CampusX line: LangGraph is for **agentic** workflows — the model (or a grader) **decides the path**.

---

## How they fit together (not rivals)

```text
                    your app
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
   LangChain bricks            LangGraph graph
   ChatCohere, @tool,          StateGraph
   retriever, PromptTemplate   nodes / edges
         │                           │
         └────────── invoke ─────────┘
```

Typical 2026 stack:

```python
# LangChain
model = ChatCohere(...)
tools = [search, weather]
retriever = vector_store.as_retriever()

# LangGraph
graph = StateGraph(State)
graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))
graph.add_conditional_edges("agent", should_continue)
```

Or the short form you already heard:

```python
from langchain.agents import create_agent
agent = create_agent(model, tools=[...])   # LangGraph under the hood
```

You still `pip install langchain langchain-cohere langgraph`. Dropping LangChain and “only using LangGraph” means you rewrite models/tools yourself. Almost nobody does that.

---

## Side-by-side

| | LangChain LCEL | LangGraph |
| --- | --- | --- |
| Abstraction | `Runnable` pipe | `StateGraph` |
| Input | string / dict | **state dict** (and reducers, e.g. append messages) |
| Branch | `RunnableBranch`, parallel dict | **conditional edges** |
| Loop | your `for`, `AgentExecutor` | edge back to a node + `recursion_limit` |
| Debug | `print_ascii()`, print `tool_calls` | graph diagram, **state after each node**, LangSmith |
| Persist | rebuild FAISS every run (your yt-chatbot) | **checkpointer**: resume thread_id |
| Human pause | no | `interrupt_before` / HITL |
| Streaming | tokens from one chain | tokens **and** node updates |
| Best at | RAG retrieve→generate, ETL-style LLM pipes | agents, CRAG, Self-RAG, multi-step research |
| Failure mode | wrong `|` wiring (dict into retriever) | wrong state keys / infinite cycle |

`RunnableBranch` in `conditional-chain.py` is **one** if/else. CRAG is “grade → maybe web search → maybe rewrite → retrieve **again**.” That second trip around the retrieve node is why people move the RAG app onto a graph.

---

## Control flow (the real difference)

**LangChain (deterministic pipe)** — you fix the path at code time:

```text
always:  retrieve → stuff → generate
```

Good when the product is “ask this PDF.” `yt-chatbot` should stay a chain until you need retries.

**LangGraph (path can depend on data)** — you fix the **rules** at code time; the **walk** depends on state:

```text
retrieve → grade
             ├─ relevant  → generate → (Self-RAG: supported?) → END
             └─ junk      → rewrite query → retrieve   ← cycle
```

Same bricks (`retriever`, `ChatCohere`). Different **skeleton**.

ReAct `AgentExecutor` is a **hidden** graph with one cycle (Thought → Action → Observation → Thought). You cannot easily insert “grade documents” in the middle. LangGraph lets you **draw** that extra node.

---

## State vs messages-you-pass-yourself

LangChain tool loop (you already felt this):

```text
messages = [Human]
ai = model.invoke(messages)
messages.append(ai)          # forgot this → Cohere 400 tool_call_id
messages.append(tool_msg)
```

LangGraph: `messages` is a **field on State** with an **add reducer**. Each node returns `{"messages": [new]}`; the graph **appends**. You do not hand-manage the list (unless you want to).

Other state is first-class, not `InjectedToolArg` + a global `state = {}`:

```text
# currency_converter.py today
state = {"conversion_rate": None}   # Python global, easy to get wrong

# LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    conversion_rate: float | None
```

The convert node reads `state["conversion_rate"]`. No hidden `InjectedToolArg` ritual unless you still want it.

---

## When to use LangChain (keep the chain)

Use **LCEL / LangChain only** when the path is **short, fixed, and one-shot**.

| Project | Why chain is enough |
| --- | --- |
| “5 facts about `{topic}`” | `simple-chain.py` |
| Summarise after a report | `sequencial-chain.py` |
| Notes + quiz in parallel | `parallel-chain.py` |
| Sentiment → canned reply | `conditional-chain.py` (`RunnableBranch`) |
| YouTube Q&A, always top-4 chunks | `yt-chatbot/chatbot.py` |
| One multiply, one search | `calling.py` (even one tool call) |

Signals you **do not** need LangGraph yet:

- One `invoke`, one answer  
- No “try again with a better query”  
- No pause for a human  
- No “resume this chat tomorrow” (or you only need a list in RAM)  
- Debugging is “print the string”

**Rule:** if you can draw it as a **straight line** (or a fan-out that merges once), stay on LangChain.

---

## When to use LangGraph (add the graph)

Use LangGraph when **the next step depends on the last result**, or the job **outlives one HTTP request**.

| Project | Why a graph |
| --- | --- |
| Currency: rate **then** convert | two tools, order matters (`currency_converter`) |
| Search + weather agent | unknown number of tool rounds (`agent.py`) |
| CRAG | retrieve → grade → (web / rewrite) → retrieve |
| Self-RAG | decide *whether* to retrieve; fact-check the answer |
| Multi-hop “compare A and B” | retrieve twice, different queries |
| Support bot that must **approve** a refund | HITL interrupt |
| Chat that survives refresh | checkpointer + `thread_id` |
| Research: plan → search → write → critique → rewrite | cycles + shared state |
| MCP + several tools + RAG | `create_agent` / ToolNode |

Signals you **do** need LangGraph:

- You wrote `for _ in range(6)` or `while tool_calls`  
- You forgot to `append(ai_message)` and Cohere 400’d  
- You want a node that is **not** “the LLM” (grader, rewriter, validator)  
- You need **persistence**, **streaming of steps**, or **time travel**  
- Two agents share a state (researcher + writer)

**Rule:** if you need a **loop**, a **grader**, or a **checkpoint**, it is a graph. LangChain bricks still sit **inside** the nodes.

---

## Same product, two designs

**YouTube transcript Q&A**

| Version | Stack | When |
| --- | --- | --- |
| v1 you have | LCEL parallel + FAISS | learning RAG, one question |
| v2 | LangGraph: retrieve → generate | add chat history / checkpoint |
| v3 | + grade docs → rewrite (CRAG) | retrieval is often wrong |
| v4 | + “should I retrieve?” (Self-RAG) | many chit-chat questions |

You do **not** throw away `CohereEmbeddings` or the splitter. You wrap them in nodes.

**Currency converter**

| Version | Stack |
| --- | --- |
| Now | `bind_tools` + manual message list + global rate |
| Next | `StateGraph` or `create_agent` + `conversion_rate` on state |

---

## `create_agent` vs `AgentExecutor` vs LCEL

Confusion in 2026 naming:

| API | Package | Engine |
| --- | --- | --- |
| `prompt \| model \| parser` | `langchain_core` | LCEL (LangChain) |
| `create_react_agent` + `AgentExecutor` | `langchain_classic` | old ReAct **text** loop |
| `create_agent` | `langchain.agents` | **LangGraph** tool-calling agent |
| `StateGraph` | `langgraph` | you draw every node |

So: “I used LangChain agents” might mean classic ReAct **or** a LangGraph. Look at the import.

For Cohere, prefer `create_agent` / a graph over `AgentExecutor`. You already saw ReAct parsing wobble (`handle_parsing_errors`).

---

## What does **not** change

- `.env` keys, `ChatCohere`, `@tool`, FAISS, retrievers  
- RAG **idea** (retrieve → stuff → generate)  
- Tool **idea** (model requests, Python executes)  
- You still debug bad chunks and bad tool descriptions  

LangGraph does not embed better or hallucinate less by magic. It only lets you **insert** a grade/retry **node**. Garbage index → garbage graph.

---

## Decision sketch

```text
Is the path a fixed pipe (maybe one branch)?
  yes → LangChain LCEL. Stop.
  no  → Does a model or grader choose the next step, more than once?
          yes → LangGraph (or create_agent if it's "just tools").
          no  → still a chain; you over-complicated it.

Need resume / HITL / shared state across days?
  yes → LangGraph checkpointer, even if the happy path is a pipe.
```

| If you catch yourself… | Use |
| --- | --- |
| `chain = a \| b \| c` and it works | LangChain |
| `messages.append` in a `while` | LangGraph |
| `RunnableBranch` with 2 arms | LangChain is fine |
| CRAG / Self-RAG / multi-agent | LangGraph |
| Interview “LangChain vs LangGraph” | bricks vs state machine |

---

## Mental model

```text
LangChain  = kitchen tools (knife, pan, recipe card)
LangGraph  = the service ticket: courses, send-back to grill, hold for chef

yt-chatbot     = one plate, one recipe          → chain
currency loop  = ticket that revisits the pass  → graph
CRAG           = expo tastes it, sends it back  → graph
create_agent   = standard ticket for "use tools"→ graph, pre-drawn
```

CampusX: **LangChain = components. LangGraph = workflows that can think and loop.** You will use **both** in one project.

---

## Map onto this repo

| You built | Library | Next move |
| --- | --- | --- |
| `masti-with-chains/` | LangChain | stay |
| `yt-chatbot/` | LangChain RAG | graph only if you add grade/retry/chat persist |
| `tools-calling/calling.py` | LangChain | stay |
| `currency_converter.py` | hand-rolled graph | rebuild in `langgraph/` |
| `agents/agent.py` | classic ReAct | `create_agent` or StateGraph |

Then this playlist: basics → workflows → chatbot → persistence → tools → RAG → CRAG / Self-RAG.

---

## Cheat sheet

| Question | Answer |
| --- | --- |
| Do I uninstall LangChain? | No |
| Is LangGraph “better RAG”? | No. Better **control** around RAG |
| Can LCEL loop? | Only if **you** loop |
| Can a graph be a straight line? | Yes, and that is overkill |
| What does `create_agent` use? | LangGraph |
| What did `AgentExecutor` use? | Its own ReAct loop (`langchain_classic`) |
| State vs scratchpad | State is typed and shared; scratchpad is a string in the ReAct prompt |
