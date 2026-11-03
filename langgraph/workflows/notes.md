# LangGraph workflows

CampusX playlist **videos 5–8**. Previous notes:

- `langchain_vs_langgraph/` — why a graph exists
- `core_concepts/` — node / edge / state / reducer / Pregel **vocabulary**

This folder is those words **as code**. Every file is the same four lines in a different **shape**:

```text
1. State   = TypedDict   (the notebook for one run)
2. Nodes   = functions   (state → dict of updates)
3. Edges   = who runs next
4. compile + invoke
```

You are not learning a new library each notebook. You are learning **four topology patterns**. Pick the pattern that matches the product.

```text
video 5  sequential     line          A → B → C
video 6  parallel       fork–join     A ─┬─ B ─► D
                                      └─ C ─┘
video 7  conditional    diamond       A → (if) → B or C
video 8  iterative      cycle         A → B ⇄ C  until stop
```

Model everywhere (unless noted): `ChatCohere("command-a-03-2025")`. LangGraph is Python — not a C++ graph engine. `StateGraph` stores an adjacency list of Python functions; Pregel walks it.

---

## How to restudy this folder

Run in this order. Each file adds **one** idea.

| Order | File | New idea |
| --- | --- | --- |
| 1 | `sequencial.ipynb` | graph with **no LLM** (BMI) |
| 2 | `basic_seq.ipynb` | one LLM node |
| 3 | `blog.py` | sequential **LLM chain** on shared state |
| 4 | `parallel.py` | fan-out / fan-in, **partial updates** |
| 5 | `essay_workflow.ipynb` | parallel **LLM judges** + `operator.add` |
| 6 | `conditional.ipynb` | `add_conditional_edges` (routing) |
| 7 | `iterative.ipynb` | **cycle** + stop rule + history lists |

After each file, draw the mermaid cell (`workflow.get_graph().draw_mermaid_png()`). If the picture does not match the edges you think you wrote, the picture is usually right.

---

## Shared machinery (every file)

### Imports you always need

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
```

Later files add:

| Extra | Why |
| --- | --- |
| `Literal` | restrict a field or a router return to a fixed set of strings |
| `Annotated` + `operator` | reducers (`operator.add` on lists / scores) |
| `BaseModel`, `Field` | structured output from Cohere |
| `load_dotenv()` | **call it** (`()`). Forgetting `()` loads nothing |

### START and END

They are **virtual**. You never write `def START`. They mark “begin here” and “nobody left to run.”

```text
START  →  your first real node
your last real node  →  END
```

### The 5-line skeleton

```python
graph = StateGraph(SomeState)
graph.add_node("name", function)
graph.add_edge(START, "name")
graph.add_edge("name", END)
workflow = graph.compile()          # freeze topology
workflow.invoke({"field": "value"}) # run it
```

`compile()` does **not** call the LLM. It builds the Pregel app. `invoke` is the run. If you change a node function, **re-run the compile cell** or you are still executing the old graph.

### Two ways a node can return

**A. Mutate and return the whole state** (`sequencial.ipynb`, `blog.py`, `basic_seq.ipynb`):

```python
def calculate_bmi(state: BMIState) -> BMIState:
    state["bmi"] = round(weight / (height ** 2), 2)
    return state
```

Works on a **line**. Dangerous in **parallel**: two nodes mutating the same dict is a race.

**B. Return a partial dict** (`parallel.py`, essay, conditional, iterative) — the LangGraph style:

```python
def calc_strike_rate(state: CricketState) -> CricketState:
    return {"strike_rate": sr}   # only the keys you changed
```

LangGraph **merges** this into state. Other keys stay. Prefer this always.

The annotation `-> BMIState` is a **hint**. Python does not check it. Returning `"err hai"` (see `blog.py`) will not crash at define-time.

### State is not validated at runtime

`TypedDict` is for you and the editor. `score: int` does **not** force Cohere to return an int. If the model returns `"**Rating: 9/10**"`, state stores a **string**. Structured output (`with_structured_output`) is how you actually constrain the model.

### Drawing the graph

```python
from IPython.display import Image
Image(workflow.get_graph().draw_mermaid_png())
```

This draws the **static** map: `add_node` / `add_edge` / `add_conditional_edges`. It does **not** execute your `if` statements. If a branch is missing from the picture, you did not **declare** it (path map or `Literal` return). Runtime can still take that branch. Picture and invoke are different layers — `iterative.ipynb` taught this the hard way.

---

## 1. Sequential — a straight line

**Files:** `sequencial.ipynb`, `basic_seq.ipynb`, `blog.py`  
**Playlist:** video 5  
**LCEL cousin:** `a | b | c`

```text
START → A → B → C → END
```

One node at a time. Each node reads state, writes fields, the next node sees them. No skip, no loop, no fork.

When it fits: outline then blog then score; retrieve then generate; BMI then label.  
When it breaks: “if the score is bad, rewrite” (that is **iterative**).

### Superstep picture (blog)

```text
superstep 0: START schedules generate_outline
superstep 1: run generate_outline   → state["outline"] = ...
superstep 2: run generate_blog      → state["blog"] = ...
superstep 3: run generate_score     → state["score"] = ...
superstep 4: END
```

---

### `sequencial.ipynb` — BMI, no LLM

The cleanest file in the folder. Learn the API here before you add Cohere.

**State**

```python
class BMIState(TypedDict):
    weight_kg: float
    height_m: float
    bmi: float
    label: str
```

You only pass `weight_kg` and `height_m` into `invoke`. The other keys appear as nodes write them.

**Nodes**

1. `calculate_bmi` — `bmi = weight / height²`, rounded to 2 decimals, written onto state.
2. `label_bmi` — reads `state["bmi"]` (must run **after** calculate, which is why the edge is a line, not a fork).

**Edges**

```text
START → calculate_bmi → label_bmi → END
```

**Invoke you already ran**

```python
{'weight_kg': 20, 'height_m': 1.2}
→ {'weight_kg': 20, 'height_m': 1.2, 'bmi': 13.89, 'label': 'bad bmi'}
```

`20 / 1.2² = 13.888… → 13.89`. Underweight. The `if` in the notebook is **medically backwards** (`bmi > 30` labelled `'good bmi'`). The graph still did its job: it ran calculate, then label. Read the `if` as “how a node uses state,” not as a BMI chart.

**What this file teaches**

- Topology is independent of LLMs.
- Order is **edges**, not the order you defined the functions.
- `invoke` returns the **full** state dict, including inputs.

---

### `basic_seq.ipynb` — one LLM node

Same shape, one node that calls Cohere.

**State:** `question`, `answer`.

```text
START → generate_answer → END
```

```python
def generate_answer(state: LLMState) -> LLMState:
    answer = model.invoke(f"answer the following question {state['question']}")
    state["answer"] = answer   # whole AIMessage, not .content
    return state
```

**Look at the output.** `answer` is an `AIMessage(...)` object, not a string. `blog.py` uses `.content` and stores a string. Know the difference:

| You store | You print |
| --- | --- |
| `model.invoke(prompt)` | `AIMessage(content="...", response_metadata=...)` |
| `model.invoke(prompt).content` | just the text |

**Cell order bug you hit:** compiling before the import cell → `NameError: StateGraph is not defined`. Notebooks remember nothing until you run the import cell. Always Run All from the top after a kernel restart.

---

### `blog.py` — sequential LLM pipeline

```text
START → generate_outline → generate_blog → generate_score → END
```

**State:** `topic`, `outline`, `blog`, `score`.

You invoke with `{'topic': 'AI in India'}`. Each node adds a field. `print(res)` dumps the **entire** state. `outline` and `blog` are huge markdown strings, so `score` sits at the **end** and is easy to miss. That is why you thought the rating was missing. It was `'score': '**Rating: 9/10**'` (a **string**, despite `score: int`).

**Why score is a string**

```python
score: int = model.invoke(prompt).content
```

The `: int` is a variable annotation. `.content` is still `str`. The model wrote a sentence. To get a real integer you need `with_structured_output` (essay / iterative) or parse the text yourself.

**Other landmines in this file**

```python
if outline == '':
    return 'err hai'   # not a dict / not BlogState
```

If outline were empty, LangGraph would try to merge a **string** into state and blow up. Dead code if outline is always generated first — still a bad return type.

It also **mutates** `state["outline"] = outline` then `return state`. Fine on a line. Copy-paste this into a parallel node and you share a mutating dict.

**`workflow.stream` vs `invoke`:** `invoke` waits for END and returns final state. `stream` yields after each superstep so you can watch outline appear, then blog, then score. Same graph.

---

## 2. Parallel — fork, then join

**Files:** `parallel.py`, `essay_workflow.ipynb`  
**Playlist:** video 6  
**LCEL cousin:** `RunnableParallel`

```text
              ┌─ node B ─┐
START ───────►├─ node C ─┼──► join_node → END
              └─ node D ─┘
```

B, C, D have **no** edge between them. They are scheduled in the **same superstep**. The join node waits at the **barrier** until all three have written. Then it runs.

**Rule that causes `InvalidUpdateError`:** two nodes in the same superstep must not write the **same last-value key**. Either:

- give them **different keys** (`language_feedback` vs `analysis_feedback`), or
- put a **reducer** on the shared key (`individual_scores: Annotated[list[int], operator.add]`).

---

### `parallel.py` — cricket stats (no LLM in the math nodes)

**State**

```python
class CricketState(TypedDict):
    runs, balls, fours, sixes     # inputs
    strike_rate                   # calc_strike_rate
    runs_boundary_per             # calc_runs_boundary_per
    bpd                           # calc_ball_per_boundary
    summary                       # summary (join)
```

**Each math node returns only its key** — this is the file to copy when you write parallel LLM nodes.

```python
return {"strike_rate": sr}          # not the whole state
return {"runs_boundary_per": b_per}
return {"bpd": bpd}
```

Three different keys → no reducer needed → no collision.

**Edges**

```text
START → calc_strike_rate         ─┐
START → calc_runs_boundary_per   ─┼→ summary → END
START → calc_ball_per_boundary   ─┘
```

**Invoke**

```python
{'runs': 100, 'balls': 70, 'fours': 3, 'sixes': 4}
```

```text
strike_rate        = 100/70 * 100     = 142.86
boundary_runs      = 3*4 + 4*6        = 36
runs_boundary_per  = 36/100 * 100     = 36.0
bpd                = 70 / (3+4)       = 10.0
```

`summary` **must** be after the fan-in. If you attached `summary` to START as well, it would run in the same superstep as the calcs and read missing keys.

**Supersteps**

```text
superstep 1: run the three calc_* nodes together
             barrier: merge three partial dicts into state
superstep 2: run summary (now all three fields exist)
```

If `summary` printed empty strike_rate, you wired the join too early.

---

### `essay_workflow.ipynb` — parallel LLM evaluators

Same cricket shape, but each fork is an LLM call with structured output. This is the UPSC-style grader.

```text
                    ┌─ evaluate_language ──► language_feedback + score
START + essay ──────┼─ evaluate_analysis ──► analysis_feedback + score
                    └─ evaluate_thought  ──► clarity_feedback  + score
                                    │
                                    ▼
                            final_evaluation
                            (average + summary)
                                    │
                                   END
```

**State**

```python
class UPSCState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float
```

Read that `Annotated` line until it is boring. Each judge returns:

```python
return {"language_feedback": res.feedback, "individual_scores": [res.score]}
```

`language_feedback` — **replace** (unique key per node).  
`individual_scores: [8]` — **add** (list concat). After three judges: `[8, 9, 9]`.  
`final_evaluation` does `sum(...) / len(...)` → `avg_score`.

If `individual_scores` had **no** reducer, three parallel writes of that key → `InvalidUpdateError: At key 'individual_scores': Can receive only one value per step`.

**Structured output**

```python
class EvaluationSchema(BaseModel):
    feedback: str
    score: int   # description 'score out of 10'
```

Cohere `with_structured_output` compiles this to JSON Schema. **`Field(ge=0, le=10)` became `minimum` / `maximum` and Cohere 400'd** (`constraint: 'minimum'`). OpenAI accepts those. Cohere does not. Put the range in the **description**, not in `ge`/`le`. `Literal` / enums are fine (sentiment, diagnosis, tweet evaluation all use them).

**Bug you hit: two nodes, one key**

Copy-paste made `evaluate_analysis` return `language_feedback` and `evaluate_thought` return `thought_feedback` (a key **not even on state**). Two writers on `language_feedback` in one superstep → `InvalidUpdateError`. Fix: each node writes **its own** feedback key, and all three write `individual_scores` **with** the reducer.

**Generous scores on `essay2`**

You wrote a broken essay on purpose. The graph **did** run it (`individual_scores` `[8, 9, 9]`, `avg_score ≈ 8.67`). The judge is the problem, not Pregel.

Why:

1. Prompts say “evaluate and score out of 10” with **no rubric / no “be harsh.”** Chat models default to 7–9.
2. `temperature=0.8` — creative, not a strict marker.
3. `score` Field has no calibration (“8+ is exceptional”).
4. `essay2` still mentions jobs, farming, Bhashini, privacy — the model grades the **topic checklist**, not the grammar.
5. `language_feedback` on that run described a *well-structured* essay with IndiaAI Mission — that is the **good** essay hallucinated from keywords.
6. `final_evaluation` **never re-reads the essay**. It averages three already-generous numbers.

To actually get a 2/10 you need a marking scheme in the prompt, few-shot 2 vs 9, and low temperature. The graph will average whatever the judges emit.

**`final_evaluation` is the join**

It waits for all three feedback strings, summarises with a normal (non-structured) `model.invoke`, averages scores. Join nodes are ordinary nodes. The “wait” is the incoming edges, not a special API.

---

## 3. Conditional — pick one path

**File:** `conditional.ipynb`  
**Playlist:** video 7  
**LCEL cousin:** `RunnableBranch` / `conditional-chain.py`

```text
                         ┌─ positive_response ──────────────► END
generate_review → find_sentiment
                         └─ run_diagnosis → negative_response ► END
```

**One** successor after `find_sentiment`, chosen by a **router function**. Not both. That is the difference from parallel.

If you wrote:

```python
graph.add_edge("find_sentiment", "positive_response")
graph.add_edge("find_sentiment", "run_diagnosis")
```

**both** would run (cricket-style fan-out). Routing is `add_conditional_edges`, not two normal edges.

---

### State and schemas

```python
class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str
```

Two Pydantic models, two structured models:

| Schema | Fields | Used in |
| --- | --- | --- |
| `Sentiment` | `sentiment: Literal["positive", "negative"]` | `find_sentiment` |
| `DiagnosisSchema` | `issue_type`, `tone`, `urgency` (all Literals) | `run_diagnosis` |

`structured_model2` must wrap **`DiagnosisSchema`**, not `Sentiment`. Otherwise diagnosis cannot return those three fields.

`run_diagnosis` stores `response.model_dump()` → a plain `dict` on state, so `negative_response` can do `diagnosis["issue_type"]`.

---

### The router

```python
def check_sentiment(state: ReviewState) -> Literal["positive_response", "run_diagnosis"]:
    if state["sentiment"] == "positive":
        return "positive_response"
    return "run_diagnosis"
```

**It returns node names, not booleans, not the sentiment string.** LangGraph looks up that string in the graph.

The `-> Literal["positive_response", "run_diagnosis"]` is **not** required for `invoke` to work. It **is** how LangGraph (without a path-map dict) knows which branches to **draw** and to treat as legal destinations. That is why the review diagram shows both arms, and why dropping Literal **and** the path map in the tweet graph left `optimize` floating.

Two equivalent APIs:

```python
# A. return node names; Literal (or path map) lists destinations
graph.add_conditional_edges("find_sentiment", check_sentiment)

# B. return labels; dict maps label → node (or END)
graph.add_conditional_edges("evaluate", route, {
    "approved": END,
    "needs_improvement": "optimize",
})
```

`A` is what `conditional.ipynb` uses. `B` is what makes mermaid print `approved` / `needs_improvement` like the CampusX screenshot.

---

### Extra edge leftover

```python
graph.add_conditional_edges("find_sentiment", check_sentiment)
...
graph.add_edge("find_sentiment", END)   # extra
```

A **normal** edge from the same node is a second exit. After sentiment, LangGraph may also schedule END as a parallel destination. The negative path still ran in your output (you got `diagnosis`), but the mermaid gets noisier and the extra line is not part of the design. The intended graph is only the conditional + the two terminal chains.

Also in this notebook: `load_dotenv` without `()`, and `model = model = ChatCohere(...)`. Harmless-ish; call `load_dotenv()`.

**`generate_review` always asks for a negative review**, so you rarely exercise `positive_response`. To test the other arm, invoke with `{'review': 'this is amazing'}` and skip generate, or change the generate prompt.

**Run you already saw:** random vacuum review → `sentiment='negative'` → `issue_type='Performance'`, `tone='frustrated'` → empathetic `response`. Routing worked.

---

## 4. Iterative — generate, grade, loop

**File:** `iterative.ipynb`  
**Playlist:** video 8  
**LCEL cousin:** `currency_converter.py` `for _ in range(6)`  
**Pattern name:** evaluator–optimizer

This is the first **cycle**. An edge goes **backwards**.

```text
START → generate → evaluate ── approved or hit max_iteration ──► END
                         │
                         └── needs_improvement ──► optimize ──► evaluate
                                                   (iteration += 1)
```

`generate` runs **once**. Later drafts come from `optimize`. `evaluate` is the hub.

---

### State

```python
class TweetState(TypedDict):
    topic: str
    tweet: str                    # current draft (replace)
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str                 # current critic note (replace)
    iteration: int                # replace; starts at 1
    max_iteration: int            # replace; you pass 3
    tweet_history: Annotated[list[str], operator.add]
    feedback_history: Annotated[list[str], operator.add]
```

**Why histories need `operator.add`**

Each loop you return `{"tweet_history": [new_tweet]}` — a **one-element list**. Without a reducer, replace would wipe history down to that one tweet. With `add`, `[t1] + [t2] + [t3]`. Same for feedback.

**Do not** return the whole history list yourself (`state["tweet_history"] + [tweet]`) **and** use `operator.add` — you would double-append. Return only the new item in a list.

`tweet` / `feedback` stay **replace**: the join node and the optimizer want the **latest** draft, not a concatenated string.

---

### Stop rule

```python
def route_evaluation(state: TweetState):
    if state["evaluation"] == "approved" or state["iteration"] >= state["max_iteration"]:
        return END
    return "optimize"
```

Two exits, same as a `while`:

- quality: critic said approved
- safety: `iteration` reached `max_iteration` even if still bad

You start `iteration` at **1** in `initial_state`. `generate` does not increment. `optimize` does `iteration + 1`. With `max_iteration=3`:

```text
iter 1: generate → evaluate   (fail) → optimize
iter 2: evaluate              (fail) → optimize
iter 3: evaluate              (fail or pass) → END   because 3 >= 3
```

That is at most 3 tweets in `tweet_history`. `recursion_limit` on the compiled graph is a **second** cap (max supersteps). Your `max_iteration` is the product-level cap; keep it.

---

### Path map vs returning END (the mermaid lesson)

This is the idea the picture fought you on.

`route_evaluation` **running** can `return END` or `return "optimize"`. Python does that at invoke time.

`draw_mermaid_png` **does not run** that `if`. It only draws destinations that were **declared**:

| You declare destinations with | Picture |
| --- | --- |
| path map `{END: END, "optimize": "optimize"}` | evaluate splits to `__end__` and `optimize` |
| `-> Literal["__end__", "optimize"]` (or node names) | same, labels = those strings |
| **neither** | START → generate → evaluate → END, and `optimize` **floats** |

The floating `optimize` box is not a different workflow. It is an undeclared road. Your current cell **does** pass the path map, so the drawing should show the loop again.

`approved` / `needs_improvement` on the CampusX screenshot are **edge labels** from a path map whose **keys** were those words. They are not extra nodes. You do **not** `add_edge("approved", END)`.

---

### Weak generator, strong critic

Same-model generate+evaluate approved on iteration 1 (generic-but-fluent tweets still score well — same lesson as `essay2`).

Split the models:

| Role | Model | Temperature | Job |
| --- | --- | --- | --- |
| `generator` | `command-r7b-12-2024` | 1.0 | first tweet + rewrites |
| `evaluator` | `command-a-03-2025` | 0 | structured approve / reject |

Generate prompt is intentionally lazy: “short casual tweet.” Critic rubric requires a hook **and** a concrete detail (number, place, company). First drafts fail; optimize reads feedback.

**Run you already have**

```text
tweet_history[0]  "AI in India is booming! Exciting times ahead..."
feedback[0]       too generic, add a statistic, sharper hook
tweet_history[1]  "India's AI sector just hit a milestone: AI startups raised $2.5B in 2023! ..."
feedback[1]       approved
iteration         2
```

The cycle did work: one reject, one rewrite, then END before hitting 3. If `command-r7b-12-2024` 404s on your key, swap the generator to `command-r-08-2024`.

---

### Supersteps for that run

```text
s0: START → generate
s1: generate     tweet=draft1, history=[draft1]
s2: evaluate     evaluation=needs_improvement, feedback=..., history=[fb1]
s3: route → optimize
    optimize     tweet=draft2, iteration=2, history=[draft1, draft2]
s4: evaluate     evaluation=approved, history=[fb1, fb2]
s5: route → END
```

A cycle is not recursion in Python. It is Pregel **waking** `evaluate` again. `recursion_limit` counts these wakes.

---

## Side-by-side: every file

| File | Pattern | LLM? | Router? | Reducer? | Cycle? |
| --- | --- | --- | --- | --- | --- |
| `sequencial.ipynb` | line | no | no | no | no |
| `basic_seq.ipynb` | line | 1 node | no | no | no |
| `blog.py` | line | 3 nodes | no | no | no |
| `parallel.py` | fork–join | no (math) | no | no (unique keys) | no |
| `essay_workflow.ipynb` | fork–join | 3 + join | no | `operator.add` on scores | no |
| `conditional.ipynb` | diamond | yes | `check_sentiment` | no | no |
| `iterative.ipynb` | loop | yes (2 models) | `route_evaluation` | `operator.add` on histories | **yes** |

---

## Return values — one page

| Return | Meaning |
| --- | --- |
| `{"strike_rate": 142.8}` | patch; other keys untouched |
| `state` after mutation | whole notebook; ok on a line |
| `{"individual_scores": [8]}` | with `operator.add` → append 8 |
| `{"tweet_history": [tweet]}` | with `operator.add` → append this draft |
| `"positive_response"` from a **router** | next **node name** (not a state update) |
| `END` from a **router** | halt |
| `"err hai"` | invalid patch; don’t |

Routers are **not nodes that update state**. They only choose the next edge. `check_sentiment` / `route_evaluation` are passed to `add_conditional_edges`, not `add_node`.

---

## Bugs you already paid for (keep this list)

1. **`ge` / `le` on Pydantic + Cohere** → HTTP 400 `minimum`. OpenAI OK. Drop constraints; describe the range.
2. **Two parallel nodes, same replace-key** → `InvalidUpdateError`. Distinct keys or a reducer.
3. **Router without path map and without `Literal`** → mermaid is a straight line; `optimize` floats. Invoke can still loop.
4. **`score: int` but `.content`** → string in state. Use structured output.
5. **`print(res)` on a blog** → rating is last; print `res["score"]`.
6. **Store `AIMessage` instead of `.content`** → ugly state (`basic_seq`).
7. **`load_dotenv` no `()`** → env not loaded.
8. **Generous LLM judge** → graph averaged 8.7 on a broken essay. Rubric + temperature 0, or a weaker generator vs a harsh evaluator.
9. **`final_evaluation` does not re-read the essay** — join nodes only see what forks wrote.
10. **Copy-paste returns the wrong key** — the parallel essay bug. Read every `return {`.
11. **Notebook cell order** — `StateGraph` NameError if compile runs before imports.
12. **Extra `add_edge(source, END)` next to a conditional** — leftover second exit.

---

## Mental model (how to look at a new graph)

Ask, in this order:

1. **What is on state?** Inputs vs things nodes will fill.
2. **Which keys are replace vs add?** If two writers can touch it in one superstep, it needs a reducer or unique names.
3. **What is the picture?** Line, fork, diamond, or loop.
4. **Who decides the next node?** Fixed edge, or a function that returns a **name**.
5. **What stops a loop?** Quality flag, counter, or both.
6. **Does mermaid match 3–5?** If not, destinations are undeclared.

```text
LCEL baton:          output of A is input of B
LangGraph notebook:  A writes fields; B reads fields; edges pick B
```

That is the whole folder.

---

## Cheat sheet

```text
StateGraph(State)              create empty graph typed by State
add_node(name, fn)             register a function
add_edge(A, B)                 always A → B  (fan-out if several from A)
add_conditional_edges(A, fn)   fn(state) → next name or END
START / END                    virtual entry / halt
compile()                      build Pregel app
invoke(dict)                   run until END, return full state
stream(...)                    yield after each superstep

replace (default)              latest write wins
Annotated[list, operator.add]  concat lists / sum numbers
Literal["a","b"]               enum for structured output AND for drawing routers

with_structured_output(Model)  force dict/Pydantic instead of prose
temperature                    0 = grader; 0.8–1.0 = writer
```

---

## What this folder is not yet

You do **not** have here (later CampusX videos / `core_concepts` cheat sheet):

- persistence / checkpointers / time travel
- streaming tokens vs `workflow.stream` of state
- `Send` / orchestrator–workers (dynamic N)
- HITL `interrupt`
- subgraphs, short/long memory

Those sit **on top** of these four shapes. If the topology is wrong, a checkpointer will only save the wrong topology.

---

## 30-second recap

```text
sequencial / basic_seq / blog   →  a pipe with a shared dict
parallel / essay                →  same superstep; unique keys or add
conditional                     →  one router, one winner
iterative                       →  router can point backwards; cap the loop
```

If you can draw those four from memory and say which reducer each state field needs, you understand this folder.
)
