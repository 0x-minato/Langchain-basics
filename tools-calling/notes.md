# Tool calling

This folder is the first time the **model can take an action**, not only write text.

Earlier folders: prompt → model → string. RAG stuffed **documents** into the prompt. Here the model emits a **structured function call**; **your Python** runs it; you send the result back.

```text
user message
    → model.bind_tools([...]).invoke(messages)
    → AIMessage.tool_calls   # name + args  (not the answer yet)
    → tool.invoke(that call)
    → ToolMessage            # result
    → append both to messages
    → invoke again
    → AIMessage.content      # the sentence for the user
```

That loop **is** a tiny agent. LangGraph later names it. `tools.py` only **defines** tools. `calling.py` runs the loop once for multiply. `currency_converter.py` loops until the model stops calling tools.

Needs `CO_API_KEY` / `COHERE_API_KEY`. Search tool also needs `pip install -U ddgs`. Currency tool needs `requests` and a live exchange-rate HTTP API. Run from the **repo root** with `venv312`.

---

## What is a tool?

A **tool** is a function the LLM is **allowed to request**, with a JSON schema for arguments.

| Piece | Who fills it |
| --- | --- |
| `name` | you (`multiply`, `convert`) |
| `description` | you — this is how the model **chooses** the tool |
| `args` / `args_schema` | Pydantic / type hints — what the model must pass |
| body (`_run` / `func`) | you — HTTP, math, search |

The model **never executes Python**. It only outputs something like:

```python
{'name': 'multiply', 'args': {'a': 10, 'b': 6}, 'id': '...', 'type': 'tool_call'}
```

You run `multiply.invoke(that)`. Then the model sees the number `60` and writes “10 times 6 is 60.”

**One line:** tool = “the model can ask for a function.” RAG = “the model can read chunks.” Different holes in the same chatbot.

### Tool vs chain vs RAG

| | Chain | RAG | Tool |
| --- | --- | --- | --- |
| Extra info | none (just prompt) | retrieved docs | **live** side effect (search, FX rate, multiply) |
| Who picks the action | you (`\|`) | retriever | **the model** (`tool_calls`) |
| Typical invoke | dict / string | question string | `{'a': 2, 'b': 3}` or a full `tool_call` dict |

### Every tool is a Runnable

`BaseTool` subclasses `RunnableSerializable`. So you get `invoke`, `|`, `batch` — same surface as prompts and chat models.

```text
Runnable  →  BaseTool  →  StructuredTool / @tool result / your subclass
```

| You call | What happens |
| --- | --- |
| `tool.invoke({'a': 2, 'b': 3})` | validate with `args_schema`, then `_run` / `func` |
| `tool._run(2, 3)` | skip schema, tracing, LLM-shaped dicts |

Do **not** chain `_run`. You can `multiply | something` because `multiply` is a runnable.

Agents, `bind_tools`, and LCEL all call **`invoke`**.

---

## `tools.py` — three ways to build a tool (+ a toolkit)

Same `multiply(a, b)` written four times so you see the APIs. Only the toolkit `print` is live at the bottom.

### 1. Built-in: `DuckDuckGoSearchRun`

```python
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
search_tool.invoke('todays geopolitics news')
```

Community tools are **already** `BaseTool`s. `invoke` takes a **string** here (the query), not a dict.

Needs the renamed package:

```powershell
pip install -U ddgs
```

Old name was `duckduckgo-search`. Missing `ddgs` → `ImportError: Could not import ddgs`.

`langchain-community is being sunset` is a **warning**, not a crash. The import still works.

### 2. `@tool` — fastest custom tool

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply 2 numbers"""
    return a * b

multiply.invoke({'a': 2, 'b': 3})  # 6
```

The decorator reads:

- function **name** → tool name
- **docstring** → description (what the LLM reads)
- **type hints** → JSON schema for `a`, `b`

No docstring → a worse tool. The model picks tools by that sentence.

Inspect:

| Attr | Meaning |
| --- | --- |
| `multiply.name` | `'multiply'` |
| `multiply.description` | docstring |
| `multiply.args` | dict of arg JSON schemas |
| `multiply.args_schema.model_json_schema()` | full JSON Schema the LLM sees |

**Do not** call `multiply.model_json_schema()`. After `@tool`, `multiply` is a **tool object** (includes a Python `func`). Pydantic cannot JSON-schema a callable:

```text
PydanticInvalidForJsonSchema: Cannot generate a JsonSchema for core_schema.CallableSchema
```

Forgot `from langchain_core.tools import tool` → `NameError: name 'tool' is not defined`.

### 3. `StructuredTool.from_function` — explicit schema

When the docstring is not enough, pass a Pydantic model:

```python
class MultiplyInput(BaseModel):
    a: int = Field(description="The first number to multiply")
    b: int = Field(description="The second number to multiply")

def multiply(a: int, b: int) -> int:
    return a * b

multiply_tool = StructuredTool.from_function(
    func=multiply,
    name='multiply',
    description='multiply 2 numbers',
    args_schema=MultiplyInput,
)
```

Same `invoke({'a': 2, 'b': 3})`. The LLM sees `Field(description=...)` in the schema. **Descriptions that say “add” while the function multiplies will make the model use it wrong.**

### 4. Subclass `BaseTool` — full control

```python
class MultiplyTool(BaseTool):
    name: str = 'multiply'
    description: str = 'multiply 2 nos'
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b
```

`_run` is **`@abstractmethod`** on `BaseTool`. You must implement it. Parameter names must match the schema (`a`, `b`). Extra `self` because it is a method.

`invoke` (from `Runnable`) validates, then calls `_run`. Optional: `_arun` for async; default runs `_run` in a thread.

In this file `args_schema` points at `MultiplyInput`, not `MultiplyInput1`. Fine for a demo; easy to edit the unused class and wonder why nothing changes.

### Pydantic v2 `Field` — required vs optional

`Field(required=True)` is **invalid**. Extra keys on `Field` warn:

```text
PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated
(Extra keys: 'required')
```

Required vs optional is **whether there is a default**:

```python
a: int = Field(description="first")                 # required
b: int = Field(default=1, description="second")     # optional, default 1
c: int | None = Field(default=None, description="") # optional null
```

On the JSON schema: required names go in the top-level `"required": ["a"]` list. Optional names are only in `properties`.

### Toolkit — a bag of tools

```python
class MathToolKit:
    def get_tools(self):
        return [add, multiply]

tools = MathToolKit().get_tools()
tools[1].invoke({'a': 2, 'b': 5})  # multiply → 10
```

Agents want a **list** of tools. A toolkit is just `get_tools() → list`. LangChain also has `BaseToolkit` with the same idea. Order in the list is what `tools[1]` means — here `add` is `[0]`, `multiply` is `[1]`.

---

## `calling.py` — bind tools, one loop

Defining a tool does not make the model use it. You must **bind** it.

```python
model = ChatCohere(model="command-a-03-2025", temperature=0.8)
model_tool = model.bind_tools([multiply])
```

`bind_tools` advertises the schema on every call. The model **may** emit `tool_calls`. It is not forced (unless you set `tool_choice`).

### Message list (required shape)

```text
[0] HumanMessage('multiply 10 with 6')
[1] AIMessage(tool_calls=[{name, args, id}])     # append the AI message
[2] ToolMessage(content='60', tool_call_id=id)   # append the tool result
[3] AIMessage(content='...')                     # final text
```

This file:

```python
messages = [query]
res = model_tool.invoke(messages)
messages.append(res)

res1 = multiply.invoke(res.tool_calls[0])
messages.append(res1)

result = model_tool.invoke(messages)
print(result.content)
```

Passing the **full tool_call dict** (with `id`) into `multiply.invoke` returns a `ToolMessage` whose `tool_call_id` matches. Cohere **requires** that match:

```text
invalid tool message: tool call id '...' not found in previous tool calls
```

That error means you appended the `ToolMessage` **without** the `AIMessage` that requested it. Always: human → **AI (calls)** → **tool (same id)** → AI again.

### `tool_calls` is not the catalog

`res.tool_calls` is **this turn’s requests**, not `[multiply]` because you bound it. Empty `[]` means “I’m done, here is `content`.” Indexing `[0]` without a check → `IndexError`.

`temperature=0.8` is noisy for tools. `currency_converter.py` uses `0`.

The commented LCEL `model_tool | append_message | multiply` does not replace the loop: the second `invoke` still needs the **AI message** in history, and `multiply` must receive a **tool_call**, not the whole `AIMessage`. Keep the explicit loop until LangGraph.

`RunnableLambda` helpers in this file are unused leftovers.

---

## `currency_converter.py` — two tools, injected arg, agent loop

Real tools: HTTP FX rate, then multiply an amount. The second arg of `convert` is **not chosen by the model**.

### Tools

**`get_conversion_factor(base, target)`**  
GET `exchangerate-api.com` pair endpoint. Stores `conversion_rate` in `state`. Returns a **string that does not include the number**.

If you return the raw JSON (`{'conversion_rate': 95.29, ...}`), the model **skips `convert`** and does `10 * 95.29` in text. `tool_calls` on the next turn is `[]`. That is valid — just not the exercise.

**`convert(base_currency_value, conversion_rate)`**  
`conversion_rate` is `Annotated[float, InjectedToolArg]`. The LLM schema **hides** that field. The model only passes `{'base_currency_value': 10}`. Python injects the live rate before `invoke`.

```python
call = {
    **call,
    "args": {**call["args"], "conversion_rate": state["conversion_rate"]},
}
convert.invoke(call)
```

### Why the first `tool_calls` had only one entry

You asked: rate **and** convert 10 USD.

First `invoke` only produced `get_conversion_factor`. That is correct:

- `convert` **needs** the first tool’s result
- `InjectedToolArg` means the model **cannot** fill the rate on turn 1
- `tool_calls` ≠ “every bound tool”

`bind_tools([get_conversion_factor, convert])` **advertises** both. The array is **what it chose this turn**.

### The loop

```python
for _ in range(6):
    ai_message = model_tool.invoke(messages)
    messages.append(ai_message)

    if not ai_message.tool_calls:
        print(ai_message.content)
        break

    for call in ai_message.tool_calls:
        ...
        messages.append(tools_by_name[name].invoke(call))
else:
    raise RuntimeError("tool loop did not finish")
```

Successful run looks like:

```text
1. tool_calls: get_conversion_factor(USD, INR)
2. ToolMessage: "Got the USD->INR rate. Now call convert..."
3. tool_calls: convert({base_currency_value: 10})   # rate injected
4. ToolMessage: 952.949
5. content: "10 USD is equal to 952.949 INR."
```

`range(6)` is a **cap**, not “always 6 times.” `_` = unused index. This task needs 3 model calls (factor, convert, final text). The `else` on `for` runs only if you **never `break`** — still calling tools after 6 turns → raise, instead of an infinite API bill.

`SystemMessage` tells the model not to multiply in prose. Still not a guarantee; hiding the rate + injecting it is what actually forces `convert`.

`tools_by_name[call["name"]].invoke(call)` is the dispatcher. Same idea as a toolkit list, keyed by name.

### Bugs this file already hit

| Symptom | Cause |
| --- | --- |
| `NameError: requests` | used `requests.get` without `import requests` |
| First turn, no `convert` in `tool_calls` | sequential tools; rate unknown / injected |
| Cohere 400: tool call id not found | appended `ToolMessage` without the `AIMessage` |
| `[]` then `IndexError` | second turn answered in text; you still did `tool_calls[0]` |
| `messages.invoke(...)` | `messages` is a **list** → `.append` |
| Model never calls `convert` | first tool returned the numeric rate |

Do **not** commit the exchange-rate API key in the URL. Put it in `.env` like Cohere.

Return type on `get_conversion_factor` was annotated `float` when it returned `response.json()` (a dict). The working version returns a `str` on purpose.

---

## Mental model

```text
bind_tools     = put the menu on the table
tool_calls     = the dish the waiter (model) ordered this round
invoke(tool)   = kitchen runs it
ToolMessage    = plate comes back (must match the ticket id)
InjectedToolArg = kitchen adds salt; the diner never orders it
range(6)       = stop if the waiter never stops ordering
```

| File | What you practise |
| --- | --- |
| `tools.py` | built-in, `@tool`, `StructuredTool`, `BaseTool._run`, toolkit |
| `calling.py` | `bind_tools`, one tool, message order |
| `currency_converter.py` | two tools, injected args, loop until `tool_calls == []` |

Next in this path: LangChain **agents** / **LangGraph** (same loop, checkpointed). CrewAI is optional later — same idea, different API.

---

## Cheat sheet

| Want | Use |
| --- | --- |
| Wrap a function fast | `@tool` + docstring + hints |
| Custom arg descriptions | `StructuredTool` + Pydantic `Field` |
| Extra state / logging | subclass `BaseTool`, implement `_run` |
| Give tools to a chat model | `model.bind_tools(list)` |
| What the LLM requested | `ai_message.tool_calls` |
| Run it | `tool.invoke(tool_call)` or `tool.invoke(args_dict)` |
| Schema the LLM sees | `tool.args_schema.model_json_schema()` |
| Hide an arg from the LLM | `Annotated[T, InjectedToolArg]` then set it in Python |
| Optional Pydantic field | `Field(default=...)` — not `required=False` |

---

## Run

```powershell
pip install -U ddgs requests
python .\tools-calling\tools.py
python .\tools-calling\calling.py
python .\tools-calling\currency_converter.py
```
