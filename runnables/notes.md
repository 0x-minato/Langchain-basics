# Runnables

Everything you pipe with `|` is a **Runnable**: prompts, models, parsers, and the helper classes in this folder.

This folder writes the same graphs as `masti-with-chains/`, but with the **class names** spelled out (`RunnableSequence`, `RunnableParallel`, …). LCEL `|` is sugar on top of these classes.

All scripts here use **Cohere** `command-a-03-2025` (`CO_API_KEY` / `COHERE_API_KEY`) and `StrOutputParser`.

There is no `chain1 & chain2` operator. Parallel LCEL is a **dict** of runnables.

---

## `sequence.py` — one after another

```python
chain = RunnableSequence(prompt, model, parser, prompt1, model, parser)
```

Same as:

```python
prompt | model | parser | prompt1 | model | parser
```

Joke on `{topic}` → explain that joke.

**Watch `invoke`:** it must be a **dict** `{'topic': 'Deep Learning'}`.  
`{'topic', 'Deep Learning'}` is a Python **set**. Sets have no keys, so the prompt cannot find `{topic}`.

The old import `from langchain.schema.runnable import ...` does not exist in this venv. Use `langchain_core.runnables`.

---

## `parallel.py` — same input, two outputs

```python
parallel_chain = RunnableParallel({
    'linkedIn': RunnableSequence(prompt1, model, parser),
    'tweet': RunnableSequence(prompt2, model, parser),
})
```

LCEL form:

```python
{
    'linkedIn': prompt1 | model | parser,
    'tweet': prompt2 | model | parser,
}
```

Same `{topic}` goes to both branches. Result is a dict. You print `res['linkedIn']` and `res['tweet']`.

This is **not** sequential: `(chain1) | (chain2)` would feed chain1’s output into chain2.

---

## `passthrough.py` — keep a copy while you also transform

`RunnablePassthrough` returns its input unchanged.

Flow:

1. `joke_chain` generates a joke (string).
2. That string enters a parallel map:
   - `'joke'` → passthrough (keep the joke)
   - `'joke-explain'` → another sequence that explains it

You end up with both the joke **and** the explanation. Without passthrough, the explain branch would consume the joke and you would lose the original text.

`prompt2` is written with `{joke}` but `input_variables=['topic']` in the file — those names should match, and the incoming value is the joke string. If explanation looks wrong, check that mapping.

---

## `lambda.py` — run your own Python

`RunnableLambda(fn)` wraps a normal function as a runnable.

```python
def count_words(text):
    return len(text.split())

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'joke-words': RunnableLambda(count_words),
})

chain = joke_chain | parallel_chain
```

Output looks like `{'joke': '...', 'joke-words': 26}` — generate once, then attach a word count without a second LLM call.

The lambda **must** take the previous step’s output as its argument. `def f():` with no args breaks when you `invoke`.

---

## `branch.py` — if / else

```python
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, summarize_chain),
    RunnablePassthrough(),   # default
)
```

1. Write a report on `{topic}`.
2. If the report is longer than 500 words → summarize.
3. Else → pass the report through.

`RunnableBranch` arguments are `(condition, runnable), (condition, runnable), default`. The default is required.

This is the same idea as `masti-with-chains/conditional-chain.py`, but the condition is **word count**, not sentiment.

---

## Cheat sheet

| Class | LCEL | Job |
| --- | --- | --- |
| `RunnableSequence` | `a \| b \| c` | series |
| `RunnableParallel` | `{"x": a, "y": b}` | fan-out, dict result |
| `RunnablePassthrough` | (class) | copy input forward |
| `RunnableLambda` | (class) | your function |
| `RunnableBranch` | (class) | if / elif / else |

---

## Run

```powershell
python .\runnables\lambda.py
python .\runnables\branch.py
python .\runnables\parallel.py
```
