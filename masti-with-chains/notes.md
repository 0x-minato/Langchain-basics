# Chains (LCEL)

This folder is the first time you **pipe** pieces together:

```text
prompt | model | parser
```

That `|` is LCEL (LangChain Expression Language). It builds a `RunnableSequence`. Same idea as `runnables/sequence.py`, but written with the operator instead of `RunnableSequence(...)`.

`load_dotenv()` is required for Gemini. Hugging Face scripts set `HF_HOME` to `D:\huggingface_cache` so weights do not land on `C:`.

`print_ascii()` needs the extra package `grandalf`. If it errors, `pip install grandalf`.

---

## `simple-chain.py` — one step

```python
chain = template | model | parser
res = chain.invoke({'topic': 'India'})
chain.get_graph().print_ascii()
```

- Prompt: “5 interesting facts about `{topic}`”.
- Model: Gemini `gemini-3.8-flash`.
- `StrOutputParser` unwraps `AIMessage` → plain string.

Without the parser, the next step would receive an `AIMessage` object. With it, you get text.

`print_ascii()` draws the graph so you can **see** the pipe.

---

## `sequencial-chain.py` — two model calls in a row

```python
chain = template1 | model | parser | template2 | model | parser
```

Flow:

1. Write a detailed report on `{topic}`.
2. Parse to string.
3. Feed that string into a summary prompt.
4. Model again, parse again.

`input()` lets you type the topic in the terminal.

**Watch the template:** `template2` uses `{topic2}` in the string but `input_variables=['topic']`. The variable names must match, and the second prompt must take whatever the parser just produced (often named `{text}` or `{topic}`). If this chain errors on “missing variable”, that mismatch is why.

---

## `parallel-chain.py` — fan-out, then merge

Two branches run on the **same** `{topic}`:

| Key | What it generates |
| --- | --- |
| `notes` | study notes |
| `quiz` | 5-question quiz |

```python
parallel_chain = RunnableParallel({
    'notes': template1 | model1 | parser,
    'quiz': template2 | model2 | parser,
})

chain = parallel_chain | merge_chain
```

`RunnableParallel` returns a **dict**. `merge_chain` expects `{notes}` and `{quiz}` — those keys must match the parallel dict keys.

There is **no** `chain1 & chain2` operator. A dict of runnables (or `RunnableParallel(...)`) **is** the LCEL for parallel.

Both `model1` and `model2` wrap the same local Qwen 3B. Parallel here means “two prompts”, not two GPUs. On a 4GB card they still share one model.

`print(chain)` dumps the chain object (can look like a huge tokenizer). Always `print(res)`.

---

## `conditional-chain.py` — classify, then branch

This is an if/else over model output.

1. **Classifier:** prompt + Cohere `command-a-03-2025` + `PydanticOutputParser` → `FeedBack(sentiment='positive'|'negative')`.
2. **Branch:**
   - positive → polite “thanks” style reply
   - negative → apology / fix style reply
   - else → `'could not find anything'`

```python
final_chain = classifier_chain | branch_chain
```

`RunnableBranch` tests **callables** against the previous output. Here `x` is a Pydantic object, so you write `x.sentiment`, not `x['sentiment']`.

Cohere needs `CO_API_KEY` or `COHERE_API_KEY` in `.env`. Older `command-r` ids 404; `command-a-03-2025` is the one that worked.

Small local models often **copy the JSON schema** instead of filling it. That is why this file uses Cohere, not Qwen, for the classifier.

---

## Mental model

| File | Pattern |
| --- | --- |
| `simple-chain.py` | A → B → C |
| `sequencial-chain.py` | A → B → C → D → … |
| `parallel-chain.py` | A splits into B and C, then D merges |
| `conditional-chain.py` | A classifies, then B **or** C |

Then `runnables/` names the same pieces: Sequence, Parallel, Passthrough, Lambda, Branch.
