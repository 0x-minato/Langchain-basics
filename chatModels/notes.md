# Chat models

This is the largest folder. You practise **chat** APIs (messages in, `AIMessage` out), then prompts, a looping chatbot, and **output parsers**.

Cloud scripts need keys in `.env`. Local Qwen scripts set:

```python
os.environ["HF_HOME"] = r"D:\huggingface_cache"
```

Weights live on `D:`. The instruct model that actually sits in that cache from earlier work is **`Qwen/Qwen2.5-3B-Instruct`**. Some files still say `Qwen2.5-Coder-3B-Instruct` (that download was removed). 4-bit `BitsAndBytesConfig` is how a 3B model fits an RTX 3050 4GB.

`ChatHuggingFace` wraps a `HuggingFacePipeline` so you can pass LangChain messages instead of raw tokenizer dicts.

---

## 1. One-shot cloud / local invokes

Same pattern everywhere: create model → `invoke` → print `.content`.

| File | Wrapper | Notes |
| --- | --- | --- |
| `chat_model.py` | `ChatOpenAI` (`gpt-4`) | `temperature=0.3`, `max_completion_tokens=10` (short answer). Needs OpenAI key. |
| `anthropic_model.py` | `ChatAnthropic` | Model string `'claude-sonnet-ljljlj'` is a placeholder — it will 404 until you put a real Claude id. |
| `gemini_model.py` | intended `ChatGoogleGenerativeAI` | Import currently says `langchain_google_gemini` (wrong). Working import is `langchain_google_genai`, as in `LLMs/llm_demo.py`. |
| `huggingface_model.py` | local Qwen **Coder** 3B | Empty `prompt = """"""` — you will get a blank-ish generation. |
| `quen_model.py` | raw `transformers` | Same Coder model, **no** LangChain. Shows `apply_chat_template` + `generate`. Good to see what LangChain hides. |

`temperature`: low = factual / JSON; high = creative. `0.8` in the chatbot is for personality.

---

## 2. Messages

Chat models think in roles:

| Class | Role |
| --- | --- |
| `SystemMessage` | personality / rules |
| `HumanMessage` | user |
| `AIMessage` | previous model reply (history) |

### `messages.py`

Builds `[SystemMessage, HumanMessage]`, invokes, then **appends** `AIMessage` so the list becomes a real conversation. `return_full_text=False` stops the pipeline from echoing the prompt back into `.content`.

### `chat_templete.py`

Sketch of `ChatPromptTemplate`. The file is unfinished: empty system string, incomplete `('human', )`, `print(prompt)` before `prompt` exists, and `messages` is never created. The **working** version of this idea is `chatbot.py`.

Templates use `{placeholders}` that you fill with `.invoke({...})` or `.format(...)`.

---

## 3. `chatbot.py` — looping character bot

A terminal chat loop with:

1. **System character** (Mika) injected through `{character}`.
2. **`MessagesPlaceholder("history")`** — previous turns.
3. Current `{input}`.
4. **Sliding window** `history[-MAX_MESSAGES:]` with `MAX_MESSAGES = 10`.

`[-N:]` means “last N messages”, not “drop N”. Without the colon, `[-10]` is a **single** message and the bot “forgets” the rest.

`load_history("chatModels/chat_history.txt")` reads few-shot lines:

```text
You: ...
Mika: ...
```

Those become `HumanMessage` / `AIMessage` so the model copies the style. Run from the **project root** so that relative path works.

`exit` breaks the loop. Type that to quit.

This is the same sliding-window idea every production chatbot uses: models have a context limit; you cannot append forever.

---

## 4. Output parsers

The model always returns text (inside `AIMessage`). Parsers turn that text into something code can use.

### `stoutparse.py` vs `stoutparser1.py`

Both do: long report on black holes → 3-line summary.

- `stoutparse.py` — two separate `invoke`s. You pass `result1.content` by hand. Easy to see, easy to mess up (`{'input': {result1.content}}` wraps the string in a **set**).
- `stoutparser1.py` — LCEL: `template1 | model | parser | template2 | model | parser`. `StrOutputParser` pulls `.content` so the second prompt receives a string.

**Point of `StrOutputParser`:** the next runnable wants text, not an `AIMessage`.

### `jsonoutputparser.py`

Asks for a person as JSON, then `JsonOutputParser`.

`get_format_instructions()` injects a JSON **schema**. Small models (Qwen 3B) often **copy the schema** (`"name": "string"`) instead of filling values. If you get `null` or invalid JSON, that is the model, not the parser class.

### `structuredoutputparser.py`

Older `StructuredOutputParser` + `ResponseSchema` (`fact1`…`fact4`). The comment in the file is the lesson: **do not** dump `get_format_instructions()` into a 3B prompt. Give a filled JSON example, then `parser.parse(raw.content)`.

### `pydanticoutputparser.py`

Best of the JSON family when it works:

1. Pydantic model `Person` with `age` `gt=18`.
2. `PydanticOutputParser`.
3. Chain: `template | Gemini | parser`.
4. `res` is a `Person` instance, not a dict.

Uses Gemini because Qwen 3B failed this. `parser.parse(...)` needs **string content**, not the whole `AIMessage`.

### `structured_bot.py`

`with_structured_output(Review)` — the model wrapper tries to emit a `Review` (themes, summary, sentiment, …). On tiny local models this often returns `None` or a blob with `include_raw=True`. Cloud models handle it better.

---

## Practical ranking (from what you already saw)

| Job | Small local 3B | Gemini / Cohere |
| --- | --- | --- |
| Chat / jokes / summaries | OK | Better |
| Strict JSON / Pydantic / branches | Often copies schema | Use these |

---

## Run examples

```powershell
python .\chatModels\chat_model.py
python .\chatModels\chatbot.py
python .\chatModels\stoutparser1.py
python .\chatModels\pydanticoutputparser.py
```

Local files need the venv with torch + bitsandbytes, and the Qwen instruct weights in `D:\huggingface_cache`.
