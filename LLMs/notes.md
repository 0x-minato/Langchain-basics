# LLMs

First contact with calling a model from Python. The file is small on purpose: load a key, pick a model, `invoke`, print.

You currently run this from **`venv312`**. Keys live in the project `.env` (never commit that file). `load_dotenv()` reads them so you do not hard-code secrets.

---

## LLM vs Chat Model (important)

LangChain has two families:

| | Old **LLM** | **Chat model** (what you actually use) |
| --- | --- | --- |
| Input | one string | list of messages (`system` / `human` / `ai`) |
| Output | string-like | `AIMessage` with `.content` |
| Examples | `OpenAI(...)` | `ChatOpenAI`, `ChatGoogleGenerativeAI`, `ChatCohere` |

`llm_demo.py` imports `OpenAI` (the old LLM wrapper) but then builds **`ChatGoogleGenerativeAI`**. The live path is the chat model. Later folders all use chat models.

`invoke("What is the capital of India")` still works on a chat model: LangChain wraps the string as a `HumanMessage`.

---

## `llm_demo.py`

```python
llm = ChatGoogleGenerativeAI(model='gemini-flash-latest')
result = llm.invoke("What is the capital of India")
print(result)
```

What happens:

1. `load_dotenv()` loads `GOOGLE_API_KEY` (or the Gemini env name your package expects).
2. LangChain sends the prompt to Google.
3. `result` is an **`AIMessage`**, not a bare string. `print(result)` dumps the whole object (content, metadata, usage). Use `result.content` when you only want the text — that is what `chatModels/chat_model.py` does.

`gemini-flash-latest` is a moving alias. If a run 404s, the model id was renamed; check Google’s current flash id.

---

## Run

From the project root, with `venv312` active:

```powershell
python .\LLMs\llm_demo.py
```

Needs a valid Google key in `.env`. No GPU required — this is a cloud API.

---

## Next

`chatModels/` does the same `invoke`, but for OpenAI, Anthropic, Gemini, Hugging Face, plus messages, prompts, and parsers.
