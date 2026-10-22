# Document loaders

Loaders turn **files and URLs** into LangChain `Document` objects:

```text
Document(page_content="the text...", metadata={...})
```

You almost always want `.page_content` (the string) before a prompt. `.metadata` holds source path, page number, URL, etc.

Loaders live in `langchain_community` (install: `pip install langchain-community`). That package is being sunset; the import still works, you just get a deprecation warning.

Always run these from the **project root** so relative paths like `doc_loaders/cricket.txt` resolve.

---

## `load()` vs `lazy_load()`

Every loader has both.

| | `load()` | `lazy_load()` |
| --- | --- | --- |
| Returns | `list[Document]` | generator of `Document`s |
| Memory | whole file / whole folder in RAM | roughly one doc at a time |
| Total CPU if you process everything | about the same | about the same |
| Starts returning data | after the last page | as soon as page 1 is parsed |

Lazy is a **RAM / streaming** choice, not a “faster on big data” choice. You still pay to parse every page you actually use. If you iterate a generator twice, you parse twice.

---

## `text_loader.py` — local `.txt` + summarize

`TextLoader` is for plain text: notes, logs, scraped dumps, `cricket.txt`.

Chain in this file:

```python
loader = TextLoader('doc_loaders/cricket.txt', encoding='utf-8')

def loader_runnable(_):
    return loader.load()[0].page_content

chain = RunnableLambda(loader_runnable) | prompt1 | model | parser
res = chain.invoke({})
```

Details that already bit this script:

- `loader.load()` returns a **list**. Use `[0].page_content`, not `.page_content` on the list.
- `RunnableLambda` always receives the `invoke` argument, even if you ignore it (`_`).
- `chain.invoke()` with **no** argument crashes. Pass `{}` if the lambda loads the file itself.
- The prompt expects `{topic}`, so the lambda should return that string (or a dict `{'topic': ...}`). Right now it returns the cricket article as the topic text — that works if `{topic}` is the whole document.
- `encoding='utf-8'` avoids Windows decode errors.

Do **not** run `python cricket.txt`. That is not Python.

---

## `py_pdf_loader.py` — one PDF

Intended class is **`PyPDFLoader`** (capital PDF). The file currently imports `PyPdfLoader`, which will fail. Typical use:

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('books/some.pdf')
docs = loader.load()          # one Document per page
# or
for page in loader.lazy_load():
    ...
```

Needs `pypdf` (`pip install pypdf`).

---

## `dir_loader.py` — whole folder of PDFs

```python
loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader=PyPDFLoader,   # some versions name this loader_cls
)
docs = loader.load()
```

`glob='*.pdf'` only matches PDFs **directly** in `books/`. Use `'**/*.pdf'` if they sit in subfolders.

This is where lazy load matters: 20 textbooks × 400 pages as one list can eat RAM. Stream with `lazy_load()` into a splitter / vector store instead.

---

## `web_base_loader.py` — fetch a page, then ask a question

`WebBaseLoader` downloads HTML and strips it with **BeautifulSoup** (`pip install beautifulsoup4 lxml requests`).

This file:

1. Loads `https://www.troves.fi`.
2. Puts page text into `{text}`.
3. Asks `{question}`: average APY.
4. Cohere answers from that text only (a tiny RAG: retrieve page → stuff into prompt).

`USER_AGENT` warning: some sites block empty user-agents. Set `os.environ["USER_AGENT"] = "Mozilla/5.0"` if you get blocked or empty `page_content`.

`loader_fn()` takes no args and is called **before** the chain, so this is not lazy LCEL — it fetches once, then invokes. That is fine for one URL.

---

## What a Document looks like

```python
docs[0].page_content   # str
docs[0].metadata       # dict: source, page, ...
```

Next step in a real app: **split** `page_content` (`text_splitting/`) → **embed** (`embeddedModels/`) → retrieve → LLM.

---

## Run

```powershell
pip install langchain-community beautifulsoup4 pypdf
python .\doc_loaders\text_loader.py
python .\doc_loaders\web_base_loader.py
```
