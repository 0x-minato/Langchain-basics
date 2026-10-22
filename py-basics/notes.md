# Python basics for LangChain

This folder is **not** LangChain yet. It is the typing you need before structured output, parsers, and `with_structured_output`.

LangChain asks models to return JSON that matches a shape. Two Python tools describe that shape:

| Tool | When it is checked | Typical use in this repo |
| --- | --- | --- |
| `TypedDict` | Only by the type checker / IDE | Quick schemas, `with_structured_output` sketches |
| Pydantic `BaseModel` | At **runtime** | `PydanticOutputParser`, real validation |

---

## `type-dict.py` — TypedDict

`TypedDict` is a **dictionary with named keys**, not a real class instance.

```python
class Person(TypedDict):
    name: str
    age: int

new_person: Person = {'name': 'kash', 'age': 22}
```

What this does:

- Tells the editor “this dict should have `name: str` and `age: int`”.
- Does **not** validate at runtime. `{'name': 123, 'age': 'nope'}` still runs.
- You still use it like a normal dict: `new_person['name']`.

Use it when you want a cheap schema. Do not use it when a bad model reply must crash loudly.

---

## `pydantic-demo.py` — Pydantic v2

Pydantic **does** validate when you build the object.

```python
class Student(BaseModel):
    name: str = 'kash'          # default if missing
    age: Optional[int] = None
    email: EmailStr             # must look like an email
    cgpa: float = Field(gt=0, lt=10, default=5)
```

In this file:

- `new_student` only has `age` and `email`. `name` becomes `'kash'`, `cgpa` becomes `5`.
- `'342'` for `age` is **coerced** to int `342`.
- `Student(**new_student)` unpacks the dict into fields.
- `dict(student)` / `student.model_dump()` → Python dict.
- `student.model_dump_json()` → JSON string (this is what parsers often need later).

`Field(...)` extras you will see again in output parsers:

- `gt` / `lt` — numeric bounds (`age` `gt=18` in `chatModels/pydanticoutputparser.py`)
- `default` — value if the model omits the key
- `description` — text LangChain injects into format instructions for the LLM

If validation fails (bad email, `cgpa=11`), Pydantic raises. That is the point.

---

## How this connects to later folders

1. Describe the shape (`Person`, `FeedBack`, `Review`).
2. Ask the model for JSON.
3. Parse with `PydanticOutputParser` or `with_structured_output`.
4. Use `.sentiment`, `.age`, etc. in a `RunnableBranch`.

TypedDict = hint. Pydantic = actual gate.
