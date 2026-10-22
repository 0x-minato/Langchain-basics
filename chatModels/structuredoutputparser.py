import os
os.environ["HF_HOME"] = r"D:\huggingface_cache"

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import BitsAndBytesConfig
from langchain_core.messages import HumanMessage
from langchain_classic.output_parsers.structured import StructuredOutputParser, ResponseSchema
from langchain_core.exceptions import OutputParserException

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    task="text-generation",
    model_kwargs={
        "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
        "device_map": "auto",
    },
    pipeline_kwargs=dict(
        do_sample=False,
        max_new_tokens=200,
        return_full_text=False
    ),
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact1', description='description about fact1'),
    ResponseSchema(name='fact2', description='description about fact2'),
    ResponseSchema(name='fact3', description='description about fact3'),
    ResponseSchema(name='fact4', description='description about fact4')
]

parser = StructuredOutputParser.from_response_schemas(schema)

# Do NOT put parser.get_format_instructions() in the prompt.
# Qwen 3B copies that schema ("fact1": string // ...) instead of real JSON.
topic = "Black Hole"
raw = model.invoke([
    HumanMessage(content=(
        f"Give 4 short facts about {topic}.\n"
    ))
])
print("RAW:\n", raw.content)

try:
    res = parser.parse(raw.content)
    print("PARSED:\n", res)
except OutputParserException as e:
    print("PARSER FAILED:\n", e)