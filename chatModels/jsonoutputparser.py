import os
os.environ["HF_HOME"] = r"D:\huggingface_cache"

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import BitsAndBytesConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-3B-Instruct",
    task="text-generation",
    model_kwargs={
        "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
        "device_map": "auto",
    },
    pipeline_kwargs=dict(
        temperature=0.1,
        do_sample=True,
        return_full_text=False
    ),
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template=(
        "Create one fictional person.\n"
        "Reply with ONLY this JSON, nothing else, no markdown:\n"
        '{{"name": "string", "age": 0, "city": "string"}}\n'
        "{instruction}"
    ),
    input_variables=[],
    partial_variables={'instruction': parser.get_format_instructions()}
)

chain = template | model | parser
result = chain.invoke({})

print(result)