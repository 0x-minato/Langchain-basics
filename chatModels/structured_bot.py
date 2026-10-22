import os
os.environ["HF_HOME"] = r"D:\huggingface_cache"

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import BitsAndBytesConfig
from typing import TypedDict, Annotated, Optional, Literal
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

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

class Review(BaseModel):
    key_themes: list[str] = Field(description="some des")
    summary: str = Field(description='some des')
    sentiment: Literal['pos', 'neg'] = Field(description='some des')
    pros: Optional[list[str]] = Field(default=None, description='some des')
    cons: Optional[list[str]] = Field(default=None, description='some des')
    name: Optional[str] = Field(default=None, description='some des')

structured_model = model.with_structured_output(Review, include_raw=True)

result = structured_model.invoke("""The matcha latte was smooth and not too sweet.
The café felt cozy on a rainy afternoon.
I would come back, though the wait was a bit long.""")

print(result)