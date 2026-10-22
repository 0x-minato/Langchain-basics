import os
os.environ["HF_HOME"] = r"D:\huggingface_cache"

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import BitsAndBytesConfig
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

# detailed report
template1 = PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables= ['topic']
)

# summary
template2 = PromptTemplate(
    template='write a 3 line summary on {input}',
    input_variables=['input']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser 

result = chain.invoke({'topic':'black hole'})

print(result)