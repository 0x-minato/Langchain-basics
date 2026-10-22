import os
os.environ["HF_HOME"] = r"D:\huggingface_cache"

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-Coder-3B-Instruct",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.2
    ),
)

model = ChatHuggingFace(llm=llm)

prompt = """"""

result = model.invoke(prompt)

print(result.content)