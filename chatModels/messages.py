import os
os.environ["HF_HOME"] = r"D:\huggingface_cache"

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-Coder-3B-Instruct",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.2,
        return_full_text=False
    ),
)

messages = [
    SystemMessage(content = "You are a helpful message"),
    HumanMessage(content = "Tell me about Langchain")
]

model = ChatHuggingFace(llm=llm)

result = model.invoke(messages)

messages.append(AIMessage(content= result.content))
print(messages)