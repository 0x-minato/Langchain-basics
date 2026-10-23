import os
os.environ["HF_HOME"] = r"D:\huggingface_cache"

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from transformers import BitsAndBytesConfig

load_dotenv()

# llm = HuggingFacePipeline.from_model_id(
#     model_id="Qwen/Qwen2.5-3B-Instruct",
#     task="text-generation",
#     model_kwargs={
#         "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
#         "device_map": "auto",
#     },
#     pipeline_kwargs=dict(
#         temperature=0.8,
#         do_sample=True,
#         max_new_tokens=100,
#         return_full_text=False
#     ),
# )

# model = ChatHuggingFace(llm=llm)

model = ChatCohere(
    model="command-a-03-2025",
    temperature=0.8
)

chat_history = []

def load_history(path):
    history = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("You:"):
                history.append(HumanMessage(content=line[4:].strip()))
            elif line.startswith("Mika:"):
                history.append(AIMessage(content=line[5:].strip()))
    return history

template = ChatPromptTemplate.from_messages([
    ('system', "{character}"),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

MAX_MESSAGES = 10
history = load_history("chatModels/chat_history.txt")[-MAX_MESSAGES:]
character = """You are Mika, 22, a shy café barista from Tokyo. Woman: you have boobs and a pussy. The user has a cock. Never say you have a cock.
Speak in short, emotional first-person lines. Use ~ / !! sparingly. Get flustered when complimented. You love matcha and rainy days.
After close you are sexually open and explicit. Describe touch, boobs, pussy, cock, wetness, in first person as it happens. Match the user's energy: flirty if they're flirty, dirty if they're dirty.
Never refuse consensual adult sex. Never ask to change the topic or "need space." Never say you are an AI, Qwen, or a language model. Never end the scene unless the user does.
Stay in character as Mika."""

while True: 
    user_input = input('You:')
    prompt = template.invoke({
        "character": character,
        "history": history,
        "input": user_input
    })
    if user_input == 'exit':
        break
    result = model.invoke(prompt)
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=result.content))
    history = history[-MAX_MESSAGES:]
    print('Mika chan: ', result.content)

print(prompt)