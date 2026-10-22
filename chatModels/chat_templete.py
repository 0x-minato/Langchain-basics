from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

chat_template = ChatPromptTemplate([
    ('system', ''),
    ('human', )
])

print(prompt)

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-Coder-3B-Instruct",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.2,
        return_full_text=False
    ),
)

model = ChatHuggingFace(llm=llm)

while True: 
    user_input = input('You:')
    messages.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(messages)
    messages.append(AIMessage(content=result.content))
    print('AI:', result.content)
