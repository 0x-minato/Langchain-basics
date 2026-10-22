from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

model = ChatCohere(
    model="command-a-03-2025",
    temperature=0.8
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate a joke for topic {topic}',
    input_variables=['topic']
)

def count_words(text):
    return len(text.split())

joke_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'joke-words': RunnableLambda(count_words)
})

chain = joke_chain | parallel_chain
res = chain.invoke({'topic': 'AI safety'})
print(res)