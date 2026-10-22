from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

model = ChatCohere(
    model="command-a-03-2025",
    temperature=0.8
)

prompt1 = PromptTemplate(
    template='Generate a linkedIn post for {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a X post for  {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'linkedIn': RunnableSequence(prompt1, model, parser),
    'tweet': RunnableSequence(prompt2, model, parser)
})

res = parallel_chain.invoke({'topic': 'Danger of AI'})
print(res['linkedIn'])
print(res['tweet'])