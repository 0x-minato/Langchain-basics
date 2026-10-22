from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

model = ChatCohere(
    model="command-a-03-2025",
    temperature=0.8
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate a Joke for topic {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate the explanation for this joke {joke}',
    input_variables=['topic']
)

joke_chain = RunnableSequence(prompt1, model, parser)
passthrough = RunnablePassthrough()

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'joke-explain': RunnableSequence(joke_chain, prompt2, model, parser)
})

final_chain = RunnableSequence(joke_chain, parallel_chain)

res = final_chain.invoke({'topic': 'Deep Learning'})
print(res['joke'])
print(res['joke-explain'])