from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

load_dotenv()

model = ChatCohere(
    model="command-a-03-2025",
    temperature=0.8
)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template='Generate a report on topic {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text {text}',
    input_variables=['text']
)

report_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_chain, branch_chain) 
res = final_chain.invoke({'topic': 'Football'})
print(res)