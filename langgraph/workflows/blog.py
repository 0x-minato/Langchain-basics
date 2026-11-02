from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_cohere import ChatCohere
from dotenv import load_dotenv

load_dotenv()

model = ChatCohere(
    model="command-a-03-2025",
    temperature=0.8
)

class BlogState(TypedDict):
    topic: str
    outline: str
    blog: str
    score: int

def generate_outline(state: BlogState) -> BlogState:
    topic = state["topic"]
    prompt = f"generate a detailed outline for topic {topic}"
    outline = model.invoke(prompt).content
    state["outline"] = outline

    return state

def generate_blog(state: BlogState) -> BlogState:
    topic = state["topic"]
    outline = state["outline"]
    if outline == '':
        return 'err hai'
    prompt = f"generate a detailed Blog for topic {topic} and its outline {outline}"
    blog = model.invoke(prompt).content
    state["blog"] = blog

    return state

def generate_score(state: BlogState) -> BlogState:
    outline = state['outline']
    blog = state['blog']
    prompt = f"Based on this outline {outline} rate my blog {blog} and generate a integer score for that rating"
    score: int = model.invoke(prompt).content 
    state['score'] = score

    return state

graph = StateGraph(BlogState)
graph.add_node('generate_outline', generate_outline)
graph.add_node('generate_blog', generate_blog)
graph.add_node('generate_score', generate_score)
graph.add_edge(START, 'generate_outline')
graph.add_edge('generate_outline', 'generate_blog')
graph.add_edge('generate_blog', 'generate_score')
graph.add_edge('generate_score', END)

workflow = graph.compile()

initial_state = {'topic': 'AI in India'}
res = workflow.invoke(initial_state)
print(res)