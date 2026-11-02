from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_cohere import ChatCohere
from dotenv import load_dotenv

load_dotenv()

model = ChatCohere(
    model="command-a-03-2025",
    temperature=0.8
)

class CricketState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int
    strike_rate: float
    runs_boundary_per: float
    bpd: float
    summary: str

def calc_strike_rate(state: CricketState) -> CricketState:
    runs = state["runs"]
    balls = state["balls"]
    sr = (runs / balls) * 100

    return {'strike_rate': sr}

def calc_runs_boundary_per(state: CricketState) -> CricketState:
    runs = state["runs"]
    fours = state["fours"]
    sixes = state["sixes"]
    boundary_runs = (fours * 4) + (sixes * 6) 
    b_per = (boundary_runs / runs) * 100

    return {'runs_boundary_per': b_per}

def calc_ball_per_boundary(state: CricketState) -> CricketState:
    balls = state["balls"]
    boundaries = state["fours"] + state["sixes"]
    bpd = balls / boundaries

    return {'bpd': bpd}

def summary(state: CricketState) -> CricketState:
    summary = f"""
    Strike Rate: {state['strike_rate']}
    Balls per boundary: {state['bpd']}
    Boundary percent: {state['runs_boundary_per']}
    """

    return {'summary': summary}

graph = StateGraph(CricketState)

graph.add_node('calc_strike_rate', calc_strike_rate)
graph.add_node('calc_runs_boundary_per', calc_runs_boundary_per)
graph.add_node('calc_ball_per_boundary', calc_ball_per_boundary)
graph.add_node('summary', summary)

# edges 
graph.add_edge(START, 'calc_strike_rate')
graph.add_edge(START, 'calc_runs_boundary_per')
graph.add_edge(START, 'calc_ball_per_boundary')

graph.add_edge('calc_strike_rate', 'summary')
graph.add_edge('calc_runs_boundary_per', 'summary')
graph.add_edge('calc_ball_per_boundary', 'summary')

graph.add_edge('summary', END)

workflow = graph.compile()

from IPython.display import Image
Image(workflow.get_graph().draw_mermaid_png())

initial_prompt = {'runs': 100, 'balls': 70, 'fours': 3, 'sixes': 4}
res = workflow.invoke(initial_prompt)
print(res)