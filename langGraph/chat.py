from typing import Annotated
from dotenv import load_dotenv
load_dotenv()
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langchain.chat_models import init_chat_model

class State(TypedDict):
    messages : Annotated[list,add_messages]

llm = init_chat_model(
    model="gemini-3.5-flash-lite",
    model_provider="google_genai"
)
# Nodes of a graph
def chatbot(state: State):
    response = llm.invoke(state.get("messages"))
    return {"messages":[response]}
def sample(state: State):
    print("\n inside sample node",state)
    return {"messages":["hey there, its a message from inside the sample node"]}

graph_builder = StateGraph(State)


# adding the node to the graph
graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("sample",sample)

# adding edges to the graph
# START -> chatbot -> sample -> END
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot","sample")
graph_builder.add_edge("sample",END)


graph = graph_builder.compile()

updated_state = graph.invoke(State({"messages":"hey there i'm rudra"}))

print("\n",updated_state)




