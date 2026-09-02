from typing import Annotated
from dotenv import load_dotenv
load_dotenv()
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.mongodb import MongoDBSaver
from langchain_core.messages import HumanMessage


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

graph_builder = StateGraph(State)


# adding the node to the graph
graph_builder.add_node("chatbot",chatbot)
# adding edges to the graph
# START -> chatbot -> sample -> END
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot",END)


def compile_graph_with_checkpoint(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)

DB_URI = "mongodb://admin:admin@localhost:27017"
with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:
    graph = compile_graph_with_checkpoint(checkpointer= checkpointer)

    config = {
        "configurable": {
            "thread_id": "rudra"
        }
    }

    for chunk in graph.stream(
        {"messages": [("user", "what is my name")]},
        config,
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
