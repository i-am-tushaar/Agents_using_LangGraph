# importing necessary libraries
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, Literal, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(r"f:\Tutorials\LangGraph_tut\.env")


class chatbot:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY")
        )
        
    def call_tool(self):
        search_tool = TavilySearchResults(
            max_results=2,
            tavily_api_key=os.getenv("TAVILY_API_KEY")
        )
        tools = [search_tool]
        self.tool_node = ToolNode(tools=tools)
        self.llm_with_tool = self.llm.bind_tools(tools)
        
    def call_model(self, state: MessagesState):
        messages = state['messages']
        response = self.llm_with_tool.invoke(messages)
        return {"messages": [response]}
    
    def router_function(self, state: MessagesState) -> Literal["tools", "__end__"]:
        messages = state['messages']
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return "__end__"
    
    def __call__(self):
        self.call_tool()
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            self.router_function,
            {"tools": "tools", "__end__": END}
        )
        workflow.add_edge("tools", "agent")
        self.app = workflow.compile()
        return self.app
        
if __name__ == "__main__":
    mybot = chatbot()
    workflow = mybot()
    response = workflow.invoke({"messages": ["who is the current prime minister of India?"]})
    print(response['messages'][-1].content)
