from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

@tool
def multiply(x: float, y: float) -> float:
    """Multiply 'x' times 'y'."""
    return x * y

@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the 'y'."""
    return x**y

@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y

prompt = ChatPromptTemplate.from_messages([
    ("system", "you're a helpful assistant"), 
    ("human", "{input}, {agent_scratchpad}"),
    # ("placeholder", "{agent_scratchpad}"),
])

tools = [multiply, exponentiate, add]


llm = ChatGroq(model="llama-3.3-70b-versatile")



agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke({"input": "what's 3 plus 5, and what is 6 raised to 4.5 and what 4 multiplied with 89", })