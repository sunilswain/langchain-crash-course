from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_redis import RedisChatMessageHistory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ---------------- Tools ----------------
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

tools = [multiply, exponentiate, add]

# ---------------- Prompt ----------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You're a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),   # memory placeholder
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# ---------------- Memory ----------------
REDIS_URL = "redis://default:yWUnjrSCDyMZcRv4jtQWKOC4BBnkuGa3@redis-18089.c11.us-east-1-2.ec2.redns.redis-cloud.com:18089"


SESSION_ID = "user_13"

chat_history = RedisChatMessageHistory(
    session_id=SESSION_ID,
    redis_url=REDIS_URL,
    ttl=60
)

# Wrap it in ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=chat_history
)

# ---------------- LLM ----------------
llm = ChatGroq(model="openai/gpt-oss-20b")

# ---------------- Agent ----------------
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory, 
    verbose=True
)

# ---------------- Run ----------------
# response = agent_executor.invoke({
#     "input": "what's 3 plus 5 raised to the 2.743. also what's 17.24 - 918.1241"
# })

print("Start chatting with the AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    ai_response = agent_executor.invoke({"input": human_input})

    print(f"AI: {ai_response}")
