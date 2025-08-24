from dotenv import load_dotenv
from redis import Redis
from langchain_redis import RedisChatMessageHistory
from langchain_groq import ChatGroq


load_dotenv()


REDIS_URL = "redis://default:yWUnjrSCDyMZcRv4jtQWKOC4BBnkuGa3@redis-18089.c11.us-east-1-2.ec2.redns.redis-cloud.com:18089"


SESSION_ID = "user_123"

chat_history = RedisChatMessageHistory(
    session_id=SESSION_ID,
    redis_url=REDIS_URL
)
print("Chat History Initialized.")
print("Current Chat History:", chat_history.messages)

# Initialize Chat Model
model = ChatGroq(model="llama-3.1-8b-instant")

print("Start chatting with the AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break

    chat_history.add_user_message(human_input)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response.content}")
