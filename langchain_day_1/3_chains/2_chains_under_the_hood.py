from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env
load_dotenv()

# Create a ChatGroq model
model = ChatGroq(model="llama-3.1-8b-instant")

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

parser = StrOutputParser()

# Create individual runnables (steps in the chain)
format_prompt = RunnableLambda(lambda x: prompt_template.invoke(x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: parser.invoke(x))

# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain
response = chain.invoke({"topic": "lawyers", "joke_count": 3})

# Output
print(response)
