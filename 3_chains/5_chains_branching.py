from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_groq import ChatGroq
from langchain.schema.runnable import RunnableLambda


# Load environment variables from .env
load_dotenv()

# Create a ChatGroq model
model = ChatGroq(model="llama-3.1-8b-instant")

# Define prompt templates for different feedback types
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Generate a thank you note for this positive feedback: {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Generate a response addressing this negative feedback: {feedback}."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a request for more details for this neutral feedback: {feedback}.",
        ),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a message to escalate this feedback to a human agent: {feedback}.",
        ),
    ]
)

# Define the feedback classification template
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human",
         "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}." \
            "output only keyword. E.g., positive, negative, neutral or escalate"),
    ]
)

def run_branch(x, keyword):
    print(f"\nRunning {keyword} branch: {keyword in x}")
    return keyword in x


# # Define the runnable branches for handling feedback
# branches = RunnableBranch(
#     (
#         lambda x: run_branch(x, "positive"),
#         positive_feedback_template | model | StrOutputParser()  # Positive feedback chain
#     ),
#     (
#         lambda x: run_branch(x, "negative"),
#         negative_feedback_template | model | StrOutputParser()  # Negative feedback chain
#     ),
#     (
#         lambda x: run_branch(x, "neutral"),
#         neutral_feedback_template | model | StrOutputParser()  # Neutral feedback chain
#     ),
#     escalate_feedback_template | model | StrOutputParser()
# )

# Define the runnable branches for handling feedback
branches = RunnableBranch(
    (
        lambda x: run_branch(x, "positive"),
        positive_feedback_template | model | StrOutputParser()  # Positive feedback chain
    ),
    (
        lambda x: run_branch(x, "negative"),
        negative_feedback_template | model | StrOutputParser()  # Negative feedback chain
    ),
    neutral_feedback_template | model | StrOutputParser()  # Neutral feedback chain
)
# Create the classification chain
classification_chain = classification_template | model | StrOutputParser()

# Combine classification and response generation into one chain

def print_n_return(x):
    print(f"1st LLM response: {x}")
    return x

chain = classification_chain | RunnableLambda(lambda x: print_n_return(x)) | branches

# Run the chain with an example review
# Good review - "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad review - "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral review - "The product is okay. It works as expected but nothing exceptional."
# Default - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

review = "I'm not sure about the product yet. Can you tell me more about its features and benefits? I want to escalate."
result = chain.invoke({"feedback": review})

# Output the result
print(result)
