import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Ensure your OPENAI_API_KEY is set in your environment
# We don't need load_dotenv() here if app.py already loaded it

def generate_follow_up_questions(question, answer):
    """
    Uses an LLM to generate relevant follow-up questions based on a question and its answer.
    """
    # Initialize the Chat LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

    # Define the prompt structure
    messages = [
        SystemMessage(
            content="You are an expert at identifying natural and relevant follow-up questions based on a user's query and the provided answer. Your goal is to anticipate what the user might want to know next."
        ),
        HumanMessage(
            content=f"""
            Here is the user's question:
            "{question}"

            Here is the answer provided by the chatbot:
            "{answer}"

            Based on this exchange, please generate three concise follow-up questions that the user might logically ask next.

            Return ONLY the questions, each on a new line. Do not include numbers, bullet points, or any introductory text.
            """
        )
    ]

    try:
        # Call the LLM to get the response
        response = llm(messages)
        # Split the response content by newlines to get a list of questions
        questions = response.content.strip().split('\n')
        # Filter out any empty strings that might result from splitting
        return [q for q in questions if q]
    except Exception as e:
        # If anything goes wrong, return an empty list
        print(f"Error generating follow-up questions: {e}")
        return []