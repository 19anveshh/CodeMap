import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


def create_llm():
    """
    Create the Groq language model.
    """

    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY was not found in the .env file."
        )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=groq_api_key
    )

    return llm


if __name__ == "__main__":

    print("Loading Groq LLM...")

    try:

        llm = create_llm()

        print("Groq LLM created successfully.")

        response = llm.invoke(
            "Explain what a Python function is in one sentence."
        )

        print("\nLLM Response:")
        print(response.content)

    except Exception as e:

        print("\nSomething went wrong.")

        print("Error:")
        print(e)