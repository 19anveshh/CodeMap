import os

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def create_embedding_model():
    """
    Load the same embedding model used
    to create the FAISS database.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embedding_model


def load_vector_store(embedding_model):
    """
    Load the saved FAISS vector store.
    """

    vector_store = FAISS.load_local(
        "data/vector_store",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vector_store


def create_retriever(vector_store):
    """
    Create a retriever from FAISS.
    """

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

    return retriever


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


def create_prompt():
    """
    Create the prompt used by the RAG system.
    """

    prompt = PromptTemplate(
        template="""
You are CodeMap, an AI assistant that helps developers
understand unfamiliar codebases.

Answer the user's question using ONLY the provided
code context.

If the answer cannot be found in the provided code,
say:

"I could not find the answer in the provided repository."

Be accurate and specific.

When possible, mention the file name where the relevant
code was found.

Code Context:
{context}

User Question:
{question}

Answer:
""",
        input_variables=[
            "context",
            "question"
        ]
    )

    return prompt


def format_documents(documents):
    """
    Convert retrieved LangChain Documents into
    text that can be given to the LLM.
    """

    formatted_documents = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown file"
        )

        content = document.page_content

        formatted_documents.append(
            f"FILE: {source}\n\n{content}"
        )

    return "\n\n".join(formatted_documents)


def create_rag_pipeline():

    print("Loading embedding model...")

    embedding_model = create_embedding_model()

    print("Embedding model loaded.")

    print("\nLoading FAISS vector store...")

    vector_store = load_vector_store(
        embedding_model
    )

    print("FAISS vector store loaded.")

    print("\nCreating retriever...")

    retriever = create_retriever(
        vector_store
    )

    print("Retriever created.")

    print("\nLoading Groq...")

    llm = create_llm()

    print("Groq loaded.")

    print("\nCreating prompt...")

    prompt = create_prompt()

    print("Prompt created.")

    return retriever, prompt, llm


if __name__ == "__main__":

    try:

        retriever, prompt, llm = create_rag_pipeline()

        question = input(
            "\nAsk a question about the code: "
        )

        print("\nSearching repository...")

        documents = retriever.invoke(question)

        print(
            f"Retrieved {len(documents)} relevant chunks."
        )

        context = format_documents(documents)

        print("\nGenerating answer...")

        chain = prompt | llm | StrOutputParser()

        answer = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

        print("\nAnswer:")
        print(answer)

        print("\nSources:")

        for document in documents:

            print(
                f"- {document.metadata.get('source')}"
            )

    except Exception as e:

        print("\nSomething went wrong.")

        print("Error:")
        print(e)