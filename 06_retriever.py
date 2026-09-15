from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def create_embedding_model():
    """
    Load the same embedding model used
    when creating the FAISS database.
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
    Create a retriever from the FAISS vector store.
    """

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

    return retriever


if __name__ == "__main__":

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

    question = input(
        "\nAsk a question about the code: "
    )

    print("\nSearching code...")

    results = retriever.invoke(question)

    print(f"\nRetrieved {len(results)} chunks.")

    for index, document in enumerate(results):

        print("\n" + "=" * 60)

        print(f"Result {index + 1}")

        print("\nSource:")
        print(document.metadata.get("source"))

        print("\nContent:")
        print(document.page_content)