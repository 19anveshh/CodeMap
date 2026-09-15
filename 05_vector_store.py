from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os


SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
}


IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build",
}


def load_code_files(repository_path):
    """
    Read source code files and convert them
    into LangChain Document objects.
    """

    documents = []

    for root, directories, files in os.walk(repository_path):

        directories[:] = [
            directory
            for directory in directories
            if directory not in IGNORED_DIRECTORIES
        ]

        for filename in files:

            extension = os.path.splitext(filename)[1].lower()

            if extension not in SUPPORTED_EXTENSIONS:
                continue

            file_path = os.path.join(root, filename)

            try:

                with open(
                    file_path,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as file:

                    content = file.read()

                relative_path = os.path.relpath(
                    file_path,
                    repository_path
                )

                document = Document(
                    page_content=content,
                    metadata={
                        "source": relative_path,
                        "file_name": filename,
                        "extension": extension,
                    }
                )

                documents.append(document)

            except Exception as e:

                print(f"Could not read {file_path}: {e}")

    return documents


def split_documents(documents):
    """
    Split code into smaller chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


def create_embedding_model():
    """
    Load the embedding model.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embedding_model


def create_vector_store(chunks, embedding_model):
    """
    Create a FAISS vector store from code chunks.
    """

    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )

    return vector_store


if __name__ == "__main__":

    repository_path = "data/repository"

    print("Reading repository...")

    documents = load_code_files(repository_path)

    print(f"Documents loaded: {len(documents)}")

    print("\nCreating chunks...")

    chunks = split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    print("\nLoading embedding model...")

    embedding_model = create_embedding_model()

    print("Embedding model loaded.")

    print("\nCreating FAISS vector store...")

    vector_store = create_vector_store(
        chunks,
        embedding_model
    )

    print("FAISS vector store created.")

    os.makedirs("data/vector_store", exist_ok=True)

    vector_store.save_local(
        "data/vector_store"
    )

    print("FAISS vector store saved successfully.")

    print("\nTesting similarity search...")

    results = vector_store.similarity_search(
        "Where is the application logic?",
        k=3
    )

    print(f"\nRetrieved {len(results)} chunks.")

    for index, result in enumerate(results):

        print("\n" + "=" * 60)

        print(f"Result {index + 1}")

        print("\nSource:")
        print(result.metadata.get("source"))

        print("\nContent:")
        print(result.page_content)