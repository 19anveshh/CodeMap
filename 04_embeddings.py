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
    Split code documents into smaller chunks.
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


def create_embeddings(chunks, embedding_model):
    """
    Convert code chunks into vector embeddings.
    """

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    vectors = embedding_model.embed_documents(texts)

    return vectors


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

    print("\nCreating embeddings...")

    vectors = create_embeddings(
        chunks,
        embedding_model
    )

    print("Embeddings created successfully.")

    print(f"Number of vectors: {len(vectors)}")

    if vectors:

        print(f"Vector dimension: {len(vectors[0])}")

        print("\nFirst vector:")
        print(vectors[0][:10])

        print("\nFirst chunk source:")
        print(chunks[0].metadata["source"])

        print("\nFirst chunk content:")
        print(chunks[0].page_content)