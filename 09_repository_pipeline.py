import os
import tempfile
import shutil

from git import Repo

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


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


def clone_repository(repo_url):
    """
    Clone the repository into a fresh temporary folder.
    """

    repository_path = tempfile.mkdtemp(
        prefix="codemap_repo_"
    )

    print("\nCloning repository...")

    Repo.clone_from(
        repo_url,
        repository_path
    )

    print("Repository cloned successfully.")

    return repository_path


def load_code_files(repository_path):
    """
    Read supported source code files.
    """

    documents = []

    for root, directories, files in os.walk(repository_path):

        directories[:] = [
            directory
            for directory in directories
            if directory not in IGNORED_DIRECTORIES
        ]

        for filename in files:

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension not in SUPPORTED_EXTENSIONS:
                continue

            file_path = os.path.join(
                root,
                filename
            )

            try:

                with open(
                    file_path,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as file:

                    content = file.read()

                if not content.strip():
                    continue

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

                print(
                    f"Could not read {file_path}: {e}"
                )

    return documents


def split_documents(documents):
    """
    Split code into smaller chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_documents(
        documents
    )


def create_embedding_model():
    """
    Load the embedding model.
    """

    print("\nLoading embedding model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Embedding model loaded.")

    return embedding_model


def create_vector_store(chunks, embedding_model):
    """
    Create FAISS vector store.
    """

    print("\nCreating FAISS vector store...")

    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )

    print(
        "FAISS vector store created successfully."
    )

    return vector_store


def save_vector_store(vector_store):
    """
    Save FAISS into a unique folder.
    """

    os.makedirs(
        "data/vector_stores",
        exist_ok=True
    )

    vector_store_path = tempfile.mkdtemp(
        prefix="faiss_",
        dir="data/vector_stores"
    )

    vector_store.save_local(
        vector_store_path
    )

    print(
        "FAISS vector store saved successfully."
    )

    print(
        f"Vector store location: {vector_store_path}"
    )

    return vector_store_path


def process_repository(repo_url):

    repository_path = None

    try:

        repository_path = clone_repository(
            repo_url
        )

        print("\nReading code files...")

        documents = load_code_files(
            repository_path
        )

        print(
            f"Documents loaded: {len(documents)}"
        )

        if not documents:

            raise ValueError(
                "No supported source code files were found."
            )

        print("\nCreating chunks...")

        chunks = split_documents(
            documents
        )

        print(
            f"Chunks created: {len(chunks)}"
        )

        if not chunks:

            raise ValueError(
                "No code chunks were created."
            )

        embedding_model = create_embedding_model()

        vector_store = create_vector_store(
            chunks,
            embedding_model
        )

        vector_store_path = save_vector_store(
            vector_store
        )

        print("\nRepository processing completed.")

        print(
            "Repository is ready for questions."
        )

        return vector_store_path

    finally:

        if repository_path and os.path.exists(
            repository_path
        ):

            try:
                shutil.rmtree(
                    repository_path,
                    ignore_errors=True
                )
            except Exception:
                pass


if __name__ == "__main__":

    print("=" * 60)
    print("CodeMap Repository Processor")
    print("=" * 60)

    repo_url = input(
        "\nPaste GitHub repository URL: "
    ).strip()

    if not repo_url:

        print(
            "\nPlease enter a GitHub repository URL."
        )

        exit()

    try:

        process_repository(
            repo_url
        )

    except Exception as e:

        print("\nSomething went wrong.")

        print("Error:")
        print(e)