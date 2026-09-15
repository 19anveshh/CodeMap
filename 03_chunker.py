from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader


def load_code_files(repository_path):
    """
    Load Python files from the repository.
    """

    loader = DirectoryLoader(
        repository_path,
        glob="**/*.py",
        show_progress=True,
        use_multithreading=True
    )

    documents = loader.load()

    return documents


def split_documents(documents):
    """
    Split source code documents into smaller chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":

    repository_path = "data/repository"

    documents = load_code_files(repository_path)

    print(f"\nOriginal documents: {len(documents)}")

    chunks = split_documents(documents)

    print(f"Total chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks[:5]):

        print("\n" + "=" * 60)
        print(f"Chunk {index + 1}")

        print("\nSource:")
        print(chunk.metadata.get("source"))

        print("\nContent:")
        print(chunk.page_content)