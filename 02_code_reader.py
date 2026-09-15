from langchain_community.document_loaders import DirectoryLoader


def load_code_files(repository_path):
    """
    Load Python files from the repository using LangChain.
    """

    loader = DirectoryLoader(
        repository_path,
        glob="**/*.py",
        show_progress=True,
        use_multithreading=True
    )

    documents = loader.load()

    return documents


if __name__ == "__main__":

    repository_path = "data/repository"

    documents = load_code_files(repository_path)

    print(f"\nTotal documents loaded: {len(documents)}")

    for document in documents:

        print("\nFile:")
        print(document.metadata.get("source"))

        print("Characters:")
        print(len(document.page_content))