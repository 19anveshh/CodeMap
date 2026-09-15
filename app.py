import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv
from git import Repo

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


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

    repository_path = tempfile.mkdtemp(
        prefix="codemap_repo_"
    )

    Repo.clone_from(
        repo_url,
        repository_path
    )

    return repository_path


def load_code_files(repository_path):

    documents = []

    for root, directories, files in os.walk(
        repository_path
    ):

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

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": relative_path,
                            "file_name": filename,
                            "extension": extension,
                        }
                    )
                )

            except Exception:
                continue

    return documents


def create_vector_store(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        documents
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )

    return vector_store


def create_llm():

    load_dotenv()

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY was not found in .env"
        )

    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=api_key
    )


def answer_question(
    vector_store,
    llm,
    question
):

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 4
        }
    )

    documents = retriever.invoke(
        question
    )

    context_parts = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        context_parts.append(
            f"FILE: {source}\n\n"
            f"{document.page_content}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = PromptTemplate(
        template="""
You are CodeMap, an AI assistant that helps
developers understand unfamiliar codebases.

Answer the question using ONLY the provided
code context.

Be accurate and concise.

Always mention the relevant file names.

If the answer cannot be found in the context,
say:

"I could not find the answer in the provided repository."

CODE CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
""",
        input_variables=[
            "context",
            "question"
        ]
    )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    sources = []

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        if source not in sources:

            sources.append(
                source
            )

    return answer, sources


st.set_page_config(
    page_title="CodeMap",
    page_icon="Code",
    layout="wide"
)


st.title("CodeMap")

st.write(
    "Understand any GitHub codebase using natural language."
)


st.subheader(
    "1. Enter GitHub Repository"
)


repo_url = st.text_input(
    "GitHub Repository URL",
    placeholder="https://github.com/user/project"
)


if st.button(
    "Analyze Repository",
    type="primary"
):

    if not repo_url:

        st.warning(
            "Please enter a GitHub repository URL."
        )

    else:

        try:

            with st.spinner(
                "Cloning and analyzing repository..."
            ):

                repository_path = clone_repository(
                    repo_url
                )

                documents = load_code_files(
                    repository_path
                )

                if not documents:

                    raise ValueError(
                        "No supported source code files found."
                    )

                vector_store = create_vector_store(
                    documents
                )

                llm = create_llm()

                st.session_state.vector_store = (
                    vector_store
                )

                st.session_state.llm = llm

                st.session_state.repository_ready = True

                st.success(
                    f"Repository analyzed successfully. "
                    f"Found {len(documents)} source files."
                )

        except Exception as e:

            st.error(
                f"Could not analyze repository: {e}"
            )


if st.session_state.get(
    "repository_ready",
    False
):

    st.divider()

    st.subheader(
        "2. Ask About the Code"
    )

    question = st.text_input(
        "Ask a question",
        placeholder="Where is authentication handled?"
    )

    if st.button(
        "Ask CodeMap",
        type="primary"
    ):

        if not question:

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                with st.spinner(
                    "Searching the codebase..."
                ):

                    answer, sources = answer_question(
                        st.session_state.vector_store,
                        st.session_state.llm,
                        question
                    )

                st.subheader(
                    "Answer"
                )

                st.write(
                    answer
                )

                st.subheader(
                    "Sources"
                )

                for source in sources:

                    st.code(
                        source
                    )

            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )