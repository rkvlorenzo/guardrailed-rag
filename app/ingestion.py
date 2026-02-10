import hashlib
import os
from functools import lru_cache

import pdfplumber
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

parent_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(parent_dir)
VECTOR_STORE_DIR = os.path.join(ROOT_DIR, os.environ["VECTOR_STORE_FOLDER"])

SUPPORTED_FORMATS = ["pdf"]

def convert_file_to_hash(file_path: str) -> str:
    """Returns SHA256 hash of a file"""
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    return hashlib.sha256(file_bytes).hexdigest()


def extract_text_from_pdf(file_path: str) -> str:
    with pdfplumber.open(file_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
        return text


def chunk_extracted_text(extracted_text: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_text(extracted_text)

def get_embedding_model():
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


def get_file_list(folder_name: str):
    try:
        folder_path = os.path.join(ROOT_DIR, folder_name)
        files = os.listdir(folder_path)
        return [file for file in files if file.split(".")[-1] in SUPPORTED_FORMATS]
    except FileNotFoundError as e:
        return []


def convert_files_to_vector():
    embeddings_model = get_embedding_model()

    files = get_file_list(os.environ["DOCS_FOLDER"])

    # Define a global vector store to handle multiple files
    global_vector_store = None
    for file in files:
        file_path = os.path.join(ROOT_DIR, os.environ["DOCS_FOLDER"], file)

        hashed_file = convert_file_to_hash(file_path=file_path)
        cache_file = os.path.join(VECTOR_STORE_DIR, f"{hashed_file}.pkl")

        if os.path.exists(cache_file):
            vector_store = FAISS.load_local(
                cache_file,
                embeddings_model,
                allow_dangerous_deserialization=True
            )
        else:
            extracted_text = extract_text_from_pdf(file_path=file_path)
            chunks = chunk_extracted_text(extracted_text)
            vector_store = FAISS.from_texts(chunks, embeddings_model)
            vector_store.save_local(cache_file)

        if global_vector_store is None:
            global_vector_store = vector_store
        else:
            global_vector_store.merge_from(vector_store)

    return global_vector_store

@lru_cache(maxsize=1)
def get_global_vector():
    return convert_files_to_vector()


def get_retriever(k=4, search_type="mmr"):
    return get_global_vector().as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )

def initialize_vector():
    get_global_vector()
