from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

# Load the embedding model (runs locally, free)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
EMBED_DIM = 384

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    # SimpleDirectoryReader can read PDFs
    reader = SimpleDirectoryReader(input_files=[path])
    docs = reader.load_data()
    texts = [d.text for d in docs if d.text]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = embedding_model.encode(texts)
    return embeddings.tolist()



# import numpy as np
# from openai import OpenAI
# from llama_index.reader.file import PDFReader
# from llama_index.core.node_parser import SentenceSplitter
# from dotenv import load_dotenv
#
# load_dotenv()
#
# client = OpenAI()
# #reader = PDFReader() #we chunk it than we embed
# EMBED_MODEL = "text-embedding-3-large"
# EMBED-DIM = 3072
#
# splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200 #represent characters not words
#                             )
#
# def load_and_chunk_pdf(path: str):
#     docs = PDFReader().load_data(file=path)
#     texts = [d.text for d in docs if getattr(d, "text", None) ]
#     chunks = []
#     for t in texts:
#         chunks.extend(splitter.split(t))
#     return chunks
#
# def embed_texts(texts: list[str]) -> list[list[float]]:
#     response = client.embeddings.create(
#         model=EMBED_MODEL,
#         input = texts,
#     )
#     return [item.embedding for item in response.data]
