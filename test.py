from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# 👉 ADD DEBUG LINE HERE
print("API KEY:", os.getenv("OPENAI_API_KEY"))

loader = PyPDFLoader("data/sample.pdf")
docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)


from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.from_documents(chunks, embeddings)

query = "What is this document about?"
results = db.similarity_search(query)

print(results[0].page_content)