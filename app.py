import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("📄 Chat with your PDF")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# Store DB in session
if "db" not in st.session_state:
    st.session_state.db = None

# When file uploaded
if uploaded_file is not None and st.session_state.db is None:
    with open("data/temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully ✅")

    loader = PyPDFLoader("data/temp.pdf")
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)

    # Save DB
    st.session_state.db = db

# If DB is ready → show question box ALWAYS
if st.session_state.db is not None:
    query = st.text_input("Ask a question from your PDF")

    if query:
        results = st.session_state.db.similarity_search(query)
        st.write("### Answer:")
        st.write(results[0].page_content)

else:
    st.info("👆 Please upload a PDF to start")