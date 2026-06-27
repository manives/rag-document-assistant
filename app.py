from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os

st.title("RAG Demo")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True)

if uploaded_files:
    if not openai_api_key:
        st.info("Please add your OpenAI API key in the sidebar to continue.")
        st.stop()
        
    os.environ["OPENAI_API_KEY"] = openai_api_key

    os.makedirs("./data", exist_ok=True)
    for file in uploaded_files:
        with open(f"./data/{file.name}", "wb") as f:
            f.write(file.getbuffer())

    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine()

    question = st.text_input("Ask something")

    if question:
        response = query_engine.query(question)

        st.write("### Answer")
        st.write(response.response)

        st.write("### Source context")
        for node in response.source_nodes:
            st.write(node.node.text)
