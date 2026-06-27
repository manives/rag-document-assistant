from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai_like import OpenAILike
import os

st.title("RAG Demo")

# Tenta pegar a chave dos Secrets do Streamlit Cloud primeiro
openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY", "")

# Se não estiver configurado nos Secrets, pede na interface
if not openrouter_api_key:
    openrouter_api_key = st.sidebar.text_input("OpenRouter API Key", type="password")

uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True)

if uploaded_files:
    if not openrouter_api_key:
        st.info("Por favor, cole sua chave do OpenRouter na barra lateral para continuar.")
        st.stop()
        
    Settings.llm = OpenAILike(
        model="minimax/minimax-m2.5:free", 
        api_key=openrouter_api_key, 
        api_base="https://openrouter.ai/api/v1",
        is_chat_model=True,
        context_window=32768
    )

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
