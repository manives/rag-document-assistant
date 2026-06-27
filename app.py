from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
import os

st.title("RAG Demo")

# Tenta ler a chave dos secrets (definitiva) primeiro
groq_api_key = st.secrets.get("GROQ_API_KEY", "")

# Se não encontrar a chave secreta, exibe o campo na barra lateral
if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Groq API Key (Gratuita)", type="password")

uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True)

if uploaded_files:
    if not groq_api_key:
        st.info("Para usar sem pagar nada, crie uma chave gratuita em https://console.groq.com/keys e cole na barra lateral (ou configure os Secrets no painel).")
        st.stop()
        
    Settings.llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key)

    os.makedirs("./data", exist_ok=True)
    for file in uploaded_files:
        with open(f"./data/{file.name}", "wb") as f:
            f.write(file.getbuffer())

    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents)

    query_engine = index.as_query_engine()

    question = st.text_input("Faça uma pergunta sobre os documentos:")

    if question:
        with st.spinner("Analisando os PDFs e gerando a resposta..."):
            response = query_engine.query(question)

        st.write("### 🤖 Resposta")
        st.write(response.response)

        with st.expander("📚 Ver trechos do documento usados como fonte"):
            for i, node in enumerate(response.source_nodes):
                st.markdown(f"**Trecho {i+1}:**")
                # Usamos st.info para que o texto fique em um bloco destacado
                # e evite que o markdown interprete caracteres especiais como títulos gigantes
                st.info(node.node.text.strip())
