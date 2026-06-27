import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

st.title("RAG Demo")

uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with open(f"./{file.name}", "wb") as f:
            f.write(file.getbuffer())

    documents = SimpleDirectoryReader("./").load_data()
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
