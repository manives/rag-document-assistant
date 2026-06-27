from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
import os
import json
import uuid

st.set_page_config(page_title="DocuMind AI", page_icon="📚")

st.title("📚 DocuMind AI")

DATA_DIR = "./data"
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")
os.makedirs(DATA_DIR, exist_ok=True)

def load_sessions():
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_sessions(sessions):
    with open(SESSIONS_FILE, "w") as f:
        json.dump(sessions, f)

# Inicializa as variáveis de estado
if "sessions" not in st.session_state:
    st.session_state.sessions = load_sessions()

def create_session():
    new_id = str(uuid.uuid4())
    st.session_state.sessions[new_id] = {
        "id": new_id,
        "title": f"Nova Conversa {len(st.session_state.sessions) + 1}",
        "messages": [],
        "files": []
    }
    st.session_state.current_session_id = new_id
    os.makedirs(os.path.join(DATA_DIR, new_id), exist_ok=True)
    save_sessions(st.session_state.sessions)
    return new_id

if "current_session_id" not in st.session_state:
    if st.session_state.sessions:
        st.session_state.current_session_id = list(st.session_state.sessions.keys())[-1]
    else:
        create_session()

current_id = st.session_state.current_session_id
current_session = st.session_state.sessions[current_id]
session_dir = os.path.join(DATA_DIR, current_id)
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.0 # Começa no 0 para evitar alucinações por padrão

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
        
    Settings.llm = Groq(
        model="llama-3.1-8b-instant", 
        api_key=groq_api_key,
        temperature=st.session_state.temperature
    )

    os.makedirs(session_dir, exist_ok=True)
    for file in uploaded_files:
        if file.name not in current_session["files"]:
            current_session["files"].append(file.name)
            save_sessions(st.session_state.sessions)
            
        with open(os.path.join(session_dir, file.name), "wb") as f:
            f.write(file.getbuffer())

if current_session["files"]:
    @st.cache_resource(show_spinner="Indexando documentos...")
    def get_query_engine(session_id, _file_names):
        folder = os.path.join("./data", session_id)
        if not os.path.exists(folder) or not os.listdir(folder):
            return None
        documents = SimpleDirectoryReader(folder).load_data()
        index = VectorStoreIndex.from_documents(documents)
        return index.as_query_engine()
        
    query_engine = get_query_engine(current_id, tuple(current_session["files"]))
else:
    query_engine = None
        
    # === CONTROLE DE CRIATIVIDADE (POPOVER) ===
    # Calcula a cor do botão baseado na temperatura (0.0 = Pastel, 1.0 = Verde Musgo)
    temp = st.session_state.temperature
    r = int(245 - (245 - 85) * temp)
    g = int(245 - (245 - 107) * temp)
    b = int(220 - (220 - 47) * temp)
    text_color = "white" if temp > 0.4 else "black"

    st.markdown(f"""
    <style>
    /* Estiliza especificamente o botão do popover na barra lateral */
    [data-testid="stSidebar"] div[data-testid="stPopover"] button {{
        background-color: rgb({r}, {g}, {b}) !important;
        color: {text_color} !important;
        border: 1px solid rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease;
    }}
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar.popover("🧠 Nível de Criatividade"):
        st.markdown("**Ajuste o volume de criatividade da IA**")
        st.slider(
            "0 = Respostas robóticas e precisas\n1 = Respostas criativas e variáveis", 
            min_value=0.0, 
            max_value=1.0, 
            step=0.1, 
            key="temperature"
        )
        st.caption("Dica: Para RAG corporativo, mantenha próximo de 0.0.")

    # Lista de sessões
    st.sidebar.markdown("---")
    if st.sidebar.button("➕ Nova Conversa", use_container_width=True):
        create_session()
        st.rerun()

    st.sidebar.markdown("### 🕒 Suas Conversas")
    for s_id, session_data in reversed(list(st.session_state.sessions.items())):
        is_current = (s_id == current_id)
        btn_label = f"🟢 {session_data['title']}" if is_current else f"💬 {session_data['title']}"
        if st.sidebar.button(btn_label, key=f"btn_{s_id}", use_container_width=True):
            st.session_state.current_session_id = s_id
            st.rerun()

    # Renderiza todas as mensagens anteriores na tela (estilo ChatGPT)
    for msg in current_session["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "sources" in msg:
                with st.expander("📚 Ver trechos do documento usados como fonte"):
                    st.markdown(msg["sources"], unsafe_allow_html=True)

    question = st.chat_input("Digite sua pergunta aqui e aperte Enter...")

    if question:
        if not query_engine:
            st.error("Por favor, faça o upload de um PDF primeiro!")
            st.stop()
            
        # Atualiza título na primeira pergunta
        if len(current_session["messages"]) == 0:
            current_session["title"] = question[:30] + ("..." if len(question) > 30 else "")
            
        # Adiciona pergunta do usuário ao histórico
        current_session["messages"].append({"role": "user", "content": question})
        save_sessions(st.session_state.sessions)
        with st.chat_message("user"):
            st.write(question)

        # Exibe a resposta do assistente
        with st.chat_message("assistant"):
            with st.spinner("Analisando os PDFs e gerando a resposta..."):
                response = query_engine.query(question)

            st.write(response.response)

            sources_html = ""
            with st.expander("📚 Ver trechos do documento usados como fonte"):
                for i, node in enumerate(response.source_nodes):
                    st.markdown(f"**Trecho {i+1}:**")
                    import html
                    safe_text = html.escape(node.node.text.strip())
                    safe_text = safe_text.replace('\n\n', '__PARAGRAPH__')
                    safe_text = safe_text.replace('\n', ' ')
                    safe_text = safe_text.replace('__PARAGRAPH__', '<br><br>')
                    
                    block = f"<div style='background-color: var(--secondary-background-color); padding: 15px; border-radius: 8px; font-size: 14px; margin-bottom: 15px; line-height: 1.5;'>{safe_text}</div>"
                    st.markdown(block, unsafe_allow_html=True)
                    sources_html += f"**Trecho {i+1}:**\n{block}\n"
            
            # Salva a resposta no histórico
            current_session["messages"].append({
                "role": "assistant", 
                "content": response.response,
                "sources": sources_html
            })
            save_sessions(st.session_state.sessions)
            
            # Força o recarregamento para atualizar o histórico lateral e o título
            st.rerun()
