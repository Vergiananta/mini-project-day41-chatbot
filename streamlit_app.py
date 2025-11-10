import streamlit as st
from typing import Optional

from src.config import load_config
from src.rag import answer_question
from src.search import SemanticSearch
from src.utils.feedback import log_feedback


st.set_page_config(page_title="Customer Assistant RAG", page_icon="💬", layout="wide")

cfg = load_config()

st.title("💬 Customer Assistant")

with st.sidebar:
    st.header("Setting")
    category = st.selectbox("Filter kategori (opsional)", options=["", "faq", "policy", "troubleshooting", "contact", "general"], index=0)
    top_k = st.slider("Top K", min_value=1, max_value=20, value=cfg["DEFAULT_TOP_K"])
    threshold = st.slider("Threshold skor gabungan", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    st.write("Model embedding:", cfg["EMBEDDING_MODEL_NAME"]) 
    st.write("Groq API configured:", bool(cfg["GROQ_API_KEY"]))


def _normalize_query(user_query: str) -> str:
    confirm_words = {"ya", "iya", "oke", "ok", "lanjut", "silakan", "ya lanjut"}
    q = (user_query or "").strip().lower()
    if q in confirm_words or (len(q.split()) <= 2):
        prev = st.session_state.get("last_user_query")
        if prev:
            return prev + " (mohon detail lanjutan)"
    return user_query

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input("Tanyakan sesuatu...")
if query:
    # Simpan pertanyaan terakhir
    st.session_state["last_user_query"] = query
    normalized = _normalize_query(query)
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("assistant"):
        with st.spinner("Mencari konteks dan menyusun jawaban..."):
            try:
                prev_answer = None
                for msg in reversed(st.session_state.chat_history):
                    if msg.get("role") == "assistant":
                        prev_answer = msg.get("content")
                        break
                prev_question = st.session_state.get("last_user_query_prev")
                res = answer_question(
                    normalized,
                    category=category or None,
                    top_k=top_k,
                    previous_answer=prev_answer,
                    previous_question=prev_question,
                )
            except Exception as e:
                st.error(f"Terjadi error: {e}")
                res = {"answer": "", "sources": []}

            st.markdown(res["answer"]) 

    st.session_state.chat_history.append({"role": "assistant", "content": res["answer"]})
    # Simpan pertanyaan sebelumnya untuk konteks di input berikutnya
    st.session_state["last_user_query_prev"] = st.session_state.get("last_user_query")

st.divider()

st.subheader("🔬 Cek hasil pencarian (analytics)")
col1, col2 = st.columns(2)
with col1:
    st.write("Preview Top-K hasil pencarian tanpa LLM")
    query2 = st.text_input("Kueri uji", value="refund policy and contact support")
    if st.button("Cari (semantic+lexical)"):
        with st.spinner("Menjalankan pencarian..."):
            try:
                search = SemanticSearch()
                results = search.query(query2, top_k=top_k, category=category or None, threshold=threshold)
                for i, r in enumerate(results, start=1):
                    st.write(f"{i}. rank≈{round(r.get('rank', 0), 3)} | cat={r.get('category','')}")
                    st.write(r["content"])
            except Exception as e:
                st.error(f"Error search: {e}")