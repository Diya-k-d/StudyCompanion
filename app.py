import streamlit as st
import sys
import os
import uuid
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pdf_reader import read_pdf
from src.vector_store import create_vector_store
from src.qa_system import answer_question, load_llm


# ------------------------------------------------
# Page config
# ------------------------------------------------
st.set_page_config(
    page_title="StudyCompanion",
    page_icon="📘",
    layout="wide"
)


# ------------------------------------------------
# Styling
# ------------------------------------------------
st.markdown("""
<style>
body {
    background:#0f172a;
}

.chat-user {
    background:#1e293b;
    padding:12px;
    border-radius:12px;
    margin-bottom:12px;
}

.chat-ai {
    background:#111827;
    padding:12px;
    border-radius:12px;
    margin-bottom:16px;
}

.title {
    font-size:36px;
    font-weight:700;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------
# Session state
# ------------------------------------------------
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:

    chat_id = str(uuid.uuid4())

    st.session_state.chats[chat_id] = {
        "messages": [],
        "index": None,
        "chunks": None,
        "title": "New Chat"
    }

    st.session_state.current_chat = chat_id


# ------------------------------------------------
# Sidebar
# ------------------------------------------------
with st.sidebar:

    st.title("📘 StudyCompanion")

    if st.button("➕ New Chat"):

        chat_id = str(uuid.uuid4())

        st.session_state.chats[chat_id] = {
            "messages": [],
            "index": None,
            "chunks": None,
            "title": "New Chat"
        }

        st.session_state.current_chat = chat_id

        st.rerun()


    st.markdown("---")

    for chat_id in list(st.session_state.chats.keys()):

        cols = st.columns([4,1])

        chat_title = st.session_state.chats[chat_id]["title"]

        with cols[0]:
            if st.button(chat_title, key=chat_id):
                st.session_state.current_chat = chat_id

        with cols[1]:
            if st.button("🗑", key=f"del{chat_id}"):

                del st.session_state.chats[chat_id]

                if len(st.session_state.chats) == 0:

                    new_chat = str(uuid.uuid4())

                    st.session_state.chats[new_chat] = {
                        "messages": [],
                        "index": None,
                        "chunks": None,
                        "title": "New Chat"
                    }

                    st.session_state.current_chat = new_chat

                else:
                    st.session_state.current_chat = list(
                        st.session_state.chats.keys()
                    )[0]

                st.rerun()


# ------------------------------------------------
# Header
# ------------------------------------------------
st.markdown('<div class="title">StudyCompanion</div>', unsafe_allow_html=True)
st.caption("Your AI partner for understanding study notes")

st.markdown("---")


# ------------------------------------------------
# Load model once
# ------------------------------------------------
@st.cache_resource
def initialize_llm():
    return load_llm()

tokenizer, model = initialize_llm()


# ------------------------------------------------
# Current chat
# ------------------------------------------------
chat = st.session_state.chats[st.session_state.current_chat]


# ------------------------------------------------
# Show previous messages
# ------------------------------------------------
for role, message in chat["messages"]:

    if role == "user":
        st.markdown(
            f'<div class="chat-user">🧑 {message}</div>',
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f'<div class="chat-ai">🤖 {message}</div>',
            unsafe_allow_html=True
        )


# ------------------------------------------------
# Input area
# ------------------------------------------------
col1, col2 = st.columns([5,1])

with col1:
    question = st.chat_input("Ask about your notes")

with col2:
    uploaded_file = st.file_uploader(
        "",
        type="pdf",
        label_visibility="collapsed",
        key=f"upload_{st.session_state.current_chat}"
    )


# ------------------------------------------------
# Process uploaded PDF
# ------------------------------------------------
if uploaded_file:

    text = read_pdf(uploaded_file)

    index, chunks = create_vector_store(text)

    chat["index"] = index
    chat["chunks"] = chunks


# ------------------------------------------------
# Question answering
# ------------------------------------------------
if question:

    if chat["title"] == "New Chat":
        chat["title"] = question[:40]

    chat["messages"].append(("user", question))

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        placeholder = st.empty()

        if chat["index"] is None:
            answer = "Please upload a PDF first."

        else:
            answer = answer_question(
                chat["index"],
                chat["chunks"],
                question
            )

        streamed = ""

        for word in answer.split():
            streamed += word + " "
            placeholder.markdown(streamed)
            time.sleep(0.01)

    chat["messages"].append(("assistant", answer))

    st.rerun()