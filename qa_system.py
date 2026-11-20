import os
import json
import urllib.request

from vector_store import search_vector_store


# ------------------------------------------------
# Model config
# ------------------------------------------------
# Hosted demo build: answers come from the Gemini API free tier
# (key lives in Streamlit secrets). The fully local Phi-3 Mini
# version is at github.com/Diya-k-d/StudyCompanion.
model_name = os.environ.get("SC_GEMINI_MODEL", "gemini-2.5-flash")


# ------------------------------------------------
# API key
# ------------------------------------------------
def get_api_key():

    # Streamlit Cloud secrets first, then environment
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass

    return os.environ.get("GEMINI_API_KEY", "")


# ------------------------------------------------
# Startup check (called once by app.py)
# ------------------------------------------------
def load_llm():

    if not get_api_key():
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it in the app's Streamlit secrets."
        )

    return True


# ------------------------------------------------
# Clean answer
# ------------------------------------------------
def clean_answer(text):

    # remove any leftover tokens
    text = text.strip()

    # stop after first 3 sentences
    sentences = text.split(".")
    text = ".".join(sentences[:3]).strip()

    if text and not text.endswith("."):
        text += "."

    return text


# ------------------------------------------------
# Answer generation
# ------------------------------------------------
def answer_question(index, chunks, question):

    # Retrieve most relevant chunk
    context = search_vector_store(index, chunks, question)

    # keep context reasonably small
    context = context[:450]


    prompt = f"""You are helping a student understand their study notes.

Instructions:
- Read the context carefully.
- Answer based mainly on the study notes.
- Explain clearly in simple textbook language.
- Do not invent unrelated information.
- Write 1-3 sentences.

Study Notes:
{context}

Question:
{question}

Answer clearly using the study notes."""


    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + model_name
        + ":generateContent"
    )

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.9,
            "maxOutputTokens": 150
        }
    }

    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": get_api_key()
        }
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))

    answer = data["candidates"][0]["content"]["parts"][0]["text"]

    answer = clean_answer(answer)

    return answer
