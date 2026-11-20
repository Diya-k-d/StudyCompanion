import os
import re
import json
import urllib.request
import urllib.error

from vector_store import search_vector_store


# ------------------------------------------------
# Model config
# ------------------------------------------------
# Hosted demo build: answers come from the Gemini API free tier
# (key lives in Streamlit secrets). The fully local Phi-3 Mini
# version is at github.com/Diya-k-d/StudyCompanion.
# The model is auto-detected from what the API key can access;
# set SC_GEMINI_MODEL in secrets to force one.
API_BASE = "https://generativelanguage.googleapis.com/v1beta"

_chosen_model = None


# ------------------------------------------------
# API key
# ------------------------------------------------
def _secret(name):

    try:
        import streamlit as st
        value = st.secrets.get(name, "")
        if value:
            return str(value)
    except Exception:
        pass

    return os.environ.get(name, "")


def get_api_key():

    # tolerate quotes / spaces if pasted into secrets that way
    return _secret("GEMINI_API_KEY").strip().strip('"').strip("'").strip()


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
# HTTP helpers
# ------------------------------------------------
def _describe_error(error):

    if isinstance(error, urllib.error.HTTPError):

        try:
            body = error.read().decode("utf-8")[:600]
        except Exception:
            body = ""

        hints = {
            400: "The key or request was rejected. Check the key was pasted in full.",
            401: "The API key was not accepted. Check GEMINI_API_KEY in secrets.",
            403: "Access refused. The key may be restricted or invalid - create a fresh one at aistudio.google.com.",
            404: "Model not found for this key.",
            429: "Free-tier quota hit. Wait a minute and try again.",
        }

        hint = hints.get(error.code, "")

        return ("Gemini API error HTTP %s. %s %s" % (error.code, hint, body)).strip()

    return "Gemini API connection error: %s" % error


def _api_post(path, payload, key):

    request = urllib.request.Request(
        API_BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": key
        }
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _api_get(path, key):

    request = urllib.request.Request(
        API_BASE + path,
        headers={"x-goog-api-key": key}
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


# ------------------------------------------------
# Rank the models this key can use
# ------------------------------------------------
def ranked_models(key):

    override = _secret("SC_GEMINI_MODEL").strip()
    if override:
        return [override]

    try:
        data = _api_get("/models?pageSize=200", key)
    except Exception as e:
        raise RuntimeError(_describe_error(e))

    blocked = (
        "tts", "image", "audio", "embedding", "omni", "robotics", "aqa",
        "banana", "transcribe", "computer-use", "antigravity",
        "deep-research", "lyria", "gemma", "nano"
    )

    candidates = []
    for m in data.get("models", []):

        name = m.get("name", "").split("/")[-1]

        if "generateContent" not in m.get("supportedGenerationMethods", []):
            continue

        if any(b in name for b in blocked):
            continue

        version = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        version_key = tuple(int(p) for p in version.group(1).split(".")) if version else (0,)

        is_flash = 1 if "flash" in name else 0
        is_preview = 1 if "preview" in name else 0

        candidates.append((is_flash, version_key, -is_preview, name))

    if not candidates:
        raise RuntimeError(
            "This API key has no text generation models available. "
            "Create a fresh key at aistudio.google.com."
        )

    candidates.sort(reverse=True)

    return [name for *_rank, name in candidates[:6]]


# ------------------------------------------------
# Clean answer
# ------------------------------------------------
def clean_answer(text):

    # remove any leftover tokens
    text = text.strip()

    # stop after first 3 sentences
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    sentences = sentences[:3]

    # drop a trailing fragment like "This" from a cut-off sentence
    if len(sentences) > 1 and len(sentences[-1].split()) < 4:
        sentences = sentences[:-1]

    text = ". ".join(sentences).strip()

    if text and not text.endswith("."):
        text += "."

    return text


# ------------------------------------------------
# Answer generation
# ------------------------------------------------
def answer_question(index, chunks, question):

    key = get_api_key()

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


    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.9,
            "maxOutputTokens": 512
        }
    }

    last_error = None
    answer = None

    for model in ranked_models(key):

        try:
            data = _api_post(
                "/models/" + model + ":generateContent",
                body,
                key
            )
        except Exception as e:
            last_error = e
            continue

        # collect every text part (thinking models may return several)
        texts = []
        for cand in data.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if part.get("text"):
                    texts.append(part["text"])

        if texts:
            answer = texts[-1]
            global _chosen_model
            _chosen_model = model
            break

        last_error = RuntimeError(
            "Model %s returned no text." % model
        )

    if answer is None:
        raise RuntimeError(_describe_error(last_error))

    answer = clean_answer(answer)

    return answer
