import os
import re
import urllib.request

import numpy as np


# ------------------------------------------------
# Embedding model: all-MiniLM-L6-v2 (ONNX, runs on CPU)
# Downloaded once from Hugging Face on first use.
# ------------------------------------------------
MODEL_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
VOCAB_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt"

CACHE_DIR = "/tmp/minilm"
MAX_LEN = 128

_session = None
_vocab = None


def _download(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def load_model():
    global _session, _vocab
    if _session is None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        model_path = _download(MODEL_URL, os.path.join(CACHE_DIR, "model.onnx"))
        vocab_path = _download(VOCAB_URL, os.path.join(CACHE_DIR, "vocab.txt"))

        import onnxruntime as ort
        _session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        with open(vocab_path, "r", encoding="utf-8") as f:
            _vocab = {line.strip(): i for i, line in enumerate(f)}

    return _session, _vocab


# ------------------------------------------------
# Minimal BERT WordPiece tokenizer (uncased)
# ------------------------------------------------
def _wordpiece(word, vocab):
    if word in vocab:
        return [vocab[word]]
    ids = []
    start = 0
    while start < len(word):
        end = len(word)
        found = None
        while end > start:
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                found = vocab[piece]
                break
            end -= 1
        if found is None:
            return [vocab["[UNK]"]]
        ids.append(found)
        start = end
    return ids


def encode(text, vocab):
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+|[^\sa-z0-9]", text)
    ids = [vocab["[CLS]"]]
    for tok in tokens:
        ids.extend(_wordpiece(tok, vocab))
    ids = ids[: MAX_LEN - 1]
    ids.append(vocab["[SEP]"])
    mask = [1] * len(ids)
    types = [0] * len(ids)
    while len(ids) < MAX_LEN:
        ids.append(vocab["[PAD]"])
        mask.append(0)
        types.append(0)
    return ids, mask, types


def embed(texts):
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    session, vocab = load_model()

    # small batches keep memory low on free hosts
    results = []
    batch_size = 32
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        all_ids, all_mask, all_types = [], [], []
        for t in batch:
            ids, mask, types = encode(t, vocab)
            all_ids.append(ids)
            all_mask.append(mask)
            all_types.append(types)

        outputs = session.run(
            None,
            {
                "input_ids": np.array(all_ids, dtype=np.int64),
                "attention_mask": np.array(all_mask, dtype=np.int64),
                "token_type_ids": np.array(all_types, dtype=np.int64),
            },
        )
        token_embeddings = outputs[0]  # (batch, seq, dim)

        mask = np.array(all_mask, dtype="float32")[:, :, None]
        summed = (token_embeddings * mask).sum(axis=1)
        counts = np.maximum(mask.sum(axis=1), 1e-12)
        results.append(summed / counts)  # mean pooling

    embeddings = np.vstack(results)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / np.maximum(norms, 1e-12)).astype("float32")


# ------------------------------------------------
# Create vector store
# ------------------------------------------------
def create_vector_store(text):

    # split text into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    chunk = ""

    for sentence in sentences:

        chunk += " " + sentence

        # create chunk around 350-400 characters
        if len(chunk) > 380:
            chunks.append(chunk.strip())
            chunk = ""

    if chunk:
        chunks.append(chunk.strip())

    # drop empty chunks (blank or image-only PDFs)
    chunks = [c for c in chunks if c]

    # cap for free-host memory; plenty for study notes
    chunks = chunks[:400]

    if not chunks:
        return None, []

    embeddings = embed(chunks)

    return embeddings, chunks


# ------------------------------------------------
# Search vector store
# ------------------------------------------------
def search_vector_store(index, chunks, question):

    q = embed([question])[0]

    # cosine similarity against every chunk
    scores = index @ q

    best = int(np.argmax(scores))

    return chunks[best]
