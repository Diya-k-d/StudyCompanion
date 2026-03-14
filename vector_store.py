from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


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

        # create chunk around 350–400 characters
        if len(chunk) > 380:
            chunks.append(chunk.strip())
            chunk = ""

    if chunk:
        chunks.append(chunk.strip())

    # create embeddings
    embeddings = model.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, chunks


# ------------------------------------------------
# Search vector store
# ------------------------------------------------
def search_vector_store(index, chunks, question):

    # encode question
    question_embedding = model.encode([question]).astype("float32")

    faiss.normalize_L2(question_embedding)

    # retrieve only the best chunk
    distances, indices = index.search(question_embedding, k=1)

    best_chunk = chunks[indices[0][0]]

    return best_chunk