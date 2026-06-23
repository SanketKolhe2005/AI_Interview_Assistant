from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def chunk_text(text, chunk_size=500):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks

def create_vector_store(chunks):

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return index

def retrieve(query, chunks, index):

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding),
        k=3
    )

    results = []

    for idx in indices[0]:
        results.append(chunks[idx])

    return "\n".join(results)