import json

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vectordb.faiss_indexer import FAISSIndexer


with open(
    "data/processed/chunked_documents.json",
    "r"
) as f:

    documents = json.load(f)


texts = [doc["chunk"] for doc in documents]


embedding_model = EmbeddingGenerator()

embeddings = embedding_model.generate_embeddings(texts)


indexer = FAISSIndexer()

indexer.create_index(
    embeddings,
    documents
)

indexer.save_index()

print("\nFAISS indexing completed.\n")