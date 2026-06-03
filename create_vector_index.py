import logging
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vectordb.faiss_store import FAISSVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PDF_PATH = "data/raw/ifc-annual-report-2024-financials.pdf"
texts = [p["text_representation"] for p in pages]

metadatas = [p["metadata"] for p in pages]

vector_store.add_documents(texts, metadatas) 

def build_index():
    logger.info("Starting vector index creation...")

    # -----------------------------
    # 1. Ingestion pipeline (FIXED)
    # -----------------------------
    pipeline = IngestionPipeline(pdf_path=PDF_PATH)
    documents = pipeline.run()

    if not documents:
        logger.error("No documents found from ingestion pipeline!")
        return

    logger.info(f"Total documents received: {len(documents)}")

    # -----------------------------
    # 2. Embeddings
    # -----------------------------
    embedding_model = EmbeddingGenerator()

    # -----------------------------
    # 3. FAISS store
    # -----------------------------
    vector_store = FAISSVectorStore(
        embedding_model.model
    )

    # -----------------------------
    # 4. Prepare data
    # -----------------------------
    texts = []
    metadatas = []

    for doc in documents:
        if isinstance(doc, dict):
            texts.append(doc.get("chunk", ""))
            metadatas.append(doc.get("metadata", {}))
        else:
            texts.append(str(doc))
            metadatas.append({})

    # -----------------------------
    # 5. Build index
    # -----------------------------
    logger.info("Adding documents to FAISS...")
    vector_store.add_documents(texts, metadatas)

    logger.info("Saving FAISS index...")
    vector_store.save_index()

    logger.info("Index creation completed successfully!")


if __name__ == "__main__":
    build_index()