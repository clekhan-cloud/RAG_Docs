from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)


class EmbeddingGenerator:

    def __init__(self):

        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        logger.info(f"Loading embedding model: {model_name}")

        self.model = HuggingFaceEmbeddings(
            model_name=model_name
        )

    # -------------------------
    # Unified embedding interface
    # -------------------------
    def embed_documents(self, texts):
        return self.model.embed_documents(texts)

    def embed_query(self, text):
        return self.model.embed_query(text)