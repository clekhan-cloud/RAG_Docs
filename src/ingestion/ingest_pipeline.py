# src/ingestion/ingest_pipeline.py

import json
import logging
from pathlib import Path

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.table_extractor import TableExtractor
from src.ingestion.image_extractor import ImageExtractor

from src.preprocessing.chunker import TextChunker

from src.embeddings.embedding_generator import EmbeddingGenerator

from src.vectordb.faiss_store import FAISSVectorStore
from src.vectordb.qdrant_store import QdrantVectorStore


logger = logging.getLogger(__name__)


class IngestPipeline:

    def __init__(
        self,
        pdf_path: str,
        db_type: str = "faiss"
    ):

        self.pdf_path = pdf_path
        self.db_type = db_type

        # =================================================
        # Components
        # =================================================
        self.pdf_parser = PDFParser(pdf_path)

        self.table_extractor = TableExtractor(
            pdf_path
        )

        self.image_extractor = ImageExtractor(
            pdf_path
        )

        self.chunker = TextChunker()

        self.embedding_model = EmbeddingGenerator()

        # =================================================
        # Vector DB Selection
        # =================================================
        if db_type == "faiss":

            self.vector_store = FAISSVectorStore(
                self.embedding_model.model
            )

        elif db_type == "qdrant":

            self.vector_store = QdrantVectorStore(
                self.embedding_model.model
            )

        else:
            raise ValueError(
                "db_type must be "
                "'faiss' or 'qdrant'"
            )

    # =====================================================
    # MASTER INGESTION PIPELINE
    # =====================================================
    def run(self):

        logger.info("===================================")
        logger.info("STARTING MASTER INGESTION PIPELINE")
        logger.info("===================================")

        # =================================================
        # STEP 1 — Extract Text
        # =================================================
        logger.info("Extracting text...")

        text_docs = self.pdf_parser.extract_text()

        logger.info(
            f"Extracted {len(text_docs)} text pages"
        )

        # =================================================
        # STEP 2 — Extract Tables
        # =================================================
        logger.info("Extracting tables...")

        table_docs = (
            self.table_extractor.extract_tables()
        )

        logger.info(
            f"Extracted {len(table_docs)} tables"
        )

        # =================================================
        # STEP 3 — Extract Images
        # =================================================
        logger.info("Extracting images...")

        image_docs = (
            self.image_extractor.extract_images()
        )

        logger.info(
            f"Extracted {len(image_docs)} images"
        )

        # =================================================
        # STEP 4 — Chunk TEXT documents only
        # =================================================
        logger.info("Chunking text documents...")

        chunked_text_docs = self.chunker.chunk_documents(
            text_docs
        )

        logger.info(
            f"Generated "
            f"{len(chunked_text_docs)} text chunks"
        )

        # =================================================
        # STEP 5 — Combine ALL documents
        # =================================================
        logger.info("Combining multimodal documents...")

        all_docs = (
            chunked_text_docs +
            table_docs +
            image_docs
        )

        logger.info(
            f"Total multimodal docs: "
            f"{len(all_docs)}"
        )

        # =================================================
        # STEP 6 — Prepare texts + metadata
        # =================================================
        texts = []
        metadatas = []

        for doc in all_docs:

            # chunker output uses "chunk"
            if "chunk" in doc:
                texts.append(doc["chunk"])

            else:
                texts.append(doc["text"])

            metadatas.append(
                doc["metadata"]
            )

        # =================================================
        # STEP 7 — Add to Vector DB
        # =================================================
        logger.info(
            f"Storing embeddings in "
            f"{self.db_type.upper()}..."
        )

        self.vector_store.add_documents(
            texts=texts,
            metadatas=metadatas
        )

        self.vector_store.save_index()

        logger.info("Vector DB storage completed")

        # =================================================
        # STEP 8 — Save processed docs
        # =================================================
        output_dir = Path(
            "data/processed"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            output_dir /
            "multimodal_documents.json"
        )

        serializable_docs = []

        for i, doc in enumerate(all_docs):

            item = dict(doc)

            # ensure chunk stored as text
            if "chunk" in item:
                item["text"] = item.pop("chunk")

            serializable_docs.append(item)

        with open(output_path, "w") as f:

            json.dump(
                serializable_docs,
                f,
                indent=2
            )

        logger.info(
            f"Saved processed docs to "
            f"{output_path}"
        )

        logger.info("===================================")
        logger.info("INGESTION PIPELINE COMPLETED")
        logger.info("===================================")

        return serializable_docs