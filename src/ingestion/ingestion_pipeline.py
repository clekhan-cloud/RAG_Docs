# =====================================================
# ingestion_pipeline.py
# =====================================================

import os
import uuid
import pandas as pd

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

# =====================================================
# INGESTION PIPELINE
# =====================================================
class IngestionPipeline:

    def __init__(self):

        self.chunk_size = 1000

        self.chunk_overlap = 200

        self.text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        )

    # =================================================
    # PROCESS TEXT
    # =================================================
    def process_text(
        self,
        text,
        page_num,
        source="IFC Annual Report"
    ):

        chunks = self.text_splitter.split_text(text)

        docs = []

        for idx, chunk in enumerate(chunks):

            docs.append({

                "id": str(uuid.uuid4()),

                "text": chunk,

                "metadata": {

                    "content_type": "text",

                    "page": page_num,

                    "chunk_id": idx,

                    "source": source
                }
            })

        return docs

    # =================================================
    # PROCESS TABLES
    # =================================================
    def process_tables(
        self,
        tables,
        page_num,
        source="IFC Annual Report"
    ):

        docs = []

        for idx, table_df in enumerate(tables):

            try:

                # =============================
                # CLEAN TABLE
                # =============================
                table_df = table_df.fillna("")

                # =============================
                # TABLE TEXT
                # =============================
                table_text = table_df.to_string()

                # =============================
                # SIMPLE TABLE SUMMARY
                # =============================
                summary = (
                    f"Financial table extracted "
                    f"from page {page_num}"
                )

                docs.append({

                    "id": str(uuid.uuid4()),

                    "text": table_text,

                    "metadata": {

                        "content_type": "table",

                        "page": page_num,

                        "chunk_id": idx,

                        "source": source,

                        "table_summary": summary,

                        "table_data": (
                            table_df.to_dict(
                                orient="records"
                            )
                        )
                    }
                })

            except Exception as e:

                print(
                    f"Table processing error: {e}"
                )

        return docs

    # =================================================
    # PROCESS IMAGES
    # =================================================
    def process_images(
        self,
        image_paths,
        page_num,
        source="IFC Annual Report"
    ):

        docs = []

        for idx, image_path in enumerate(image_paths):

            try:

                caption = (
                    f"Financial chart/image "
                    f"from page {page_num}"
                )

                docs.append({

                    "id": str(uuid.uuid4()),

                    "text": caption,

                    "metadata": {

                        "content_type": "image",

                        "page": page_num,

                        "chunk_id": idx,

                        "source": source,

                        "image_path": image_path,

                        "image_caption": caption
                    }
                })

            except Exception as e:

                print(
                    f"Image processing error: {e}"
                )

        return docs

    # =================================================
    # COMBINE ALL MODALITIES
    # =================================================
    def build_documents(
        self,
        text="",
        tables=None,
        images=None,
        page_num=0
    ):

        all_docs = []

        # =============================================
        # TEXT DOCS
        # =============================================
        if text:

            all_docs.extend(
                self.process_text(
                    text=text,
                    page_num=page_num
                )
            )

        # =============================================
        # TABLE DOCS
        # =============================================
        if tables:

            all_docs.extend(
                self.process_tables(
                    tables=tables,
                    page_num=page_num
                )
            )

        # =============================================
        # IMAGE DOCS
        # =============================================
        if images:

            all_docs.extend(
                self.process_images(
                    image_paths=images,
                    page_num=page_num
                )
            )

        return all_docs