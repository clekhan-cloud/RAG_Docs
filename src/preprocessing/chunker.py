# src/processing/chunker.py

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


class TextChunker:

    # =====================================================
    # INIT
    # =====================================================
    def __init__(
        self,
        chunk_size=500,
        chunk_overlap=100
    ):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    ""
                ]
            )
        )

    # =====================================================
    # CLEAN TEXT
    # =====================================================
    def clean_text(
        self,
        text
    ):

        if not text:
            return ""

        # remove extra spaces
        text = " ".join(
            text.split()
        )

        return text.strip()

    # =====================================================
    # CHUNK DOCUMENTS
    # =====================================================
    def chunk_documents(
        self,
        documents
    ):

        """
        Creates page-aware chunks.

        INPUT:
        [
            {
                "text": "...",
                "metadata": {
                    "content_type": "table",
                    "page": 42,
                    "table_summary": "...",
                     "table_data": {...}
                    "source": "IFC_2024.pdf"
                }
            }
        ]

        OUTPUT:
        [
            {
                "text": "...",
                "metadata": {
                    "content_type": "image",
                    "page": 55,
                    "image_path": "...",
                    "image_caption": "..."
                    "source": "IFC_2024.pdf",
                }
            }
        ]
        """

        chunked_docs = []

        # -------------------------------------------------
        # PROCESS EACH DOCUMENT
        # -------------------------------------------------
        for doc_id, doc in enumerate(documents):

            text = self.clean_text(
                doc.get("text", "")
            )

            if not text:
                continue

            metadata = doc.get(
                "metadata",
                {}
            )

            # ---------------------------------------------
            # PAGE-AWARE METADATA
            # ---------------------------------------------
            page = metadata.get(
                "page",
                "NA"
            )

            source = metadata.get(
                "source",
                "Unknown"
            )

            content_type = metadata.get(
                "content_type",
                "text"
            )

            # ---------------------------------------------
            # SPLIT INTO CHUNKS
            # ---------------------------------------------
            chunks = self.splitter.split_text(
                text
            )

            # ---------------------------------------------
            # CREATE CHUNK OBJECTS
            # ---------------------------------------------
            for idx, chunk in enumerate(chunks):

                chunk = self.clean_text(chunk)

                if not chunk:
                    continue

                chunk_start = (
                    idx *
                    (
                        self.chunk_size -
                        self.chunk_overlap
                    )
                )

                chunk_end = (
                    chunk_start +
                    len(chunk)
                )

                chunk_doc = {

                    # IMPORTANT
                    "text": chunk,

                    # -------------------------------------
                    # PAGE-AWARE METADATA
                    # -------------------------------------
                    "metadata": {

                        # preserve existing metadata
                        **metadata,

                        # source tracking
                        "source": source,

                        # page grounding
                        "page": page,

                        # content type
                        "content_type": content_type,

                        # chunk tracking
                        "chunk_id": idx,

                        # optional offsets
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_end,

                        # optional doc tracking
                        "document_id": doc_id
                    }
                }

                # -----------------------------------------
                # PRESERVE IMAGE INFORMATION
                # -----------------------------------------
                if "image_path" in doc:
                    chunk_doc["image_path"] = doc[
                        "image_path"
                    ]

                if "image_caption" in doc:
                    chunk_doc["image_caption"] = doc[
                        "image_caption"
                    ]

                chunked_docs.append(
                    chunk_doc
                )

        print(
            f"✅ Created "
            f"{len(chunked_docs)} chunks"
        )

        return chunked_docs

    # =====================================================
    # DEBUG PREVIEW
    # =====================================================
    def preview_chunks(
        self,
        chunks,
        n=3
    ):

        for i, chunk in enumerate(chunks[:n]):

            print("\n" + "=" * 60)

            print(
                f"CHUNK {i+1}"
            )

            print("=" * 60)

            print(
                chunk["text"][:300]
            )

            print("\nMETADATA:")

            print(
                chunk["metadata"]
            )