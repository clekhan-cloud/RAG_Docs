# src/ingestion/table_extractor.py

import pdfplumber
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class TableExtractor:

    def __init__(self, pdf_path: str):

        self.pdf_path = pdf_path
        self.pdf_name = Path(pdf_path).name

    # =====================================================
    # EXTRACT TABLES
    # =====================================================
    def extract_tables(self):

        logger.info(f"Extracting tables from: {self.pdf_path}")

        extracted_tables = []

        with pdfplumber.open(self.pdf_path) as pdf:

            for page_num, page in enumerate(pdf.pages):

                try:

                    tables = page.extract_tables()

                    if not tables:
                        continue

                    for table_idx, table in enumerate(tables):

                        if not table:
                            continue

                        # -----------------------------------
                        # Convert table rows into text
                        # -----------------------------------
                        table_text_rows = []

                        for row in table:

                            # remove None values safely
                            clean_row = [
                                str(cell).strip()
                                if cell is not None
                                else ""
                                for cell in row
                            ]

                            row_text = " | ".join(clean_row)

                            table_text_rows.append(row_text)

                        table_text = "\n".join(table_text_rows)

                        # skip empty tables
                        if not table_text.strip():
                            continue

                        extracted_tables.append({

                            "text": table_text,

                            "metadata": {
                                "page": page_num + 1,
                                "source": self.pdf_name,
                                "content_type": "table",
                                "table_id": table_idx + 1
                            }
                        })

                except Exception as e:

                    logger.warning(
                        f"Failed extracting tables from page "
                        f"{page_num + 1}: {e}"
                    )

        logger.info(
            f"Extracted {len(extracted_tables)} tables"
        )

        return extracted_tables