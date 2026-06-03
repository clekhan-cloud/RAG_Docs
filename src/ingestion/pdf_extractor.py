import fitz
from pathlib import Path
from src.utils.logger import logger


class PDFExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.doc = None

    def load_pdf(self):
        """
        Load PDF document.
        """
        try:
            self.doc = fitz.open(self.pdf_path)
            logger.info(f"Loaded PDF: {self.pdf_path}")
        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            raise

    def extract_text(self):
        """
        Extract text from all pages.
        """
        if self.doc is None:
            self.load_pdf()

        extracted_pages = []

        try:
            for page_num in range(len(self.doc)):
                page = self.doc[page_num]

                text = page.get_text("text")

                metadata = {
                    "page_number": page_num + 1,
                    "source": Path(self.pdf_path).name
                }

                extracted_pages.append({
                    "text": "...".join(text.splitlines()),
                    "metadata": metadata
                })

            logger.info(f"Extracted text from {len(extracted_pages)} pages")

            return extracted_pages

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise

    def extract_document_metadata(self):
        """
        Extract PDF metadata.
        """
        if self.doc is None:
            self.load_pdf()

        try:
            metadata = self.doc.metadata

            logger.info("Extracted document metadata")

            return metadata

        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            raise