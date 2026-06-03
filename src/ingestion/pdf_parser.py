# src/ingestion/pdf_parser.py

import fitz  # PyMuPDF


class PDFParser:

    # =====================================================
    # INIT
    # =====================================================
    def __init__(self, pdf_path):

        self.pdf_path = pdf_path

    # =====================================================
    # PARSE PDF
    # =====================================================
    def parse(self):

        documents = []

        pdf = fitz.open(self.pdf_path)

        for page_num in range(len(pdf)):

            page = pdf[page_num]

            text = page.get_text()

            if text.strip():

                documents.append({

                    "text": text,

                    "metadata": {

                        "page": page_num + 1,

                        "source": self.pdf_path,

                        "content_type": "text"
                    }
                })

        pdf.close()

        print(
            f"✅ Extracted {len(documents)} pages from PDF"
        )

        return documents