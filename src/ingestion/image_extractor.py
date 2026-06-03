# src/ingestion/image_extractor.py

import fitz  # PyMuPDF
import os
import logging
from pathlib import Path

from src.llm.gemini_client import GeminiClient


logger = logging.getLogger(__name__)


class ImageExtractor:

    def __init__(
        self,
        pdf_path: str,
        output_dir: str = "data/images"
    ):

        self.pdf_path = pdf_path
        self.output_dir = output_dir

        self.pdf_name = Path(pdf_path).name

        # create image folder
        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        # Gemini captioning client
        self.gemini = GeminiClient()

    # =====================================================
    # EXTRACT IMAGES + GENERATE CAPTIONS
    # =====================================================
    def extract_images(self):

        logger.info(
            f"Extracting images from {self.pdf_path}"
        )

        doc = fitz.open(self.pdf_path)

        extracted_images = []

        # ---------------------------------------------
        # Iterate pages
        # ---------------------------------------------
        for page_index in range(len(doc)):

            page = doc[page_index]

            image_list = page.get_images(full=True)

            # skip pages without images
            if not image_list:
                continue

            # -----------------------------------------
            # Extract each image
            # -----------------------------------------
            for img_index, img in enumerate(image_list):

                try:

                    xref = img[0]

                    base_image = doc.extract_image(xref)

                    image_bytes = base_image["image"]

                    image_ext = base_image["ext"]

                    image_filename = (
                        f"page_{page_index+1}"
                        f"_img_{img_index+1}.{image_ext}"
                    )

                    image_path = os.path.join(
                        self.output_dir,
                        image_filename
                    )

                    # ---------------------------------
                    # Save image
                    # ---------------------------------
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)

                    # ---------------------------------
                    # Generate caption using Gemini
                    # ---------------------------------
                    try:

                        caption = self.gemini.caption_image(
                            image_path
                        )

                    except Exception as caption_error:

                        logger.warning(
                            f"Caption failed for "
                            f"{image_filename}: "
                            f"{caption_error}"
                        )

                        caption = (
                            "No caption generated"
                        )

                    # ---------------------------------
                    # Store multimodal document
                    # ---------------------------------
                    extracted_images.append({

                        "text": caption,

                        "image_path": image_path,

                        "image_caption": caption,

                        "metadata": {

                            "page": page_index + 1,

                            "source": self.pdf_name,

                            "content_type": "image",

                            "image_id": img_index + 1
                        }
                    })

                except Exception as e:

                    logger.warning(
                        f"Failed extracting image "
                        f"from page {page_index+1}: {e}"
                    )

        doc.close()

        logger.info(
            f"Extracted {len(extracted_images)} images"
        )

        return extracted_images