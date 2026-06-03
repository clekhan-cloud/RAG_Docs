class PageBuilder:

    def build(self, texts, tables, images):

        pages = {}

        # -------------------------
        # 1. TEXT
        # -------------------------
        for t in texts:
            page = t["metadata"]["page"]

            if page not in pages:
                pages[page] = {
                    "page": page,
                    "text": "",
                    "tables": [],
                    "images": []
                }

            pages[page]["text"] += t["text"] + "\n"

        # -------------------------
        # 2. TABLES
        # -------------------------
        for tb in tables:
            page = tb["page"]

            if page not in pages:
                pages[page] = {
                    "page": page,
                    "text": "",
                    "tables": [],
                    "images": []
                }

            pages[page]["tables"].append(tb.get("table_text", ""))

        # -------------------------
        # 3. IMAGES (CAPTIONS)
        # -------------------------
        for img in images:
            page = img["page"]

            if page not in pages:
                pages[page] = {
                    "page": page,
                    "text": "",
                    "tables": [],
                    "images": []
                }

            pages[page]["images"].append(img.get("caption", ""))

        # -------------------------
        # 4. FINAL TEXT REPRESENTATION
        # -------------------------
        final_pages = []

        for p in pages.values():

            combined = ""

            combined += "TEXT:\n" + p["text"] + "\n\n"

            if p["tables"]:
                combined += "TABLES:\n" + "\n".join(p["tables"]) + "\n\n"

            if p["images"]:
                combined += "IMAGES:\n" + "\n".join(p["images"])

            final_pages.append({
                "text_representation": combined,
                "metadata": {
                    "page": p["page"],
                    "type": "multimodal_page"
                }
            })

        return final_pages