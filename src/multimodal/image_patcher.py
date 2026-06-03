from PIL import Image

class ImagePatcher:

    def split(self, image_path, grid=2):
        img = Image.open(image_path)

        w, h = img.size
        patches = []

        patch_w = w // grid
        patch_h = h // grid

        for i in range(grid):
            for j in range(grid):

                box = (
                    j * patch_w,
                    i * patch_h,
                    (j + 1) * patch_w,
                    (i + 1) * patch_h
                )

                patch = img.crop(box)
                patches.append(patch)

        return patches
    def ask(self, query, mode="text"):

        if mode == "text":
            docs = self.retriever.multi_hop_retrieve(query)

        elif mode == "multimodal":
            docs = self.retriever.retrieve_multimodal(query)

        elif mode == "vision":
            docs = self.retriever.vision_retrieve(query, self.vision_store)

        context = "\n".join([d["text"] for d in docs])
        return self.gemini.generate_answer(query, context)