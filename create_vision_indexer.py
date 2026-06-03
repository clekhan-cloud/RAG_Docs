from src.embeddings.vision_embedding import VisionEmbeddingModel
from src.multimodal.image_patcher import ImagePatcher
from src.vectordb.vision_faiss import VisionFAISSStore

import os


def build():

    embedder = VisionEmbeddingModel()
    patcher = ImagePatcher()
    store = VisionFAISSStore(dim=512)

    image_dir = "data/pages"

    all_embeddings = []
    metadata = []

    for img_name in os.listdir(image_dir):

        path = os.path.join(image_dir, img_name)

        patches = patcher.split(path, grid=2)

        for i, patch in enumerate(patches):

            emb = embedder.embed_image(patch)

            all_embeddings.append(emb)

            metadata.append({
                "image": img_name,
                "patch_id": i
            })

    store.add(all_embeddings, metadata)

    print("Vision index created successfully!")


if __name__ == "__main__":
    build()