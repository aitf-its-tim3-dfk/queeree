import torch
import json
import numpy as np
import hnswlib
from sentence_transformers import SentenceTransformer
import os


def unpack_binary_embeddings(emb):
    # If it is packed (e.g., shape (N, 1024/8)), unpack it to (N, 1024)
    # If it's already (N, 1024), it might just be int8 with 0/1 or -1/1.
    if emb.dtype == np.uint8 or emb.dtype == np.int8:
        if emb.shape[1] < 1000:  # highly likely packed
            unpacked = np.unpackbits(emb.view(np.uint8), axis=1)
            # convert 0/1 to -1/1 for inner product pseudo-hamming
            return unpacked.astype(np.float32) * 2 - 1
        else:
            # Maybe it's already -1 and 1 or 0 and 1
            float_emb = emb.astype(np.float32)
            if np.all(float_emb >= 0):  # it's 0 and 1
                return float_emb * 2 - 1
            return float_emb
    return emb.astype(np.float32)


def main():
    print("Loading laws.json...")
    with open("data/laws.json", "r", encoding="utf-8") as f:
        laws = json.load(f)

    print("Loading Model...")
    model = SentenceTransformer(
        "perplexity-ai/pplx-embed-v1-0.6B",
        trust_remote_code=True,
        model_kwargs={"dtype": torch.bfloat16},
    )

    titles = [law["pasal"] for law in laws]
    passages = [law["description"] for law in laws]

    print("Encoding titles...")
    title_embs = model.encode(titles, quantization="binary")
    print("Title embeddings raw shape/dtype:", title_embs.shape, title_embs.dtype)
    title_embs = unpack_binary_embeddings(title_embs)
    print("Title embeddings unpacked:", title_embs.shape, title_embs.dtype)

    print("Encoding passages...")
    passage_embs = model.encode(passages, quantization="binary")
    passage_embs = unpack_binary_embeddings(passage_embs)

    dim = title_embs.shape[1]
    num_elements = len(laws)

    print(f"Initializing hnswlib index (dim={dim})...")
    # We use inner product 'ip' for cosine/hamming correlation on {-1, 1} vectors
    p_index = hnswlib.Index(space="ip", dim=dim)
    p_index.init_index(max_elements=num_elements, ef_construction=200, M=16)

    # Insert items
    p_index.add_items(passage_embs, np.arange(num_elements))

    # Save the index to disk
    os.makedirs("data", exist_ok=True)
    p_index.save_index("data/law_passages.bin")

    np.save("data/law_title_embs.npy", title_embs)

    with open("data/law_metadata.json", "w", encoding="utf-8") as f:
        json.dump(laws, f, indent=2)

    print("Successfully built index and saved to data/")


if __name__ == "__main__":
    main()
