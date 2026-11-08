# scripts/02_embed.py
#
# Turns every paper (title + abstract) into a 768-dimensional vector
# with the allenai/specter2_base model and saves all vectors to
# embeddings/embeddings.npy.
#
# Important: row i in embeddings.npy belongs to row i in arxiv_subset.parquet.

import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_PARQUET = "data/arxiv_subset.parquet"
OUTPUT_DIR    = "embeddings"
OUTPUT_FILE   = os.path.join(OUTPUT_DIR, "embeddings.npy")
MODEL_NAME    = "allenai/specter2_base"
BATCH_SIZE    = 64

# 1. Load the dataset
df = pd.read_parquet(INPUT_PARQUET)
print(f"Loaded papers: {len(df)}")

# 2. Prepare texts: "title [SEP] abstract" — the input format specter2 was trained on
texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()
print(f"\nExample input text:\n{texts[0][:200]}...")

# 3. Load the model
# Expected warning: "No sentence-transformers model found ... Creating a new one
# with mean pooling." specter2_base is a plain HuggingFace model, so
# sentence-transformers adds mean pooling on its own. This is not an error.
model = SentenceTransformer(MODEL_NAME)
print(f"\nModel device: {model.device}")          # mps = Apple GPU, cpu = processor
print(f"Max input length (tokens): {model.max_seq_length}")

# 4. Encode all texts in batches, with a progress bar, normalized to length 1
embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True,
)

# 5. Checks
assert len(embeddings) == len(df), "Number of vectors does not match number of papers!"

print(f"\nTexts encoded:       {len(embeddings)}")
print(f"Embedding dimension: {embeddings.shape[1]}")              # expected 768
print(f"Norm of the 1st embedding: {np.linalg.norm(embeddings[0]):.6f}")  # expected ~1.0
print(f"Data type: {embeddings.dtype}")

# 6–7. Save (create the folder if it doesn't exist)
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.save(OUTPUT_FILE, embeddings)
print(f"\nSaved to {OUTPUT_FILE}")