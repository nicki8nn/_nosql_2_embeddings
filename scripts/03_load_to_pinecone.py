# Creates the Pinecone index "arxiv-papers"
# and uploads all 10,000 paper vectors with their metadata, in batches.
#
# Vector id "paper_<i>" = row i in arxiv_subset.parquet = row i in embeddings.npy.

import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INPUT_PARQUET    = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
METRIC     = "cosine"   # embeddings are normalized -> cosine = dot product
BATCH_SIZE = 200        # Pinecone recommends batches of up to 200 vectors

# Client initialization
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# 1. Create the index (if it doesn't exist) and connect to it
existing_indexes = pc.list_indexes().names()

if INDEX_NAME not in existing_indexes:
    print(f"Creating index '{INDEX_NAME}' (dim={VECTOR_DIM}, metric={METRIC})...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric=METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # the only region on Starter
    )

    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)
    print("Index is ready.")
else:
    print(f"Index '{INDEX_NAME}' already exists — connecting to it.")

index = pc.Index(INDEX_NAME)

# 2. Load the data
df = pd.read_parquet(INPUT_PARQUET)
embeddings = np.load(INPUT_EMBEDDINGS)

assert len(df) == len(embeddings), "Parquet and embeddings have different lengths!"
assert embeddings.shape[1] == VECTOR_DIM, f"Expected dim {VECTOR_DIM}, got {embeddings.shape[1]}"
print(f"Papers: {len(df)}, embeddings: {embeddings.shape}")


# 3. Prepare one record for Pinecone
def make_record(i: int, row: pd.Series, vector: np.ndarray) -> dict:
    """
    Pinecone metadata accepts only simple types (str, int, float, bool, list of str),
    so numpy/pandas types are converted to plain Python with str() / int().
    """
    return {
        "id": f"paper_{i}",
        "values": vector.tolist(),
        "metadata": {
            "arxiv_id": str(row["id"]),
            "title":    str(row["title"]),
            "abstract": str(row["abstract"])[:500],  # metadata limit: 40 KB per vector
            "authors":  str(row["authors"])[:200],
            "year":     int(row["year"]),
            "category": str(row["category"]),
        },
    }


# 4. Upload in batches with a progress bar
for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Uploading to Pinecone"):
    end = min(start + BATCH_SIZE, len(df))
    batch = [
        make_record(i, df.iloc[i], embeddings[i])
        for i in range(start, end)
    ]
    index.upsert(vectors=batch)

# 5. Total number of vectors in the index
for _ in range(30):
    total = index.describe_index_stats()["total_vector_count"]
    if total >= len(df):
        break
    time.sleep(2)

print(f"\nTotal vectors in index '{INDEX_NAME}': {total}")
print(index.describe_index_stats())