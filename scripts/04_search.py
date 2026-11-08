# Part 3 — search queries:
#   1. Pure semantic search in Pinecone
#   2. Search with metadata filters (A: recent cs.LG, B: before 2015)
#   3. Comparison of similarity metrics on local embeddings:
#      cosine vs dot product vs L2, plus an experiment with NON-normalized vectors

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5

EMBEDDINGS_FILE     = "embeddings/embeddings.npy"       # normalized (from 02_embed.py)
EMBEDDINGS_RAW_FILE = "embeddings/embeddings_raw.npy"   # NOT normalized (created here once)

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")  # for the full abstract

CURRENT_YEAR = int(df["year"].max())



# Helper functions


def encode_query(text: str, normalize: bool = True) -> np.ndarray:
    """Turn a query into a 768-dim vector — the same model and settings as for the papers."""
    return model.encode(text, normalize_embeddings=normalize, convert_to_numpy=True)


def row_from_id(vector_id: str) -> pd.Series:
    """'paper_123' -> row 123 of the parquet file (full abstract lives there)."""
    return df.iloc[int(vector_id.split("_")[1])]


def print_paper(rank: int, row: pd.Series, score: float, score_name: str = "score") -> None:
    print(f"{rank}. [{score_name} = {score:.4f}] {row['title']}")
    print(f"   {row['category']} | {row['year']} | arXiv:{row['id']}")
    print(f"   {row['abstract'][:200]}...")


def pinecone_search(query: str, top_k: int = TOP_K, filter: dict | None = None) -> list:
    """Search the Pinecone index; returns the list of matches."""
    result = index.query(
        vector=encode_query(query).tolist(),
        top_k=top_k,
        include_metadata=True,
        filter=filter,
    )
    return result["matches"]


def print_matches(matches: list) -> None:
    if not matches:
        print("   (no results)")
    for rank, m in enumerate(matches, start=1):
        print_paper(rank, row_from_id(m["id"]), m["score"], "cosine")


def header(text: str) -> None:
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)


# 1. Pure semantic search


header("1. SEMANTIC SEARCH (no filters)")

semantic_queries = [
    "teaching machines to recognize objects in pictures",
    "protecting user data when training models on many devices",
]

for q in semantic_queries:
    print(f"\nQuery: \"{q}\"\n")
    print_matches(pinecone_search(q))


# 2. Search with filters

header("2. SEARCH WITH FILTERS")

rl_query = "reinforcement learning agent learns a policy from rewards"

# Baseline: the same query without any filter, to see what the filters change
print(f"\nQuery: \"{rl_query}\"")
print("\n--- Baseline: no filter ---\n")
print_matches(pinecone_search(rl_query))

# A: last 5 years (2022–2026) AND category cs.LG
filter_a = {
    "year":     {"$gte": CURRENT_YEAR - 4},
    "category": {"$eq": "cs.LG"},
}
print(f"\n--- A: cs.LG, year >= {CURRENT_YEAR - 4} ---")
print(f"    filter = {filter_a}\n")
matches_a = pinecone_search(rl_query, filter=filter_a)
print_matches(matches_a)

# B: older papers (before 2015), any category
filter_b = {"year": {"$lt": 2015}}
print("\n--- B: year < 2015, any category ---")
print(f"    filter = {filter_b}\n")
matches_b = pinecone_search(rl_query, filter=filter_b)
print_matches(matches_b)

# Short numeric summary for the README comparison
def summary(name: str, matches: list) -> None:
    rows = [row_from_id(m["id"]) for m in matches]
    scores = [m["score"] for m in matches]
    years = [int(r["year"]) for r in rows]
    cats = sorted({r["category"] for r in rows})
    if scores:
        print(f"{name}: scores {min(scores):.4f}–{max(scores):.4f}, "
              f"years {min(years)}–{max(years)}, categories {cats}")

print("\n--- Filter comparison ---")
summary("A (cs.LG, recent)", matches_a)
summary("B (before 2015)  ", matches_b)


# 3. Comparing similarity metrics on local embeddings

header("3. SIMILARITY METRICS ON LOCAL EMBEDDINGS")

emb = np.load(EMBEDDINGS_FILE)  # (10000, 768), normalized
print(f"Loaded embeddings: {emb.shape}, "
      f"norms min/max = {np.linalg.norm(emb, axis=1).min():.4f} / "
      f"{np.linalg.norm(emb, axis=1).max():.4f}")


def cosine_scores(docs: np.ndarray, q: np.ndarray) -> np.ndarray:
    """cos = (a·b) / (|a|·|b|)  — higher = more similar"""
    return (docs @ q) / (np.linalg.norm(docs, axis=1) * np.linalg.norm(q))


def dot_scores(docs: np.ndarray, q: np.ndarray) -> np.ndarray:
    """a·b  — higher = more similar"""
    return docs @ q


def l2_distances(docs: np.ndarray, q: np.ndarray) -> np.ndarray:
    """|a - b|  — LOWER = more similar"""
    return np.linalg.norm(docs - q, axis=1)


def top_indices(values: np.ndarray, k: int, lower_is_better: bool = False) -> np.ndarray:
    order = np.argsort(values)            # ascending
    if not lower_is_better:
        order = order[::-1]               # descending for similarities
    return order[:k]


def compare_metrics(docs: np.ndarray, q: np.ndarray, label: str) -> dict:
    """Print top-5 for each metric and return the index lists for comparison."""
    metrics = {
        "cosine": (cosine_scores(docs, q), False),
        "dot":    (dot_scores(docs, q),    False),
        "L2":     (l2_distances(docs, q),  True),
    }
    tops = {}
    for name, (values, lower_better) in metrics.items():
        idx = top_indices(values, TOP_K, lower_is_better=lower_better)
        tops[name] = list(idx)
        print(f"\n--- {label}: top-{TOP_K} by {name} ---")
        for rank, i in enumerate(idx, start=1):
            print_paper(rank, df.iloc[i], values[i], name)

    print(f"\n>>> {label}: same top-{TOP_K} (incl. order)?")
    print(f"    cosine vs dot: {tops['cosine'] == tops['dot']}")
    print(f"    cosine vs L2:  {tops['cosine'] == tops['L2']}")
    return tops


metric_query = semantic_queries[0]
print(f"\nQuery: \"{metric_query}\"")

# 3a. Normalized embeddings (as stored in Pinecone)
q_norm = encode_query(metric_query, normalize=True)
tops_local = compare_metrics(emb, q_norm, "NORMALIZED")

# Sanity check: does Pinecone return the same top-5 as local cosine?
pinecone_ids = [int(m["id"].split("_")[1]) for m in pinecone_search(metric_query)]
print(f"\n>>> Pinecone top-{TOP_K} == local cosine top-{TOP_K}: "
      f"{pinecone_ids == tops_local['cosine']}")

# 3b. Experiment: what if embeddings were NOT normalized?
# We encode all papers once more with normalize_embeddings=False
# (~1–2 min on mps) and cache the result, so later runs are fast.
if os.path.exists(EMBEDDINGS_RAW_FILE):
    emb_raw = np.load(EMBEDDINGS_RAW_FILE)
else:
    print("\nEncoding papers WITHOUT normalization (one time, cached afterwards)...")
    texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()
    emb_raw = model.encode(texts, batch_size=64, show_progress_bar=True,
                           normalize_embeddings=False, convert_to_numpy=True)
    np.save(EMBEDDINGS_RAW_FILE, emb_raw)

raw_norms = np.linalg.norm(emb_raw, axis=1)
print(f"\nNON-normalized embeddings: norms min = {raw_norms.min():.2f}, "
      f"max = {raw_norms.max():.2f}, mean = {raw_norms.mean():.2f}")

q_raw = encode_query(metric_query, normalize=False)
compare_metrics(emb_raw, q_raw, "NOT NORMALIZED")