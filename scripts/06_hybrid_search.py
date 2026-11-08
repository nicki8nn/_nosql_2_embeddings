# Hybrid search: BM25 + vector search, fused with
# Reciprocal Rank Fusion (RRF).
#
#   RRF score(doc) = sum over lists of  1 / (k + rank_in_that_list)
#   rank starts at 1; a doc missing from a list gets nothing from that list.

import os
import re
 
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
 
load_dotenv()
 
INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 10   # take more candidates, so RRF has something to re-rank
SHOW  = 5    # how many results to print
RRF_K = 60   # standard value from the original RRF paper
 
df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)
 
 
# 1. Local BM25 index over titles + abstracts
 
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "from", "at", "as", "is", "are", "was", "were", "be", "been", "this", "that",
    "these", "those", "it", "its", "we", "our", "they", "their", "can", "which",
}
 
 
def tokenize(text: str) -> list[str]:
    """Lowercase, keep words/numbers, drop very common words."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]
 
 
corpus = (df["title"] + " " + df["abstract"]).tolist()
bm25 = BM25Okapi([tokenize(doc) for doc in corpus])
 
# Extra variant for the experiment with the author query:
# the same index, but with the authors field added to every document.
corpus_auth = (df["title"] + " " + df["abstract"] + " " + df["authors"]).tolist()
bm25_auth = BM25Okapi([tokenize(doc) for doc in corpus_auth])
 
 
# 2. Pinecone + specter2_base for vector search
 
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)
model = SentenceTransformer(MODEL_NAME)
 

# 3. Reciprocal Rank Fusion
 
def rrf(ranked_lists: list[list[tuple[int, float]]], k: int = RRF_K) -> list[tuple[int, float]]:
    """
    Only the RANK of a document is used, not its score — so BM25 scores
    (e.g. 12.7) and cosine scores (e.g. 0.86) don't need to be on the same scale.
    """
    fused: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc, _score) in enumerate(ranked, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
 
 
# 4. Search functions — each returns a ranked list of (row_index, score)
 
def bm25_search(query: str, top_k: int = TOP_K, engine: BM25Okapi = bm25) -> list[tuple[int, float]]:
    scores = engine.get_scores(tokenize(query))
    top = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top if scores[i] > 0]  # score 0 = no word matched
 
 
def vector_search(query: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
    q = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    matches = index.query(vector=q.tolist(), top_k=top_k)["matches"]
    return [(int(m["id"].split("_")[1]), float(m["score"])) for m in matches]
 
 
def hybrid_search(query: str, k: int = RRF_K, top_k: int = TOP_K):
    bm = bm25_search(query, top_k)
    vec = vector_search(query, top_k)
    return rrf([bm, vec], k=k), bm, vec
 
 
# Printing
 
def rank_in(results: list[tuple[int, float]], doc: int) -> str:
    for r, (d, _) in enumerate(results, start=1):
        if d == doc:
            return f"#{r}"
    return "—"
 
 
def print_list(title: str, results, score_name: str, bm=None, vec=None) -> None:
    print(f"\n  --- {title} ---")
    if not results:
        print("  (no results)")
    for r, (doc, score) in enumerate(results[:SHOW], start=1):
        row = df.iloc[doc]
        extra = ""
        if bm is not None:
            new = ""
            in_bm_top = any(d == doc for d, _ in bm[:SHOW])
            in_vec_top = any(d == doc for d, _ in vec[:SHOW])
            if not in_bm_top and not in_vec_top:
                new = "  <-- NOT in top-5 of either method"
            extra = f"  (BM25 {rank_in(bm, doc)}, vector {rank_in(vec, doc)}){new}"
        print(f"  {r}. [{score_name} = {score:.4f}] {row['title'][:75]}{extra}")
        print(f"     {row['category']} | {row['year']} | {row['authors'][:60]}")
 
 
def header(text: str) -> None:
    print("\n" + "=" * 100)
    print(text)
    print("=" * 100)
 
 
# 5–6. Demo queries
 
queries = [
    ("exact term",        "BERT fine-tuning"),
    ("author name",       "Yann LeCun convolutional networks"),
    ("paraphrase",        "making computers understand human emotions from text"),
    ("paraphrase (mine)", "protecting user data when training models on many devices"),
]
 
summary_rows = []
 
for label, q in queries:
    header(f"[{label}] \"{q}\"")
    hybrid, bm, vec = hybrid_search(q)
    print_list("BM25", bm, "bm25")
    print_list("Vector (Pinecone)", vec, "cosine")
    print_list(f"Hybrid RRF (k={RRF_K})", hybrid, "rrf", bm, vec)
 
    summary_rows.append({
        "query": label,
        "BM25 #1":   df.iloc[bm[0][0]]["title"][:45] if bm else "—",
        "Vector #1": df.iloc[vec[0][0]]["title"][:45],
        "Hybrid #1": df.iloc[hybrid[0][0]]["title"][:45],
        "overlap BM25∩vector (top-10)": len({d for d, _ in bm} & {d for d, _ in vec}),
    })
 

# Experiment: the author query — are there LeCun papers, and does BM25 see authors?
 
header("EXPERIMENT: author names are not in the BM25 index")
 
lecun = df[df["authors"].str.contains("LeCun", case=False)]
print(f"Papers with 'LeCun' among the authors in the sample: {len(lecun)}")
for _, row in lecun.iterrows():
    print(f"  - {row['title'][:80]} ({row['year']})")
 
author_query = "Yann LeCun convolutional networks"
print_list("BM25 on title + abstract (as in the assignment)", bm25_search(author_query), "bm25")
print_list("BM25 on title + abstract + AUTHORS", bm25_search(author_query, engine=bm25_auth), "bm25")
 
 
# Experiment: k = 60 vs k = 1
 
header("EXPERIMENT: RRF parameter k (60 vs 1)")
 
print("\nContribution of one list to the RRF score, by rank:")
print(f"  {'rank':>4} | {'k=60':>8} | {'k=1':>8}")
for r in [1, 2, 3, 5, 10]:
    print(f"  {r:>4} | {1/(60 + r):>8.4f} | {1/(1 + r):>8.4f}")
 
for label, q in queries:
    _, bm, vec = hybrid_search(q)
    top60 = [d for d, _ in rrf([bm, vec], k=60)[:SHOW]]
    top1  = [d for d, _ in rrf([bm, vec], k=1)[:SHOW]]
    print(f"\n[{label}] \"{q}\"  — same top-{SHOW}: {top60 == top1}")
    for r in range(SHOW):
        t60 = df.iloc[top60[r]]["title"][:42] if r < len(top60) else "—"
        t1  = df.iloc[top1[r]]["title"][:42]  if r < len(top1)  else "—"
        print(f"  {r+1}. k=60: {t60:<42} | k=1: {t1}")
 

# Summary
 
header("SUMMARY: #1 result per method")
print(pd.DataFrame(summary_rows).to_string(index=False))