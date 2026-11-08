# Part 4 — Chunking.
# Takes the 30 papers with the longest abstracts, splits them into chunks with
# two strategies, embeds every chunk, uploads each strategy into its own
# Pinecone index and compares search results.
#
#   Fixed-size: N words per chunk, with a small overlap between chunks
#   Semantic:   whole sentences, added one by one until a word limit is reached

import os
import re
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768

N_PAPERS       = 30
FIXED_SIZE     = 50    # words per fixed-size chunk
FIXED_OVERLAP  = 10    # words shared between neighbouring fixed-size chunks
SEMANTIC_MAX   = 60    # max words per semantic chunk
BATCH_SIZE     = 100
TOP_K          = 5

INDEXES = {
    "fixed":    "arxiv-chunks-fixed",
    "semantic": "arxiv-chunks-semantic",
}

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
model = SentenceTransformer(MODEL_NAME)
df = pd.read_parquet("data/arxiv_subset.parquet")


def header(text: str) -> None:
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)


# 1. The 30 papers with the longest abstracts

df["n_words"] = df["abstract"].str.split().str.len()
long_df = df.nlargest(N_PAPERS, "n_words").reset_index(drop=True)

header(f"1. {N_PAPERS} LONGEST ABSTRACTS")
print(f"Abstract length in this group: {long_df['n_words'].min()}–{long_df['n_words'].max()} words "
      f"(whole dataset: median {int(df['n_words'].median())} words)\n")
for _, row in long_df.iterrows():
    print(f"  {row['n_words']:4d} words | {row['category']:8s} | {row['title'][:70]}")



# 2. Two chunking strategies

def fixed_size_chunks(text: str, size: int = FIXED_SIZE, overlap: int = FIXED_OVERLAP) -> list[str]:
    """
    Cut the text every `size` words. Each new chunk starts `size - overlap` words
    after the previous one, so the last `overlap` words are repeated.
    Ignores sentence boundaries -> sentences can be cut in the middle.
    """
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start:start + size]
        chunks.append(" ".join(chunk))
        if start + size >= len(words):   # this chunk already reaches the end
            break
    return chunks


def split_sentences(text: str) -> list[str]:
    """
    Split after . ! ? when the next word starts with a capital letter or digit.
    Simple rule — it can still be fooled by abbreviations like 'Fig. 2'.
    """
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def semantic_chunks(text: str, max_words: int = SEMANTIC_MAX) -> list[str]:
    """
    Add whole sentences to the current chunk until the next sentence would
    exceed `max_words`, then start a new chunk. Sentences are never cut.
    (A single sentence longer than max_words becomes its own chunk.)
    """
    chunks, current, current_len = [], [], 0
    for sentence in split_sentences(text):
        n = len(sentence.split())
        if current and current_len + n > max_words:
            chunks.append(" ".join(current))
            current, current_len = [], 0
        current.append(sentence)
        current_len += n
    if current:
        chunks.append(" ".join(current))
    return chunks


STRATEGIES = {
    "fixed":    fixed_size_chunks,
    "semantic": semantic_chunks,
}

# Build all chunks as rows of a table
chunk_tables = {}
for name, chunk_fn in STRATEGIES.items():
    rows = []
    for _, paper in long_df.iterrows():
        for n, text in enumerate(chunk_fn(paper["abstract"])):
            rows.append({
                "id":        f"{paper['id'].replace('/', '_')}_{name}_{n}",
                "arxiv_id":  paper["id"],
                "title":     paper["title"],
                "text":      text,
                "chunk_num": n,
                "year":      int(paper["year"]),
                "category":  paper["category"],
            })
    chunk_tables[name] = pd.DataFrame(rows)


# 2a. Comparison of the two strategies

header("2. CHUNKING STATISTICS")

ENDS_SENTENCE = re.compile(r"[.!?]\s*$")
STARTS_SENTENCE = re.compile(r"^[A-Z0-9]")

for name, chunks in chunk_tables.items():
    lengths = chunks["text"].str.split().str.len()
    cut_end = (~chunks["text"].str.contains(ENDS_SENTENCE)).sum()
    cut_start = (~chunks["text"].str.contains(STARTS_SENTENCE)).sum()
    print(f"\n{name.upper()}:")
    print(f"  chunks total:            {len(chunks)}  ({len(chunks) / N_PAPERS:.1f} per paper)")
    print(f"  words per chunk:         min {lengths.min()}, mean {lengths.mean():.1f}, max {lengths.max()}")
    print(f"  chunks ending mid-sentence:   {cut_end}")
    print(f"  chunks starting mid-sentence: {cut_start}")

# Shows how ONE paper is split by both strategies
example = long_df.iloc[0]
print(f"\n--- Example: how one abstract is split ---")
print(f"Paper: {example['title']}\n")
for name in STRATEGIES:
    print(f"[{name.upper()}]")
    for _, c in chunk_tables[name][chunk_tables[name]["arxiv_id"] == example["id"]].iterrows():
        print(f"  #{c['chunk_num']} ({len(c['text'].split())} words): "
              f"{c['text'][:90]} ... {c['text'][-60:]}")
    print()


# 2b. Experiment: how overlap changes the number of chunks

header("3. EXPERIMENT: OVERLAP vs NUMBER OF CHUNKS (fixed size = 50 words)")

total_words = long_df["n_words"].sum()
print(f"Total words in the 30 abstracts: {total_words}\n")
print(f"{'overlap':>8} | {'chunks':>6} | {'words stored':>12} | {'duplication':>11}")
for overlap in [0, 10, 25, 40]:
    chunks = [c for text in long_df["abstract"] for c in fixed_size_chunks(text, FIXED_SIZE, overlap)]
    stored = sum(len(c.split()) for c in chunks)
    print(f"{overlap:>8} | {len(chunks):>6} | {stored:>12} | {stored / total_words:>10.2f}x")


# 3–5. Create indexes, embed chunks, upload in batches

header("4. EMBEDDING AND UPLOADING CHUNKS")

existing = pc.list_indexes().names()
indexes = {}

for name, index_name in INDEXES.items():
    if index_name not in existing:
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    indexes[name] = pc.Index(index_name)

for name, chunks in chunk_tables.items():
    # Only the chunk text is embedded (no title), so chunks of the same paper
    vectors = model.encode(
        chunks["text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    index = indexes[name]
    for start in tqdm(range(0, len(chunks), BATCH_SIZE), desc=f"Uploading {name}"):
        batch = []
        for i in range(start, min(start + BATCH_SIZE, len(chunks))):
            c = chunks.iloc[i]
            batch.append({
                "id": c["id"],
                "values": vectors[i].tolist(),
                "metadata": {
                    "arxiv_id":  c["arxiv_id"],
                    "title":     c["title"],
                    "text":      c["text"],
                    "chunk_num": int(c["chunk_num"]),
                    "year":      int(c["year"]),
                    "category":  c["category"],
                },
            })
        index.upsert(vectors=batch)

    for _ in range(30):
        count = index.describe_index_stats()["total_vector_count"]
        if count >= len(chunks):
            break
        time.sleep(2)
    print(f"Index '{INDEXES[name]}': {count} vectors\n")


# 6. Search over chunks

def search_chunks(query: str, strategy: str, top_k: int = TOP_K) -> list:
    q = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    return indexes[strategy].query(vector=q.tolist(), top_k=top_k, include_metadata=True)["matches"]


header("5. SEARCH OVER CHUNKS")

# Queries are generic on purpose: the indexes only hold the 30 papers listed above.
test_queries = [
    "deep neural network for image recognition",
    "privacy and security of machine learning models",
    "theoretical convergence guarantees of an optimization algorithm",
]

for q in test_queries:
    print(f"\nQuery: \"{q}\"")
    for strategy in STRATEGIES:
        print(f"\n  --- {strategy.upper()} ---")
        for rank, m in enumerate(search_chunks(q, strategy), start=1):
            md = m["metadata"]
            print(f"  {rank}. [cosine = {m['score']:.4f}] {md['title'][:70]} (chunk #{int(md['chunk_num'])})")
            print(f"     \"{md['text'][:160]}...\"")