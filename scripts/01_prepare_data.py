# scripts/01_prepare_data.py
#
# Changes compared to the template from the assignment:
#   1. Sampling: instead of the FIRST 10,000 lines of the file (all from ~2007,
#      mostly physics), we take a random sample over the WHOLE file:
#      8,000 papers from Computer Science (cs.*) + 2,000 from other fields.
#      Reason: the filter search in Part 3 needs recent cs.LG papers
#      ("reinforcement learning, last 5 years") and older ones (before 2015).
#   2. Text cleaning: real line breaks and double spaces are removed.
#   3. Authors: space between surname and initials ("Berger E. L.").

import json
import os
import random

import pandas as pd
from tqdm import tqdm

INPUT_FILE  = "arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE = "data/arxiv_subset.parquet"

N_CS    = 8_000   # papers from cs.* categories
N_OTHER = 2_000   # papers from all other categories
RANDOM_STATE = 42  # fixed seed -> same sample on every run

os.makedirs("data", exist_ok=True)
random.seed(RANDOM_STATE)


def clean_text(text: str) -> str:
    """Remove line breaks and repeated spaces: 'a\\n  b' -> 'a b'."""
    return " ".join(text.split())


def extract_year(paper: dict) -> int:
    """
    Year of the FIRST version = publication date on arXiv.
    Format of 'created': "Mon, 2 Apr 2007 19:18:42 GMT" -> split()[3] = "2007"
    Fallback: update_date "YYYY-MM-DD".
    """
    try:
        versions = paper.get("versions", [])
        if versions:
            return int(versions[0]["created"].split()[3])
    except (IndexError, ValueError, KeyError):
        pass
    return int(paper.get("update_date", "2000-01-01")[:4])


def format_authors(paper: dict) -> str:
    """authors_parsed = [["Surname", "Initials", ""], ...] -> "Surname I., Surname I." """
    parsed = paper.get("authors_parsed", [])
    if parsed:
        parts = []
        for entry in parsed[:10]:  # max 10 authors
            last  = entry[0].strip() if len(entry) > 0 else ""
            first = entry[1].strip() if len(entry) > 1 else ""
            if last:
                parts.append(f"{last} {first}".strip())
        return ", ".join(parts)
    return clean_text(paper.get("authors", ""))


def reservoir_add(reservoir: list, item: dict, seen: int, size: int) -> None:
    """
    Reservoir sampling (Algorithm R): a uniform random sample of `size` items
    from a stream whose length we don't know in advance.
    `seen` = how many items of this group we have seen so far (including this one).
    Memory stays small: we never keep more than `size` papers.
    """
    if len(reservoir) < size:
        reservoir.append(item)
    else:
        j = random.randint(0, seen - 1)
        if j < size:
            reservoir[j] = item


cs_sample, other_sample = [], []
seen_cs, seen_other = 0, 0

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading dataset"):
        line = line.strip()
        if not line:
            continue
        paper = json.loads(line)

        abstract = clean_text(paper.get("abstract", ""))
        title    = clean_text(paper.get("title", ""))
        if not abstract or not title:
            continue

        # "cs.LG cs.AI" -> the first category is the primary one
        primary_category = paper.get("categories", "unknown").split()[0]

        record = {
            "id":       paper["id"],
            "title":    title,
            "abstract": abstract,
            "authors":  format_authors(paper),
            "year":     extract_year(paper),
            "category": primary_category,
        }

        if primary_category.startswith("cs."):
            seen_cs += 1
            reservoir_add(cs_sample, record, seen_cs, N_CS)
        else:
            seen_other += 1
            reservoir_add(other_sample, record, seen_other, N_OTHER)

df = pd.DataFrame(cs_sample + other_sample)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)  # shuffle

print(f"\nPapers in the full file: cs = {seen_cs}, other = {seen_other}")
print(f"Papers in the sample: {len(df)}")
print("\nTop-10 categories:")
print(df["category"].value_counts().head(10))
print("\nPapers per year (last 10 years):")
print(df["year"].value_counts().sort_index().tail(10))
print(f"\nBefore 2015: {(df['year'] < 2015).sum()} papers")
print(f"cs.LG, last 5 years: {((df['category'] == 'cs.LG') & (df['year'] >= df['year'].max() - 4)).sum()} papers")
print("\nExample record:")
print(df.iloc[0].to_dict())

df.to_parquet(OUTPUT_FILE, index=False)
print(f"\nSaved to {OUTPUT_FILE}")