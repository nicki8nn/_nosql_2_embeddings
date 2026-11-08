# Semantic Search over Scientific Papers (arXiv)

"NoSQL and Vector Databases".

I built a semantic search over 10,000 arXiv papers: raw texts → `allenai/specter2_base` embeddings → Pinecone index → semantic search, filtered search, metric comparison, chunking and (optional) hybrid search.

## Repository structure

```
.
├── .env                    # Pinecone API key (not in the repo)
├── .gitignore
├── requirements.txt
├── data/                   # arxiv_subset.parquet (not in the repo)
├── embeddings/             # embeddings.npy (not in the repo)
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_embed.py
│   ├── 03_load_to_pinecone.py
│   ├── 04_search.py
│   ├── 05_chunking.py
│   └── 06_hybrid_search.py
├── output_04.txt           # full outputs of scripts 04–06
├── output_05.txt
├── output_06.txt
└── README.md
```

`.gitignore`:

```
.env
.venv/
data/
embeddings/
arxiv-metadata-oai-snapshot.json
arxiv.zip
__pycache__/
.DS_Store
```

## How to run

I worked on macOS in VS Code, Python 3.11+, with a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/01_prepare_data.py
python scripts/02_embed.py
# ... and so on, in order up to 06
```

I downloaded `arxiv-metadata-oai-snapshot.json` from Kaggle (Cornell-University/arxiv) manually and put it in the project root. It is not committed (~4 GB).

---

## Part 1 — Data preparation and tool selection

### 1.1. Dataset preparation

#### Why I changed the sampling

The template takes the **first 10,000 lines** of the file. Before running it, I checked what those lines are. The file is sorted by arXiv ID, and the ID starts with year and month (`0704.0001` = April 2007):

```
0704.0001 Mon, 2 Apr 2007 19:18:42 GMT
0704.0002 Sat, 31 Mar 2007 02:26:18 GMT
0704.0003 Sun, 1 Apr 2007 20:46:54 GMT
```

Counting over the first 10,000 lines:

```
Years: Counter({'2007': 10000})
Top categories: [('astro-ph', 1838), ('hep-th', 680), ('hep-ph', 671), ('quant-ph', 564), ('gr-qc', 350)]
```

All 10,000 papers are from 2007 and the top-5 categories are all physics, no `cs.*` at all. With this sample, filter A in Part 3 ("RL, cs.LG, last 5 years") would return nothing, and filter B ("before 2015") would return everything.

#### What I changed in `01_prepare_data.py`

1. **A random sample from the whole file**: 8,000 papers from `cs.*` and 2,000 from other categories. The 80/20 split is my choice, not arXiv's real distribution (cs is about 26% there). I chose it because the queries and filters in the assignment are about ML, and the 20% other fields still give the "any category" filter something to work with.
2. **Reservoir sampling** to do this without loading 3.2M papers into memory: the file is read line by line, and every paper has the same chance to end up in the sample. `RANDOM_STATE = 42` makes the sample reproducible.
3. **Text cleaning** with `" ".join(text.split())`, because the template's `.replace("\\n", " ")` didn't remove the real line breaks.
4. **Authors** with a space between surname and initials (`Berger E. L.` instead of `BergerE. L.`).

#### Output

```
Reading dataset: 3233620it [00:48, 66922.67it/s]

Papers in the full file: cs = 846739, other = 2386881
Papers in the sample: 10000

Top-10 categories:
cs.CV    1585
cs.LG    1400
cs.CL     861
cs.AI     499
cs.RO     444
cs.IT     379
cs.CR     313
cs.SE     222
cs.DC     192
cs.DS     190

Papers per year (last 10 years):
2017     307
2018     411
2019     516
2020     608
2021     674
2022     737
2023     860
2024    1075
2025    1365
2026    1556

Before 2015: 1403 papers
cs.LG, last 5 years: 986 papers

Saved to data/arxiv_subset.parquet
```

Now both filters have data: 986 cs.LG papers from 2022–2026 and 1,403 papers before 2015.

### 1.2. Tool selection

#### 1. Pinecone vs Qdrant vs Chroma

| | Pinecone | Qdrant | Chroma |
|---|---|---|---|
| Deployment | Managed cloud only (serverless), no self-hosting | Self-hosted (Docker/Kubernetes) or Qdrant Cloud | Inside the Python process, self-hosted server, or Chroma Cloud |
| License | Proprietary | Apache 2.0 | Apache 2.0 |
| Performance | Scales to very large volumes without administration | Rust core, fast search and payload filtering | Easiest start, less proven at large scale |

For this homework Pinecone was the easiest: I didn't have to run any server, the index was created with one call from the script, and the free tier is enough for 10,000 vectors. What I don't like about it is that it is proprietary and cloud-only — my data can live only in Pinecone's cloud (Starter even only in AWS us-east-1), and if I wanted to leave, I would have to rebuild everything somewhere else.

That's why, for a real client project with personal data, I would choose **Qdrant**: it is open source, I can run it on our own server in the EU (important for GDPR), and it has strong metadata filtering. **Chroma** I would use for quick prototypes and experiments in a notebook, because it runs directly inside the Python script, without a separate server — but I wouldn't build a large production system on it. So my choice is: Pinecone when I want zero infrastructure, Qdrant when I need control over the data, Chroma for a quick local start.

#### 2. Why specter2_base and not all-MiniLM-L6-v2

`all-MiniLM-L6-v2` is a general-purpose model trained on everyday sentence pairs, with 384 dimensions and an input limit of about 256 tokens. For scientific abstracts that is a weak point — the vocabulary is specialized and many abstracts are longer than that.

`specter2_base` was built for scientific papers. From the [model card](https://huggingface.co/allenai/specter2_base):

- it was trained on "over 6M triplets of scientific paper citations" — a paper, a paper it cites and a paper it doesn't cite. So it learned that papers connected by citations should be close to each other, which is exactly what a paper search needs;
- it is fine-tuned from `allenai/scibert`, with an input limit of 512 tokens and 768 dimensions;
- the task formats it was trained for: Classification, Regression, Proximity (Retrieval), Adhoc Search;
- the input is title + `[SEP]` + abstract, which is why `02_embed.py` builds the text this way.

In the card's own evaluation, SPECTER2 Base also beats BM-25 on citation recommendation (MDCR, MAP 38.0 vs 33.7).

One thing I noticed: the card says that "for general embedding purposes" `allenai/specter2` should be used — the base model with the proximity adapter. The assignment uses the base model without an adapter, so we don't use the best version for retrieval.

#### 3. Recommended similarity metric

The model card does **not** name a similarity metric. What it shows is how the embedding is built: from the first token, `output.last_hidden_state[:, 0, :]`. The original SPECTER paper trained the model with a triplet loss based on L2 distance.

This matters because in Pinecone the metric is fixed when the index is created and can't be changed later. Since I normalize the embeddings, cosine, dot product and L2 give the same ranking (I show this in Part 3 and derive it in Part 6). I chose `cosine`, because its values are easy to read (−1 to 1) and it doesn't depend on vector length even if I forgot to normalize somewhere.

Another detail: loading `specter2_base` through `SentenceTransformer` creates **mean pooling** (average of all tokens), not the `[CLS]` token from the model card — the warning "Creating a new one with mean pooling" appears in my output. I kept the assignment's approach, but I'm aware of the difference.

#### Pinecone Starter limits

The assignment says the free tier has one index. The current Starter plan allows up to 5 serverless indexes, 2 GB storage, 2M write units and 1M read units per month, AWS us-east-1 only. So the three indexes I needed (`arxiv-papers`, `arxiv-chunks-fixed`, `arxiv-chunks-semantic`) fit.

### 1.3. Embeddings

`02_embed.py` builds `title + " [SEP] " + abstract` for every paper, encodes all of them in batches of 64 with `normalize_embeddings=True`, and saves the vectors to `embeddings/embeddings.npy`. Row *i* in the `.npy` file is row *i* in the parquet file, and the later scripts rely on this order.

#### Output

```
Loaded papers: 10000
No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.
Model device: mps:0
Max input length (tokens): 512
Batches: 100%|██████████| 157/157 [01:16<00:00,  2.07it/s]

Texts encoded:       10000
Embedding dimension: 768
Norm of the 1st embedding: 1.000000
Data type: float32

Saved to embeddings/embeddings.npy
```

It ran on my Mac's GPU (`mps`) in 1 min 16 s.

#### Why is cosine similarity equal to dot product for normalized embeddings?

$$\cos(a, b) = \frac{a \cdot b}{\|a\| \, \|b\|}$$

After normalization every vector has length 1, so the denominator is $1 \cdot 1 = 1$ and $\cos(a, b) = a \cdot b$. It's not "almost the same" — it's the same number. My output confirms the vectors are unit length (norm = 1.000000). In Part 3 the cosine and dot scores came out identical (0.8615, 0.8521, …). For a large index this means dot product is the cheaper way to get the same result, without computing lengths.

---

## Part 2 — Loading data into Pinecone

`03_load_to_pinecone.py` creates the serverless index `arxiv-papers` (768 dimensions, `cosine`, AWS us-east-1) and uploads the vectors in batches of 200. Each vector has the id `paper_<i>`, where *i* is the row in the parquet file — so after a search I can always get the full abstract from there. Metadata: `arxiv_id`, `title`, `abstract` (first 500 characters), `authors` (first 200), `year`, `category`.

#### Output

```
Creating index 'arxiv-papers' (dim=768, metric=cosine)...
Index is ready.
Papers: 10000, embeddings: (10000, 768)
Uploading to Pinecone: 100%|██████████| 50/50 [01:25<00:00,  1.70s/it]

Total vectors in index 'arxiv-papers': 10000
{'dimension': 768,
 'index_fullness': 0.0,
 'namespaces': {'': {'vector_count': 10000}},
 'total_vector_count': 10000}
```

Pinecone updates the statistics with a short delay, so the script waits until the count reaches 10,000 before printing it.

---

## Part 3 — Search queries

The full output of `04_search.py` is in the [Appendix](#appendix--full-script-outputs) and in [`output_04.txt`](output_04.txt).

### 3.1. Semantic search

**"teaching machines to recognize objects in pictures"**

| # | cosine | Title | Category | Year |
|---|---|---|---|---|
| 1 | 0.8614 | Vision Transformers: State of the Art and Research Challenges | cs.CV | 2022 |
| 2 | 0.8521 | Reproduction Report on "Learn to Pay Attention" | cs.LG | 2018 |
| 3 | 0.8504 | Machine-learnable Sets | cs.LG | 2026 |
| 4 | 0.8500 | Morphological Image Analysis and Feature Extraction for … Defect Detection and Classification Models | cs.CV | 2023 |
| 5 | 0.8498 | A Novel Biologically Mechanism-Based Visual Cognition Model … | cs.CV | 2016 |

The query has no technical terms, and still 4 of 5 results are about visual recognition. Only #3 is weak — its abstract talks about things being "easily recognized" and "learned", but not about images.

**"protecting user data when training models on many devices"** — my own query, federated learning without the word "federated".

| # | cosine | Title | Category | Year |
|---|---|---|---|---|
| 1 | 0.8809 | Pretraining Data Exposure in Large Language Models: A Survey of Membership Inference … | cs.CL | 2026 |
| 2 | 0.8789 | Enhancing Privacy in ControlNet and Stable Diffusion via Split Learning | cs.LG | 2024 |
| 3 | 0.8745 | Rethinking Membership Inference Attacks Against Transfer Learning | cs.CR | 2025 |
| 4 | 0.8725 | Securing Graph Neural Networks in MLaaS … | cs.CR | 2023 |
| 5 | 0.8723 | Synthetic ID Card Image Generation for Improving Presentation Attack Detection | cs.CV | 2022 |

The model got the "privacy" part (and split learning at #2 is close to federated learning), but no result is really about federated learning. I checked whether such papers exist in my sample:

```
df[df['abstract'].str.contains('federated', case=False)]
→ 113 rows
```

There are **113**, and none made it into the top-5. So this is a real miss. My explanation: "protecting user data" pulls the query toward the privacy/security papers, and "many devices" is too weak to pull it toward federated learning. It may also matter that the base model learned from paper title + abstract, not from short questions — the card has a separate `adhoc_query` adapter for that.

I also noticed that all scores are high (0.85–0.88) and very close, even for weak results. So for this model the ranking matters, not the absolute score. I think this comes from mean pooling in the base model: all scientific texts end up in a narrow area of the vector space.

### 3.2. Search with filters

Query: **"reinforcement learning agent learns a policy from rewards"**

| | No filter | A: cs.LG, year ≥ 2022 | B: year < 2015 |
|---|---|---|---|
| Scores | 0.9032–0.9141 | 0.8876–0.9068 | 0.8208–0.8258 |
| Years | 2019–2023 | 2022–2024 | 2005–2014 |
| Categories | cs.LG, cs.RO | cs.LG | cs.AI, cs.CE, cs.LG, cs.RO, cs.SE |
| Really about RL | 5 / 5 | 5 / 5 | ~2 / 5 |

**Filter A:** two papers stayed from the unfiltered top-5 (they already matched the conditions), the cs.RO and older papers were removed, and three new cs.LG papers came in. The scores barely dropped, because my sample has many recent RL papers.

**Filter B:** the scores dropped to ~0.82 and only about two results relate to RL (planning under uncertainty, an agent learning from experience). The rest are crowd teaching, genetic networks, and a withdrawn paper whose abstract is just "This is article is taken out". Before 2015 I have only 1,403 papers and very few about RL — but Pinecone still returns the 5 nearest ones. Vector search has no "nothing found"; in a real system I would add a minimum score.

### 3.3. Comparing similarity metrics

Same query as in 3.1, calculated locally with numpy.

**Normalized embeddings** (all norms = 1.0000):

| # | cosine | dot | L2 | Title |
|---|---|---|---|---|
| 1 | 0.8615 | 0.8615 | 0.5264 | Vision Transformers … |
| 2 | 0.8521 | 0.8521 | 0.5438 | Reproduction Report on "Learn to Pay Attention" |
| 3 | 0.8504 | 0.8504 | 0.5469 | Machine-learnable Sets |
| 4 | 0.8499 | 0.8499 | 0.5478 | Morphological Image Analysis … |
| 5 | 0.8498 | 0.8498 | 0.5482 | A Novel Biologically Mechanism-Based Visual Cognition Model … |

```
cosine vs dot: True
cosine vs L2:  True
Pinecone top-5 == local cosine top-5: True
```

**Experiment — non-normalized embeddings.** I encoded all papers again with `normalize_embeddings=False`. The vector lengths were then **13.42–20.07** (mean 16.91).

| # | cosine | dot | L2 |
|---|---|---|---|
| 1 | Vision Transformers … | **withdrawn paper** ("This is article is taken out") | Vision Transformers … |
| 2 | "Learn to Pay Attention" | "Learn to Pay Attention" | Machine-learnable Sets |
| 3 | Machine-learnable Sets | **Generalised 3D Morton and Hilbert Orderings** (one-sentence abstract) | Morphological Image Analysis … |
| 4 | Morphological Image Analysis … | **Tree structured sparse coding on cubes** (one-sentence abstract) | "Learn to Pay Attention" |
| 5 | Visual Cognition Model … | **Overview of AI Grading of Physics Olympiad Exams** | Visual Cognition Model … |

```
cosine vs dot: False
cosine vs L2:  False
```

### Answers to the Part 3 questions

**Do the top-5 for cosine and dot product match, and why?**
Yes, same papers, same order, same scores. My vectors are normalized, so cosine = dot product (see 1.3).

**Do the results differ for L2, and why?**
With normalized vectors, no: same top-5, same order, only sorted the other way (smaller = closer). For unit vectors $\|a-b\| = \sqrt{2 - 2\cos(a,b)}$, so a higher cosine always means a smaller distance. I checked it on my #1 result: $\sqrt{2 - 2 \cdot 0.8615} = 0.5264$, exactly the L2 value in my output. The derivation is in Part 6.

**What would happen if the embeddings were not normalized?**
My experiment shows it:
- **cosine** didn't change at all, because it divides by the lengths anyway;
- **dot product** broke: 4 of 5 results were new and irrelevant, and #1 was the withdrawn paper (dot = 314.5). The "winners" have one-sentence abstracts. My interpretation is that with mean pooling a very short text gets a longer vector, and dot product rewards length instead of meaning;
- **L2** kept the same 5 papers here but changed their order ("Learn to Pay Attention" fell from #2 to #4), because it now also measures the difference in length.

So with dot product or L2 the embeddings have to be normalized. I use cosine in Pinecone, which is safe either way.

**Data quality note:** the withdrawn paper appeared twice (filter B and non-normalized dot). Next time I would drop papers with very short or "withdrawn" abstracts in `01_prepare_data.py`.

---

## Part 4 — Chunking

The full output of `05_chunking.py` is in the [Appendix](#appendix--full-script-outputs) and in [`output_05.txt`](output_05.txt).

| | Fixed-size | Semantic |
|---|---|---|
| My rule | 50 words per chunk, 10 words overlap | Whole sentences until the next one would go over 60 words |
| Sentence boundaries | Ignored | Kept |

I embedded only the chunk text, without the title. With the title in every chunk, all chunks of one paper would end up almost the same vector, and I wanted to find the specific part of a text.

The 30 longest abstracts have **299–363 words** (median of the whole dataset: 166) and come from very different fields — supply chain, astrophysics, graph theory, LLM jailbreaking, stock prediction. The longest is around the model's 512-token limit, not far above it, so for my data chunking is mostly practice for really long documents.

### Statistics

| | Fixed-size | Semantic |
|---|---|---|
| Chunks | 241 (8.0 per paper) | 204 (6.8 per paper) |
| Words per chunk | min 19, mean 47.1, max 50 | min 9, mean 45.2, max 79 |
| Ending mid-sentence | **206 of 241 (85%)** | **0** |
| Starting mid-sentence | **189 of 241 (78%)** | **0** |

Semantic chunks vary more: 79 words when a single sentence was already longer than 60, 9 words when the last sentence was short.

The same abstract ("Traceability as an integral part of supply chain logistics management") split both ways:
- **fixed** chunk #1 starts with *"about the product location, its processing history, raw materials, etc at each point…"* — no subject, just a fragment;
- **semantic** chunks follow the structure of the abstract: #0 starts with *"Purpose:"*, #5 with *"Design/methodology/approach:"*, #6 with *"Findings:"*.

### Search over chunks

| Query | Fixed | Semantic |
|---|---|---|
| "deep neural network for image recognition" | **#1 = stock market LSTM** chunk (0.8506), then the image/GAN paper | **#1–2 = image/GAN paper** (0.8355, 0.8254), then the LSTM paper |
| "privacy and security of machine learning models" | Pb-Hash, Multi Stage Screening, pathogen attack, LSTM ×2 | Pb-Hash, pathogen attack, Multi Stage Screening, LSTM, Bayesian active learning |
| "theoretical convergence guarantees of an optimization algorithm" | Multi Stage Screening, Approximation Algorithms, Sleeping Model ×2, Bayesian AL | Multi Stage Screening, Approximation Algorithms ×2, Sleeping Model ×2 |

- For **image recognition** the fixed index ranked a stock price chunk first, just because it contains "Neural Network". The semantic index ranked the only real image paper first.
- For **privacy** there is no relevant paper among the 30, and both indexes still returned 5 results — the same effect as filter B in Part 3.
- For **convergence** both found theory papers, but the semantic results were more coherent, while the fixed ones included fragments like *"per hire) at the end of the final stage…"*.
- The fixed index often had the higher top score but the wrong paper, so scores from two different indexes can't be compared.
- One paper often took several places in the top-5. In a real system I would group by `arxiv_id`.

### Overlap experiment

Fixed size 50 words, 30 abstracts, 9,230 words:

| Overlap | Step | Chunks | Words stored | Duplication |
|---|---|---|---|---|
| 0 | 50 | 207 | 9,230 | 1.00x |
| 10 | 40 | 241 | 11,340 | 1.23x |
| 25 | 25 | 359 | 17,455 | 1.89x |
| 40 | 10 | 818 | 40,750 | 4.41x |

### Answers to the Part 4 questions

**Which strategy gives more meaningful chunks?**
Semantic. Every chunk is complete sentences (0 cut against 206), it followed the logical parts of structured abstracts, and it found the right paper for the image query where fixed-size didn't. The downside: chunk length isn't controlled (9–79 words), and my simple regex sentence splitter could be fooled by abbreviations like "Fig. 2" or "et al.".

**Are there cut sentences, and how does it affect the embeddings?**
Yes, in fixed-size chunking 85% of chunks end and 78% start mid-sentence. A cut sentence loses its meaning — *"about the product location, its processing history…"* has no subject, so its embedding only represents a few keywords, and the idea of the sentence is split between two vectors. The 10-word overlap gives the next chunk a bit of context, but the cut just moves further.

**How does overlap affect the number of chunks and the coverage?**
The step is *size − overlap*, so more overlap means more chunks: 207 → 241 → 359 → 818. All variants cover 100% of the text; overlap only increases the chance that an idea crossing a border appears complete in one chunk. The price is duplication (4.41x at overlap 40), more embeddings, more vectors in Pinecone and more near-duplicates in the results. I would keep overlap at 10–20% of the chunk size.

---

## Part 5 — Hybrid search (optional)

`06_hybrid_search.py` builds a local BM25 index over title + abstract, takes the top-10 from BM25 and the top-10 from Pinecone, and combines them with Reciprocal Rank Fusion:

$$\text{RRF}(d) = \sum_{\text{lists}} \frac{1}{k + \text{rank}(d)}, \qquad k = 60$$

RRF uses only the position, not the score — BM25 scores (around 19.6) and cosine (around 0.87) can't be added. Besides the three required queries, I added my federated learning query from Part 3. The full output is in the [Appendix](#appendix--full-script-outputs) and in [`output_06.txt`](output_06.txt).

### Comparison: 4 queries × 3 methods

| Query | BM25 | Vector | Hybrid RRF | Shared in top-10 |
|---|---|---|---|---|
| "BERT fine-tuning" | ✅ 5/5 about BERT | ❌ 0/5 — RL, radio signals, curiosity | ⚠️ 3/5 — alternates both lists | 0 |
| "Yann LeCun convolutional networks" | ⚠️ CNN papers, no LeCun | ⚠️ CNN papers, no LeCun | ⚠️ CNN papers, ranked by agreement | 3 |
| "making computers understand human emotions from text" | ⚠️ ~3/5 — "AGI for Education", text simplification | ✅ 5/5 — emotion recognition, empathy, valence/arousal | ✅ #1 found by both; 4/5 relevant | 1 |
| "protecting user data when training models on many devices" | ✅ 2 federated learning papers (#1, #5) | ❌ privacy papers, no federated learning | ✅ federated learning paper at #2 | 1 |

### Answers to the Part 5 questions

**Which method gave the best result, and why?**
It depended on the query.
- **"BERT fine-tuning"** — BM25, clearly. Vector search found nothing about BERT. Two technical words don't give the model much "meaning" to work with, while for BM25 the rare word "bert" is a strong signal.
- **Emotions** — vector search. It found papers about valence, arousal and empathy that don't share words with the query; BM25 matched generic words like "human" and "text".
- **My federated query** — BM25 surprised me: without the word "federated" it still found two federated learning papers, because they use exactly "user", "data", "training", "devices".
- **Hybrid** was never the best and never the worst. It brought federated learning into the top-5 where vector search had none, but for BERT it was worse than BM25 alone, because RRF can't tell which list is the good one.

**Are there documents in the hybrid top-5 that are not in the top-5 of either method?**
Not in my run — every hybrid result was in at least one top-5. But RRF re-ranked by agreement. In the author query, "Deep Unitary Convolutional Neural Networks" (vector #3, BM25 #6) moved up to hybrid #2, above the BM25 #1:

- in both lists, at #6 and #3: 1/66 + 1/63 = 0.0310
- only in BM25, at #1: 1/61 = 0.0164

So a paper at #6 in both lists (2/66 = 0.0303) could appear in the hybrid top-5 without being in either top-5. In my queries the lists overlapped too little for that (0, 3, 1 and 1 shared papers). With no overlap at all (BERT), RRF just alternates the lists with equal scores (0.0164, 0.0164, 0.0161, …).

**How does k affect the result (k = 60 vs k = 1)?**

| Rank | k = 60 | k = 1 |
|---|---|---|
| 1 | 0.0164 | 0.5000 |
| 2 | 0.0161 | 0.3333 |
| 5 | 0.0154 | 0.1667 |
| 10 | 0.0143 | 0.0909 |

With k = 60 the values are almost flat, so what counts is being in both lists. With k = 1, rank 1 is worth 5.5× more than rank 10, so being #1 in one list counts more. In my results the top-5 was the same for 3 of 4 queries (too little overlap to make a difference). Only the author query changed: with k = 1 "Convexified CNNs" (BM25 #1, 0.500) went from #4 to #2, above "Deep Unitary CNNs" (1/7 + 1/4 = 0.393), and "Upright adjustment" dropped out. I think k = 60 makes more sense, because one method's #1 can be completely wrong (vector #1 for "BERT fine-tuning"), while agreement between two different methods is more reliable.

### Experiment: author names

BM25 is built on title + abstract only, as the assignment says, so author names aren't in it. Neither method found any LeCun paper. But my sample has two:

```
Papers with 'LeCun' among the authors in the sample: 2
  - Revisiting Feature Prediction for Learning Visual Representations from Video (2024)
  - Compact and Optimal Deep Learning with Recurrent Parameter Generators (2021)
```

When I added `authors` to the BM25 documents, both came out at **#1 and #2**. So BM25 is good at names only if the names are indexed. Vector search can't find them either way. I would handle author search as a metadata filter or a separate keyword field.

---

## Part 6 — Analysis and conclusions

### 1. Semantic search vs keyword search

In my results, semantic search worked when the query **described an idea in other words** than the papers use: "recognize objects in pictures" found Vision Transformers and image classification (Part 3), "human emotions from text" found papers about valence, arousal and empathy (Part 5). BM25 can't do that, because it only counts words that are actually in the document.

Keyword search worked better when the query had **the exact words the papers use**. For "BERT fine-tuning" BM25 found 5 BERT papers and vector search 0. For federated learning, vector search didn't put any of the 113 papers into the top-5, while BM25 found two even without the word "federated" in the query. Author names worked only with BM25 over the authors field (both LeCun papers at #1 and #2).

My rule: **if the user knows the exact words — a term, an acronym, a name, an ID — keyword search is better; if they describe a concept in their own words, semantic search is better. For a normal search box, where I don't know which kind of query will come, I would use hybrid search**, because in my tests it was never the worst. Year, category or author should be exact filters, not part of the query.

### 2. Chunk size

**Too small (10–15 words):** a chunk is one sentence or less. In Part 4 I saw fragments like *"about the product location, its processing history, raw materials, etc"* — without context the embedding only represents a few keywords. Search gets noisy (the stock market chunk that matched "neural network"), the same paper repeats in the top-k, and the number of vectors grows: already 241 chunks for 30 abstracts at 50 words, and it would be over 600 at 10–15 words.

**Too large (500+ words):** my model's limit is 512 tokens (`Max input length (tokens): 512` in my `02_embed.py` output), and 500 words of scientific text are usually more than 512 tokens. Everything after the limit is **cut off without any warning** and can never be found. And with mean pooling, a long text about several topics becomes one "average" vector that is a bit similar to everything and very similar to nothing. The result is also a big block of text where the user still has to look for the relevant part.

I don't think there's one optimal size. It depends on the model's token limit, on the queries (short factual questions need small precise chunks, topic search can use larger ones) and on the documents (cutting along sentences, paragraphs or sections is better than cutting by word count). For my data, whole abstracts (median 166 words) were already a good unit, and for longer texts the ~45-word semantic chunks worked better than the fixed ones. I would start with a few hundred tokens and a small overlap, then adjust with real queries.

### 3. A mismatched metric: euclidean index with normalized vectors

The ranking would be **exactly the same** as with cosine; only the scores would look different.

$$\|a - b\|^2 = (a - b)\cdot(a - b) = \|a\|^2 - 2\,a\cdot b + \|b\|^2$$

For normalized vectors $\|a\| = \|b\| = 1$ and $a \cdot b = \cos(a, b)$, so:

$$\|a - b\|^2 = 2 - 2\cos(a,b) \qquad\Rightarrow\qquad \|a - b\| = \sqrt{2 - 2\cos(a,b)}$$

$\sqrt{2 - 2x}$ decreases as $x$ grows, so a higher cosine always means a smaller distance — sorting by L2 ascending gives the same list as sorting by cosine descending. Identical vectors give distance 0, orthogonal ones $\sqrt{2} \approx 1.41$, opposite ones 2. My numbers confirm it: $\sqrt{2 - 2 \cdot 0.8615} = 0.5264$, and the top-5 by cosine, dot and L2 was identical (Part 3.3).

What could still go wrong:
- **reading the scores** — with euclidean, smaller is better, so a cosine-style threshold like "score > 0.8" would keep the wrong results;
- **vectors that aren't normalized** — then this equivalence is gone. In my experiment L2 changed the order of the top-5 and dot product put a withdrawn paper at #1. That's why I used normalized embeddings with the cosine metric.

### 4. Pinecone Starter limits

Starter gives up to 5 serverless indexes, 2 GB storage, 2M write units and 1M read units per month, only AWS us-east-1, and 1 project. I also read that inactive Starter indexes are paused after about 3 weeks.

I didn't hit any limit. I used 3 of 5 indexes, and my vectors are tiny: 10,000 × 768 × 4 bytes ≈ 31 MB, plus metadata that I kept short on purpose. The assignment says only one index is allowed — if that were true, I would have used namespaces inside one index. What could affect me is the inactivity pause if the mentor runs the scripts weeks later, and the US-only region for data that must stay in the EU.

**With 10 million papers** I would have to change almost everything:
- **Storage:** 10M × 768 × 4 bytes ≈ **31 GB** of vectors, 15× the free limit. I would need a paid plan, or make the vectors smaller (a 384-dimensional model, float16, quantization), or move to self-hosted Qdrant/Milvus.
- **Embeddings:** my Mac did about 130 papers per second (10,000 in 76 s), so 10M would take about **21 hours**. I would run it in parallel on cloud GPUs and save the results in parts, not in one `.npy` in memory.
- **Upload:** 50,000 requests of 200 vectors, or Pinecone's bulk import, with retries for failed batches.
- **Metadata:** full abstracts in a normal database or object storage, fetched by id after the search — like my parquet file, just bigger.
- **BM25:** `rank_bm25` keeps everything in memory and scores every document for every query. For 10M I would use Elasticsearch/OpenSearch or Pinecone's sparse vectors.
- **Data preparation:** my reservoir sampling already reads the file as a stream, so it would scale. For 10M I would take the whole file, but drop withdrawn and empty papers first.

---

## Appendix — full script outputs
Outputs of `01`–`03` are shown in full in Parts 1–2 above. The outputs of `04`–`06` are long, so they are collapsed here (click to expand) and also saved as `output_04.txt`, `output_05.txt`, `output_06.txt`.

<details>
<summary><b>04_search.py</b></summary>

```
No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.

==========================================================================================
1. SEMANTIC SEARCH (no filters)
==========================================================================================

Query: "teaching machines to recognize objects in pictures"

1. [cosine = 0.8614] Vision Transformers: State of the Art and Research Challenges
   cs.CV | 2022 | arXiv:2207.03041
   Transformers have achieved great success in natural language processing. Due to the powerful capability of self-attention mechanism in transformers, researchers develop the vision transformers for a v...
2. [cosine = 0.8521] Reproduction Report on "Learn to Pay Attention"
   cs.LG | 2018 | arXiv:1812.04650
   We have successfully implemented the "Learn to Pay Attention" model of attention mechanism in convolutional neural networks, and have replicated the results of the original paper in the categories of ...
3. [cosine = 0.8504] Machine-learnable Sets
   cs.LG | 2026 | arXiv:2606.28947
   In this study we present a formal definition of large discrete sets having, informally, three properties: theirelements are easily recognized, easily generated, and the latter tasks are easily learne...
4. [cosine = 0.8500] Morphological Image Analysis and Feature Extraction for Reasoning with AI-based Defect Detection and Classification Models
   cs.CV | 2023 | arXiv:2307.11643
   As the use of artificial intelligent (AI) models becomes more prevalent in industries such as engineering and manufacturing, it is essential that these models provide transparent reasoning behind thei...
5. [cosine = 0.8498] A Novel Biologically Mechanism-Based Visual Cognition Model--Automatic Extraction of Semantics, Formation of Integrated Concepts and Re-selection Features for Ambiguity
   cs.CV | 2016 | arXiv:1603.07886
   Integration between biology and information science benefits both fields. Many related models have been proposed, such as computational visual cognition models, computational motor control models, int...

Query: "protecting user data when training models on many devices"

1. [cosine = 0.8809] Pretraining Data Exposure in Large Language Models: A Survey of Membership Inference, Data Contamination, and Security Implications
   cs.CL | 2026 | arXiv:2605.26133
   Large Language Models (LLMs) have become the predominant paradigm in NLP, advancing both research and industry.As model sizes and pretraining data grow, concerns about Pretraining Data Exposure (PDE)...
2. [cosine = 0.8789] Enhancing Privacy in ControlNet and Stable Diffusion via Split Learning
   cs.LG | 2024 | arXiv:2409.08503
   With the emerging trend of large generative models, ControlNet is introduced to enable users to fine-tune pre-trained models with their own data for various use cases. A natural question arises: how c...
3. [cosine = 0.8745] Rethinking Membership Inference Attacks Against Transfer Learning
   cs.CR | 2025 | arXiv:2501.11577
   Transfer learning, successful in knowledge translation across related tasks, faces a substantial privacy threatfrom membership inference attacks (MIAs). These attacks, despite posing significant risk...
4. [cosine = 0.8725] Securing Graph Neural Networks in MLaaS: A Comprehensive Realization of Query-based IntegrityVerification
   cs.CR | 2023 | arXiv:2312.07870
   The deployment of Graph Neural Networks (GNNs) within Machine Learning as a Service (MLaaS) has opened up new attack surfaces and an escalation in security concerns regarding model-centric attacks. Th...
5. [cosine = 0.8723] Synthetic ID Card Image Generation for Improving Presentation Attack Detection
   cs.CV | 2022 | arXiv:2211.00098
   Currently, it is ever more common to access online services for activities which formerly required physical attendance. From banking operations to visa applications, a significant number of processes ...

==========================================================================================
2. SEARCH WITH FILTERS
==========================================================================================

Query: "reinforcement learning agent learns a policy from rewards"

--- Baseline: no filter ---

1. [cosine = 0.9141] Guiding Robot Exploration in Reinforcement Learning via Automated Planning
   cs.RO | 2020 | arXiv:2004.11456
   Reinforcement learning (RL) enables an agent to learn from trial-and-error experiences toward achieving long-term goals; automated planning aims to compute plans for accomplishing tasks using action k...
2. [cosine = 0.9068] Backward Learning for Goal-Conditioned Policies
   cs.LG | 2023 | arXiv:2312.05044
   Can we learn policies in reinforcement learning without rewards? Can we learn a policy just by trying to reach a goal state? We answer these questions positively by proposing a multi-step procedure th...
3. [cosine = 0.9052] Macro Action Reinforcement Learning with Sequence Disentanglement using Variational Autoencoder
   cs.LG | 2019 | arXiv:1903.09366
   One problem in the application of reinforcement learning to real-world problems is the curse of dimensionality on the action space. Macro actions, a sequence of primitive actions, have been studied to...
4. [cosine = 0.9038] Recursive Reasoning Graph for Multi-Agent Reinforcement Learning
   cs.LG | 2022 | arXiv:2203.02844
   Multi-agent reinforcement learning (MARL) provides an efficient way for simultaneously learning policies for multiple agents interacting with each other. However, in scenarios requiring complex intera...
5. [cosine = 0.9032] Multi-Agent Reinforcement Learning for Unmanned Aerial Vehicle Coordination by Multi-Critic Policy Gradient Optimization
   cs.RO | 2020 | arXiv:2012.15472
   Recent technological progress in the development of Unmanned Aerial Vehicles (UAVs) together with decreasing acquisition costs make the application of drone fleets attractive for a wide variety of tas...

--- A: cs.LG, year >= 2022 ---
    filter = {'year': {'$gte': 2022}, 'category': {'$eq': 'cs.LG'}}

1. [cosine = 0.9068] Backward Learning for Goal-Conditioned Policies
   cs.LG | 2023 | arXiv:2312.05044
   Can we learn policies in reinforcement learning without rewards? Can we learn a policy just by trying to reach a goal state? We answer these questions positively by proposing a multi-step procedure th...
2. [cosine = 0.9038] Recursive Reasoning Graph for Multi-Agent Reinforcement Learning
   cs.LG | 2022 | arXiv:2203.02844
   Multi-agent reinforcement learning (MARL) provides an efficient way for simultaneously learning policies for multiple agents interacting with each other. However, in scenarios requiring complex intera...
3. [cosine = 0.9008] Weighted Maximum Entropy Inverse Reinforcement Learning
   cs.LG | 2022 | arXiv:2208.09611
   We study inverse reinforcement learning (IRL) and imitation learning (IM), the problems of recovering a reward or policy function from expert's demonstrated trajectories. We propose a new way to impro...
4. [cosine = 0.8918] Sample Efficient Reinforcement Learning by Automatically Learning to Compose Subtasks
   cs.LG | 2024 | arXiv:2401.14226
   Improving sample efficiency is central to Reinforcement Learning (RL), especially in environments where the rewards are sparse. Some recent approaches have proposed to specify reward functions as manu...
5. [cosine = 0.8876] Which Experiences Are Influential for Your Agent? Policy Iteration with Turn-over Dropout
   cs.LG | 2023 | arXiv:2301.11168
   In reinforcement learning (RL) with experience replay, experiences stored in a replay buffer influence the RL agent's performance. Information about the influence is valuable for various purposes, inc...

--- B: year < 2015, any category ---
    filter = {'year': {'$lt': 2015}}

1. [cosine = 0.8258] Near-Optimally Teaching the Crowd to Classify
   cs.LG | 2014 | arXiv:1402.2092
   How should we present training examples to learners to teach them classification rules? This is a natural problem when training workers for crowdsourcing labeling tasks, and is also motivated by chall...
2. [cosine = 0.8241] Cost Sensitive Reachability Heuristics for Handling State Uncertainty
   cs.AI | 2012 | arXiv:1207.1350
   While POMDPs provide a general platform for non-deterministic conditional planning under a variety of quality metrics they have limited scalability. On the other hand, non-deterministic conditional pl...
3. [cosine = 0.8215] From General Systems to Soft Systems to Soft Computing: Applications for Large and Complex Real World Systems
   cs.SE | 2005 | arXiv:cs/0511018
   This is article is taken out....
4. [cosine = 0.8214] Evolutionary algorithms in genetic regulatory networks model
   cs.CE | 2012 | arXiv:1205.1986
   Genetic Regulatory Networks (GRNs) plays a vital role in the understanding of complex biological processes. Modeling GRNs is significantly important in order to reveal fundamental cellular processes, ...
5. [cosine = 0.8208] Experience-Based Planning with Sparse Roadmap Spanners
   cs.RO | 2014 | arXiv:1410.1950
   We present an experienced-based planning framework called Thunder that learns to reduce computation time required to solve high-dimensional planning problems in varying environments. The approach is e...

--- Filter comparison ---
A (cs.LG, recent): scores 0.8876–0.9068, years 2022–2024, categories ['cs.LG']
B (before 2015)  : scores 0.8208–0.8258, years 2005–2014, categories ['cs.AI', 'cs.CE', 'cs.LG', 'cs.RO', 'cs.SE']

==========================================================================================
3. SIMILARITY METRICS ON LOCAL EMBEDDINGS
==========================================================================================
Loaded embeddings: (10000, 768), norms min/max = 1.0000 / 1.0000

Query: "teaching machines to recognize objects in pictures"

--- NORMALIZED: top-5 by cosine ---
1. [cosine = 0.8615] Vision Transformers: State of the Art and Research Challenges
   cs.CV | 2022 | arXiv:2207.03041
   Transformers have achieved great success in natural language processing. Due to the powerful capability of self-attention mechanism in transformers, researchers develop the vision transformers for a v...
2. [cosine = 0.8521] Reproduction Report on "Learn to Pay Attention"
   cs.LG | 2018 | arXiv:1812.04650
   We have successfully implemented the "Learn to Pay Attention" model of attention mechanism in convolutional neural networks, and have replicated the results of the original paper in the categories of ...
3. [cosine = 0.8504] Machine-learnable Sets
   cs.LG | 2026 | arXiv:2606.28947
   In this study we present a formal definition of large discrete sets having, informally, three properties: theirelements are easily recognized, easily generated, and the latter tasks are easily learne...
4. [cosine = 0.8499] Morphological Image Analysis and Feature Extraction for Reasoning with AI-based Defect Detection and Classification Models
   cs.CV | 2023 | arXiv:2307.11643
   As the use of artificial intelligent (AI) models becomes more prevalent in industries such as engineering and manufacturing, it is essential that these models provide transparent reasoning behind thei...
5. [cosine = 0.8498] A Novel Biologically Mechanism-Based Visual Cognition Model--Automatic Extraction of Semantics, Formation of Integrated Concepts and Re-selection Features for Ambiguity
   cs.CV | 2016 | arXiv:1603.07886
   Integration between biology and information science benefits both fields. Many related models have been proposed, such as computational visual cognition models, computational motor control models, int...

--- NORMALIZED: top-5 by dot ---
1. [dot = 0.8615] Vision Transformers: State of the Art and Research Challenges
   cs.CV | 2022 | arXiv:2207.03041
   Transformers have achieved great success in natural language processing. Due to the powerful capability of self-attention mechanism in transformers, researchers develop the vision transformers for a v...
2. [dot = 0.8521] Reproduction Report on "Learn to Pay Attention"
   cs.LG | 2018 | arXiv:1812.04650
   We have successfully implemented the "Learn to Pay Attention" model of attention mechanism in convolutional neural networks, and have replicated the results of the original paper in the categories of ...
3. [dot = 0.8504] Machine-learnable Sets
   cs.LG | 2026 | arXiv:2606.28947
   In this study we present a formal definition of large discrete sets having, informally, three properties: theirelements are easily recognized, easily generated, and the latter tasks are easily learne...
4. [dot = 0.8499] Morphological Image Analysis and Feature Extraction for Reasoning with AI-based Defect Detectionand Classification Models
   cs.CV | 2023 | arXiv:2307.11643
   As the use of artificial intelligent (AI) models becomes more prevalent in industries such as engineering and manufacturing, it is essential that these models provide transparent reasoning behind thei...
5. [dot = 0.8498] A Novel Biologically Mechanism-Based Visual Cognition Model--Automatic Extraction of Semantics, Formation of Integrated Concepts and Re-selection Features for Ambiguity
   cs.CV | 2016 | arXiv:1603.07886
   Integration between biology and information science benefits both fields. Many related models have been proposed, such as computational visual cognition models, computational motor control models, int...

--- NORMALIZED: top-5 by L2 ---
1. [L2 = 0.5264] Vision Transformers: State of the Art and Research Challenges
   cs.CV | 2022 | arXiv:2207.03041
   Transformers have achieved great success in natural language processing. Due to the powerful capability of self-attention mechanism in transformers, researchers develop the vision transformers for a v...
2. [L2 = 0.5438] Reproduction Report on "Learn to Pay Attention"
   cs.LG | 2018 | arXiv:1812.04650
   We have successfully implemented the "Learn to Pay Attention" model of attention mechanism in convolutional neural networks, and have replicated the results of the original paper in the categories of ...
3. [L2 = 0.5469] Machine-learnable Sets
   cs.LG | 2026 | arXiv:2606.28947
   In this study we present a formal definition of large discrete sets having, informally, three properties: theirelements are easily recognized, easily generated, and the latter tasks are easily learne...
4. [L2 = 0.5478] Morphological Image Analysis and Feature Extraction for Reasoning with AI-based Defect Detection and Classification Models
   cs.CV | 2023 | arXiv:2307.11643
   As the use of artificial intelligent (AI) models becomes more prevalent in industries such as engineering and manufacturing, it is essential that these models provide transparent reasoning behind thei...
5. [L2 = 0.5482] A Novel Biologically Mechanism-Based Visual Cognition Model--Automatic Extraction of Semantics, Formation of Integrated Concepts and Re-selection Features for Ambiguity
   cs.CV | 2016 | arXiv:1603.07886
   Integration between biology and information science benefits both fields. Many related models have been proposed, such as computational visual cognition models, computational motor control models, int...

>>> NORMALIZED: same top-5 (incl. order)?
    cosine vs dot: True
    cosine vs L2:  True

>>> Pinecone top-5 == local cosine top-5: True

Encoding papers WITHOUT normalization (one time, cached afterwards)...
Batches: 100%|███████████████████████████████████████████████████████████████████| 157/157 [01:15<00:00,  2.09it/s]

NON-normalized embeddings: norms min = 13.42, max = 20.07, mean = 16.91

--- NOT NORMALIZED: top-5 by cosine ---
1. [cosine = 0.8615] Vision Transformers: State of the Art and Research Challenges
   cs.CV | 2022 | arXiv:2207.03041
   Transformers have achieved great success in natural language processing. Due to the powerful capability of self-attention mechanism in transformers, researchers develop the vision transformers for a v...
2. [cosine = 0.8521] Reproduction Report on "Learn to Pay Attention"
   cs.LG | 2018 | arXiv:1812.04650
   We have successfully implemented the "Learn to Pay Attention" model of attention mechanism in convolutional neural networks, and have replicated the results of the original paper in the categories of ...
3. [cosine = 0.8504] Machine-learnable Sets
   cs.LG | 2026 | arXiv:2606.28947
   In this study we present a formal definition of large discrete sets having, informally, three properties: theirelements are easily recognized, easily generated, and the latter tasks are easily learne...
4. [cosine = 0.8499] Morphological Image Analysis and Feature Extraction for Reasoning with AI-based Defect Detection and Classification Models
   cs.CV | 2023 | arXiv:2307.11643
   As the use of artificial intelligent (AI) models becomes more prevalent in industries such as engineering and manufacturing, it is essential that these models provide transparent reasoning behind thei...
5. [cosine = 0.8498] A Novel Biologically Mechanism-Based Visual Cognition Model--Automatic Extraction of Semantics, Formation of Integrated Concepts and Re-selection Features for Ambiguity
   cs.CV | 2016 | arXiv:1603.07886
   Integration between biology and information science benefits both fields. Many related models have been proposed, such as computational visual cognition models, computational motor control models, int...

--- NOT NORMALIZED: top-5 by dot ---
1. [dot = 314.4849] From General Systems to Soft Systems to Soft Computing: Applications for Large and Complex Real World Systems
   cs.SE | 2005 | arXiv:cs/0511018
   This is article is taken out....
2. [dot = 306.1501] Reproduction Report on "Learn to Pay Attention"
   cs.LG | 2018 | arXiv:1812.04650
   We have successfully implemented the "Learn to Pay Attention" model of attention mechanism in convolutional neural networks, and have replicated the results of the original paper in the categories of ...
3. [dot = 300.9658] Generalised 3D Morton and Hilbert Orderings
   cs.CG | 2023 | arXiv:2309.15199
   This document describes algorithms for generating general Morton and Hilbert orderings for three-dimensional data volumes....
4. [dot = 298.2936] Tree structured sparse coding on cubes
   cs.IT | 2013 | arXiv:1301.3590
   A brief description of tree structured sparse coding on the binary cube....
5. [dot = 297.8528] Overview of AI Grading of Physics Olympiad Exams
   cs.AI | 2025 | arXiv:2505.02121
   Automatically grading the diverse range of question types in high school physics problem is a challenge that requires automated grading techniques from different fields. We report the findings of a Sy...

--- NOT NORMALIZED: top-5 by L2 ---
1. [L2 = 9.9285] Vision Transformers: State of the Art and Research Challenges
   cs.CV | 2022 | arXiv:2207.03041
   Transformers have achieved great success in natural language processing. Due to the powerful capability of self-attention mechanism in transformers, researchers develop the vision transformers for a v...
2. [L2 = 10.2374] Machine-learnable Sets
   cs.LG | 2026 | arXiv:2606.28947
   In this study we present a formal definition of large discrete sets having, informally, three properties: theirelements are easily recognized, easily generated, and the latter tasks are easily learne...
3. [L2 = 10.3515] Morphological Image Analysis and Feature Extraction for Reasoning with AI-based Defect Detectionand Classification Models
   cs.CV | 2023 | arXiv:2307.11643
   As the use of artificial intelligent (AI) models becomes more prevalent in industries such as engineering and manufacturing, it is essential that these models provide transparent reasoning behind thei...
4. [L2 = 10.3565] Reproduction Report on "Learn to Pay Attention"
   cs.LG | 2018 | arXiv:1812.04650
   We have successfully implemented the "Learn to Pay Attention" model of attention mechanism in convolutional neural networks, and have replicated the results of the original paper in the categories of ...
5. [L2 = 10.3625] A Novel Biologically Mechanism-Based Visual Cognition Model--Automatic Extraction of Semantics, Formation of Integrated Concepts and Re-selection Features for Ambiguity
   cs.CV | 2016 | arXiv:1603.07886
   Integration between biology and information science benefits both fields. Many related models have been proposed, such as computational visual cognition models, computational motor control models, int...

>>> NOT NORMALIZED: same top-5 (incl. order)?
    cosine vs dot: False
    cosine vs L2:  False
```

</details>

<details>
<summary><b>05_chunking.py</b></summary>

```
No sentence-transformers model found with name allenai/specter2_base. Creating a new one with mean pooling.

==========================================================================================
1. 30 LONGEST ABSTRACTS
==========================================================================================
Abstract length in this group: 299–363 words (whole dataset: median 166 words)

   363 words | cs.CY    | Traceability as an integral part of supply chain logistics management:
   328 words | cs.DS    | Engineering Faster Sorters for Small Sets of Items
   319 words | astro-ph.GA | Analysis of Far-UV Data of Central Stars of Planetary Nebulae: Occurre
   314 words | math-ph  | The split and approximate split property in 2D systems: stability and 
   312 words | cs.GT    | On a Bounded Budget Network Creation Game
   311 words | math.CO  | H-colouring bipartite graphs
   310 words | cs.CR    | Depth, Not Breadth: Best-of-N Jailbreaking Beyond Surface Noise
   310 words | cs.DC    | Sleeping Model: Local and Dynamic Algorithms
   309 words | cs.LG    | A Bayesian Active Learning Approach to Comparative Judgement
   308 words | cs.DC    | When the KV Cache Is Slower Than the GPU: Scaling Remote KV Tiers for 
   308 words | cs.CE    | Compressible aerodynamics and rigid-body support motion in post-flutte
   307 words | astro-ph | Warm gas accretion onto the Galaxy
   304 words | cs.DS    | HashGraph -- Scalable Hash Tables Using A Sparse Graph Data Structure
   304 words | astro-ph.SR | Molecules in the transition disk orbiting T Cha
   304 words | cs.SI    | Approximation Algorithms for Coordinating Ad Campaigns on Social Netwo
   304 words | cs.LG    | Pb-Hash: Partitioned b-bit Hashing
   303 words | cs.DS    | Conflict Resolution in Multiple Access Channels Supporting Simultaneou
   303 words | cs.AI    | Winning by Peeking: Unenforced Budgets and Test-Set Selection Inflate 
   302 words | astro-ph.EP | Is ozone a reliable proxy for molecular oxygen? III. The impact of CH$
   302 words | astro-ph.EP | Resolving the hot dust around HD69830 and eta Corvi with MIDI and VISI
   301 words | cs.DM    | Tree-layout based graph classes: proper chordal graphs
   301 words | cs.LG    | Extreme Volatility Prediction in Stock Market: When GameStop meets Lon
   301 words | cs.DM    | On the Expansion of Group-Based Lifts
   301 words | cs.CV    | The applicability of transperceptual and deep learning approaches to t
   301 words | cs.LG    | Multi Stage Screening: Enforcing Fairness and Maximizing Efficiency in
   301 words | cs.SE    | A Large-Scale Empirical Study of COVID-19 Contact Tracing Mobile App R
   300 words | cs.CR    | A Wolf in Sheep's Clothing: Spreading Deadly Pathogens Under the Disgu
   300 words | astro-ph.SR | Numerical studies on the link between radioisotopic signatures on Eart
   300 words | cs.GT    | Discrete Preference Games in Heterogeneous Social Networks: Subverted 
   299 words | cs.IR    | Don't read, just look: Main content extraction from web pages using vi

==========================================================================================
2. CHUNKING STATISTICS
==========================================================================================

FIXED:
  chunks total:            241  (8.0 per paper)
  words per chunk:         min 19, mean 47.1, max 50
  chunks ending mid-sentence:   206
  chunks starting mid-sentence: 189

SEMANTIC:
  chunks total:            204  (6.8 per paper)
  words per chunk:         min 9, mean 45.2, max 79
  chunks ending mid-sentence:   0
  chunks starting mid-sentence: 0

--- Example: how one abstract is split ---
Paper: Traceability as an integral part of supply chain logistics management: an analytical review

[FIXED]
  #0 (50 words): Purpose: Supply chain has become very complex today. There are multiple stakeholders at va ... product location, its processing history, raw materials, etc
  #1 (50 words): about the product location, its processing history, raw materials, etc at each point so as ...  across the business supply chain from source to destination
  #2 (50 words): effectiveness across the business supply chain from source to destination to achieve their ... ream at each point. All the stakeholders in the supply chain
  #3 (50 words): at each point. All the stakeholders in the supply chain have different objectives to imple ... l these actors are also different and are to be shared among
  #4 (50 words): actors are also different and are to be shared among all to run the supply chain smoothly. ... t traceability solutions. The purpose of this research is to
  #5 (50 words): current traceability solutions. The purpose of this research is to conduct thorough study  ... business supply chain. The study will be helpful for further
  #6 (50 words): business supply chain. The study will be helpful for further researchers to propose a trac ... The scope and the objectives of the research is set based on
  #7 (50 words): and the objectives of the research is set based on which the papers are filtered as per th ... e industries since a long for their supply chain management.
  #8 (43 words): the industries since a long for their supply chain management. After proper analysis of th ... ive the proper recommendations and perspectives of our work.

[SEMANTIC]
  #0 (60 words): Purpose: Supply chain has become very complex today. There are multiple stakeholders at va ... , etc at each point so as to control the production process.
  #1 (35 words): Companies need to develop global and effective strategies to sustain in the competitive ma ... from source to destination to achieve their strategic goals.
  #2 (57 words): Traceability has become one of the integrated parts of the supply chain logistics manageme ... facturers, distributers or wholesalers have their own role).
  #3 (51 words): The information generated and needed from all these actors are also different and are to b ... challenge to be addressed in current traceability solutions.
  #4 (57 words): The purpose of this research is to conduct thorough study of the literatures available on  ... rsto propose a traceability meta-model to address the gaps.
  #5 (54 words): Design/methodology/approach: The relevant literatures with keywords supply chain managemen ...  screened papers are done and the recommendations are given.
  #6 (49 words): Findings: Traceability solution is popular among the industries since a long for their sup ... ive the proper recommendations and perspectives of our work.


==========================================================================================
3. EXPERIMENT: OVERLAP vs NUMBER OF CHUNKS (fixed size = 50 words)
==========================================================================================
Total words in the 30 abstracts: 9230

 overlap | chunks | words stored | duplication
       0 |    207 |         9230 |       1.00x
      10 |    241 |        11340 |       1.23x
      25 |    359 |        17455 |       1.89x
      40 |    818 |        40750 |       4.41x

==========================================================================================
4. EMBEDDING AND UPLOADING CHUNKS
==========================================================================================
Creating index 'arxiv-chunks-fixed'...
Creating index 'arxiv-chunks-semantic'...
Batches: 100%|███████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  5.11it/s]
Uploading fixed: 100%|███████████████████████████████████████████████████████████████| 3/3 [00:04<00:00,  1.42s/it]
Index 'arxiv-chunks-fixed': 241 vectors

Batches: 100%|███████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  5.94it/s]
Uploading semantic: 100%|████████████████████████████████████████████████████████████| 3/3 [00:03<00:00,  1.26s/it]
Index 'arxiv-chunks-semantic': 204 vectors


==========================================================================================
5. SEARCH OVER CHUNKS
==========================================================================================

Query: "deep neural network for image recognition"

  --- FIXED ---
  1. [cosine = 0.8506] Extreme Volatility Prediction in Stock Market: When GameStop meets Lon (chunk #2)
     "ETF fund) from February 2002 to February 2021. For strategy 1, we develop a Long Short-term Memory (LSTM) Neural Network to predict stock prices recurrently wit..."
  2. [cosine = 0.8422] The applicability of transperceptual and deep learning approaches to t (chunk #2)
     "architectures mimicking the native ones. The proposed method uses both traditional imagery, and sound generated from each image as a method of rapidly comparing..."
  3. [cosine = 0.8227] The applicability of transperceptual and deep learning approaches to t (chunk #5)
     "image is divided into 16 squares to maintain the limit of 64 $\times$ 64 pixels required by the GAN. We reconstruct the 2D stacks of artificially generated data..."
  4. [cosine = 0.8171] The applicability of transperceptual and deep learning approaches to t (chunk #3)
     "generative adversarial network (GAN) on the 2D image stacks. The impact of the training set of images on thesimilarity of the artificial to the original datase..."
  5. [cosine = 0.8103] Extreme Volatility Prediction in Stock Market: When GameStop meets Lon (chunk #7)
     "our study sheds more light into the field of extreme volatility predictions based on LSTMs to outperform buying and holding strategy...."

  --- SEMANTIC ---
  1. [cosine = 0.8355] The applicability of transperceptual and deep learning approaches to t (chunk #2)
     "The proposed method uses both traditional imagery, and sound generated from each image as a method of rapidly comparing and contrasting the porosity and pore si..."
  2. [cosine = 0.8254] The applicability of transperceptual and deep learning approaches to t (chunk #3)
     "The impact of the training set of images on the similarity of the artificial to the original dataset was assessed by analyzing two samples...."
  3. [cosine = 0.8109] Extreme Volatility Prediction in Stock Market: When GameStop meets Lon (chunk #5)
     "We hope that our study sheds more light into the field of extreme volatility predictions based on LSTMs to outperform buying and holding strategy...."
  4. [cosine = 0.8022] Extreme Volatility Prediction in Stock Market: When GameStop meets Lon (chunk #2)
     "For our strategy 2, we develop an LSTM autoencoder network specifically designed to trade only on consolidation breakouts after predicting anomalies in the stoc..."
  5. [cosine = 0.8010] Extreme Volatility Prediction in Stock Market: When GameStop meets Lon (chunk #1)
     "We investigate GME stock for its volatility and compare it to SPY as a benchmark (since it is a less volatile ETF fund) from February 2002 to February 2021. For..."

Query: "privacy and security of machine learning models"

  --- FIXED ---
  1. [cosine = 0.8530] Pb-Hash: Partitioned b-bit Hashing (chunk #5)
     "be a good addition to the family of hashing methods/applications and benefit industrial practitioners. We verify the effectiveness of Pb-Hash in machine learnin..."
  2. [cosine = 0.8079] Multi Stage Screening: Enforcing Fairness and Maximizing Efficiency in (chunk #6)
     "per hire) at the end of the final stage. We also present examples showing that the solution space is non-convex, which motivate our exact and (FPTAS) approximat..."
  3. [cosine = 0.8027] A Wolf in Sheep's Clothing: Spreading Deadly Pathogens Under the Disgu (chunk #6)
     "control over the negative pressure and show that the \textit{forged} pressure can be varied within 12 - 33 Pa. Our attack is also capable of attacking multiple ..."
  4. [cosine = 0.7952] Extreme Volatility Prediction in Stock Market: When GameStop meets Lon (chunk #6)
     "success rate and \$2.10 from strategy 2 with 48% success rate. Overall, buying and holding outperforms all deep-learning assisted prediction models in our study..."
  5. [cosine = 0.7946] Extreme Volatility Prediction in Stock Market: When GameStop meets Lon (chunk #7)
     "our study sheds more light into the field of extreme volatility predictions based on LSTMs to outperform buying and holding strategy...."

  --- SEMANTIC ---
  1. [cosine = 0.8445] Pb-Hash: Partitioned b-bit Hashing (chunk #4)
     "We expect Pb-Hash would be a good addition to the family of hashing methods/applications and benefit industrial practitioners. We verify the effectiveness of Pb..."
  2. [cosine = 0.8051] A Wolf in Sheep's Clothing: Spreading Deadly Pathogens Under the Disgu (chunk #6)
     "Moreover, we demonstrate our attack at a real-world NPR located in an anonymous bioresearch facility, which is FDA approved and follows CDC guidelines. We also ..."
  3. [cosine = 0.8036] Multi Stage Screening: Enforcing Fairness and Maximizing Efficiency in (chunk #6)
     "We also present examples showing that the solution space is non-convex, which motivate our exact and (FPTAS)approximation algorithms for maximizing the linear ..."
  4. [cosine = 0.7943] Extreme Volatility Prediction in Stock Market: When GameStop meets Lon (chunk #5)
     "We hope that our study sheds more light into the field of extreme volatility predictions based on LSTMs to outperform buying and holding strategy...."
  5. [cosine = 0.7924] A Bayesian Active Learning Approach to Comparative Judgement (chunk #4)
     "Furthermore, we demonstrate how the entire approach may provide transparency by providing the user insights into how it is making its decisions and, at the same..."

Query: "theoretical convergence guarantees of an optimization algorithm"

  --- FIXED ---
  1. [cosine = 0.8379] Multi Stage Screening: Enforcing Fairness and Maximizing Efficiency in (chunk #6)
     "per hire) at the end of the final stage. We also present examples showing that the solution space is non-convex, which motivate our exact and (FPTAS) approximat..."
  2. [cosine = 0.8291] Approximation Algorithms for Coordinating Ad Campaigns on Social Netwo (chunk #5)
     "practical; however, we show that specifically under the Independent Cascade model, LP rounding and Reverse Reachability techniques can be combined to obtain a 1..."
  3. [cosine = 0.8227] Sleeping Model: Local and Dynamic Algorithms (chunk #7)
     "is the neighborhood independence of the input graph. Graphs with bounded neighborhood independence are well studied with several results in recent years for sev..."
  4. [cosine = 0.8149] Sleeping Model: Local and Dynamic Algorithms (chunk #5)
     "a higher awake complexity for lower clock round complexity. We note that the awake time incurred is not thatsignificant. We offer dynamic algorithms in the sle..."
  5. [cosine = 0.8056] A Bayesian Active Learning Approach to Comparative Judgement (chunk #3)
     "that should be compared next in an informative manner, but some existing methods are known to have created their own bias within results inflating the reliabili..."

  --- SEMANTIC ---
  1. [cosine = 0.8345] Multi Stage Screening: Enforcing Fairness and Maximizing Efficiency in (chunk #6)
     "We also present examples showing that the solution space is non-convex, which motivate our exact and (FPTAS)approximation algorithms for maximizing the linear ..."
  2. [cosine = 0.8312] Approximation Algorithms for Coordinating Ad Campaigns on Social Netwo (chunk #4)
     "The 1-1/e approximation algorithm for matroid-constrained problem is far from practical; however, we show that specifically under the Independent Cascade model,..."
  3. [cosine = 0.8208] Approximation Algorithms for Coordinating Ad Campaigns on Social Netwo (chunk #2)
     "Our main result is to show that under a broad class of influence models, the problem can be reduced to maximizing a submodular function subject to two matroid c..."
  4. [cosine = 0.8201] Sleeping Model: Local and Dynamic Algorithms (chunk #6)
     "Specifically, our algorithm requires $O(K)$ awake time where $K$ is the neighborhood independence of the input graph. Graphs with bounded neighborhood independe..."
  5. [cosine = 0.8182] Sleeping Model: Local and Dynamic Algorithms (chunk #3)
     "The clock round complexity of that algorithm is $O(\Delta^2)$. In this work we offer three algorithms for the bf O-LOCAL class of problems with a trade off betw..."
```

</details>

<details>
<summary><b>06_hybrid_search.py (optional)</b></summary>

```

====================================================================================================
[exact term] "BERT fine-tuning"
====================================================================================================

  --- BM25 ---
  1. [bm25 = 19.6443] Utilizing BERT Intermediate Layers for Aspect Based Sentiment Analysis and 
     cs.CL | 2020 | Song Youwei, Wang Jiahai, Liang Zhiwei, Liu Zhiyue, Jiang Ta
  2. [bm25 = 14.7991] It's not Greek to mBERT: Inducing Word-Level Translations from Multilingual
     cs.CL | 2020 | Gonen Hila, Ravfogel Shauli, Elazar Yanai, Goldberg Yoav
  3. [bm25 = 14.0840] Annotated Job Ads with Named Entity Recognition
     cs.CL | 2023 | Stollenwerk Felix, Fastlund Niklas, Nyqvist Anna, Öhman Joey
  4. [bm25 = 14.0223] Robust Transfer Learning with Pretrained Language Models through Adapters
     cs.CL | 2021 | Han Wenjuan, Pang Bo, Wu Yingnian
  5. [bm25 = 13.8981] Enabling Natural Zero-Shot Prompting on Encoder Models via Statement-Tuning
     cs.CL | 2024 | Elshabrawy Ahmed, Huang Yongxin, Gurevych Iryna, Aji Alham F

  --- Vector (Pinecone) ---
  1. [cosine = 0.8693] Deep Reinforcement Learning Radio Control and Signal Detection with KeRLym,
     cs.LG | 2016 | O'Shea Timothy J., Clancy T. Charles
  2. [cosine = 0.8679] Boredom-driven curious learning by Homeo-Heterostatic Value Gradients
     cs.AI | 2018 | Yu Yen, Chang Acer Y. C., Kanai Ryota
  3. [cosine = 0.8626] A Survey of Slow Thinking-based Reasoning LLMs using Reinforced Learning an
     cs.AI | 2025 | Pan Qianjun, Ji Wenkai, Ding Yuyang, Li Junsong, Chen Shilia
  4. [cosine = 0.8605] Beyond Fixed Morphologies: Learning Graph Policies with Trust Region Compen
     cs.LG | 2025 | Gallien Thomas
  5. [cosine = 0.8585] PolySwarm: A Multi-Agent Large Language Model Framework for Prediction Mark
     cs.AI | 2026 | Barot Rajat M., Borkhatariya Arjun S.

  --- Hybrid RRF (k=60) ---
  1. [rrf = 0.0164] Utilizing BERT Intermediate Layers for Aspect Based Sentiment Analysis and   (BM25 #1, vector —)
     cs.CL | 2020 | Song Youwei, Wang Jiahai, Liang Zhiwei, Liu Zhiyue, Jiang Ta
  2. [rrf = 0.0164] Deep Reinforcement Learning Radio Control and Signal Detection with KeRLym,  (BM25 —, vector #1)
     cs.LG | 2016 | O'Shea Timothy J., Clancy T. Charles
  3. [rrf = 0.0161] It's not Greek to mBERT: Inducing Word-Level Translations from Multilingual  (BM25 #2, vector —)
     cs.CL | 2020 | Gonen Hila, Ravfogel Shauli, Elazar Yanai, Goldberg Yoav
  4. [rrf = 0.0161] Boredom-driven curious learning by Homeo-Heterostatic Value Gradients  (BM25 —, vector #2)
     cs.AI | 2018 | Yu Yen, Chang Acer Y. C., Kanai Ryota
  5. [rrf = 0.0159] Annotated Job Ads with Named Entity Recognition  (BM25 #3, vector —)
     cs.CL | 2023 | Stollenwerk Felix, Fastlund Niklas, Nyqvist Anna, Öhman Joey

====================================================================================================
[author name] "Yann LeCun convolutional networks"
====================================================================================================

  --- BM25 ---
  1. [bm25 = 11.1225] Convexified Convolutional Neural Networks
     cs.LG | 2016 | Zhang Yuchen, Liang Percy, Wainwright Martin J.
  2. [bm25 = 10.2732] Re-presenting a Story by Emotional Factors using Sentimental Analysis Metho
     cs.CL | 2016 | Jo Hwiyeol, Moon Yohan, Kim Jong In, Ryu Jeong
  3. [bm25 = 9.5839] A Hybrid Method for Training Convolutional Neural Networks
     cs.NE | 2020 | Lopes Vasco, Fazendeiro Paulo
  4. [bm25 = 9.5257] R-FCN: Object Detection via Region-based Fully Convolutional Networks
     cs.CV | 2016 | Dai Jifeng, Li Yi, He Kaiming, Sun Jian
  5. [bm25 = 9.3683] Norm-based Generalization Bounds for Compositionally Sparse Neural Networks
     cs.LG | 2023 | Galanti Tomer, Xu Mengjia, Galanti Liane, Poggio Tomaso

  --- Vector (Pinecone) ---
  1. [cosine = 0.8845] KAConvNet: Kolmogorov-Arnold Convolutional Networks for Vision Recognition
     cs.CV | 2026 | Liu Zhaoxiang, Ma Zhicheng, Zhao Kaikai, Wang Kai, Lian Shig
  2. [cosine = 0.8771] The High-Dimensional Geometry of Binary Neural Networks
     cs.LG | 2017 | Anderson Alexander G., Berg Cory P.
  3. [cosine = 0.8747] Deep Unitary Convolutional Neural Networks
     cs.LG | 2021 | Chang Hao-Yuan, Wang Kang L.
  4. [cosine = 0.8743] Reproduction Report on "Learn to Pay Attention"
     cs.LG | 2018 | Shugliashvili Levan, Soselia Davit, Amashukeli Shota, Koberi
  5. [cosine = 0.8717] Upright adjustment with graph convolutional networks
     cs.CV | 2024 | Jung Raehyuk, Cho Sungmin, Kwon Junseok

  --- Hybrid RRF (k=60) ---
  1. [rrf = 0.0311] KAConvNet: Kolmogorov-Arnold Convolutional Networks for Vision Recognition  (BM25 #8, vector #1)
     cs.CV | 2026 | Liu Zhaoxiang, Ma Zhicheng, Zhao Kaikai, Wang Kai, Lian Shig
  2. [rrf = 0.0310] Deep Unitary Convolutional Neural Networks  (BM25 #6, vector #3)
     cs.LG | 2021 | Chang Hao-Yuan, Wang Kang L.
  3. [rrf = 0.0299] Upright adjustment with graph convolutional networks  (BM25 #9, vector #5)
     cs.CV | 2024 | Jung Raehyuk, Cho Sungmin, Kwon Junseok
  4. [rrf = 0.0164] Convexified Convolutional Neural Networks  (BM25 #1, vector —)
     cs.LG | 2016 | Zhang Yuchen, Liang Percy, Wainwright Martin J.
  5. [rrf = 0.0161] Re-presenting a Story by Emotional Factors using Sentimental Analysis Metho  (BM25 #2, vector —)
     cs.CL | 2016 | Jo Hwiyeol, Moon Yohan, Kim Jong In, Ryu Jeong

====================================================================================================
[paraphrase] "making computers understand human emotions from text"
====================================================================================================

  --- BM25 ---
  1. [bm25 = 13.6638] EmoGenie: Benchmarking and Evaluating Expressed and Evoked Emotion in T2I G
     cs.CV | 2026 | Xing Zixuan, Gao Lancheng, Li Shengyan, Duan Huiyu, Cai Zhao
  2. [bm25 = 13.1114] Performance Evaluation of Supervised Machine Learning Techniques for Effici
     cs.IR | 2019 | Asghar Muhammad Zubair, Subhan Fazli, Imran Muhammad, Kundi 
  3. [bm25 = 12.9892] AGI: Artificial General Intelligence for Education
     cs.AI | 2023 | Latif Ehsan, Mai Gengchen, Nyaaba Matthew, Wu Xuansheng, Liu
  4. [bm25 = 12.8882] Do Text Simplification Systems Preserve Meaning? A Human Evaluation via Rea
     cs.CL | 2023 | Agrawal Sweta, Carpuat Marine
  5. [bm25 = 12.7351] Evaluation of Generative Models for Emotional 3D Animation Generation in VR
     cs.HC | 2025 | Chhatre Kiran, Guarese Renan, Matviienko Andrii, Peters Chri

  --- Vector (Pinecone) ---
  1. [cosine = 0.8839] Performance Evaluation of Supervised Machine Learning Techniques for Effici
     cs.IR | 2019 | Asghar Muhammad Zubair, Subhan Fazli, Imran Muhammad, Kundi 
  2. [cosine = 0.8815] "We care": Improving Code Mixed Speech Emotion Recognition in Customer-Care
     cs.AI | 2023 | Abhishek N V S, Bhattacharyya Pushpak
  3. [cosine = 0.8762] Emotion Recognition and Generation: A Comprehensive Review of Face, Speech,
     cs.LG | 2025 | Mobbs Rebecca, Makris Dimitrios, Argyriou Vasileios
  4. [cosine = 0.8745] Datasets for Valence and Arousal Inference: A Survey
     cs.HC | 2025 | Schneider Helen, Pavlitska Svetlana, Gremmelmaier Helen, Zöl
  5. [cosine = 0.8739] EmpathBERT: A BERT-based Framework for Demographic-aware Empathy Prediction
     cs.LG | 2021 | Guda Bhanu Prakash Reddy, Garimella Aparna, Chhaya Niyati

  --- Hybrid RRF (k=60) ---
  1. [rrf = 0.0325] Performance Evaluation of Supervised Machine Learning Techniques for Effici  (BM25 #2, vector #1)
     cs.IR | 2019 | Asghar Muhammad Zubair, Subhan Fazli, Imran Muhammad, Kundi 
  2. [rrf = 0.0164] EmoGenie: Benchmarking and Evaluating Expressed and Evoked Emotion in T2I G  (BM25 #1, vector —)
     cs.CV | 2026 | Xing Zixuan, Gao Lancheng, Li Shengyan, Duan Huiyu, Cai Zhao
  3. [rrf = 0.0161] "We care": Improving Code Mixed Speech Emotion Recognition in Customer-Care  (BM25 —, vector #2)
     cs.AI | 2023 | Abhishek N V S, Bhattacharyya Pushpak
  4. [rrf = 0.0159] AGI: Artificial General Intelligence for Education  (BM25 #3, vector —)
     cs.AI | 2023 | Latif Ehsan, Mai Gengchen, Nyaaba Matthew, Wu Xuansheng, Liu
  5. [rrf = 0.0159] Emotion Recognition and Generation: A Comprehensive Review of Face, Speech,  (BM25 —, vector #3)
     cs.LG | 2025 | Mobbs Rebecca, Makris Dimitrios, Argyriou Vasileios

====================================================================================================
[paraphrase (mine)] "protecting user data when training models on many devices"
====================================================================================================

  --- BM25 ---
  1. [bm25 = 13.9603] Federated Unbiased Learning to Rank
     cs.IR | 2021 | Li Chang, Ouyang Hua
  2. [bm25 = 13.6827] Random Deployment of Data Collectors for Serving Randomly-Located Sensors
     cs.NI | 2011 | Kwon Taesoo, Cioffi John. M.
  3. [bm25 = 13.2187] GPU backed Data Mining on Android Devices
     cs.DC | 2021 | Fritze Robert, Plant Claudia
  4. [bm25 = 12.9121] GRAIMATTER Green Paper: Recommendations for disclosure control of trained M
     cs.LG | 2022 | Jefferson Emily, Liley James, Malone Maeve, Reel Smarti, Cre
  5. [bm25 = 12.5994] Towards Practical Watermark for Deep Neural Networks in Federated Learning
     cs.CR | 2021 | Li Fang-Qi, Wang Shi-Lin, Liew Alan Wee-Chung

  --- Vector (Pinecone) ---
  1. [cosine = 0.8809] Pretraining Data Exposure in Large Language Models: A Survey of Membership 
     cs.CL | 2026 | Tong Ziyi, Sun Feifei, Nguyen Le Minh
  2. [cosine = 0.8789] Enhancing Privacy in ControlNet and Stable Diffusion via Split Learning
     cs.LG | 2024 | Yao Dixi
  3. [cosine = 0.8745] Rethinking Membership Inference Attacks Against Transfer Learning
     cs.CR | 2025 | Wu Cong, Chen Jing, Fang Qianru, He Kun, Zhao Ziming, Ren Ha
  4. [cosine = 0.8725] Securing Graph Neural Networks in MLaaS: A Comprehensive Realization of Que
     cs.CR | 2023 | Wu Bang, Yuan Xingliang, Wang Shuo, Li Qi, Xue Minhui, Pan S
  5. [cosine = 0.8723] Synthetic ID Card Image Generation for Improving Presentation Attack Detect
     cs.CV | 2022 | Benalcazar Daniel, Tapia Juan E., Gonzalez Sebastian, Busch 

  --- Hybrid RRF (k=60) ---
  1. [rrf = 0.0307] Pretraining Data Exposure in Large Language Models: A Survey of Membership   (BM25 #10, vector #1)
     cs.CL | 2026 | Tong Ziyi, Sun Feifei, Nguyen Le Minh
  2. [rrf = 0.0164] Federated Unbiased Learning to Rank  (BM25 #1, vector —)
     cs.IR | 2021 | Li Chang, Ouyang Hua
  3. [rrf = 0.0161] Random Deployment of Data Collectors for Serving Randomly-Located Sensors  (BM25 #2, vector —)
     cs.NI | 2011 | Kwon Taesoo, Cioffi John. M.
  4. [rrf = 0.0161] Enhancing Privacy in ControlNet and Stable Diffusion via Split Learning  (BM25 —, vector #2)
     cs.LG | 2024 | Yao Dixi
  5. [rrf = 0.0159] GPU backed Data Mining on Android Devices  (BM25 #3, vector —)
     cs.DC | 2021 | Fritze Robert, Plant Claudia

====================================================================================================
EXPERIMENT: author names are not in the BM25 index
====================================================================================================
Papers with 'LeCun' among the authors in the sample: 2
  - Revisiting Feature Prediction for Learning Visual Representations from Video (2024)
  - Compact and Optimal Deep Learning with Recurrent Parameter Generators (2021)

  --- BM25 on title + abstract (as in the assignment) ---
  1. [bm25 = 11.1225] Convexified Convolutional Neural Networks
     cs.LG | 2016 | Zhang Yuchen, Liang Percy, Wainwright Martin J.
  2. [bm25 = 10.2732] Re-presenting a Story by Emotional Factors using Sentimental Analysis Metho
     cs.CL | 2016 | Jo Hwiyeol, Moon Yohan, Kim Jong In, Ryu Jeong
  3. [bm25 = 9.5839] A Hybrid Method for Training Convolutional Neural Networks
     cs.NE | 2020 | Lopes Vasco, Fazendeiro Paulo
  4. [bm25 = 9.5257] R-FCN: Object Detection via Region-based Fully Convolutional Networks
     cs.CV | 2016 | Dai Jifeng, Li Yi, He Kaiming, Sun Jian
  5. [bm25 = 9.3683] Norm-based Generalization Bounds for Compositionally Sparse Neural Networks
     cs.LG | 2023 | Galanti Tomer, Xu Mengjia, Galanti Liane, Poggio Tomaso

  --- BM25 on title + abstract + AUTHORS ---
  1. [bm25 = 17.1372] Compact and Optimal Deep Learning with Recurrent Parameter Generators
     cs.CV | 2021 | Wang Jiayun, Chen Yubei, Yu Stella X., Cheung Brian, LeCun Y
  2. [bm25 = 15.8300] Revisiting Feature Prediction for Learning Visual Representations from Vide
     cs.CV | 2024 | Bardes Adrien, Garrido Quentin, Ponce Jean, Chen Xinlei, Rab
  3. [bm25 = 11.1044] Convexified Convolutional Neural Networks
     cs.LG | 2016 | Zhang Yuchen, Liang Percy, Wainwright Martin J.
  4. [bm25 = 9.6865] Re-presenting a Story by Emotional Factors using Sentimental Analysis Metho
     cs.CL | 2016 | Jo Hwiyeol, Moon Yohan, Kim Jong In, Ryu Jeong
  5. [bm25 = 9.6505] A Hybrid Method for Training Convolutional Neural Networks
     cs.NE | 2020 | Lopes Vasco, Fazendeiro Paulo

====================================================================================================
EXPERIMENT: RRF parameter k (60 vs 1)
====================================================================================================

Contribution of one list to the RRF score, by rank:
  rank |     k=60 |      k=1
     1 |   0.0164 |   0.5000
     2 |   0.0161 |   0.3333
     3 |   0.0159 |   0.2500
     5 |   0.0154 |   0.1667
    10 |   0.0143 |   0.0909

[exact term] "BERT fine-tuning"  — same top-5: True
  1. k=60: Utilizing BERT Intermediate Layers for Asp | k=1: Utilizing BERT Intermediate Layers for Asp
  2. k=60: Deep Reinforcement Learning Radio Control  | k=1: Deep Reinforcement Learning Radio Control 
  3. k=60: It's not Greek to mBERT: Inducing Word-Lev | k=1: It's not Greek to mBERT: Inducing Word-Lev
  4. k=60: Boredom-driven curious learning by Homeo-H | k=1: Boredom-driven curious learning by Homeo-H
  5. k=60: Annotated Job Ads with Named Entity Recogn | k=1: Annotated Job Ads with Named Entity Recogn

[author name] "Yann LeCun convolutional networks"  — same top-5: False
  1. k=60: KAConvNet: Kolmogorov-Arnold Convolutional | k=1: KAConvNet: Kolmogorov-Arnold Convolutional
  2. k=60: Deep Unitary Convolutional Neural Networks | k=1: Convexified Convolutional Neural Networks
  3. k=60: Upright adjustment with graph convolutiona | k=1: Deep Unitary Convolutional Neural Networks
  4. k=60: Convexified Convolutional Neural Networks  | k=1: Re-presenting a Story by Emotional Factors
  5. k=60: Re-presenting a Story by Emotional Factors | k=1: The High-Dimensional Geometry of Binary Ne

[paraphrase] "making computers understand human emotions from text"  — same top-5: True
  1. k=60: Performance Evaluation of Supervised Machi | k=1: Performance Evaluation of Supervised Machi
  2. k=60: EmoGenie: Benchmarking and Evaluating Expr | k=1: EmoGenie: Benchmarking and Evaluating Expr
  3. k=60: "We care": Improving Code Mixed Speech Emo | k=1: "We care": Improving Code Mixed Speech Emo
  4. k=60: AGI: Artificial General Intelligence for E | k=1: AGI: Artificial General Intelligence for E
  5. k=60: Emotion Recognition and Generation: A Comp | k=1: Emotion Recognition and Generation: A Comp

[paraphrase (mine)] "protecting user data when training models on many devices"  — same top-5: True
  1. k=60: Pretraining Data Exposure in Large Languag | k=1: Pretraining Data Exposure in Large Languag
  2. k=60: Federated Unbiased Learning to Rank        | k=1: Federated Unbiased Learning to Rank
  3. k=60: Random Deployment of Data Collectors for S | k=1: Random Deployment of Data Collectors for S
  4. k=60: Enhancing Privacy in ControlNet and Stable | k=1: Enhancing Privacy in ControlNet and Stable
  5. k=60: GPU backed Data Mining on Android Devices  | k=1: GPU backed Data Mining on Android Devices

====================================================================================================
SUMMARY: #1 result per method
====================================================================================================
            query                                       BM25 #1                                     Vector #1                                     Hybrid #1  overlap BM25∩vector (top-10)
       exact term Utilizing BERT Intermediate Layers for Aspect Deep Reinforcement Learning Radio Control and Utilizing BERT Intermediate Layers for Aspect                             0
      author name     Convexified Convolutional Neural Networks KAConvNet: Kolmogorov-Arnold Convolutional Ne KAConvNet: Kolmogorov-Arnold Convolutional Ne                             3
       paraphrase EmoGenie: Benchmarking and Evaluating Express Performance Evaluation of Supervised Machine  Performance Evaluation of Supervised Machine                              1
paraphrase (mine)           Federated Unbiased Learning to Rank Pretraining Data Exposure in Large Language M Pretraining Data Exposure in Large Language M                             1
```

</details>