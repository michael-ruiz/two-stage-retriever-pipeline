# Two-Stage Retriever Pipeline

A high-throughput semantic search engine demonstrating enterprise-grade information retrieval on the [MS MARCO](https://microsoft.github.io/msmarco/) passage dataset.

## How It Works

The pipeline splits retrieval into two stages to balance speed and accuracy:

**Stage 1 -- Bi-Encoder Retrieval**
Embeds 500K passages with [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) and loads them into a FAISS inner-product index. At query time, the same model encodes the query and FAISS returns the top 100 candidates in milliseconds.

**Stage 2 -- Cross-Encoder Re-Ranking**
A [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) model scores every (query, passage) pair from Stage 1. Because cross-encoders see query and passage together, they capture deeper relevance signals. The candidates are re-sorted and the top 5 results are returned.

## Setup

```bash
conda create -n ret-env python=3.11 -y
conda activate ret-env
pip install -r requirements.txt
```

## Usage

```bash
python pipeline.py
```

The script loads the corpus, builds the FAISS index, and runs a sample query (`"how many ounces in a gallon"`). Output includes per-stage latency and the final top 5 passages.

## Project Structure

```
pipeline.py        # Full retrieval pipeline (data ingestion, indexing, search, re-ranking)
requirements.txt   # Python dependencies
```

## Dependencies

- **faiss-cpu** -- Approximate nearest-neighbor search
- **sentence-transformers** -- Bi-encoder and cross-encoder models
- **datasets** -- Hugging Face dataset loading (MS MARCO)
- **torch** -- PyTorch backend
