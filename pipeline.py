import time

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder


class TwoStageRetriever:
    """Two-stage semantic search pipeline: Bi-Encoder retrieval + Cross-Encoder re-ranking."""

    BI_ENCODER_MODEL = "all-MiniLM-L6-v2"
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CORPUS_SIZE = 500_000

    def __init__(self):
        self.passages = []
        self.index = None
        self.bi_encoder = SentenceTransformer(self.BI_ENCODER_MODEL)
        self.cross_encoder = CrossEncoder(self.CROSS_ENCODER_MODEL)

    def load_data(self):
        """Pull a 10,000-row sample from the MS MARCO passage retrieval dataset."""
        print(f"Loading MS MARCO dataset ({self.CORPUS_SIZE} passages)...")
        dataset = load_dataset("ms_marco", "v1.1", split="train")

        passages = []
        seen = set()
        for row in dataset:
            for passage in row["passages"]["passage_text"]:
                text = passage.strip()
                if text and text not in seen:
                    seen.add(text)
                    passages.append(text)
                    if len(passages) >= self.CORPUS_SIZE:
                        break
            if len(passages) >= self.CORPUS_SIZE:
                break

        self.passages = passages
        print(f"Loaded {len(self.passages)} unique passages.")

    def build_index(self):
        """Encode the corpus with the bi-encoder and build a FAISS index."""
        print("Encoding corpus with bi-encoder...")
        embeddings = self.bi_encoder.encode(
            self.passages, show_progress_bar=True, convert_to_numpy=True
        )
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors (dim={dimension}).")

    def stage1_retrieve(self, query, top_k=100):
        """Stage 1: Bi-Encoder retrieval via FAISS. Returns (candidates, latency_ms)."""
        start = time.perf_counter()

        query_embedding = self.bi_encoder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, top_k)

        candidates = [
            {"text": self.passages[idx], "bi_score": float(scores[0][i])}
            for i, idx in enumerate(indices[0])
            if idx < len(self.passages)
        ]

        latency_ms = (time.perf_counter() - start) * 1000
        return candidates, latency_ms

    def stage2_rerank(self, query, candidates, top_k=5):
        """Stage 2: Cross-Encoder re-ranking. Returns (top_results, latency_ms)."""
        start = time.perf_counter()

        pairs = [[query, c["text"]] for c in candidates]
        scores = self.cross_encoder.predict(pairs)

        for i, candidate in enumerate(candidates):
            candidate["cross_score"] = float(scores[i])

        ranked = sorted(candidates, key=lambda x: x["cross_score"], reverse=True)

        latency_ms = (time.perf_counter() - start) * 1000
        return ranked[:top_k], latency_ms


if __name__ == "__main__":
    pipeline = TwoStageRetriever()

    pipeline.load_data()
    pipeline.build_index()

    query = "how many ounces in a gallon"
    print(f"\nQuery: \"{query}\"\n")
    print("=" * 80)

    candidates, stage1_time = pipeline.stage1_retrieve(query, top_k=100)
    print(f"Stage 1 (Bi-Encoder Retrieval): {stage1_time:.2f} ms — {len(candidates)} candidates")

    top_results, stage2_time = pipeline.stage2_rerank(query, candidates, top_k=5)
    print(f"Stage 2 (Cross-Encoder Re-rank): {stage2_time:.2f} ms — Top {len(top_results)} selected")

    print(f"\nTotal search latency: {stage1_time + stage2_time:.2f} ms")
    print("=" * 80)
    print("\nTop 5 Results:\n")

    for rank, result in enumerate(top_results, 1):
        print(f"  [{rank}] (cross-encoder: {result['cross_score']:.4f} | bi-encoder: {result['bi_score']:.4f})")
        print(f"      {result['text'][:200]}")
        print()
