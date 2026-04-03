# -two-stage-retriever-pipeline
Two-stage semantic search pipeline. Stage 1: Bi-Encoder with FAISS for high-speed candidate retrieval (O(log N) latency) Stage 2: Cross-Encoder for deep contextual re-ranking, optimizing the tradeoff between search throughput and absolute precision over the MS MARCO dataset.
