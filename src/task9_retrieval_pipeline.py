"""
Task 9 - Retrieval Pipeline Hoan Chinh.

Ket hop semantic search + lexical search + reranking + PageIndex fallback
thanh mot pipeline thong nhat.

Logic:
    1. Chay semantic_search + lexical_search song song
    2. Merge ket qua (RRF hoac weighted fusion)
    3. Rerank
    4. Neu top result score < threshold -> fallback sang PageIndex
    5. Return top_k results
"""

try:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:
    from task5_semantic_search import semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Neu best score < threshold -> fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "mmr"  # "cross_encoder" | "mmr" | "rrf"


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoan chinh voi fallback logic.

    Pipeline:
        Query
          -> Semantic Search -> results_dense
          -> Lexical Search  -> results_sparse
          ->
          -> Merge (RRF) -> merged_results
          -> Rerank -> reranked_results
          ->
          -> If best_score < threshold:
                -> PageIndex Vectorless -> fallback_results

    Args:
        query: Cau truy van
        top_k: So luong ket qua cuoi cung
        score_threshold: Nguong diem toi thieu cho hybrid results
        use_reranking: Co ap dung reranking hay khong

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoac 'pageindex'
        }
    """
    dense_results = semantic_search(query, top_k=top_k * 2)
    sparse_results = lexical_search(query, top_k=top_k * 2)

    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    for item in merged:
        item["source"] = "hybrid"

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            item["source"] = "hybrid"
    else:
        final_results = merged[:top_k]

    if not final_results or final_results[0]["score"] < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
        except Exception:
            fallback = []
        return fallback

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hinh phat cho toi tang tru trai phep chat ma tuy",
        "Nghe si nao bi bat vi su dung ma tuy nam 2024",
        "Luat phong chong ma tuy 2021 quy dinh gi ve cai nghien",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
