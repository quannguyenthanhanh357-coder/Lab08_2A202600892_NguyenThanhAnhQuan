"""
Task 7 - Reranking Module.

Chon 1 trong cac phuong phap:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoac Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tu implement
    - RRF (Reciprocal Rank Fusion): tu implement

Neu dung MMR hoac RRF, dam bao hieu va giai thich duoc co che.
"""

import hashlib
from typing import Optional


def _embed_text(text: str, dim: int = 128) -> list[float]:
    vector = [0.0] * dim
    tokens = text.lower().split()
    if not tokens:
        return vector

    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        index = int(digest[:8], 16) % dim
        vector[index] += 1.0

    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates su dung cross-encoder model.

    Args:
        query: Cau truy van
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: So luong ket qua sau rerank

    Returns:
        List of top_k candidates, re-scored va sorted by rerank_score descending.
    """
    query_terms = set(query.lower().split())
    reranked = []
    for candidate in candidates:
        content_terms = set(candidate["content"].lower().split())
        overlap = len(query_terms & content_terms)
        score = 0.7 * float(candidate.get("score", 0.0)) + 0.3 * (
            overlap / max(len(query_terms), 1)
        )
        reranked.append({**candidate, "score": float(score)})
    return sorted(reranked, key=lambda item: item["score"], reverse=True)[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance - chon candidates vua relevant vua diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding cua query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: So luong ket qua
        lambda_param: Trade-off giua relevance (1.0) va diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    def cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
        numerator = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return numerator / (norm_a * norm_b)

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            relevance = cosine_sim(query_embedding, candidates[idx].get("embedding", []))

            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = cosine_sim(
                    candidates[idx].get("embedding", []),
                    candidates[sel_idx].get("embedding", []),
                )
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for i in selected:
        item = {**candidates[i]}
        item["score"] = float(
            cosine_sim(query_embedding, item.get("embedding", []))
        )
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion - gop ket qua tu nhieu ranker.

    RRF(d) = sum 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (moi list tu 1 ranker)
        top_k: So luong ket qua cuoi cung
        k: Smoothing constant (default=60, tu paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores = {}
    content_map = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "mmr",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Cau truy van
        candidates: Danh sach candidates tu retrieval
        top_k: So luong ket qua sau rerank
        method: Phuong phap reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        prepared_candidates = []
        for candidate in candidates:
            item = {**candidate}
            item["embedding"] = item.get("embedding", _embed_text(item.get("content", "")))
            prepared_candidates.append(item)
        query_embedding = _embed_text(query)
        return rerank_mmr(query_embedding, prepared_candidates, top_k)
    elif method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Dieu 248: Toi tang tru trai phep chat ma tuy", "score": 0.8, "metadata": {}},
        {"content": "Nghe si X bi bat vi su dung ma tuy", "score": 0.7, "metadata": {}},
        {"content": "Hinh phat tu tu 2-7 nam cho toi tang tru", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hinh phat tang tru ma tuy", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
