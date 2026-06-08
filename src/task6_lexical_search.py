"""
Task 6 - Lexical Search Module (BM25).

Mac dinh su dung BM25. Neu dung phuong phap khac (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hay giai thich co che trong buoi demo -> +5 bonus.

Cai dat:
    pip install rank-bm25

BM25 hoat dong the nao:
    - Term Frequency (TF): tu xuat hien nhieu trong document -> diem cao
    - Inverse Document Frequency (IDF): tu hiem -> quan trong hon
    - Document length normalization: document dai khong bi uu tien qua muc
    - Formula: score(q,d) = sum IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path

# TODO: Load corpus tu data/standardized/ hoac tu vector store
CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}


def build_bm25_index(corpus: list[dict]):
    """
    Xay dung BM25 index tu corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [doc["content"].lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tim kiem tu khoa su dung BM25.

    Args:
        query: Cau truy van
        top_k: So luong ket qua toi da

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global CORPUS

    if not CORPUS:
        standardized_dir = Path(__file__).parent.parent / "data" / "standardized"
        for md_file in standardized_dir.rglob("*.md"):
            content = md_file.read_text(encoding="utf-8").strip()
            if not content:
                continue
            CORPUS.append(
                {
                    "content": content,
                    "metadata": {
                        "source": md_file.name,
                        "type": "legal" if "legal" in str(md_file) else "news",
                        "path": str(md_file.relative_to(standardized_dir)),
                    },
                }
            )

    if not CORPUS:
        return []

    bm25 = build_bm25_index(CORPUS)
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append(
                {
                    "content": CORPUS[idx]["content"],
                    "score": float(scores[idx]),
                    "metadata": CORPUS[idx]["metadata"],
                }
            )
    return results


if __name__ == "__main__":
    results = lexical_search("Dieu 248 tang tru trai phep chat ma tuy", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
