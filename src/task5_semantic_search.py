"""
Task 5 - Semantic Search Module.

Viet module tim kiem ngu nghia (dense retrieval) tren vector store.

Yeu cau:
    - Input: query string + top_k
    - Output: danh sach chunks co score, sorted descending
    - Phai tuong thich voi embedding model va vector store o Task 4
"""

import os

from dotenv import load_dotenv

try:
    from .task4_chunking_indexing import EMBEDDING_MODEL
except ImportError:
    from task4_chunking_indexing import EMBEDDING_MODEL

load_dotenv()


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tim kiem ngu nghia su dung vector similarity.

    Args:
        query: Cau truy van
        top_k: So luong ket qua toi da

    Returns:
        List of {
            'content': str,      # Noi dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    import weaviate
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing package 'openai'. Install it with: pip install openai"
        ) from exc
    from weaviate.classes.init import AdditionalConfig, Auth, Timeout
    from weaviate.classes.query import MetadataQuery

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for semantic search.")

    client_openai = OpenAI(api_key=api_key)
    query_embedding = client_openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    ).data[0].embedding

    weaviate_url = os.getenv("WEAVIATE_URL", "").strip()
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY", "").strip()
    if not weaviate_url or not weaviate_api_key:
        raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY are required for Weaviate Cloud.")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
        additional_config=AdditionalConfig(timeout=Timeout(init=30)),
    )

    collection = client.collections.get("DrugLawDocs")
    results = collection.query.near_vector(
        near_vector=query_embedding,
        limit=top_k,
        return_metadata=MetadataQuery(distance=True),
    )

    formatted = [
        {
            "content": obj.properties["content"],
            "score": 1 - float(obj.metadata.distance or 0.0),
            "metadata": {
                "source": obj.properties.get("source", ""),
                "type": obj.properties.get("doc_type", ""),
                "path": obj.properties.get("path", ""),
                "chunk_index": obj.properties.get("chunk_index", 0),
            },
        }
        for obj in results.objects
    ]
    client.close()
    return sorted(formatted, key=lambda item: item["score"], reverse=True)


if __name__ == "__main__":
    results = semantic_search("hinh phat cho toi tang tru ma tuy", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
