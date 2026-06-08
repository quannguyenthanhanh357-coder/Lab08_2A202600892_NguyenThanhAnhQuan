"""
Task 4 - Chunking & Indexing vao Vector Store.

Huong dan:
    1. Doc toan bo markdown files tu data/standardized/
    2. Chon 1 chunking strategy (giai thich ly do)
    3. Chon 1 embedding model (giai thich ly do)
    4. Index vao vector store (Weaviate khuyen cao)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toan, pho bien
    - MarkdownHeaderTextSplitter: tot cho file co heading
    - SemanticChunker: dung embedding de tach (nang cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhe)
    - BAAI/bge-m3 (1024 dim, multilingual, tot cho tieng Viet)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyen cao: ho tro hybrid search built-in)
    - ChromaDB (don gian, local)
    - FAISS (chi dense search)

Cai dat:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_CACHE_PATH = Path(__file__).parent.parent / "data" / "index_cache.json"


# =============================================================================
# CONFIGURATION - Giai thich lua chon cua ban trong comment
# =============================================================================

# Chon RecursiveCharacterTextSplitter vi day la cach chunk an toan, pho bien,
# giu duoc ngu canh tot hon cat cung theo fixed-size thuan tuy.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

# Chon OpenAI text-embedding-3-small vi de trien khai bang API key,
# chi phi hop ly va kich thuoc 1536 phu hop retrieval.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Chon Weaviate dung theo khuyen nghi README va ho tro hybrid search.
VECTOR_STORE = "weaviate"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Doc toan bo markdown files tu data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy da chon.

    Returns:
        List of {'content': str, 'metadata': dict} - moi item la 1 chunk
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        split_text = splitter.split_text
    except ModuleNotFoundError:
        def split_text(text: str) -> list[str]:
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + CHUNK_SIZE, len(text))
                if end < len(text):
                    split_at = max(
                        text.rfind("\n\n", start, end),
                        text.rfind("\n", start, end),
                        text.rfind(". ", start, end),
                        text.rfind(" ", start, end),
                    )
                    if split_at > start:
                        end = split_at
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(chunk_text)
                if end >= len(text):
                    break
                start = max(end - CHUNK_OVERLAP, start + 1)
            return chunks

    chunks = []
    for doc in documents:
        splits = split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": i,
                        "chunk_size": CHUNK_SIZE,
                        "chunk_overlap": CHUNK_OVERLAP,
                    },
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toan bo chunks bang model da chon.

    Returns:
        Moi chunk dict duoc them key 'embedding': list[float]
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for Task 4 embedding.")

    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing package 'openai'. Install it with: pip install openai"
        ) from exc

    client = OpenAI(api_key=api_key)
    texts = [c["content"] for c in chunks]
    embedded_chunks = []

    batch_size = 100
    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch_texts)
        for chunk, emb in zip(chunks[batch_start:batch_start + batch_size], response.data):
            item = {"content": chunk["content"], "metadata": dict(chunk["metadata"])}
            item["embedding"] = emb.embedding
            embedded_chunks.append(item)
    return embedded_chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Luu chunks vao vector store da chon.
    """
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property
    from weaviate.classes.init import AdditionalConfig, Auth, Timeout

    weaviate_url = os.getenv("WEAVIATE_URL", "").strip()
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY", "").strip()
    if not weaviate_url or not weaviate_api_key:
        raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY are required for Weaviate Cloud.")

    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_api_key),
        additional_config=AdditionalConfig(timeout=Timeout(init=30)),
    )

    collection_name = "DrugLawDocs"

    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)

    collection = client.collections.create(
        name=collection_name,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="doc_type", data_type=DataType.TEXT),
            Property(name="path", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
        ],
    )

    with collection.batch.dynamic() as batch:
        for chunk in chunks:
            batch.add_object(
                properties={
                    "content": chunk["content"],
                    "source": chunk["metadata"].get("source", ""),
                    "doc_type": chunk["metadata"].get("type", ""),
                    "path": chunk["metadata"].get("path", ""),
                    "chunk_index": int(chunk["metadata"].get("chunk_index", 0)),
                },
                vector=chunk["embedding"],
            )

    INDEX_CACHE_PATH.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    client.close()


def run_pipeline():
    """Chay toan bo pipeline: load -> chunk -> embed -> index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
