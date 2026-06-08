"""
Task 8 - PageIndex Vectorless RAG.

Dang ky tai khoan tai: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phep RAG ma khong can vector store - su dung
structural understanding cua document thay vi embedding.

Cai dat:
    pip install pageindex

Huong dan:
    1. Dang ky account tai pageindex.ai
    2. Lay API key
    3. Upload documents
    4. Query su dung PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import requests

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_API_URL = os.getenv("PAGEINDEX_API_URL", "https://api.pageindex.ai")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """
    Upload toan bo markdown documents len PageIndex.
    """
    uploaded = []
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        with open(md_file, "rb") as file_handle:
            response = requests.post(
                f"{PAGEINDEX_API_URL}/markdown/",
                headers={"api_key": PAGEINDEX_API_KEY},
                files={"file": file_handle},
                timeout=60,
            )
        response.raise_for_status()
        uploaded.append(response.json())
        print(f"  Uploaded: {md_file.name}")
    return uploaded


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval su dung PageIndex.
    Dung lam fallback khi hybrid search khong co ket qua tot.

    Args:
        query: Cau truy van
        top_k: So luong ket qua toi da

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Danh dau nguon retrieval
        }
    """
    response = requests.post(
        f"{PAGEINDEX_API_URL}/chat/completions",
        headers={
            "api_key": PAGEINDEX_API_KEY,
            "Content-Type": "application/json",
        },
        json={
            "messages": [{"role": "user", "content": query}],
            "enable_citations": True,
            "stream": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()
    answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not answer:
        return []

    return [
        {
            "content": answer,
            "score": 1.0,
            "metadata": {"provider": "pageindex"},
            "source": "pageindex",
        }
    ][:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("Hay set PAGEINDEX_API_KEY trong file .env")
        print("  Dang ky tai: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hinh phat su dung ma tuy", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
