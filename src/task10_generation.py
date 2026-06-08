"""
Task 10 - Generation Co Citation.

Huong dan:
    1. Chon top_k, top_p phu hop (giai thich ly do)
    2. Sap xep lai chunks sau reranking de tranh "lost in the middle"
    3. Inject context vao prompt
    4. Yeu cau LLM tra loi co citation
    5. Neu khong du evidence -> "I cannot verify this information"
"""

import os
from dotenv import load_dotenv

load_dotenv()

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION - Giai thich lua chon
# =============================================================================

# top_k: So chunks dua vao context
# Chon 5 vi: du evidence ma khong qua dai gay lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xac suat tich luy cho token generation
# Chon 0.9 vi: du diverse nhung khong qua random
TOP_P = 0.9

# temperature: Do ngau nhien cua output
# Chon 0.3 vi: RAG can factual, it sang tao
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luat Phong chong ma tuy 2021, Dieu 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Toi khong the xac minh thong tin nay tu nguon hien co' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tranh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sap xep chunks de tranh "lost in the middle" effect.

    LLM nho tot thong tin o DAU va CUOI prompt, quen thong tin o GIUA.
    Strategy: dat chunks quan trong nhat o dau va cuoi, kem quan trong o giua.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered de maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    reordered = []
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])
    start_index = len(chunks) - 1 if len(chunks) % 2 == 0 else len(chunks) - 2
    for i in range(start_index, 0, -2):
        reordered.append(chunks[i])

    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thanh context string cho prompt.
    Moi chunk co label source de LLM co the cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        context_parts.append(
            f"[Document {i} | Source: {source} | Type: {doc_type}]\n"
            f"{chunk['content']}\n"
        )
    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation co citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder de tranh lost in the middle
        3. Format context voi source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Cau hoi cua user

    Returns:
        {
            'answer': str,           # Cau tra loi co citation
            'sources': list[dict],   # Cac chunks da dung
            'retrieval_source': str  # 'hybrid' hoac 'pageindex'
        }
    """
    try:
        chunks = retrieve(query, top_k=top_k)
    except Exception:
        chunks = []

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    answer = "Toi khong the xac minh thong tin nay tu nguon hien co."

    if reordered:
        source = reordered[0].get("metadata", {}).get("source", "Nguon hien co")
        excerpt = " ".join(reordered[0].get("content", "").split())[:300].strip()
        if excerpt:
            answer = f"{excerpt} [{source}]"

    user_message = f"""Context:\n{context}\n\n---\n\nQuestion: {query}"""

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            answer = response.choices[0].message.content or answer
        except Exception:
            pass

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid") if chunks else "none",
    }


if __name__ == "__main__":
    test_queries = [
        "Hinh phat cho toi tang tru trai phep chat ma tuy theo phap luat Viet Nam?",
        "Nhung nghe si nao da bi bat vi lien quan toi ma tuy?",
        "Quy trinh cai nghien bat buoc theo Luat Phong chong ma tuy 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
