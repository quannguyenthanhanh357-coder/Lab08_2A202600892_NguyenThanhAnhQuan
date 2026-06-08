"""Configurable RAG backend for the group Streamlit chatbot."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from openai import OpenAI
from rank_bm25 import BM25Okapi

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
LOCAL_VECTOR_DIR = PROJECT_ROOT / "group_project" / ".local_vectorstore"
EMBED_BATCH_SIZE = 64
LLM_MODEL = "gpt-4o-mini"

EMBEDDING_OPTIONS: dict[str, dict[str, Any]] = {
    "text-embedding-3-small": {"dimensions": 1536},
    "text-embedding-3-large": {"dimensions": 3072},
}

SPLITTER_OPTIONS = ["recursive", "markdown_header", "semantic"]
RERANK_OPTIONS = ["none", "rrf", "mmr", "cross_encoder"]
VECTOR_STORE_OPTIONS = ["local_numpy", "weaviate_cloud"]
MAX_TOOL_TURNS = 3

OUT_OF_DOMAIN_MSG = (
    "Tôi không phải chatbot trong lĩnh vực này. "
    "Tôi chỉ hỗ trợ các câu hỏi về pháp luật/chất cấm/ma tuý "
    "và tin tức nghệ sĩ liên quan."
)

AGENT_SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi tiếng Việt, giới hạn trong DATASET sau:
- Văn bản pháp luật Việt Nam về ma tuý và các chất cấm.
- Bài báo về nghệ sĩ/người nổi tiếng Việt Nam liên quan tới ma tuý.

# Công cụ (tool):
Bạn có một tool tên `search_context` để tìm ngữ cảnh trong dataset trên.

## Khi NÀO gọi `search_context`:
- Khi câu hỏi liên quan tới pháp luật/chất cấm/ma tuý, hoặc
- Khi câu hỏi liên quan tới nghệ sĩ/người nổi tiếng dính líu ma tuý, hoặc
- Khi cần dữ kiện cụ thể (điều luật, hình phạt, tên người, sự kiện, thời gian)
  mà chỉ có thể xác minh từ dataset.

## Khi NÀO KHÔNG gọi `search_context`:
- Khi câu hỏi nằm ngoài chủ đề trên (ví dụ: lập trình/code, thời tiết, nấu ăn,
  toán học, thể thao, du lịch, chuyện phiếm...).
- Trong trường hợp này, TUYỆT ĐỐI KHÔNG gọi tool và trả lời CHÍNH XÁC câu sau:
  "{out_of_domain}"

# Quy tắc trả lời (sau khi đã có context từ tool):
- Chỉ dùng thông tin trong context tool trả về.
- Mỗi nhận định/sự kiện PHẢI có citation trong ngoặc, ví dụ
  [Luật Phòng chống ma tuý 2021, Điều 3] hoặc [VTC News, 2026].
- Nếu context không đủ để trả lời, nói rõ:
  "Tôi không thể xác minh thông tin này từ nguồn hiện có".
- Trình bày rõ ràng theo đoạn.
""".replace("{out_of_domain}", OUT_OF_DOMAIN_MSG)

SEARCH_CONTEXT_TOOL = {
    "type": "function",
    "function": {
        "name": "search_context",
        "description": (
            "Tìm kiếm ngữ cảnh liên quan trong dataset gồm văn bản pháp luật về "
            "ma tuý/chất cấm và bài báo về nghệ sĩ liên quan tới ma tuý. "
            "CHỈ gọi khi câu hỏi thuộc các chủ đề này. KHÔNG gọi cho câu hỏi "
            "ngoài lĩnh vực (lập trình, thời tiết, nấu ăn...)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Truy vấn tìm kiếm, nên cụ thể và bằng tiếng Việt.",
                },
            },
            "required": ["query"],
        },
    },
}


@dataclass(frozen=True)
class RAGConfig:
    splitter: str
    embedding_model: str
    reranker: str
    vector_store: str = "local_numpy"
    chunk_size: int = 900
    chunk_overlap: int = 120
    score_threshold: float = 0.25

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_OPTIONS[self.embedding_model]["dimensions"]

    @property
    def collection_name(self) -> str:
        # Weaviate free tier thường chỉ cho 1 collection, nên dùng tên cố định.
        return "DrugLawDocs"

    @property
    def local_index_path(self) -> Path:
        payload = (
            f"{self.splitter}|{self.embedding_model}|{self.chunk_size}|"
            f"{self.chunk_overlap}"
        )
        short_hash = hashlib.md5(payload.encode("utf-8")).hexdigest()[:8]
        return LOCAL_VECTOR_DIR / f"index_{short_hash}.json"


class ConfigurableRAG:
    def __init__(self, config: RAGConfig):
        self.config = config
        self._bm25: BM25Okapi | None = None
        self._bm25_corpus: list[dict] = []
        self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._local_payload_cache: dict | None = None

    def _get_weaviate_client(self):
        import weaviate
        from weaviate.auth import AuthApiKey

        cluster_url = os.getenv("WEAVIATE_URL", "")
        api_key = os.getenv("WEAVIATE_API_KEY", "")
        if not cluster_url or not api_key or "xxx" in cluster_url or "xxx" in api_key:
            raise RuntimeError(
                "Weaviate Cloud chưa được cấu hình. "
                "Vui lòng điền WEAVIATE_URL và WEAVIATE_API_KEY trong .env."
            )
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_url,
            auth_credentials=AuthApiKey(api_key),
            skip_init_checks=True,
        )

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            response = self._openai.embeddings.create(
                model=self.config.embedding_model,
                input=batch,
                dimensions=self.config.embedding_dim,
            )
            vectors.extend([item.embedding for item in response.data])
        return vectors

    def _embed_query(self, query: str) -> list[float]:
        response = self._openai.embeddings.create(
            model=self.config.embedding_model,
            input=query,
            dimensions=self.config.embedding_dim,
        )
        return response.data[0].embedding

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def _load_local_payload(self) -> dict:
        if self._local_payload_cache is not None:
            return self._local_payload_cache
        path = self.config.local_index_path
        if not path.exists():
            return {"chunks": [], "vectors": []}
        self._local_payload_cache = json.loads(path.read_text(encoding="utf-8"))
        return self._local_payload_cache

    def _save_local_payload(self, payload: dict):
        self.config.local_index_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.local_index_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        self._local_payload_cache = payload

    def load_documents(self) -> list[dict]:
        docs = []
        for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
            content = md_file.read_text(encoding="utf-8").strip()
            if not content:
                continue
            doc_type = "legal" if "legal" in md_file.parts else "news"
            docs.append(
                {
                    "content": content,
                    "metadata": {
                        "source": md_file.name,
                        "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                        "type": doc_type,
                    },
                }
            )
        return docs

    def _recursive_split(self, text: str) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)

    def _markdown_header_split(self, text: str) -> list[str]:
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]
        )
        sections = header_splitter.split_text(text)
        chunks: list[str] = []
        for section in sections:
            body = section.page_content.strip()
            if not body:
                continue
            chunks.extend(self._recursive_split(body))
        if not chunks:
            return self._recursive_split(text)
        return chunks

    def _semantic_split(self, text: str) -> list[str]:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_openai import OpenAIEmbeddings

        semantic_splitter = SemanticChunker(
            embeddings=OpenAIEmbeddings(
                model=self.config.embedding_model,
                dimensions=self.config.embedding_dim,
            ),
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=85,
        )
        semantic_chunks = semantic_splitter.split_text(text)
        final_chunks: list[str] = []
        for chunk in semantic_chunks:
            if len(chunk) <= int(self.config.chunk_size * 1.1):
                final_chunks.append(chunk)
            else:
                final_chunks.extend(self._recursive_split(chunk))
        return final_chunks

    def chunk_documents(self, docs: list[dict]) -> list[dict]:
        chunks: list[dict] = []
        for doc in docs:
            text = doc["content"]
            if self.config.splitter == "recursive":
                split_texts = self._recursive_split(text)
            elif self.config.splitter == "markdown_header":
                split_texts = self._markdown_header_split(text)
            else:
                split_texts = self._semantic_split(text)

            for idx, chunk_text in enumerate(split_texts):
                cleaned = chunk_text.strip()
                if not cleaned:
                    continue
                chunks.append(
                    {
                        "content": cleaned,
                        "metadata": {**doc["metadata"], "chunk_index": idx},
                    }
                )
        return chunks

    def index_documents(self, force_rebuild: bool = False) -> dict:
        docs = self.load_documents()
        if not docs:
            raise RuntimeError("Không tìm thấy tài liệu markdown trong data/standardized.")

        chunks = self.chunk_documents(docs)
        vectors = self._embed_texts([item["content"] for item in chunks])
        self._bm25 = None
        self._bm25_corpus = []

        if self.config.vector_store == "local_numpy":
            payload = {"chunks": chunks, "vectors": vectors}
            if force_rebuild or not self.config.local_index_path.exists():
                self._save_local_payload(payload)
            return {
                "collection": str(self.config.local_index_path.name),
                "documents": len(docs),
                "chunks": len(chunks),
                "status": "indexed_local",
            }

        from weaviate.classes.config import Configure, DataType, Property, VectorDistances

        client = self._get_weaviate_client()
        try:
            exists = client.collections.exists(self.config.collection_name)
            if exists and force_rebuild:
                client.collections.delete(self.config.collection_name)
                exists = False

            if not exists:
                collection = client.collections.create(
                    name=self.config.collection_name,
                    vector_config=Configure.Vectors.self_provided(
                        vector_index_config=Configure.VectorIndex.hfresh(
                            distance_metric=VectorDistances.COSINE
                        )
                    ),
                    properties=[
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="source", data_type=DataType.TEXT),
                        Property(name="path", data_type=DataType.TEXT),
                        Property(name="doc_type", data_type=DataType.TEXT),
                        Property(name="chunk_index", data_type=DataType.INT),
                    ],
                )
                with collection.batch.dynamic() as batch:
                    for chunk, vector in zip(chunks, vectors):
                        meta = chunk["metadata"]
                        batch.add_object(
                            properties={
                                "content": chunk["content"],
                                "source": meta.get("source", ""),
                                "path": meta.get("path", ""),
                                "doc_type": meta.get("type", ""),
                                "chunk_index": meta.get("chunk_index", 0),
                            },
                            vector=vector,
                        )

                failed = collection.batch.failed_objects
                if failed:
                    raise RuntimeError(
                        f"Indexing thất bại {len(failed)} objects vào Weaviate."
                    )

            return {
                "collection": self.config.collection_name,
                "documents": len(docs),
                "chunks": len(chunks),
                "status": "ready" if exists else "indexed_weaviate",
            }
        finally:
            client.close()

    def _get_collection_objects(self) -> list[dict]:
        if self.config.vector_store == "local_numpy":
            payload = self._load_local_payload()
            return payload.get("chunks", [])

        client = self._get_weaviate_client()
        try:
            if not client.collections.exists(self.config.collection_name):
                return []
            collection = client.collections.get(self.config.collection_name)
            docs = []
            for obj in collection.iterator():
                docs.append(
                    {
                        "content": obj.properties.get("content", ""),
                        "metadata": {
                            "source": obj.properties.get("source", ""),
                            "path": obj.properties.get("path", ""),
                            "type": obj.properties.get("doc_type", ""),
                            "chunk_index": obj.properties.get("chunk_index", 0),
                        },
                    }
                )
            return docs
        finally:
            client.close()

    def _ensure_bm25(self):
        if self._bm25 is not None:
            return
        corpus = self._get_collection_objects()
        self._bm25_corpus = corpus
        tokenized = [self._tokenize(item["content"]) for item in corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def semantic_search(self, query: str, top_k: int) -> list[dict]:
        if not query.strip():
            return []

        query_vector = self._embed_query(query)
        if self.config.vector_store == "local_numpy":
            payload = self._load_local_payload()
            chunks = payload.get("chunks", [])
            vectors = payload.get("vectors", [])
            if not chunks or not vectors:
                return []

            scored = []
            for chunk, vector in zip(chunks, vectors):
                score = self._cosine_similarity(query_vector, vector)
                scored.append(
                    {
                        "content": chunk["content"],
                        "score": float(score),
                        "metadata": chunk["metadata"],
                        "retrieval_method": "semantic_local",
                    }
                )
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]

        from weaviate.classes.query import MetadataQuery

        client = self._get_weaviate_client()
        try:
            if not client.collections.exists(self.config.collection_name):
                return []
            collection = client.collections.get(self.config.collection_name)
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                return_metadata=MetadataQuery(distance=True),
            )
            results = []
            for obj in response.objects:
                distance = obj.metadata.distance if obj.metadata else 1.0
                score = max(0.0, 1.0 - float(distance))
                results.append(
                    {
                        "content": obj.properties.get("content", ""),
                        "score": score,
                        "metadata": {
                            "source": obj.properties.get("source", ""),
                            "path": obj.properties.get("path", ""),
                            "type": obj.properties.get("doc_type", ""),
                            "chunk_index": obj.properties.get("chunk_index", 0),
                        },
                        "retrieval_method": "semantic",
                    }
                )
            return results
        finally:
            client.close()

    def lexical_search(self, query: str, top_k: int) -> list[dict]:
        self._ensure_bm25()
        if not self._bm25 or not self._bm25_corpus:
            return []

        scores = self._bm25.get_scores(self._tokenize(query))
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results = []
        for idx in ranked_indices[:top_k]:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(
                {
                    "content": self._bm25_corpus[idx]["content"],
                    "score": score,
                    "metadata": self._bm25_corpus[idx]["metadata"],
                    "retrieval_method": "lexical",
                }
            )
        return results

    @staticmethod
    def _rrf_merge(rank_lists: list[list[dict]], k: int = 60) -> list[dict]:
        merged_scores: dict[str, float] = {}
        merged_item: dict[str, dict] = {}

        for rank_list in rank_lists:
            for rank, item in enumerate(rank_list, start=1):
                key = item["content"]
                merged_scores[key] = merged_scores.get(key, 0.0) + 1.0 / (k + rank)
                merged_item[key] = item

        sorted_items = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
        output = []
        for content, score in sorted_items:
            item = dict(merged_item[content])
            item["score"] = score
            item["retrieval_method"] = "hybrid"
            output.append(item)
        return output

    def _rerank_mmr(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        if not candidates:
            return []
        query_vec = self._embed_query(query)
        doc_vectors = self._embed_texts([item["content"] for item in candidates])
        selected: list[int] = []
        remaining = list(range(len(candidates)))
        lambda_param = 0.7

        while remaining and len(selected) < top_k:
            best_idx = remaining[0]
            best_score = float("-inf")
            for idx in remaining:
                relevance = self._cosine_similarity(query_vec, doc_vectors[idx])
                max_sim = 0.0
                for selected_idx in selected:
                    max_sim = max(
                        max_sim,
                        self._cosine_similarity(doc_vectors[idx], doc_vectors[selected_idx]),
                    )
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            selected.append(best_idx)
            remaining.remove(best_idx)

        results = []
        for idx in selected:
            item = dict(candidates[idx])
            item["score"] = self._cosine_similarity(query_vec, doc_vectors[idx])
            item["retrieval_method"] = "hybrid+mmr"
            results.append(item)
        return results

    def _rerank_cross_encoder(
        self, query: str, candidates: list[dict], top_k: int
    ) -> list[dict]:
        if not candidates:
            return []

        docs_block = "\n\n".join(
            f"ID {idx}:\n{item['content'][:1200]}" for idx, item in enumerate(candidates)
        )
        response = self._openai.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Score each candidate document for relevance to query. "
                        "Return strict JSON with schema "
                        "{\"scores\": [{\"id\": int, \"score\": float}]}. "
                        "score must be between 0 and 1."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nDocuments:\n{docs_block}",
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
            score_map = {
                int(item["id"]): float(item["score"])
                for item in parsed.get("scores", [])
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            score_map = {
                idx: candidates[idx].get("score", 0.0) for idx in range(len(candidates))
            }

        reranked = []
        for idx, candidate in enumerate(candidates):
            item = dict(candidate)
            item["score"] = score_map.get(idx, candidate.get("score", 0.0))
            item["retrieval_method"] = "hybrid+cross_encoder"
            reranked.append(item)
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(self, query: str, top_k: int = 6) -> list[dict]:
        fetch_k = max(top_k * 2, 8)
        dense = self.semantic_search(query, top_k=fetch_k)
        sparse = self.lexical_search(query, top_k=fetch_k)
        merged = self._rrf_merge([dense, sparse])[:fetch_k]

        if not merged:
            return []

        if self.config.reranker == "none":
            final = merged
        elif self.config.reranker == "rrf":
            final = merged
        elif self.config.reranker == "mmr":
            final = self._rerank_mmr(query, merged, top_k=top_k)
        else:
            final = self._rerank_cross_encoder(query, merged, top_k=top_k)

        filtered = [item for item in final if item["score"] >= self.config.score_threshold]
        return (filtered or final)[:top_k]

    @staticmethod
    def reorder_for_llm(chunks: list[dict]) -> list[dict]:
        if len(chunks) <= 2:
            return chunks
        front = chunks[::2]
        back = chunks[-1 if len(chunks) % 2 == 0 else -2 : 0 : -2]
        return front + back

    def _context_for_generation(self, chunks: list[dict]) -> str:
        blocks = []
        for idx, chunk in enumerate(chunks, start=1):
            meta = chunk.get("metadata", {})
            blocks.append(
                f"[Doc {idx}] source={meta.get('source', 'unknown')} | "
                f"path={meta.get('path', 'unknown')} | "
                f"type={meta.get('type', 'unknown')}\n"
                f"{chunk['content']}"
            )
        return "\n\n---\n\n".join(blocks)

    def rewrite_with_memory(self, question: str, history: list[dict]) -> str:
        if not history:
            return question

        recent = history[-6:]
        convo = []
        for turn in recent:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            convo.append(f"{role}: {content}")

        response = self._openai.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the last user question into a standalone query in Vietnamese. "
                        "Use conversation context only when needed. "
                        "Return only the rewritten query text."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Conversation:\n"
                        + "\n".join(convo)
                        + f"\n\nLast user question:\n{question}"
                    ),
                },
            ],
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten or question

    def search_context(self, query: str, top_k: int = 6) -> dict:
        """Tool: retrieval trên dataset, trả về context đã format + chunks."""
        sources = self.retrieve(query, top_k=top_k)
        if not sources:
            return {"context": "", "chunks": []}
        ordered = self.reorder_for_llm(sources)
        return {"context": self._context_for_generation(ordered), "chunks": ordered}

    def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        use_memory: bool = True,
        top_k: int = 6,
    ) -> dict:
        """
        Agentic answer: LLM tự quyết định có gọi tool `search_context` hay không.

        - Câu hỏi thuộc dataset → gọi tool, trả lời có citation.
        - Câu hỏi ngoài lĩnh vực → không gọi tool, trả lời từ chối.
        """
        chat_history = history or []
        retrieval_query = (
            self.rewrite_with_memory(question, chat_history) if use_memory else question
        )

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        collected_chunks: list[dict] = []
        used_search_tool = False

        for _ in range(MAX_TOOL_TURNS):
            response = self._openai.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=[SEARCH_CONTEXT_TOOL],
                tool_choice="auto",
                temperature=0.2,
                top_p=0.9,
            )
            message = response.choices[0].message
            tool_calls = message.tool_calls or []

            if not tool_calls:
                return {
                    "answer": message.content or "",
                    "sources": collected_chunks,
                    "retrieval_query": retrieval_query,
                    "used_search_tool": used_search_tool,
                }

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )

            for tool_call in tool_calls:
                if tool_call.function.name != "search_context":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": "Unknown tool.",
                        }
                    )
                    continue

                used_search_tool = True
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                search_query = args.get("query") or retrieval_query
                result = self.search_context(search_query, top_k=top_k)
                collected_chunks = result["chunks"] or collected_chunks
                tool_payload = result["context"] or (
                    "Không tìm thấy ngữ cảnh phù hợp trong dataset."
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_payload,
                    }
                )

        final = self._openai.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
            top_p=0.9,
        )
        return {
            "answer": final.choices[0].message.content or "",
            "sources": collected_chunks,
            "retrieval_query": retrieval_query,
            "used_search_tool": used_search_tool,
        }
