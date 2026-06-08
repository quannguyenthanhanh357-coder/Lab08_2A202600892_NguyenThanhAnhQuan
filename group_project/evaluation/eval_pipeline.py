"""
RAG Evaluation Pipeline.

Su dung DeepEval / RAGAS / TruLens de danh gia chat luong RAG pipeline.
Chon 1 framework va implement day du.

Yeu cau:
    1. Load golden_dataset.json (>=15 Q&A pairs)
    2. Chay RAG pipeline tren tung question
    3. Evaluate voi 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sanh A/B it nhat 2 configs
    5. Export results ra results.md
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from statistics import mean

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "group_project") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "group_project"))

from rag_chatbot import ConfigurableRAG, RAGConfig


def load_golden_dataset() -> list[dict]:
    """Load golden dataset tu JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class EvalRAGPipeline:
    """Thin wrapper de eval theo 2 config retrieval khac nhau."""

    def __init__(self, name: str, retrieval_mode: str, reranker: str):
        self.name = name
        self.retrieval_mode = retrieval_mode
        self.rag = ConfigurableRAG(
            RAGConfig(
                splitter="recursive",
                embedding_model="text-embedding-3-small",
                reranker=reranker,
                vector_store="local_numpy",
                chunk_size=900,
                chunk_overlap=120,
                score_threshold=0.25,
            )
        )

    def prepare(self):
        self.rag.index_documents(force_rebuild=False)

    def _retrieve(self, question: str, top_k: int = 6) -> list[dict]:
        if self.retrieval_mode == "dense_only":
            results = self.rag.semantic_search(question, top_k=top_k)
            for item in results:
                item["retrieval_method"] = "semantic_dense_only"
            return results
        return self.rag.retrieve(question, top_k=top_k)

    def generate_with_citation(self, question: str) -> dict:
        sources = self._retrieve(question, top_k=6)
        ordered = self.rag.reorder_for_llm(sources)
        context = self.rag._context_for_generation(ordered) if ordered else ""

        fallback_answer = "Toi khong the xac minh thong tin nay tu nguon hien co."
        if ordered:
            source = ordered[0].get("metadata", {}).get("source", "Nguon hien co")
            excerpt = " ".join(ordered[0].get("content", "").split())[:320].strip()
            if excerpt:
                fallback_answer = f"{excerpt} [{source}]"

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key or not context:
            return {
                "answer": fallback_answer,
                "sources": ordered,
                "retrieval_source": ordered[0].get("retrieval_method", "none") if ordered else "none",
            }

        try:
            response = self.rag._openai.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                top_p=0.9,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tra loi bang tieng Viet. Moi nhan dinh phai co citation "
                            "dang [ten_nguon]. Neu context khong du, noi ro khong the xac minh."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}",
                    },
                ],
            )
            answer = response.choices[0].message.content or fallback_answer
        except Exception:
            answer = fallback_answer

        return {
            "answer": answer,
            "sources": ordered,
            "retrieval_source": ordered[0].get("retrieval_method", "none") if ordered else "none",
        }


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline su dung DeepEval.

    pip install deepeval
    """
    raise NotImplementedError("Requirement 2 is implemented with RAGAS, not DeepEval.")


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def _score_config(config_name: str, rag_pipeline: EvalRAGPipeline, golden_dataset: list[dict]) -> dict:
    from datasets import Dataset
    from langchain_openai import OpenAIEmbeddings

    try:
        from ragas import evaluate
    except ModuleNotFoundError as exc:
        if exc.name != "langchain_community.chat_models.vertexai":
            raise

        # Compatibility shim for current langchain-community build used in this environment.
        shim = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:  # pragma: no cover - import shim only
            pass

        shim.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = shim
        from ragas import evaluate

    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    raw_outputs = []

    for item in golden_dataset:
        result = rag_pipeline.generate_with_citation(item["question"])
        contexts = [chunk["content"] for chunk in result["sources"]]
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(item["expected_answer"])
        raw_outputs.append(
            {
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "expected_context": item.get("expected_context", ""),
                "answer": result["answer"],
                "contexts": contexts,
                "retrieval_source": result["retrieval_source"],
            }
        )

    dataset = Dataset.from_dict(eval_data)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        embeddings=embeddings,
    )
    frame = result.to_pandas()
    rows = frame.to_dict(orient="records")
    metric_names = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

    aggregates = {
        metric: float(mean([row[metric] for row in rows if row.get(metric) is not None]))
        for metric in metric_names
    }

    for row, raw in zip(rows, raw_outputs):
        raw.update(row)

    return {
        "config_name": config_name,
        "aggregates": aggregates,
        "rows": raw_outputs,
    }


def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline su dung RAGAS.

    pip install ragas
    """
    rag_pipeline.prepare()
    return _score_config(rag_pipeline.name, rag_pipeline, golden_dataset)


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline su dung TruLens.

    pip install trulens
    """
    raise NotImplementedError("Requirement 2 is implemented with RAGAS, not TruLens.")


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(rag_pipeline, golden_dataset: list[dict]):
    """
    So sanh A/B giua it nhat 2 configs.

    Goi y configs de so sanh:
    - Config A: hybrid search + reranking
    - Config B: dense-only (khong reranking)
    - Config C: hybrid search + PageIndex fallback
    """
    configs = {
        "Config A (hybrid + mmr)": EvalRAGPipeline(
            name="Config A (hybrid + mmr)",
            retrieval_mode="hybrid",
            reranker="mmr",
        ),
        "Config B (dense-only)": EvalRAGPipeline(
            name="Config B (dense-only)",
            retrieval_mode="dense_only",
            reranker="none",
        ),
    }

    results = {}
    for config_name, pipeline in configs.items():
        pipeline.prepare()
        results[config_name] = _score_config(config_name, pipeline, golden_dataset)

    return results


# =============================================================================
# Export Results
# =============================================================================

def _failure_stage(row: dict) -> tuple[str, str]:
    faithfulness = row.get("faithfulness", 0.0) or 0.0
    answer_relevancy = row.get("answer_relevancy", 0.0) or 0.0
    context_recall = row.get("context_recall", 0.0) or 0.0
    context_precision = row.get("context_precision", 0.0) or 0.0

    metric_map = {
        "retrieval_recall": context_recall,
        "retrieval_precision": context_precision,
        "generation_grounding": faithfulness,
        "answer_quality": answer_relevancy,
    }
    worst_stage = min(metric_map, key=metric_map.get)

    if worst_stage == "retrieval_recall":
        return "Retrieval Recall", "Retriever chua lay du evidence can thiet."
    if worst_stage == "retrieval_precision":
        return "Retrieval Precision", "Retriever lay nhieu context chua thuc su huu ich."
    if worst_stage == "generation_grounding":
        return "Generation Faithfulness", "Cau tra loi chua bam sat context da truy xuat."
    return "Answer Relevance", "Cau tra loi chua tap trung dung vao cau hoi."


def export_results(results: dict, comparison: dict):
    """Export evaluation results to results.md"""
    config_a = comparison["Config A (hybrid + mmr)"]["aggregates"]
    config_b = comparison["Config B (dense-only)"]["aggregates"]
    primary_rows = comparison["Config A (hybrid + mmr)"]["rows"]

    scored_rows = []
    for row in primary_rows:
        metrics = [
            row.get("faithfulness", 0.0) or 0.0,
            row.get("answer_relevancy", 0.0) or 0.0,
            row.get("context_recall", 0.0) or 0.0,
            row.get("context_precision", 0.0) or 0.0,
        ]
        row["_average"] = mean(metrics)
        scored_rows.append(row)

    worst_three = sorted(scored_rows, key=lambda item: item["_average"])[:3]

    content = "# RAG Evaluation Results\n\n"
    content += "## Framework su dung\n\n"
    content += "> RAGAS\n\n---\n\n"
    content += "## Overall Scores\n\n"
    content += "| Metric | Config A (hybrid + mmr) | Config B (dense-only) | Delta |\n"
    content += "|--------|-------------------------|-----------------------|-------|\n"

    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        delta = config_a[metric] - config_b[metric]
        content += (
            f"| {metric} | {config_a[metric]:.4f} | {config_b[metric]:.4f} | {delta:+.4f} |\n"
        )

    avg_a = mean(config_a.values())
    avg_b = mean(config_b.values())
    content += f"| **average** | {avg_a:.4f} | {avg_b:.4f} | {avg_a - avg_b:+.4f} |\n"
    content += "\n---\n\n"

    content += "## A/B Comparison Analysis\n\n"
    content += "**Config A:** hybrid retrieval (semantic + BM25) ket hop MMR reranking.\n\n"
    content += "**Config B:** dense-only retrieval, khong reranking.\n\n"
    if avg_a >= avg_b:
        content += (
            "**Ket luan:** Config A tot hon hoac on dinh hon Config B. "
            "Hybrid retrieval giup tang recall, trong khi MMR giam trung lap va giu ngu canh da dang.\n\n"
        )
    else:
        content += (
            "**Ket luan:** Config B tot hon trong lan chay nay. "
            "Can xem lai threshold va tham so reranking cua Config A de tranh loai bo qua nhieu ket qua tot.\n\n"
        )
    content += "---\n\n"

    content += "## Worst Performers (Bottom 3)\n\n"
    content += "| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |\n"
    content += "|---|----------|-------------|-----------|--------|---------------|------------|\n"
    for idx, row in enumerate(worst_three, 1):
        stage, cause = _failure_stage(row)
        content += (
            f"| {idx} | {row['question']} | {row.get('faithfulness', 0.0):.4f} | "
            f"{row.get('answer_relevancy', 0.0):.4f} | {row.get('context_recall', 0.0):.4f} | "
            f"{stage} | {cause} |\n"
        )

    content += "\n---\n\n"
    content += "## Recommendations\n\n"
    content += "### Cai tien 1\n"
    content += "**Action:** Tang chat luong golden dataset bang cach them expected_context chi tiet hon theo dieu/khoan hoac ten bai bao.\n"
    content += "**Expected impact:** Giup danh gia context recall va context precision chinh xac hon.\n\n"
    content += "### Cai tien 2\n"
    content += "**Action:** Thu nghiem them config rerank cross-encoder de so sanh voi MMR tren tap query tin tuc.\n"
    content += "**Expected impact:** Co the tang answer relevancy cho cac cau hoi can phan biet su kien va nhan vat gan nhau.\n\n"
    content += "### Cai tien 3\n"
    content += "**Action:** Dieu chinh chunk size/overlap rieng cho legal va news thay vi dung mot cau hinh chung.\n"
    content += "**Expected impact:** Tang context recall voi van ban luat dai va giam nhieu chunk thua o bai bao ngan.\n"

    RESULTS_PATH.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    pipeline = EvalRAGPipeline(
        name="Config A (hybrid + mmr)",
        retrieval_mode="hybrid",
        reranker="mmr",
    )
    results = evaluate_with_ragas(pipeline, golden_dataset)
    comparison = compare_configs(pipeline, golden_dataset)
    export_results(results, comparison)
    print(f"Exported evaluation report to: {RESULTS_PATH}")
