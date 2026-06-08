"""Streamlit app for the group RAG chatbot deliverable."""

from __future__ import annotations

import streamlit as st

from rag_chatbot import (
    EMBEDDING_OPTIONS,
    RERANK_OPTIONS,
    SPLITTER_OPTIONS,
    VECTOR_STORE_OPTIONS,
    ConfigurableRAG,
    RAGConfig,
)


def _init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None
    if "active_config_key" not in st.session_state:
        st.session_state.active_config_key = ""


def _build_config(
    splitter: str,
    embedding_model: str,
    reranker: str,
    vector_store: str,
    chunk_size: int,
    chunk_overlap: int,
    score_threshold: float,
) -> RAGConfig:
    return RAGConfig(
        splitter=splitter,
        embedding_model=embedding_model,
        reranker=reranker,
        vector_store=vector_store,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        score_threshold=score_threshold,
    )


def _config_key(config: RAGConfig) -> str:
    return (
        f"{config.splitter}|{config.embedding_model}|{config.reranker}|{config.vector_store}|"
        f"{config.chunk_size}|{config.chunk_overlap}|{config.score_threshold}"
    )


def _render_sources():
    st.subheader("Source documents đã dùng")
    sources = st.session_state.last_sources
    if not sources:
        st.info("Chưa có source nào. Hãy gửi câu hỏi để hệ thống retrieval dữ liệu.")
        return

    for idx, source in enumerate(sources, start=1):
        meta = source.get("metadata", {})
        title = (
            f"#{idx} | {meta.get('source', 'unknown')} | "
            f"score={source.get('score', 0.0):.3f} | "
            f"{source.get('retrieval_method', 'hybrid')}"
        )
        with st.expander(title):
            st.markdown(
                f"- **Path:** `{meta.get('path', 'unknown')}`\n"
                f"- **Type:** `{meta.get('type', 'unknown')}`\n"
                f"- **Chunk index:** `{meta.get('chunk_index', 0)}`"
            )
            st.write(source.get("content", ""))


def main():
    st.set_page_config(page_title="DrugLaw RAG Chatbot", page_icon=":speech_balloon:")
    _init_state()

    st.title("RAG Chatbot — Pháp luật ma tuý & tin tức nghệ sĩ")
    st.caption("Yêu cầu 1: Streamlit + citation + memory + source documents")

    with st.sidebar:
        st.header("Pipeline Config")
        splitter = st.selectbox(
            "Chunking splitter",
            options=SPLITTER_OPTIONS,
            index=0,
            help="Lựa chọn strategy tách chunk trước khi index.",
        )
        embedding_model = st.selectbox(
            "Embedding model",
            options=list(EMBEDDING_OPTIONS.keys()),
            index=0,
        )
        vector_store = st.selectbox(
            "Vector Store",
            options=VECTOR_STORE_OPTIONS,
            index=0,
            help="Dùng local_numpy để tránh giới hạn collection trên Weaviate free tier.",
        )
        st.caption(f"Đang dùng: `{vector_store}`")

        reranker = st.selectbox(
            "Reranking",
            options=RERANK_OPTIONS,
            index=3,
            help="none/rrf/mmr/cross_encoder",
        )
        chunk_size = st.slider("Chunk size", min_value=400, max_value=1800, value=900)
        chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=300, value=120)
        score_threshold = st.slider(
            "Score threshold",
            min_value=0.0,
            max_value=0.8,
            value=0.25,
            step=0.05,
        )
        top_k = st.slider("Top-K retrieval", min_value=3, max_value=10, value=6)
        use_memory = st.checkbox("Dùng conversation memory cho follow-up", value=True)

        config = _build_config(
            splitter=splitter,
            embedding_model=embedding_model,
            reranker=reranker,
            vector_store=vector_store,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            score_threshold=score_threshold,
        )
        config_key = _config_key(config)

        if st.button("Build/Rebuild index", type="primary"):
            with st.spinner(f"Đang load/chunk/embed/index lên {vector_store}..."):
                engine = ConfigurableRAG(config)
                stats = engine.index_documents(force_rebuild=True)
                st.session_state.rag_engine = engine
                st.session_state.active_config_key = config_key
            st.success(
                "Index hoàn tất: "
                f"{stats['documents']} docs, {stats['chunks']} chunks, "
                f"collection={stats['collection']}"
            )

        if st.button("Xoá hội thoại hiện tại"):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.rerun()

    if (
        st.session_state.rag_engine is None
        or st.session_state.active_config_key != config_key
    ):
        st.info(
            "Hãy bấm `Build/Rebuild Weaviate index` ở sidebar sau khi chọn config "
            "để bắt đầu chat."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Đặt câu hỏi về luật ma tuý hoặc tin tức liên quan...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        if st.session_state.rag_engine is None:
            warning = "Chưa có index. Vui lòng Build/Rebuild index trước khi chat."
            st.session_state.messages.append({"role": "assistant", "content": warning})
            with st.chat_message("assistant"):
                st.markdown(warning)
            _render_sources()
            return

        with st.chat_message("assistant"):
            with st.spinner("Đang retrieve + generate câu trả lời có citation..."):
                result = st.session_state.rag_engine.answer(
                    question=question,
                    history=st.session_state.messages[:-1],
                    use_memory=use_memory,
                    top_k=top_k,
                )
            st.markdown(result["answer"])
            st.caption(f"Retrieval query: `{result['retrieval_query']}`")

        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"]}
        )
        st.session_state.last_sources = result["sources"]

    _render_sources()


if __name__ == "__main__":
    main()
