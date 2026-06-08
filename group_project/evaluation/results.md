# RAG Evaluation Results

## Framework su dung

> RAGAS

---

## Overall Scores

| Metric | Config A (hybrid + mmr) | Config B (dense-only) | Delta |
|--------|-------------------------|-----------------------|-------|
| faithfulness | 0.8222 | 0.7883 | +0.0339 |
| answer_relevancy | 0.4117 | 0.3723 | +0.0394 |
| context_recall | 0.5556 | 0.4667 | +0.0889 |
| context_precision | 0.6511 | 0.6195 | +0.0316 |
| **average** | 0.6102 | 0.5617 | +0.0485 |

---

## A/B Comparison Analysis

**Config A:** hybrid retrieval (semantic + BM25) ket hop MMR reranking.

**Config B:** dense-only retrieval, khong reranking.

**Ket luan:** Config A tot hon hoac on dinh hon Config B. Hybrid retrieval giup tang recall, trong khi MMR giam trung lap va giu ngu canh da dang.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Ma túy "nước vui" hay "bùa lưỡi" thuộc nhóm chất nào và có tác hại ra sao? | 0.5000 | 0.3120 | 0.0000 | Retrieval Recall | Retriever chua lay du evidence can thiet. |
| 2 | Người bị quản lý sau cai nghiện ma túy tại nơi cư trú có thời hạn bao lâu? | 0.6667 | 0.0000 | 0.0000 | Retrieval Recall | Retriever chua lay du evidence can thiet. |
| 3 | Các cơ sở kinh doanh dịch vụ (như quán bar, karaoke) để xảy ra tình trạng sử dụng ma túy bị xử lý thế nào? | 1.0000 | 0.4833 | 0.0000 | Retrieval Recall | Retriever chua lay du evidence can thiet. |

---

## Recommendations

### Cai tien 1
**Action:** Tang chat luong golden dataset bang cach them expected_context chi tiet hon theo dieu/khoan hoac ten bai bao.
**Expected impact:** Giup danh gia context recall va context precision chinh xac hon.

### Cai tien 2
**Action:** Thu nghiem them config rerank cross-encoder de so sanh voi MMR tren tap query tin tuc.
**Expected impact:** Co the tang answer relevancy cho cac cau hoi can phan biet su kien va nhan vat gan nhau.

### Cai tien 3
**Action:** Dieu chinh chunk size/overlap rieng cho legal va news thay vi dung mot cau hinh chung.
**Expected impact:** Tang context recall voi van ban luat dai va giam nhieu chunk thua o bai bao ngan.
