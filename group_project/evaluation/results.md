# RAG Evaluation Results

## Framework su dung

> RAGAS

---

## Overall Scores

| Metric | Config A (hybrid + mmr) | Config B (dense-only) | Delta |
|--------|-------------------------|-----------------------|-------|
| faithfulness | 0.8095 | 0.8700 | -0.0605 |
| answer_relevancy | 0.3916 | 0.3884 | +0.0031 |
| context_recall | 0.6111 | 0.6333 | -0.0222 |
| context_precision | 0.7732 | 0.7313 | +0.0418 |
| **average** | 0.6463 | 0.6558 | -0.0094 |

---

## A/B Comparison Analysis

**Config A:** hybrid retrieval (semantic + BM25) ket hop MMR reranking.

**Config B:** dense-only retrieval, khong reranking.

**Ket luan:** Config B tot hon trong lan chay nay. Can xem lai threshold va tham so reranking cua Config A de tranh loai bo qua nhieu ket qua tot.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Người bị quản lý sau cai nghiện ma túy tại nơi cư trú có thời hạn bao lâu? | 0.7500 | 0.0000 | 0.0000 | Retrieval Recall | Retriever chua lay du evidence can thiet. |
| 2 | Tội tổ chức sử dụng trái phép chất ma túy có mức phạt cao nhất là bao nhiêu? | 0.5000 | 0.3926 | 0.0000 | Retrieval Recall | Retriever chua lay du evidence can thiet. |
| 3 | Cơ quan nào có thẩm quyền ra quyết định đưa người nghiện ma túy vào cơ sở cai nghiện bắt buộc? | 0.6667 | 0.4469 | 0.0000 | Retrieval Recall | Retriever chua lay du evidence can thiet. |

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
