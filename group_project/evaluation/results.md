# RAG Evaluation Results

## Framework su dung

> RAGAS

---

## Muc tieu

- Danh gia RAG pipeline tren bo QA tu tao >= 15 cau hoi
- Do 4 metric:
  - faithfulness
  - answer_relevancy
  - context_recall
  - context_precision
- So sanh A/B giua:
  - Config A: hybrid retrieval + MMR reranking
  - Config B: dense-only retrieval

---

## Cach tao ket qua

Chay:

```bash
python group_project/evaluation/eval_pipeline.py
```

Script se:

1. Load `golden_dataset.json`
2. Khoi tao pipeline A/B
3. Chay RAGAS evaluation
4. Ghi de file nay bang bang diem va phan tich moi nhat

---

## Trang thai hien tai

File nay la template/report dau vao trong repo. Sau khi chay `eval_pipeline.py`,
noi dung se duoc cap nhat thanh:

- Bang diem tong hop
- So sanh A/B
- 3 truong hop worst performers
- De xuat cai tien

---

## Ghi chu

- Golden dataset duoc tao thu cong dua tren van ban phap luat va cac bai bao trong `data/standardized/`
- Evaluation pipeline su dung `local_numpy` de tranh phu thuoc vao Weaviate khi danh gia nhom
- Neu muon chay RAGAS day du, can cai them cac goi lien quan va cung cap `OPENAI_API_KEY`
