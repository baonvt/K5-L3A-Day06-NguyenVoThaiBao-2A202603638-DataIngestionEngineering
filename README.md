# Day 06 · Data Ingestion Engineering

Lab cá nhân: biến tài liệu thành kho tri thức có thể truy xuất, rồi đo ảnh hưởng của cách chuẩn bị dữ liệu.

## Bản chạy trong repo này

Pipeline đã được hiện thực với **NASA Systems Engineering Handbook Rev 2**, embedding local thật (FastEmbed/ONNX) và Qdrant có lưu bền vững. Không cần API key hoặc Docker. PDF, manifest, bộ câu hỏi, trace và báo cáo chạy nằm trong repo.

```powershell
cd starter
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.run
python -m src.server
```

Lần đầu, model embedding và binary Qdrant chính thức sẽ được tải về máy. Khi server hiện `Uvicorn running`, mở [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), health check tại `/healthz`, hoặc chạy một benchmark query đã viết sẵn tại `/simulate/q17?strategy=C`. Server dùng Qdrant và model thật; lỗi backend được trả về, không fallback sang mock. Với collection hiện có, Qdrant thực hiện cosine full scan chính xác (894–902 points, dưới ngưỡng 10.000; chưa có HNSW-indexed vectors).

## Bằng chứng và bài nộp

- [Hướng dẫn chạy và API](starter/README.md)
- [Corpus và manifest nguồn/quyền/trang](data/manifest.json)
- [20 câu hỏi, split 15 dev / 5 test và gold evidence](benchmark/questions.json)
- [Trace top-3 và chỉ số retrieval](results/top3_traces.jsonl), [bảng kết quả](results/benchmark_results.csv), [tóm tắt lần chạy](results/run_summary.json)
- [Bản ghi kiểm tra serve thật](results/serve_demo.json), [audit và nhãn reviewer](results/manual_audit_template.csv), [tóm tắt audit](results/manual_audit_summary.json), [báo cáo cá nhân](report/individual/2A202603638_NguyenVoThaiBao.md)
- [Ghi chú môi trường và phương án dự phòng](results/run_environment_notes.md)

Audit CSV ghi nhãn, lý do và reviewer cho 90 mẫu. Khi cập nhật audit, từ `starter` chạy `python -m src.refresh_audit` để đồng bộ summary/báo cáo mà không tính lại embedding. Trước khi nộp, hoàn thành checklist trong [SUBMISSION.md](SUBMISSION.md). Các định nghĩa và tiêu chí xem [GUIDE.md](GUIDE.md), [BENCHMARK.md](BENCHMARK.md), [RUBRIC.md](RUBRIC.md) và [RULES.md](RULES.md).
