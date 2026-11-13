# Báo cáo cá nhân — Day 6

**Họ tên:** Nguyen Vo Thai Bao · **MSSV:** 2A202603638 · **Lớp:** L3A

> Báo cáo được tạo từ lần chạy thật. Audit hiện trạng: Reviewer-labelled audit; author verification recorded in CSV.

## Dữ liệu và pipeline
- Chủ đề: NASA Systems Engineering Handbook, Revision 2 (tiếng Anh).
- Nguồn: NASA NTRS [bản ghi](https://ntrs.nasa.gov/citations/20170001761), [PDF](https://ntrs.nasa.gov/api/citations/20170001761/downloads/20170001761.pdf); hồ sơ ghi “Work of the U.S. Government. Public Use Permitted”. Trích văn bản, không trích ảnh/figure. SHA-256: `3153ae2e53e29452d5997efafe280a5f05cd21b43a047e988a17e1dd5207a38e`.
- Quy tắc trang: trang PDF vật lý 14–324, bỏ trang trống/figure-only/ít hơn 400 ký tự và mục Reserved; giữ `301` trang nội dung hữu ích. Manifest: `data/manifest.json`; audit từng trang: `results/ingestion_page_audit.csv`.
- Chuẩn hóa Unicode NFKC, bỏ header/footer, nối dòng ngắt, giữ ranh giới đoạn; metadata gồm `document_id`, `source`, trang PDF vật lý và in, chapter, section, language, document_type. Trang thưa được ghi lý do loại.
- Embedding: `FastEmbed / ONNX Runtime CPU`, `sentence-transformers/all-MiniLM-L6-v2`, 384 chiều, CPU, chi phí API $0. Vector store: Qdrant 1.20.0, cosine, persistent path `results/qdrant`. Runtime: Python 3.12.10; fastembed 0.9.1, onnxruntime 1.31.1, fastapi 0.162.5, uvicorn 0.54.0, pypdf 6.21.4.
- Chế độ tìm kiếm thực đo: cosine full scan chính xác với payload filter. Collection hiện tại có A=894, B=902, C=902 points; Qdrant báo indexed vectors 0, 0, 0 và ngưỡng full scan 10.000. Vectors được lưu bền vững.
- Ghi chú môi trường và phương án dự phòng: `results/run_environment_notes.md`.

## Cấu hình và benchmark
- A: fixed windows 1000 ký tự, overlap 120; B: nhóm paragraph tối đa 1000; C: cùng chunks và vectors như B, chỉ bật lọc `chapter` nếu benchmark đưa filter.
- Benchmark cố định: 20 câu (15 dev / 5 test), 5 câu có filter. Gold evidence được kiểm tra tự động tồn tại đúng trang nguồn trước khi chạy.
- Khi nạp lại, code xóa collection cùng tên rồi tạo lại trước khi ghi points; đây là cơ chế ngăn tích lũy dữ liệu cũ. Thời gian C trong bảng là thời gian tăng thêm vì tái dùng vector B.
- Quy tắc chọn cấu hình trước khi xem test: ưu tiên Evidence Hit@3 trên dev; nếu hòa, chọn Hit@1 trên dev; nếu vẫn hòa, chọn ít tài nguyên/ít chunk hơn. Test chỉ dùng để báo cáo.

## Kết quả truy xuất
| Strategy | Split | Hit@1 | Evidence Hit@3 | Chunks | Ingest (s) | Cost (USD) |
|---|---|---:|---:|---:|---:|---:|
| A | dev | 2/15 | 5/15 | 894 | 48.254 | 0 |
| A | test | 5/5 | 5/5 | 894 | 48.254 | 0 |
| B | dev | 3/15 | 5/15 | 902 | 53.599 | 0 |
| B | test | 5/5 | 5/5 | 902 | 53.599 | 0 |
| C | dev | 3/15 | 6/15 | 902 | 1.14 | 0 |
| C | test | 5/5 | 5/5 | 902 | 1.14 | 0 |

Phân bố độ dài chunk:
| Strategy | Chunks | Total chars | Min | Median | Mean | Max | <100 chars |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 894 | 757950 | 127 | 993.0 | 847.82 | 999 | 0 |
| B | 902 | 729322 | 10 | 992.0 | 808.56 | 1000 | 14 |
| C | 902 | 729322 | 10 | 992.0 | 808.56 | 1000 | 14 |

Evidence Hit@3 tự động là phép so khớp evidence đã chuẩn hóa với text trong chunk ở đúng nguồn/trang; đây là proxy truy xuất, không phải độ đúng của đáp án. Nhãn grounding bên dưới là phần đọc top-3 riêng.

Trace đầy đủ: `results/top3_traces.jsonl`; tóm tắt: `results/retrieval_summary.json`; lần chạy: `results/run_summary.json`.

### Lỗi top-3 trên test
- Không có miss top-3 trên câu test trong lần chạy này.

## Audit theo nhãn reviewer trong phiếu
- Coherence trung bình trên 10 trang đầu, thang 0–2: A: 0.9 (n=10); B: 1 (n=10); C: 1 (n=10).
- Metadata: 45/45 bản ghi có document ID, source, page và chapter đúng theo manifest. C tuân thủ filter ở 3/3 câu test có filter. Filter chỉ áp dụng với câu C có filter input; chapter giúp giới hạn kết quả theo phần handbook, nhưng metadata đúng không đảm bảo câu trả lời đủ.
- Grounding top-3 trên 5 câu test: A: 5 điểm 2, 0 điểm 1, 0 điểm 0; trung bình 2/2; B: 5 điểm 2, 0 điểm 1, 0 điểm 0; trung bình 2/2; C: 5 điểm 2, 0 điểm 1, 0 điểm 0; trung bình 2/2. Đây là đánh giá đủ bằng chứng để trả lời và truy nguồn, không phải độ chính xác của câu trả lời LLM.
Reviewer và lý do được lưu ở từng dòng trong CSV. Summary được tính lại từ các nhãn đó; người nộp chịu trách nhiệm xác nhận chúng phản ánh đúng phần đã đọc và tiêu chí trong GUIDE.

## Phân tích và lỗi quan sát được
- A → B: đổi fixed window sang paragraph grouping; Hit@1 dev tăng từ 2/15 lên 3/15, Evidence Hit@3 dev giữ 5/15, số chunk tăng từ 894 lên 902.
- B → C: giữ nguyên chunk và vector, chỉ áp dụng filter khi có đầu vào. Evidence Hit@3 dev tăng từ 5/15 lên 6/15; khác biệt là q13 được lọc vào chapter 4.0 và có evidence trong top-3. Trên test, cả ba cấu hình đều 5/5.
- Chọn C theo tiêu chí đã chốt trước: dev Evidence Hit@3 cao nhất (6/15 so với A 5/15 và B 5/15). Lợi thế chỉ là một câu; test 5 câu quá nhỏ để suy rộng. C dùng lại embedding của B, nên thời gian ingest C là thời gian tăng thêm, không phải build độc lập.

Ví dụ miss top-3 ở dev của C:
- q01: câu hỏi về các thành phần của system trả về trang 285, 273, 100; top-3 không có trang nguồn 16.
- q03: top-3 có một đoạn ở trang 20 bắt đầu sau định nghĩa cần tìm; đoạn nói về lặp qua các cấp sản phẩm nhưng bỏ phần phân biệt iterative trên cùng sản phẩm với recursive qua các lớp.

Coherence thấp chủ yếu do chunk cắt ở ranh giới ký tự; trang 21 bị dẹt bảng khiến quan hệ cột khó hiểu. B/C còn 14 chunk dưới 100 ký tự; một chunk dài 10 ký tự chỉ là mẩu ô bảng `(~$10-50M)`. Metadata `chapter` vừa giúp filter vừa truy ngược vị trí, nhưng filter không khắc phục câu hỏi thiếu từ khóa/diễn đạt khác.

Ngưỡng 400 ký tự cũng loại một số trang có đoạn văn ngắn (ví dụ PDF trang 15, 51, 115 và 281); chúng được ghi ở page audit nhưng không được truy xuất. 301 trang còn lại vẫn vượt xa mức tối thiểu 30.


## Giới hạn
- Chỉ một handbook và 5 câu test; không suy rộng kết quả. Dev/test đều lấy từ cùng một handbook, nên chủ đề và thuật ngữ có thể chồng lấn. A/B/C dùng embedding, corpus, metric và câu hỏi giống nhau; B/C khác ở metadata filter. C tái sử dụng vectors của B vì chunks giống hệt nhau. Thời gian là đo trên máy chạy, không đại diện phần cứng khác.
- Trạng thái audit: Reviewer-labelled audit; author verification recorded in CSV.
