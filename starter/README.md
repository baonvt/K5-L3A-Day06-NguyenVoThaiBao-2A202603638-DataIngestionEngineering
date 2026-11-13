# Pipeline local · FastEmbed + Qdrant

Starter này đã được hoàn thiện để chạy lab trên máy cá nhân. Nó đọc PDF, giữ nguồn/trang/chapter/section, tạo embedding bằng model ONNX thật, ghi index Qdrant bền vững trên đĩa, chạy ba chiến lược và lưu trace top-3. Không cần PostgreSQL, Docker hoặc API key. Qdrant chạy local trên `127.0.0.1:6333`; binary chính thức v1.20.0 được tự tải lần đầu. Không có vector giả; nếu model, index hay nguồn bị lỗi, chương trình dừng với lỗi rõ ràng.

## Điều kiện và cài đặt

- Python 3.11–3.14.
- Mạng ở lần cài dependencies và lần tải model đầu tiên. Model `sentence-transformers/all-MiniLM-L6-v2` khoảng 90 MB, tiếng Anh, chạy CPU.
- Windows PowerShell:

```powershell
cd starter
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m src.run
```

macOS/Linux: the Python commands are the same, but start a local Qdrant v1.20.0 server at `127.0.0.1:6333` first; automatic Qdrant binary download is currently implemented for Windows only. If Qdrant is hosted elsewhere, set `QDRANT_URL` in `.env` to its HTTP URL.

```bash
cd starter
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
python -m src.run
```

Lệnh đầu tiên tải model FastEmbed về cache máy. Những lần sau model và PDF được dùng từ local. Có thể đổi các đường dẫn, model và kích thước chunk trong `starter/.env`; không cần key. Giữ nguyên cấu hình model, corpus và benchmark khi so sánh A/B/C.

## Chạy API demo

Sau khi `python -m src.run` tạo ba collection, mở terminal thứ hai và chạy:

```powershell
cd starter
.venv\Scripts\Activate.ps1
python -m src.server
```

API bind mặc định `127.0.0.1:8000`:

- `GET /healthz` kiểm tra ba index và số vector.
- `GET /simulate/q17?strategy=C` chạy câu benchmark `q17`, chỉ truyền câu hỏi và filter metadata đã cung cấp; đáp án/gold không đi vào truy vấn.
- `POST /search` với body `{"question":"What does product integration do?","strategy":"C","filters":{"chapter":"5.0"},"top_k":3}` chạy truy xuất trực tiếp.
- `GET /docs` mở giao diện thử API.

Chiến lược A/B là baseline không filter; API từ chối filter nếu chọn A hoặc B. C dùng filter chỉ khi có filter input. Dừng server bằng `Ctrl+C`.

## Corpus và quyền sử dụng

Corpus lấy từ [NASA NTRS record 20170001761](https://ntrs.nasa.gov/citations/20170001761); [PDF Rev 2](https://ntrs.nasa.gov/api/citations/20170001761/downloads/20170001761.pdf) được lưu ở `../data/nasa_systems_engineering_handbook_rev2.pdf`. NTRS ghi đây là tác phẩm của Chính phủ Hoa Kỳ và cho phép sử dụng công khai. Pipeline chỉ trích văn bản, không trích ảnh/figure. Manifest ghi URL, dấu SHA-256, quy tắc trang và metadata từng trang.

Corpus dùng trang PDF vật lý 14–324. Trang dưới 400 ký tự, trang trống/figure-only và phần Reserved được ghi lý do trong `../results/ingestion_page_audit.csv` rồi bỏ khỏi corpus. References bắt đầu từ trang vật lý 325 và không được tính. Số trang hữu ích thực tế được ghi trong `../data/manifest.json` sau khi chạy.

## Cấu hình thí nghiệm

| Chiến lược | Chunking | Truy xuất |
|---|---|---|
| A | Cửa sổ 1.000 ký tự, overlap 120 | Không metadata filter |
| B | Gom paragraph, tối đa 1.000 ký tự | Không metadata filter |
| C | Chunk và vector giống B | Filter theo `chapter` khi câu hỏi có filter input |

Cùng model, corpus, metric cosine và Qdrant store cho cả ba. Qdrant lưu dữ liệu trong `../results/qdrant/storage` và bind chỉ trên `127.0.0.1`. Trong lần chạy đã ghi nhận, mỗi collection có 894–902 points, dưới `full_scan_threshold=10000`, và `indexed_vectors_count=0`; vì vậy Qdrant dùng cosine full scan chính xác, không phải HNSW approximate. Khi collection vượt ngưỡng, Qdrant có thể xây HNSW. Payload filter theo chapter được áp dụng trong truy vấn C. Nạp lại sẽ xóa và tạo lại collection theo strategy nên không giữ chunk cũ hay nhân đôi; embeddings của B được tái sử dụng ở C vì văn bản chunk giống hệt.

## Artifacts

Sau `python -m src.run`:

- `../data/manifest.json` — nguồn, quyền, hash và trang/metadata.
- `../results/ingestion_page_audit.csv` — số ký tự và lý do giữ/bỏ từng trang.
- `../results/chunk_statistics.json` — số lượng và phân bố độ dài.
- `../results/benchmark_results.csv` và `../results/retrieval_summary.json` — Hit@1 và Evidence Hit@3 tách dev/test.
- `../results/top3_traces.jsonl` — câu hỏi, gold evidence, filter, top-3 text/score/metadata và hit tự động.
- `../results/manual_audit_template.csv` — coherence, metadata accuracy và grounding; nhãn ban đầu do Codex hỗ trợ, cần reviewer đọc và xác nhận. Nhãn được giữ qua lần chạy lại nếu mẫu không đổi.
- `../results/manual_audit_summary.json` — phân bố nhãn audit sơ bộ và kiểm tra metadata/filter.
- `../results/run_summary.json` — phiên bản model/store, cấu hình, thời gian và chi phí.
- `../report/individual/2A202603638_NguyenVoThaiBao.md` — báo cáo cá nhân đã đặt đúng tên nộp.

Sau khi tự đọc và cập nhật đủ 90 dòng audit (điểm, lý do, reviewer), chạy `python -m src.refresh_audit` từ thư mục `starter`. Lệnh kiểm tra nhãn và cập nhật summary/báo cáo mà không tính lại embedding.

Grounding là độ đủ của bằng chứng truy xuất, không phải độ đúng của câu trả lời LLM. Hit@3 tự động không thay cho nhãn coherence, metadata accuracy hay grounding do người nộp xác minh. Xem [GUIDE.md](../GUIDE.md) để kiểm tra lại nhãn sơ bộ, phân tích 5 khía cạnh và nộp bài.
