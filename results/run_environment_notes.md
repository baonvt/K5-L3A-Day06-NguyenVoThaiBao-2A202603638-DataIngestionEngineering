# Ghi chú môi trường chạy

- Pipeline cuối chạy trên Windows với Python 3.12.10, model FastEmbed/ONNX CPU và Qdrant server 1.20.0 local.
- Docker daemon không hoạt động trong môi trường này. Import Qdrant Python client cũng bị Windows chặn khi nạp native gRPC DLL; không tìm cách vượt chặn. Thay vào đó, dùng Qdrant binary chính thức và gọi HTTP REST bằng `httpx`.
- Phương án REST vẫn dùng Qdrant thật, lưu bền vững tại `results/qdrant`; không dùng mock vector hoặc backend thay thế. Health, benchmark simulation và search đều trả HTTP 200.
- Không gọi embedding API, nên không có lỗi provider/quota hoặc chi phí API; API key không cần cho cấu hình local. Các trang PDF bị loại được liệt kê riêng trong `results/ingestion_page_audit.csv`.
- Không còn bước ingestion/retrieval nào chưa chạy trong lần chạy được ghi ở `results/run_summary.json`.
