# Notebook tham khảo cho nhóm

Đây là **13 bản chỉ đọc**, tạo ngày 21/10/2026 từ notebook release lịch sử. Mở trực tiếp trên GitHub để xem luồng các cell ngoài; tất cả cells là Markdown, không chạy training hoặc resume.

Các bản này bỏ outputs, metadata gốc, thông tin tài khoản/storage, thread identity và bundle runtime base64. Mã các cell ngoài được giữ dưới dạng code excerpts có định dạng lại, SPEC chỉ giữ trường khoa học/provenance được chọn. **Không chứa code runtime bên trong bundle đã bỏ và không phải notebook chạy được.** Bản gốc được maintainer giữ nguyên, xác định bằng SHA256 trong [provenance.json](provenance.json).

## v3.2 — seed 43

Audit 20/10 mới xác nhận notebook đã được chuẩn bị; chưa xác nhận actual execution của năm run này.

- [PhoBERT ST sentiment43](v3.2-seed43/neu_esc_phobert_st_sentiment_s43.ipynb)
- [PhoBERT ST topic43](v3.2-seed43/neu_esc_phobert_st_topic_s43.ipynb)
- [ViSoBERT MTL43](v3.2-seed43/neu_esc_visobert_mtl_s43.ipynb)
- [ViSoBERT ST sentiment43](v3.2-seed43/neu_esc_visobert_st_sentiment_s43.ipynb)
- [ViSoBERT ST topic43](v3.2-seed43/neu_esc_visobert_st_topic_s43.ipynb)

## v3.1 — PhoBERT seed 42

Final evidence trong audit còn thiếu; preprocessing VnCoreNLP của nhóm khác author-processed input. Không hiểu tên release là metrics PASS.

- [PhoBERT ST sentiment42](v3.1-phobert-seed42/neu_esc_phobert_st_sentiment_s42.ipynb)
- [PhoBERT ST topic42](v3.1-phobert-seed42/neu_esc_phobert_st_topic_s42.ipynb)
- [PhoBERT MTL42](v3.1-phobert-seed42/neu_esc_phobert_mtl_s42.ipynb)
- [PhoBERT MTL+SMART42](v3.1-phobert-seed42/neu_esc_phobert_mtl_smart_s42.ipynb)

## direct-v2 — ViSoBERT W1–W4

Hai ST seed42 có metrics đã được audit; MTL42 có completion receipt nhưng thiếu final predictions trong audit, SMART46 có tiến triển lịch sử nhưng final chưa biết. Notebook source không tự chứng minh toàn bộ byte runtime đã thực chạy.

- [W1 MTL+SMART46](direct-v2-visobert/neu_esc_W1_visobert_mtl_smart_s46.ipynb)
- [W2 ST sentiment42](direct-v2-visobert/neu_esc_W2_visobert_st_sentiment_s42.ipynb)
- [W3 ST topic42](direct-v2-visobert/neu_esc_W3_visobert_st_topic_s42.ipynb)
- [W4 MTL42](direct-v2-visobert/neu_esc_W4_visobert_mtl_s42.ipynb)

Trạng thái khoa học: [PROGRESS.md](../../docs/PROGRESS.md). Bảng chi tiết: [run-status](../../reports/progress/run-status-20261020.csv). Đây là mốc audit 20/10/2026, không phải trạng thái GPU/Drive trực tiếp.

Để đọc implementation nhóm, xem [src/neu_esc](../../src/neu_esc) và [configs/experiments](../../configs/experiments). Code trên main là scientific release nền; runtime adapter trong bundle lịch sử có thể khác và không được thay bằng main chỉ vì cùng nhãn release. Muốn chạy mới cần protocol, config, storage và run namespace riêng được duyệt; không có claim exact Table 6 replication.
