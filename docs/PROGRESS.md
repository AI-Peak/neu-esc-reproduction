# Tiến độ NEU-ESC cho thành viên mới

Cập nhật tài liệu: 21/10/2026. **Trạng thái khoa học lấy từ audit ngày 20/10/2026; đây không phải trạng thái runtime/Drive trực tiếp.** Không có training, inference hoặc tính lại metrics trong lượt chuẩn bị trang này.

## Đã làm được gì?

- Pipeline data/model/loss/training/evaluation, 22 experiment configs, CPU fixtures và môi trường khóa đã có trên repo. Release `caaa76228bde261fd33bba15e0c742d77967fdb2` có [CI CPU thành công](https://github.com/AI-Peak/neu-esc-reproduction/actions/runs/38214475551). Remote main kiểm ngày 21/10: `4abdfe3f078778bc2273ce99ee1495466ff4114e`.
- Audit trước đã đối chiếu dữ liệu công khai: 32.966 mẫu, splits train 23.048 / validation 3.305 / test 6.613. Input PhoBERT dùng VnCoreNLP của nhóm vẫn khác author-processed CSV ở 9.630 mẫu.
- Audit đã kiểm lại metrics từ saved predictions của **7 run**. Các kết quả được giữ với provenance và giới hạn, chưa gọi là tái lập chính xác Table 6.
- Notebook và runtime adapters đã được tạo cho nhiều wave. Việc có notebook hoặc tên `release` không chứng minh run đã hoàn tất hay phương pháp được duyệt.

## Trạng thái theo nhóm run

| Nhóm | Bằng chứng tại audit 20/10 | Cách đọc |
|---|---|---|
| ViSoBERT MTL+SMART seed 42 lịch sử 13/10 và T4 14/10 | 2 run có metrics được kiểm lại; T4 thiếu một phần logs/config final local | Giữ riêng, cùng seed nên không tính thành 2 seeds độc lập |
| ViSoBERT MTL+SMART A100 seeds 43,44,45 | 3 run có predictions/metrics đã kiểm | Chỉ báo subgroup 3 seeds; chưa phải đủ 5 seeds |
| ViSoBERT ST sentiment42, ST topic42 | 2 run có metrics/selection được kiểm | Giữ với nhãn implementation nhóm có điều chỉnh |
| ViSoBERT MTL42 | Có completion receipt step 3620; thiếu final prediction bytes trong audit | Chưa xác minh metrics; không dùng snapshot RUNNING cũ làm trạng thái hiện tại |
| ViSoBERT MTL+SMART46 | Có evidence tiến triển lịch sử, final chưa biết | Cần final artifacts; không suy là đang chạy hoặc đã thất bại |
| 4 PhoBERT seed42 | Evidence khác nhau: handoff completion, recovery, training progress hoặc notebook intent | Chưa có đủ final predictions để xác minh; phải ghi preprocessing variant |
| 5 notebook v3.2 seed43 | Đã có bản notebook bàn giao; actual execution chưa xác định trong audit | Không ghi completed chỉ vì notebook đã build |
| T4 SMART43 bị retired | Hồ sơ ghi retired | Không đưa vào benchmark hoặc tự resume |
| Smoke/pilot/fixture | Kiểm tra kỹ thuật | Không tính vào kết quả tái lập |

Chi tiết từng run: [run-status-20261020.csv](../reports/progress/run-status-20261020.csv). Các nhãn PASS là kết luận của audit nguồn, không phải xác nhận mới ngày 21/10. Hash và nguồn: [source-provenance.json](../reports/progress/source-provenance.json). Full evidence còn do maintainer giữ; bảng CSV này không tự thay thế predictions/checkpoint/source bytes.

## Notebook nằm ở đâu và có dùng được ngay không?

Xem [danh mục notebook](../reports/progress/notebook-catalog-20261021.csv) và [kết luận kiểm tra notebook](NOTEBOOKS.md). Đường dẫn trong catalog là vị trí tương đối trong workspace của maintainer, **không phải file đã có khi clone repo**.

Các notebook nguyên bản gắn với run IDs, storage và cơ chế resume cũ. Repo có [13 bản notebook tham khảo chỉ đọc](../notebooks/reference/README.md), được tạo từ các release v3.2/v3.1/direct-v2. Chúng giữ mã các cell ngoài dưới dạng Markdown, bỏ bundle nhúng và thông tin tài khoản/storage; không cung cấp notebook để bấm Run all. Một bản chạy mới cần cấu hình storage riêng, provenance mới và protocol được duyệt.

## Các phần phương pháp đang chờ chốt

Code nhóm dùng SMART continuous embeddings và các thay đổi reduction/dropout/projection; public code có vấn đề gradient noise qua `.long()`. Public MTL evaluator đảo weighted F1, nhưng chưa biết evaluator thực sự tạo Table 6. MLM reduction/denominator và cách bật SMART còn xung đột; PhoBERT preprocessing, batch/accumulation, early stopping và checkpoint selection cũng khác hoặc thiếu căn cứ lịch sử.

Vì vậy, `method_mode: paper_method` trong config là tên lịch sử của implementation nhóm, không phải chứng nhận protocol tác giả. Không dùng notebook hoặc score gần paper để tự giải quyết các quyết định này. Không có cấu hình full run mới nào được trang tiến độ này phê duyệt.

## Thành viên có thể nhận việc ngay

- Đối chiếu paper/public code và cập nhật decision log theo từng khác biệt.
- Chuẩn hóa docs/citations và kiểm data identity, license, model/tokenizer pins.
- Review evaluator/ID joins và catalog artifacts theo run/cohort/generation.
- Thu thập final metadata/predictions còn thiếu theo quyền truy cập được giao; không cần tải toàn bộ weights hay retrain để lấp báo cáo.

Giữ kết quả cũ để truy vết. Không chọn seed/checkpoint theo test, không đổi config frozen hoặc nhập code/data tác giả vào repo khi chưa rõ quyền phân phối.
