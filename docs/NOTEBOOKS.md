# Notebook training: catalog và trạng thái công bố

Ngày kiểm tra 21/10/2026: 91 notebook files, 21 embedded archives duy nhất và 1.222 archive members được đọc tĩnh. Không execute notebook hoặc sử dụng secrets. Chưa phát hiện credential-pattern trong phạm vi nội dung này; kết quả regex không bảo đảm không có secrets, đặc biệt payload tạo động hoặc tải qua mạng.

## Kết luận

**Chưa chọn notebook nguyên bản nào để public ngay làm entrypoint.** Repo công bố catalog/hash, bảng tiến độ và [13 reading derivatives](../notebooks/reference/README.md). Các derivative chỉ có Markdown, gồm code excerpts của outer cells, không chứa runtime bundle hay thông tin storage/account. Muốn có notebook chạy được cần một bản được review riêng; bản gốc giữ bất biến.

| Bộ | Đánh giá | Phần cần xử lý |
|---|---|---|
| v3.2, 5 run seed43 | Ứng viên tốt để làm bản tham khảo sau xử lý; 0 saved outputs | Storage owner email/folder ID trong SPEC và runner/specs, constants trong runner/storage.py; historical run IDs; chưa có execution evidence trong audit |
| v3.1, 4 PhoBERT seed42 | Ứng viên archive tham khảo sau xử lý; 0 saved outputs | Cùng loại binding storage/run cũ; preprocessing variant; final evidence còn thiếu |
| direct-v2, 4 ViSoBERT W1–W4 | Cần xử lý nhiều thông tin vận hành hơn | Account email, thread/profile IDs, đường dẫn local, storage/resume bindings; bản source notebook không tự chứng minh byte từng chạy |
| 6 notebook trong repo local | Campaign cũ, chỉ catalog trước | A100 preflight/pilot/full-run gates, T4 recovery/ownership; không promote chỉ vì đang ở thư mục notebooks |
| Root notebooks và recovery/candidates | Archive local | Bản before-custom-oauth có 15 output items; có Drive links/metadata trong một số bản; tránh trùng lịch sử |

Email/Drive IDs không đồng nghĩa là token đăng nhập. Vấn đề là chúng gắn notebook public vào tài khoản, storage và run của campaign cũ, không phù hợp làm cấu hình dùng chung cho người mới. Không in các giá trị đó vào catalog.

## Khi tạo derivative notebook để publish

1. Chọn một mục đích rõ: archive tham khảo hay template chạy mới. Archive phải ghi historical/not approved; template phải có protocol/config/cohort mới được duyệt.
2. Xử lý cả notebook JSON **và bundle base64 nhúng**, kể cả `runner/specs/*.json`, `runner/storage.py`, metadata, outputs và attachment nếu có. Chỉ clear outputs hoặc sửa markdown chưa đủ.
3. Tách storage/account settings khỏi source bằng cấu hình môi trường do người vận hành cung cấp; giữ cơ chế xác minh owner/quyền, không bỏ ownership check cho dễ chạy.
4. Không tái sử dụng run IDs/storage namespaces cũ cho run mới. Thay bytes bundle thì dựng lại hash/manifest và ghi rõ derivative, không giả là release nguyên bản.
5. Kiểm secrets/privacy/license, dependencies, CPU/static tests và provenance; không tự training chỉ để duyệt publish. Không bundle dataset/checkpoints/tokenizer caches/upstream code chưa rõ quyền.

Không có notebook nào bị sửa hoặc xóa trong lượt review này. [Catalog](../reports/progress/notebook-catalog-20261021.csv) giữ original notebook/bundle hashes để truy vết. Source/protocol decisions và evidence khoa học xem [PROGRESS.md](PROGRESS.md).
