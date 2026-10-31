# Audit Table 6 và Figure 3

Nguồn: paper NEU-ESC, arXiv:2506.23524, trang PDF 18 (Table 6) và 19 (Figure 3).
SHA-256 PDF: `bdc37ae84e6653ca60b325ccb54f7f655fb4cd1ee606d0f9a8a5a15772e7189c`.

Đã render trực tiếp hai trang từ PDF bằng Poppler và đọc hình. 126 ô số Table 6 đã đối chiếu hình và khớp chính xác text trích xuất từ PDF; 116 ô Figure 3 đã đọc trực tiếp. Không có ô chưa xác minh. CSV dùng nhãn hàng thật và nhãn cột dự đoán theo hình.

| Task | N | Trace | Accuracy từ Figure 3 (%) | Table 6 (%) | Figure 3 − Table 6 (pp) |
|---|---:|---:|---:|---:|---:|
| sentiment | 6613 | 3619 | 54.72554060 | 83.58 | -28.85445940 |
| topic | 6613 | 2074 | 31.36246787 | 79.80 | -48.43753213 |

Mỗi ma trận chứa 6.613 mẫu. Không cộng hai task để báo N của một ma trận. Accuracy tính bằng trace/N trước khi làm tròn.

**Kết luận:** số đếm Figure 3 bất nhất với accuracy Table 6 của ViSoBERT 2-task & SMART. Chưa chứng minh nguyên nhân. Sai thứ tự predictions/labels, nhầm checkpoint và khác dữ liệu chỉ là giả thuyết; không dùng các giả thuyết để sửa số paper.

Các dấu in đậm và dấu * của Table 6 được lưu riêng trong `table6_metadata.json`, không lẫn vào số. Alias `mbert` bám ma trận kế hoạch nhưng Table 6 chỉ ghi BERT_base; alias `claude4` giữ nguyên Claude 4 như bảng. Module này không xác nhận checkpoint hoặc API snapshot.

Seed 42 được chọn trước cho CM tái lập so với Figure 3. CM mọi seed phải giữ lại; nếu cộng 5 seed đủ test thì N=33.065 mỗi task. Chưa có predictions benchmark trong module này.

Phép tính được tạo bằng `neu_esc.evaluation.audit_reference` của pipeline cũ, nay đã gỡ khỏi repo (bản gốc ở commit `d626a86`). Có thể kiểm lại trực tiếp từ hai file `figure3_*_counts.csv`: accuracy = tổng đường chéo / tổng ma trận.
