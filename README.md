# NEU-ESC reproduction

Repo này tái lập kết quả Table 6 của paper [NEU-ESC](https://arxiv.org/abs/2506.23524) trên Kaggle GPU T4. Đây là bước đầu của đề tài DPL302m của Group 5.

## Kế hoạch: 8 setup × 5 seed (42–46) = 40 run

| Setup | PhoBERT-base-v2 | ViSoBERT |
|---|---|---|
| Single-task sentiment | ✓ | ✓ |
| Single-task topic | ✓ | ✓ |
| 2-task (encoder dùng chung + 2 head) | ✓ | ✓ |
| 2-task + SMART | ✓ | ✓ |

Runner bám code công khai của tác giả ([hung20gg/multi-task-bert@b5637cc](https://github.com/hung20gg/multi-task-bert/tree/b5637cc0a8a7b5e88e794d180c11902405548eb5)):
- hyperparameter, head và loss 0.5/0.5 lấy theo code tác giả;
- SMART theo đúng code tác giả, đã kiểm khớp từng bit;
- PhoBERT đọc `*_processed.csv` (bản đã tách từ của tác giả).

Kết quả chỉ được gọi là "tái lập theo code công khai", không phải khôi phục đúng thí nghiệm của paper.

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| [kaggle/](kaggle/README.md) | Runner, queue chạy 2 GPU, kiểm tra kỹ thuật, tổng hợp kết quả, notebook smoke và notebook chạy thật. **Bắt đầu từ đây.** |
| [docs/FINDINGS.md](docs/FINDINGS.md) | Phát hiện về dữ liệu, code tác giả, Table 6 và Figure 3; giới hạn của tuyên bố |
| `docs/papers/` | Paper và proposal. Chỉ có trên máy, không commit |
| [reports/reference/](reports/reference/) | Số liệu Table 6 và Figure 3 chép từ paper, phần audit Figure 3 |
| [reports/method/reference/](reports/method/reference/) | Bản sao code tác giả, dùng cho `kaggle/check_author_smart.py` |
| [reports/source/](reports/source/) | Hash tải dữ liệu, thống kê nhãn, đối chiếu CSV gốc với bản `_processed` |
| `reports/kaggle_results/` | Sẽ chứa kết quả 40 run sau khi chạy `kaggle/aggregate.py` |

## Dữ liệu

Không commit dữ liệu, vì repo của tác giả không có license phân phối. Runner tự tải 6 file CSV từ repo tác giả và kiểm SHA-256 (`python kaggle/neu_esc_kaggle.py fetch --data-dir …`). Thư mục `data/raw/` và `runs/` đã được `.gitignore`.

## Pipeline cũ

Pipeline cũ (`src/neu_esc`, configs, tests, scripts, CI, các notebook tham khảo, tài liệu module) được viết cho Colab A100. Nó đã được gỡ khỏi repo trước khi chạy tái lập vì ba lý do:
- SMART khác code tác giả;
- PhoBERT tự tách từ, lệch 29% mẫu so với bản của tác giả;
- chạy fp32 deterministic, quá chậm trên T4.

Bản gốc vẫn còn trong git history ở commit `d626a86`.

## Quy định

- Không push khi chưa được người phụ trách repo đồng ý.
- Không chọn seed hay checkpoint theo test.
- Điều kiện đi tiếp giữa các seed là `kaggle/check_runs.py` báo PASS, không phải điểm có gần paper hay không.
