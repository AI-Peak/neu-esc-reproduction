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
| [src_baseline/](src_baseline/README.md) | Code baseline (reproduce NEU-ESC): runner, queue chạy 2 GPU, kiểm tra kỹ thuật, tổng hợp kết quả, notebook smoke và notebook chạy thật. **Bắt đầu từ đây.** |
| [src_improvement/](src_improvement/README.md) | Code cải tiến: baseline + CR-SD (`selfdistill`) + CR-SD + D (`crsd_ens`). |
| `kaggle/` | Chỉ chứa API token của các account Kaggle (gitignored, không commit). |
| `docs/papers/` | Paper và proposal. Chỉ có trên máy, không commit |
| [reports/baseline_reproduce/](reports/baseline_reproduce/README.md) | **Báo cáo baseline reproduce**: số chốt, nguyên nhân lệch so với paper, metrics 15 seed, số liệu paper, bản sao code tác giả, hash dữ liệu |
| [reports/improvement/](reports/improvement/) | Kết quả cải tiến CR-SD / CR-SD + D so với baseline |

## Dữ liệu

Không commit dữ liệu, vì repo của tác giả không có license phân phối. Runner tự tải 6 file CSV từ repo tác giả và kiểm SHA-256 (`python src_baseline/neu_esc_kaggle.py fetch --data-dir …`). Thư mục `data/raw/` và `runs/` đã được `.gitignore`.

## Quy định

- Không push khi chưa được người phụ trách repo đồng ý.
- Không chọn seed hay checkpoint theo test.
- Điều kiện đi tiếp giữa các seed là `src_baseline/check_runs.py` báo PASS, không phải điểm có gần paper hay không.
