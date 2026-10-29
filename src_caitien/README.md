# src_caitien: code cải tiến

Thư mục này là bản sao của [src_baseline/](../src_baseline/README.md) cộng các cải tiến của nhóm. Runner `neu_esc_caitien.py` vẫn chạy được mọi setup của baseline (`st_sentiment`, `st_topic`, `mtl`, `mtl_smart`) và cho kết quả giống hệt `src_baseline/neu_esc_kaggle.py`. Dữ liệu, backbone, siêu tham số chung, cách chọn checkpoint và các file output giống baseline; xem README của baseline.

| File | Nội dung |
|---|---|
| `neu_esc_caitien.py` | Runner: `fetch`, `prefetch`, `train`, `ensemble-targets` |
| `run_queue.py` | Chạy nhiều job, mỗi GPU một job; `--folds N` để chạy các fold teacher |
| `check_runs.py` | Kiểm tra kỹ thuật các run (PASS/FAIL) |
| `aggregate_caitien.py` | Tổng hợp kết quả, so với baseline (Welch t-test, main effect) |

## Các setup

**Ablation giai thừa (kết quả âm).** Tên có dạng `{head}_{opt}_{loss}`, tổng cộng 12 setup:

- `head`: `sep` (hai head riêng) hoặc `fac` (Factorized + Interaction).
- `opt`: `sum`, `pcgrad` hoặc `patgrad`.
- `loss`: `ce` hoặc `cbf` (CB-Focal).

`sep_sum_ce` cho ra logits giống hệt `mtl` của baseline. Kết quả 120 run nằm trong `reports/caitien_results/summary.md`: không cấu hình nào cải thiện có ý nghĩa.

**CR-SD (`selfdistill`).** Kiến trúc giữ nguyên `sep_sum_ce`. Khi train, mỗi câu đi qua student hai lần với dropout khác nhau (R-Drop, α = 1). Một bản EMA của trọng số (μ = 0.999) làm teacher và student học theo nó từ epoch 2 (β = 1). Chọn checkpoint, test và dùng thật đều bằng trọng số EMA. Mỗi run ghi thêm kết quả của student (`metrics.json` → `student`, kèm `student_logits.npz`).

**CR-SD + D (`crsd_ens`).** Giống CR-SD, thêm một teacher thứ hai cố định: dự đoán *out-of-fold* của một ensemble trên tập train. Từ epoch 1, student được kéo về phân phối của ensemble (γ = 1). Inference vẫn chỉ là một model.

Out-of-fold nghĩa là mỗi câu train chỉ được dự đoán bởi model *không* thấy câu đó lúc train, nên nhãn mềm mang thông tin thật chứ không phải nhãn đã học thuộc.

## Chạy CR-SD + D (3 bước)

```bash
# 1. Teacher: chia train thành 5 fold (cố định, FOLD_SEED = 2026), mỗi run train trên 4 fold và dự đoán fold còn lại -> oof.npz
python src_caitien/run_queue.py --setups mtl --folds 5 --seeds 42 --backbones phobert visobert --data-dir DATA --out-dir runs/teachers

# 2. Gộp xác suất out-of-fold của mọi run (có thể trộn backbone và seed) thành một file
python src_caitien/neu_esc_caitien.py ensemble-targets runs/teachers/*-f?of5 --out runs/teachers/ens_targets.npz

# 3. Train CR-SD + D
python src_caitien/run_queue.py --setups crsd_ens --seeds 42 43 --backbones phobert visobert --data-dir DATA --out-dir runs/caitien -- --ens-targets runs/teachers/ens_targets.npz
```

Ở bước 2, mọi dòng train phải có ít nhất một dự đoán out-of-fold, nên mỗi bộ fold phải đủ cả 5 fold. Bước 3 kiểm tra file target được tạo từ đúng train CSV (SHA-256).

Lệnh chạy CR-SD thường và lệnh tổng hợp:

```bash
python src_caitien/run_queue.py --setups selfdistill --seeds 42 43 --data-dir DATA --out-dir runs/caitien
python src_caitien/aggregate_caitien.py runs/caitien runs/s42 runs/s43 runs/s44 runs/s45 runs/s46 --out reports/caitien_results
```

## Test nhanh trên CPU

```bash
python src_caitien/run_queue.py --setups mtl --folds 2 --seeds 42 --backbones phobert --data-dir DATA --out-dir runs/cpu --no-prefetch -- --model-path TINY --epochs 1 --max-steps-per-epoch 3 --limit-eval 200
python src_caitien/neu_esc_caitien.py ensemble-targets runs/cpu/phobert-mtl-s42-f0of2 runs/cpu/phobert-mtl-s42-f1of2 --out runs/cpu/ens.npz
python src_caitien/neu_esc_caitien.py train --backbone phobert --setup crsd_ens --seed 42 --data-dir DATA --out-dir runs/cpu --model-path TINY --epochs 2 --max-steps-per-epoch 3 --limit-eval 200 --ens-targets runs/cpu/ens.npz
```

`TINY` là một model nhỏ tự tạo (cùng tokenizer với PhoBERT), chỉ dùng để kiểm tra code chạy được.
