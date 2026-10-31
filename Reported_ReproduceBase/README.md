# Báo cáo tình trạng Baseline Reproduce (NEU-ESC, arXiv 2506.23524)

Nhóm 5, DPL302m. Cập nhật 2026-10-31.

## 1. Kết luận

**Nhóm đã reproduce đúng baseline.**
- Code chạy theo đúng code gốc của tác giả.
- Kết quả trung bình lệch paper khoảng 0.4 điểm.
- Xu hướng giống paper: SMART giúp tăng so với MTL.

Hai con số baseline chốt dùng để so sánh với phần cải tiến:

| Task | Baseline reproduce | Cấu hình | Paper |
|---|---|---|---|
| **Sentiment** | **82.85 ± 0.24** | PhoBERT MTL + SMART, 15 seed | 83.26 |
| **Topic** | **79.35 ± 0.16** | ViSoBERT MTL + SMART, 15 seed | 79.80 |

Cách tính hai số này như sau:
- Chọn checkpoint theo tập **validation**, như paper mô tả.
- Đánh giá **một lần** trên test.
- Lấy **trung bình ± độ lệch chuẩn** qua 15 lần chạy với 15 seed khác nhau (42–56).
- Không nhìn test để chọn epoch, không chọn seed may mắn.

Mốc lớn nhất trong paper là 83.66 / 79.80 (§3).
- Ô topic 79.80 chính là dòng ViSoBERT MTL+SMART trong bảng trên. Trung bình nhóm thấp hơn 0.45 điểm, và 0/15 lần chạy đạt tới (§3).
- Ô sentiment 83.66 là dòng PhoBERT MTL+MLM+SMART. Nhóm đã chạy dòng này nhưng không đạt (§3).

## 2. Kết quả đầy đủ (test accuracy %, trung bình ± độ lệch chuẩn)

| Backbone | Setup | Sentiment | Paper | Δ | Topic | Paper | Δ |
|---|---|---|---|---|---|---|---|
| PhoBERT | Single-task | 82.27 ± 0.40 | 81.75 | +0.52 | 78.16 ± 0.58 | 78.65 | −0.49 |
| PhoBERT | MTL | 82.39 ± 0.71 | 82.88 | −0.49 | 78.75 ± 0.36 | 79.25 | −0.50 |
| PhoBERT | MTL + SMART | **82.85 ± 0.24** | 83.26 | −0.41 | 79.05 ± 0.17 | 79.31 | −0.26 |
| PhoBERT | MTL + MLM | 82.42 ± 0.26 | 82.67 | −0.25 | 78.38 ± 0.64 | 78.48 | −0.10 |
| PhoBERT | MTL + MLM + SMART | 82.77 ± 0.06 | 83.66 | −0.89 | 79.09 ± 0.18 | 79.49 | −0.40 |
| ViSoBERT | Single-task | 82.30 ± 0.64 | 82.78 | −0.48 | 78.11 ± 0.42 | 77.94 | +0.17 |
| ViSoBERT | MTL | 82.04 ± 0.43 | 82.63 | −0.59 | 78.22 ± 0.55 | 78.56 | −0.34 |
| ViSoBERT | MTL + SMART | 82.71 ± 0.18 | 83.58 | −0.87 | **79.35 ± 0.16** | 79.80 | −0.45 |
| ViSoBERT | MTL + MLM | 81.89 ± 0.29 | 82.23 | −0.34 | 78.04 ± 0.20 | 78.38 | −0.34 |
| ViSoBERT | MTL + MLM + SMART | 82.83 ± 0.14 | 83.12 | −0.29 | 79.46 ± 0.23 | 79.25 | +0.21 |

Số lần chạy: 15 seed cho Single-task, MTL và MTL+SMART; 5 seed cho hai dòng MLM.

Cách đọc bảng:
- Độ lệch trung bình (giá trị tuyệt đối) là **0.42 điểm**. Biên độ giữa các seed (max − min) đã lên tới 1–2 điểm. Vậy độ lệch so với paper nhỏ hơn mức dao động tự nhiên của một lần chạy.
- Hầu hết các ô của nhóm **thấp hơn paper một chút**. Đây là điều dễ hiểu: paper chỉ báo **một lần chạy** cho mỗi dòng, còn nhóm báo **trung bình nhiều lần**. Số của một lần chạy được chọn để báo cáo thường cao hơn số trung bình.

## 3. Đã kiểm tra những nguyên nhân gì

Nhóm dùng 10 tài khoản Kaggle (GPU T4) và chạy tổng cộng **170 lần**. Kết quả theo từng giả thuyết:

| # | Giả thuyết | Cách kiểm | Kết quả |
|---|---|---|---|
| 1 | Paper chỉ chạy 1 lần, gặp may | Tăng từ 5 lên 15 seed | **Đúng với phần lớn các ô.** Số của paper nằm trong vùng dao động của các seed của nhóm. |
| 2 | Tác giả chọn epoch bằng cách nhìn test (code lưu mọi epoch, không ghi tiêu chí chọn) | Lưu kết quả test ở mọi epoch, so 4 cách chọn | Nếu nhìn test thì MTL tăng thêm khoảng +0.3 đến +0.9. **Không giải thích được 2 ô lệch lớn** (xem dưới). |
| 3 | fp32 (tác giả) khác fp16 (nhóm) | Chạy lại MTL và MTL+SMART bằng fp32, cùng 5 seed | **Không ảnh hưởng.** Chênh lệch từ −0.28 đến +0.41, trung bình gần 0. |
| 4 | SMART bị cài sai | Đặt nhiễu lên embedding như SMART gốc của Jiang et al. | Accuracy **sụp xuống khoảng 72%**. Code gốc mới là bản đúng để reproduce (giải thích dưới). |
| 5 | Dòng MLM chưa được chạy | Cài đúng trainer 3 head của tác giả | **Đã reproduce.** MTL+MLM khớp paper; MTL+MLM+SMART thấp hơn (xem dưới). |

### Hai ô lệch lớn: không phải lỗi của nhóm

| Ô | Paper | Trung bình nhóm | Số lần chạy đạt paper | Mức cao nhất có thể* |
|---|---|---|---|---|
| ViSoBERT MTL+SMART, sentiment | 83.58 | 82.71 ± 0.18 | 0/15 | 83.09 |
| PhoBERT MTL+MLM+SMART, sentiment | 83.66 | 82.77 ± 0.06 | 0/5 | 83.25 |

\*Mức cao nhất có thể: lấy epoch có test cao nhất cho riêng task đó, rồi lấy seed tốt nhất. Đây là kiểu "gian lận tối đa", chỉ dùng để chứng minh, không dùng để báo cáo.

Kể cả khi gian lận tối đa, nhóm vẫn không chạm được hai số này. Ngoài ra:
- Đổi sang fp32 không giúp được.
- Code tác giả công bố không cho ra được số này.

Theo nguyên tắc nhóm đặt ra từ đầu, **hai ô này được kết luận là do phía tác giả.** Có thể họ đã chạy một phiên bản code, dữ liệu hoặc cấu hình khác với bản công bố. Từ những gì đã công bố, nhóm không thể sửa được.

### Những điểm lạ trong code của tác giả (nhóm phát hiện)

1. **SMART thực chất không phải SMART.**
   - Code cộng nhiễu vào *token ID* chứ không phải vào embedding, rồi ép kiểu `.long()`. Khi đó nhiều token bị đổi thành token có ID liền kề.
   - Vì vậy phần "SMART" của paper thực ra là **thay token ngẫu nhiên rồi bắt mô hình dự đoán giống nhau** (consistency regularization).
   - Nếu đặt nhiễu đúng lên embedding theo thuật toán của chính họ, nhiễu bị chuẩn hóa về độ lớn khoảng 1, lớn hơn embedding (khoảng 0.05) rất nhiều, nên mô hình hỏng.
   - Nhóm reproduce theo **code gốc**, vì bản này cho đúng xu hướng tăng như paper: ViSoBERT sentiment +0.67 (paper +0.95), topic +1.13 (paper +1.24).
2. **ViSoBERT + MLM không chạy được với code gốc.** Code chọn token ngẫu nhiên trong khoảng 0–64000, theo kích thước từ vựng của PhoBERT. Từ vựng ViSoBERT nhỏ hơn nhiều (khoảng 15k), nên sẽ báo lỗi. Nhóm giới hạn trong từ vựng của ViSoBERT; đây là sai khác duy nhất và có ghi lại trong config.
3. **Trainer MLM luôn bật SMART**, vì dòng `if is_smart` đã bị comment. Dòng "MTL+MLM không SMART" trong paper vì vậy không thể sinh ra từ code công bố nếu không sửa code. Nhóm đã tắt SMART để chạy dòng này.
4. **MTL không chọn checkpoint tự động.** Code lưu mọi epoch từ epoch 4 và không ghi tiêu chí chọn. Ngoài ra code không đặt seed, nên chính tác giả cũng không thể chạy lại ra đúng số cũ.

## 4. Ý nghĩa cho phần cải tiến

- So sánh cải tiến với **82.85 (sentiment)** và **79.35 (topic)**, theo đúng cách tính ở trên: chọn theo val và lấy trung bình nhiều seed.
- Trong bảng chính, nên so **cùng backbone, cùng setup** (ví dụ PhoBERT MTL+SMART với PhoBERT CR-SD) để phép so sánh công bằng.
- Phát hiện số 1 ở §3 cho thấy cái lợi thật sự của "SMART" là **consistency regularization**. Điều này củng cố hướng cải tiến CR-SD: R-Drop, EMA teacher và self-distillation đều thuộc nhóm phương pháp này.

## Phụ lục: số liệu kiểm tra

Mỗi ô ghi theo dạng **sentiment / topic**, test accuracy %.

### A. Cách chọn checkpoint (giả thuyết 2)

Các run có lưu test ở mọi epoch:
- 10 seed cho MTL và MTL+SMART.
- 5 seed cho hai dòng MLM.

Cách đọc các cột:
- **Val**: cách của nhóm, chọn epoch theo tập validation.
- **Nhìn test**: chọn epoch có test trung bình hai task cao nhất.
- **Trần**: lấy epoch tốt nhất cho riêng từng task, rồi lấy seed tốt nhất.

| Backbone | Setup | Val (trung bình) | Nhìn test (trung bình) | Trần | Paper |
|---|---|---|---|---|---|
| PhoBERT | MTL | 82.43 / 78.73 | 83.02 / 78.93 | 83.43 / 79.36 | 82.88 / 79.25 |
| PhoBERT | MTL + SMART | 82.82 / 79.07 | 82.88 / 79.12 | 83.29 / 79.40 | 83.26 / 79.31 |
| PhoBERT | MTL + MLM | 82.42 / 78.38 | 83.09 / 78.98 | 83.35 / 79.36 | 82.67 / 78.48 |
| PhoBERT | MTL + MLM + SMART | 82.77 / 79.09 | 83.02 / 79.14 | **83.25** / 79.40 | **83.66** / 79.49 |
| ViSoBERT | MTL | 81.92 / 78.45 | 82.88 / 78.67 | 83.32 / 79.04 | 82.63 / 78.56 |
| ViSoBERT | MTL + SMART | 82.68 / 79.38 | 82.80 / 79.50 | **83.09** / 79.75 | **83.58** / 79.80 |
| ViSoBERT | MTL + MLM | 81.89 / 78.04 | 82.60 / 78.55 | 83.20 / 78.87 | 82.23 / 78.38 |
| ViSoBERT | MTL + MLM + SMART | 82.83 / 79.46 | 82.85 / 79.71 | 83.15 / 79.96 | 83.12 / 79.25 |

### B. fp32 so với fp16 (giả thuyết 3)

Cùng 5 seed 42–46, chạy một lần bằng fp16 và một lần bằng fp32.

| Backbone | Setup | fp16 | fp32 | Chênh lệch |
|---|---|---|---|---|
| PhoBERT | MTL | 82.30 / 78.80 | 82.08 / 78.52 | −0.22 / −0.28 |
| PhoBERT | MTL + SMART | 82.90 / 79.02 | 82.86 / 78.95 | −0.05 / −0.07 |
| ViSoBERT | MTL | 82.30 / 77.77 | 82.25 / 78.19 | −0.05 / +0.41 |
| ViSoBERT | MTL + SMART | 82.77 / 79.28 | 82.92 / 79.43 | +0.15 / +0.15 |

### C. SMART đặt nhiễu trên embedding (giả thuyết 4)

5 seed 42–46.

| Backbone | MTL | SMART theo code gốc | SMART trên embedding |
|---|---|---|---|
| PhoBERT | 82.30 / 78.80 | 82.90 / 79.02 | 72.49 / 72.38 |
| ViSoBERT | 82.30 / 77.77 | 82.77 / 79.28 | 72.21 / 70.96 |

### Dữ liệu và cách tạo lại

- `gap_runs.csv`: kết quả từng run (170 run).
- Lệnh tạo lại các bảng: `python src_baseline/analyze_gap.py`. Lệnh in bảng ra màn hình và ghi lại file CSV.
- Run gốc nằm trong:
  - `runs/s42..s46`: 5 seed gốc.
  - `runs/gap/a`: thêm 10 seed và các dòng MLM.
  - `runs/gap/b`: fp32 và SMART trên embedding.
