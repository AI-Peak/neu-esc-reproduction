# Phát hiện khi tái lập NEU-ESC

Tài liệu này gộp bốn audit cũ: `data_audit.md`, `method_audit.md`, `evaluation.md` và `source_audit.md` (ngày 11–12/10/2026). Nó bổ sung thêm các điểm đã kiểm lại ngày 21/10/2026 khi viết runner [kaggle/](../kaggle/README.md).

- **Bản đầy đủ của bốn file cũ:** xem trong git history, commit `d626a86`.
- **Bằng chứng máy đọc được:** nằm trong [reports/](../reports/).
- **Tài liệu quy trình của pipeline cũ** (hợp đồng module, môi trường Colab, gate, handoff, tiến độ run cũ): đã xoá; bản gốc còn trong git history ở commit `d626a86`.

## 1. Nguồn

| Nguồn | Revision / hash | Ghi chú |
|---|---|---|
| Paper [arXiv 2506.23524](https://arxiv.org/abs/2506.23524) | PDF SHA-256 `bdc37ae84e6653ca…` | 24 trang; Table 6 ở trang 18, Figure 3 ở trang 19 (số trang PDF) |
| Code tác giả [hung20gg/multi-task-bert](https://github.com/hung20gg/multi-task-bert/tree/b5637cc0a8a7b5e88e794d180c11902405548eb5) | `b5637cc0a8a7b5e88e794d180c11902405548eb5` (18/07/2024) | Commit có trước paper. **Chưa chứng minh đây là code tạo Table 6**; README của repo có số khác Table 6. Bản sao các file liên quan: [reports/method/reference/](../reports/method/reference/) |
| HF dataset [hung20gg/NEU-ESC](https://huggingface.co/datasets/hung20gg/NEU-ESC) | revision quan sát `daf543ad…` | Dataset gated (`gated=auto`), metadata license Apache-2.0. Tải CSV không có token bị HTTP 401, nên chưa so nội dung với bản GitHub |
| CSV trong repo tác giả (bản đang dùng) | xem bảng hash ở mục 2 | Chính là các file mà script của tác giả đọc |

**Quyền phân phối:** repo GitHub của tác giả không có LICENSE, và license Apache-2.0 trên HF chưa chắc áp dụng cho bản GitHub. Vì vậy dữ liệu chỉ tải về máy (`data/.gitignore`), không commit lên repo. PDF cũng bị `.gitignore` vì lý do tương tự.

## 2. Dữ liệu

| File | Số mẫu | SHA-256 |
|---|---:|---|
| `train_set.csv` | 23.048 | `8a18a73ca21105e0b424829cc41395719be98dd7f01bf3d962eca86e7a1501ec` |
| `val_set.csv` | 3.305 | `0b2567fe5374cf9b8d22de92ce71d3f9bfa036dffac46fa6f7a5650e85c81169` |
| `test_set.csv` | 6.613 | `cdd81b2c7b96752f9f45d05e231682fcd6c380f40273dac521425ceff71ab312` |
| `train_set_processed.csv` | 23.048 | `7ca23b39e8e917de5c81745b0be2f4bb0e085fdc6173882479ecb2ed8e9ebe96` |
| `val_set_processed.csv` | 3.305 | `948578e4b92df56117fd38d8edaee8d98f11adbd38611d6faee655008adea626` |
| `test_set_processed.csv` | 6.613 | `bf850a67db6f9dd48df2adc5aa181e63e052e526361a1e9f0e6179bf582dc2e4` |

Tổng 32.966 mẫu, khớp Table 3. Mỗi CSV có các cột `text`, `sentiment`, `classification`; trong đó `classification` là nhãn topic. Nhãn lưu dạng `0.0` nhưng đều là số nguyên hợp lệ.

**Nhãn:**
- Sentiment: `0 Neutral, 1 Positive, 2 Negative, 3 Toxic`.
- Topic: `0 Spam, 1 News, 2 Academic, 3 Other, 4 Service, 5 Jobs & Recruitment, 6 Personal Affairs, 7 Social Affairs, 8 Help & Share, 9 Club & Events`.

| Sentiment | train | val | test | Tổng |
|---|---:|---:|---:|---:|
| Neutral | 15.936 | 2.279 | 4.558 | 22.773 |
| Positive | 2.897 | 417 | 834 | 4.148 |
| Negative | 3.630 | 524 | 1.046 | 5.200 |
| Toxic | 585 | 85 | 175 | 845 |

| Topic | train | val | test |
|---|---:|---:|---:|
| Spam | 281 | 40 | 84 |
| News | 628 | 91 | 183 |
| Academic | 7.355 | 1.053 | 2.104 |
| Other | 10.078 | 1.442 | 2.882 |
| Service | 1.647 | 237 | 474 |
| Jobs & Recruitment | 564 | 81 | 163 |
| Personal Affairs | 1.032 | 148 | 298 |
| Social Affairs | 536 | 78 | 155 |
| Help & Share | 465 | 68 | 137 |
| Club & Events | 462 | 67 | 133 |

**Chất lượng dữ liệu:**
- Không có dòng rỗng hay nhãn sai.
- Không có bản trùng chính xác, cả trong cùng split lẫn giữa các split, kể cả sau khi chuẩn hoá NFC và khoảng trắng. Chưa kiểm câu gần trùng.
- Text đã được tác giả làm sạch: gần như toàn bộ là chữ thường, bản gốc không có dấu `_`. Không có script cleaning hay từ điển viết tắt để tái tạo bước này.

**Lỗi trình bày trong paper.** Giữ nguyên dữ liệu, không sửa để khớp với các lỗi này:
- Table 1 ghi Negative 5.250, dữ liệu thật là 5.200. Dùng số 5.250 thì tổng Table 1 thành 33.016, lệch 50 so với Table 3.
- Đoạn văn trang 9–10 nói có "nine" topic, nhưng Table 2 có 10 topic, với `3 = Other`.

**Input của PhoBERT:**
- Script của tác giả đọc `*_processed.csv` (đã tách từ) khi tên model có chữ `vinai`; các model khác đọc CSV gốc.
- Bản processed khớp với bản gốc từng dòng và từng cặp nhãn, không lệch dòng nào.
- Có 317 dòng (train 223, val 26, test 68) không khôi phục được bản gốc chỉ bằng cách bỏ `_`, do tách lại dấu câu, dấu thanh hoặc khác token.
- Không tìm thấy script tách từ của tác giả.
- Pipeline cũ của nhóm tự chạy VnCoreNLP và cho kết quả khác bản của tác giả ở 9.630/32.966 mẫu (29,21%). **Runner Kaggle dùng `_processed.csv` của tác giả.**

**ID mẫu:** có dạng `{split}:{dòng:06d}:{12 ký tự đầu SHA-256 của CSV gốc}`, ví dụ `test:000000:cdd81b2c7b96`. Pipeline cũ và runner Kaggle dùng cùng định dạng này. Vì bản processed khớp bản gốc từng dòng, PhoBERT và ViSoBERT dùng chung ID.

## 3. Code tác giả so với paper

Paper không ghi hyperparameter. Các giá trị dưới đây lấy từ `train.py`, `trainmultihead.py`, `trainer/base_trainer.py`, `Trainer/head_trainer.py` và `utils/`.

| Hạng mục | Paper | Code công khai |
|---|---|---|
| Single-task | Fine-tune từng task riêng | `AutoModelForSequenceClassification`, CE |
| Head 2-task | Encoder, linear, layer norm, hai head | `[CLS]` → Linear → SiLU → LayerNorm → Linear(4) / Linear(10). `Dropout(0.2)` được khai báo nhưng không dùng trong `forward` |
| Loss 2-task | Cộng loss các head | `0.5·L_s + 0.5·L_t` ở trainer 2 head. Trainer 3 head (MLM) dùng `L_s + L_t + L_MLM/σ`, tức trọng số 1/1; tham số `percentage` ở trainer này chỉ xuất hiện trong tên file |
| Batch, độ dài | Không ghi | batch 128, max_length 128, tokenizer slow, pad tới max_length |
| Optimizer single-task | Không ghi | AdamW lr 2e-5 (eps 1e-8, weight decay 0.01), 15 epoch, LinearLR theo epoch giảm về 0.2, early stopping patience 4 theo val macro-F1 |
| Optimizer 2-task | Không ghi | AdamW lr 2e-5, eps 5e-9, 20 epoch (comment ghi 19 nhưng vòng lặp chạy 20), LinearLR giảm về 0.1 |
| Chọn checkpoint 2-task | Không ghi | Lưu mọi epoch từ epoch 4; script không có bước chọn (ví dụ `predict.py` nạp epoch 6) |
| Precision, seed | Không ghi | fp32, không đặt seed |

**SMART.**
- Paper mô tả nhiễu nhỏ trên embedding. Code lại cộng nhiễu vào **token ID** rồi gọi `.long()` ([utils__loss_function.py:82](../reports/method/reference/utils__loss_function.py:82)).
- Vì `.long()` cắt đồ thị tính toán, gradient theo nhiễu luôn là `None`; bước "adversarial" thực chất chỉ chuẩn hoá nhiễu ngẫu nhiên.
- Hệ quả: khoảng 49% token bị đổi thành ID liền trước. Sau đó code tính sym-KL giữa dự đoán sạch và dự đoán trên input bị nhiễu, với `reduction='sum'` và hệ số 0.02. Gradient theo tham số model vẫn chảy qua lượt forward cuối.
- Runner Kaggle tái tạo đúng hành vi này, khớp từng bit ([check_author_smart.py](../src_baseline/check_author_smart.py)), và gọi nó là "SMART theo code công khai".
- Pipeline cũ (`paper_method`) thay bằng nhiễu embedding liên tục, nên là một cohort khác.

**MLM** (không nằm trong 8 setup đang chạy):
- Paper định nghĩa σ lúc là số token trung bình, lúc là số token được mask trung bình.
- Code lấy mean CE rồi chia thêm cho số token mask trung bình một lần nữa.
- Trainer 3 head luôn gọi SMART dù cờ tắt, vì dòng `if self.is_smart` bị comment ([Trainer__mlm_3head_trainer.py:75](../reports/method/reference/Trainer__mlm_3head_trainer.py:75)). Table 6 lại tách riêng MLM và MLM + SMART.
- Head MLM có kích thước vocab cố định 64001.

**Revision model** (khóa ngày 11/10/2026; không biết revision mà paper đã dùng):

| Backbone | Checkpoint | Revision |
|---|---|---|
| PhoBERT-base-v2 | `vinai/phobert-base-v2` | `86cd7fd4c148980922ac11a2cf5e257f2ba639e1` |
| ViSoBERT | `uitnlp/visobert` | `196a62afad9cbe4f52a54aabad828b13f0eec59a` |
| XLM-R base | `FacebookAI/xlm-roberta-base` | `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089` |
| BERT base | `google-bert/bert-base-multilingual-uncased` (lấy từ `train.py`; paper chỉ ghi BERTbase) | `7cbf9a625e29989f6b9c6c2fa68234c304f7e38f` |
| vELECTRA | `FPTAI/velectra-base-discriminator-cased` | `f594fee501933d92d8c13285c969495681d0cc00` |

PhoBERT ở revision đã khóa chỉ có `pytorch_model.bin`. Nếu để mặc định, transformers sẽ tải `model.safetensors` từ một nhánh PR chuyển đổi, tức là một revision khác revision đã khóa. Runner Kaggle đặt `DISABLE_SAFETENSORS_CONVERSION=1` để tránh việc này.

## 4. Đánh giá, Table 6 và Figure 3

- **Số liệu Table 6** được chép ở [reports/reference/table6.csv](../reports/reference/table6.csv): 21 hàng, 126 ô, đã đối chiếu với ảnh trang 18. Mỗi hàng single-task ghép từ hai run riêng, `st_sentiment` và `st_topic`.
- **Weighted-F1 của evaluator 2-task bị đảo đối số.**
  - Code gọi `f1_score(pred, true)` ([Trainer__head_trainer.py:140](../reports/method/reference/Trainer__head_trainer.py:140)), nên wF1 được tính theo trọng số của phân phối dự đoán.
  - Evaluator single-task dùng đúng thứ tự. Accuracy và macro-F1 không bị ảnh hưởng vì đối xứng.
  - Chưa biết Table 6 có dùng evaluator lỗi này không. Runner báo `weighted_f1` (đúng, là metric chuẩn) và `weighted_f1_author_mtl` (chỉ để chẩn đoán).
- **Figure 3 không khớp Table 6.**
  - Mỗi confusion matrix có N = 6.613. Đường chéo cho accuracy 54,73% (sentiment) và 31,36% (topic), trong khi Table 6 ghi 83,58% và 79,80%.
  - Tổng mỗi hàng khớp đúng phân phối nhãn thật của tập test.
  - Đường chéo gần đúng bằng mức kỳ vọng nếu dự đoán không liên quan gì tới nhãn: 3.637 so với 3.619 quan sát (sentiment), 2.103 so với 2.074 (topic). Chi-square lần lượt là 21,5 (df 9) và 89,9 (df 81).
  - Giả thuyết, chưa chứng minh: dự đoán bị lệch thứ tự so với nhãn. Loader của tác giả bật `shuffle=True` cho cả tập test ([utils__dataloader.py:60](../reports/method/reference/utils__dataloader.py:60)) và không trả về index mẫu.
  - Số trong Table 6 không bị ảnh hưởng, vì evaluator lấy dự đoán và nhãn trong cùng một batch.
  - Bản chép số: [reports/reference/figure3_audit.md](../reports/reference/figure3_audit.md).
- **Quy ước báo cáo:** confusion matrix có hàng là nhãn thật, cột là dự đoán. Macro-F1 lấy trung bình trên đủ 4 hoặc 10 nhãn. Báo mean ± sample std theo seed. Không chọn seed hay checkpoint theo test.

## 5. Không xác minh được: giới hạn của tuyên bố

- Code, config, seed và số lần chạy cho từng hàng Table 6; mỗi số là một run hay số tổng hợp.
- Tiêu chí chọn checkpoint 2-task của tác giả.
- Revision dataset trên HF và revision model mà paper đã dùng; tool và từ điển dùng cho cleaning và tách từ.
- Code nào thực sự sinh ra Figure 3.

Vì vậy kết quả của nhóm chỉ được gọi là **tái lập theo code công khai của tác giả**, không phải khôi phục đúng thí nghiệm của paper. Các câu hỏi cần gửi tác giả nằm trong tài liệu nhận xét của nhóm (mục 5).

## 6. Bằng chứng trong `reports/`

| Thư mục | Nội dung |
|---|---|
| [reports/source/](../reports/source/) | Sổ nguồn và hash tải về, thống kê dữ liệu, đối chiếu bản gốc với bản processed, so dữ liệu với paper |
| [reports/method/reference/](../reports/method/reference/) | Bản sao code tác giả, kèm hash upstream |
| [reports/reference/](../reports/reference/) | Bản chép Table 6 và Figure 3, ảnh trang 18–19, audit Figure 3 |

Các audit khác của pipeline cũ (backbone/tokenizer, gradient SMART kiểu embedding, gate tích hợp, trạng thái run cũ) đã gỡ khỏi repo; bản gốc còn trong git history ở commit `d626a86`.
