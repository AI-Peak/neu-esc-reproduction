# Pipeline dữ liệu NEU-ESC v1

Module 01; kiểm tra thực thi CPU ngày 2026-10-11. Không có training/model metrics trong tài liệu này. Manifest chính: `data/manifests/dataset.json`; config: `configs/data/dataset.yaml`.

## Quy tắc dữ liệu

Giữ train=23048, validation=3305, test=6613. Decode UTF-8 strict; BOM encoding được bỏ nếu có. Cột chính xác `text,sentiment,classification`; rename `classification` thành `topic`. Nhãn nguồn gồm giá trị dạng `0.0`; chỉ chấp nhận số hữu hạn có giá trị nguyên trong [0,3]/[0,9], chuyển chính xác sang integer. Null, empty text/label, nhãn không nguyên/ngoài range, encoding sai và schema cột sai đều fail. Không drop, dedup, resplit hay relabel.

Text prepared `text` giữ nguyên string lấy từ CSV, không strip/lowercase/NFC/expand acronyms/remove emoji thêm lần nữa. Nguyên văn byte nguồn giữ trong `data/raw/github/dataset/`. Tên `raw` trong API không có nghĩa dữ liệu chưa được tác giả clean.

ID: `{split}:{source_row:06d}:{raw_sha256[:12]}`; source_row zero-based không tính header. Ví dụ đầu split test: `test:000000:cdd81b2c7b96`. Duplicate text có ID khác nhau; số row và raw file hash bảo đảm identity theo nguồn. Prefix hash 12 ký tự là lựa chọn công bố của pipeline; manifest giữ SHA-256 đầy đủ.

JSONL UTF-8, mỗi record có `sample_id,split,source_row,text,text_segmented,sentiment,topic`. Serialization canonical: `ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False`; mỗi record kết thúc LF. Manifest digest SHA-256 canonical JSON bỏ duy nhất `manifest_digest`, không tính pretty-print bytes. Config hash dùng canonical parsed YAML; manifest lưu riêng implementation hashes. Paths raw/processed trong manifest tương đối với thư mục chứa manifest, không có đường dẫn máy cá nhân.

## Audit dữ liệu thật

| Split | Rows/IDs | Null/empty | Invalid nhãn | Mean whitespace words | Max words |
|---|---:|---:|---:|---:|---:|
| train | 23048 | 0 | 0 | 24.813259 | 1578 |
| validation | 3305 | 0 | 0 | 25.124054 | 832 |
| test | 6613 | 0 | 0 | 26.022078 | 1468 |

32966 IDs duy nhất toàn bộ. Không có UTF-8 replacement character, không có text khác NFC. Raw có 0 dòng chứa underscore, gần như toàn bộ lowercase (train 3 và validation 1 dòng có uppercase; test 0). CSV không phải văn bản scrape nguyên thủy: đã clean/punctuate, phù hợp README HF và mô tả paper, nhưng không có script đủ để xác minh từng bước ban đầu.

| Sentiment | train | validation | test | Tổng |
|---|---:|---:|---:|---:|
| 0 Neutral | 15936 | 2279 | 4558 | 22773 |
| 1 Positive | 2897 | 417 | 834 | 4148 |
| 2 Negative | 3630 | 524 | 1046 | 5200 |
| 3 Toxic | 585 | 85 | 175 | 845 |

| Topic | train | validation | test |
|---|---:|---:|---:|
| 0 Spam | 281 | 40 | 84 |
| 1 News | 628 | 91 | 183 |
| 2 Academic | 7355 | 1053 | 2104 |
| 3 Other | 10078 | 1442 | 2882 |
| 4 Service | 1647 | 237 | 474 |
| 5 Jobs & Recruitment | 564 | 81 | 163 |
| 6 Personal Affairs | 1032 | 148 | 298 |
| 7 Social Affairs | 536 | 78 | 155 |
| 8 Help & Share | 465 | 68 | 137 |
| 9 Club & Events | 462 | 67 | 133 |

Mỗi split có bảng joint 4×10 trong `reports/source/data_statistics.json`. Không có exact duplicate trong split hay giữa bất kỳ cặp split; so sánh `NFC + collapse whitespace` cũng không tìm thấy duplicate. Chuẩn hóa này chỉ phục vụ audit, không đổi input. Không làm kiểm tra near-duplicate ngữ nghĩa, vì zero exact duplicate không loại trừ nội dung gần giống nhau. `duplicate_groups.jsonl` hiện rỗng; test synthetic chứng minh detector phát hiện và giữ cả duplicate có nhãn mâu thuẫn.

## Đối chiếu CSV `_processed` của tác giả

Pinned `train.py`/`trainmultihead.py` chọn `_processed.csv` khi model name có `vinai`, các backbone khác đọc CSV chưa segment. Không tìm thấy generator cleaning/segmentation hay dictionary viết tắt trong toàn bộ Python files/EDA notebook của commit.

| Split | Rows/label pairs alignment | Dòng `_processed` có underscore | Raw khác processed | Khác sau bỏ underscore/collapse whitespace | VnCoreNLP mới khác tác giả |
|---|---|---:|---:|---:|---:|
| train | 23048 / 0 label mismatch | 17024 | 17041 | 223 | 6723 |
| validation | 3305 / 0 label mismatch | 2455 | 2457 | 26 | 1005 |
| test | 6613 / 0 label mismatch | 4891 | 4896 | 68 | 1902 |

317 dòng không thể khôi phục raw bằng chỉ bỏ underscore và collapse whitespace: có retokenization punctuation, dấu thanh hoặc khác nội dung token. Giữ row indices trong `raw_processed_alignment.json` và `data_statistics.json`. Row count/label alignment là bằng chứng mạnh cho paired comparison, nhưng không chứng minh identity ở các dòng text bị đổi. Không chạy segmenter trên `_processed`, không dùng bản này âm thầm như raw.

Tổng VnCoreNLP khác author processed: 9630/32966 (khoảng 29.21%). Không khẳng định tác giả dùng VnCoreNLP hay một tool khác chỉ từ underscore. Các file tác giả được giữ để đối chiếu `reference_code` nếu cần cohort riêng; pipeline chính hiện không nhập chúng làm text training. Đối chứng dùng author CSV cần config/cohort riêng do đầu mối phối hợp; không trộn với manifest này.

## Preprocessing/segmentation được implement

Version `neu-esc-data-v1.0.0`, cleaning `identity`. VnCoreNLP 1.2/RDRSegmenter pin commit `62bbc58fe5d113c898eae112656be97dcf50b3a0`; exact JAR/dictionary/rule hashes ở lock và manifest. Lựa chọn theo PhoBERT upstream; phiên bản paper chưa xác minh. Pipeline JVM annotator `wseg` áp dụng tokenizer/normalizer bên trong VnCoreNLP, rồi join sentence outputs bằng một ASCII space vào `text_segmented`. Text gốc, row, split, hai nhãn và IDs giữ nguyên; không chia một sample thành nhiều sample.

Java bridge source được embed trong Python package và materialize trong cache resource, nên wheel không cần package-data Java riêng. Cần **JDK 11+** cho source launcher; upstream toolkit chỉ cần Java 8+, nhưng wrapper này có yêu cầu cao hơn được công bố. Windows đã chạy Java 22.0.2. Bridge chỉ sửa resource locator public `vn.pipeline.Utils.jarDir` về đường dẫn cwd thật, để tránh lỗi `%20` của upstream ở workspace có dấu cách; không sửa thuật toán/dictionary/JAR. Dùng UTF-8 cho cả pipe và JVM. Một request/response base64 mỗi sample tránh mất ID do newline/tab; startup logs không thể trở thành text output. Timeout/process failure/empty output đều fail, không fallback raw.

## CLI và recipe cho module 04/07

Sau khi module 04 cài package với dependency PyYAML (Windows smoke đã dùng 6.0.3), tại repo root:

```text
python -m neu_esc.data.resources --config configs/data/dataset.yaml
python -m neu_esc.data.prepare --config configs/data/dataset.yaml --output data/processed
python tests/data/verify_real.py
```

Recipe cài JDK trên Colab/Linux do 04 sở hữu: JDK >=11, ví dụ `apt-get install -y openjdk-17-jdk-headless`; xác minh `java --version`, rồi chạy resource CLI để tải đúng JAR/dictionary/rules theo hash. Không dùng `py_vncorenlp.download_model()` tải alias mới nhất thay lock. Package chỉ cần PyYAML, stdlib và JDK cho data; không cần pretrained encoder, torch, pandas hay tokenizer để prepare/test. Tool recipe download khoảng 28MB, không cần toàn bộ POS/NER/parser models. Linux/Colab chưa chạy thực tế trong module 01.

Trên Windows nếu Java chưa có PATH, set `NEU_ESC_JAVA` về executable JDK. Lệnh thực tế đã dùng `C:/Program Files/Java/jdk-22/bin/java.exe`. Giá trị đường dẫn này không lưu trong manifest digest. `--offline` chỉ reuse nguồn đã có/checksum đúng. Prepare không tự cài JDK hoặc tự tải tài nguyên segmentation khi thiếu: recipe resources chạy trước, lỗi segmentation được báo riêng.

Restart cùng config/code/resources tái tạo cùng processed hashes, IDs và manifest digest, không viết lại file đã đúng. Nếu raw/processed bị sửa hoặc target thuộc cohort khác, fail và giữ file để điều tra. Cohort khác cần **cả `--output` và `--manifest-path` mới**; không overwrite manifest đã dùng cho training. Publish manifest cuối cùng sau JSONL/checks để loader không nhận tập file thiếu checksum. Không có rollback transaction xuyên nhiều file; consumer verify hashes để phát hiện bộ chưa hoàn tất. Không chạy prepare đồng thời ở cùng target.

## API

```python
from neu_esc.data.dataset import load_split
records = load_split("data/manifests/dataset.json", "test", "segmented")
```

API trả list records; `text` là variant đã chọn, `text_raw` giữ unsegmented text, `text_segmented` giữ segmented text; IDs/nhãn/source_row/split vẫn có. `text_variant` chỉ nhận `raw|segmented`. Loader kiểm manifest digest, byte hash split, số dòng, integer labels, split/source_row/order/ID. Segmented thiếu/null/empty thì DataError; PhoBERT không được fallback sang raw. Training đọc train/validation và không clean lại.

## Kiểm tra đã chạy

- Hai lần full prepare cùng phiên bản cuối trên 32966 mẫu bằng CPU: digest và cả ba processed hashes giữ nguyên. Hash nguồn trước/sau giữ nguyên. Manifest thật có segmented text đủ cả ba split.
- `python -m unittest discover -s tests/data -v`: 15 tests PASS; fixture kiểm duplicate, null/invalid labels, encoding/header, mappings, counts, source/manifest/processed tampering, ID/order, thiếu segmentation, changed cohort và restart.
- `.venv/Scripts/python.exe -m pytest tests/data -q -o cache_dir=runs/smoke/data/pytest_cache`: 15 passed, 23 subtests passed; không tải pretrained model.
- `python tests/data/verify_real.py`: PASS, 32966 unique IDs; đối chiếu từng row với raw source, đủ hai text variants, không đổi labels/split/order; kết quả trong `runs/smoke/data/verification.json`.

Manifest/hash cuối và allowlist release nằm trong `handoffs/01-data.md`. Technical input gate có thể dùng với notice alternative revision; gate **exact HF/paper equivalence chưa đạt**. Full training GPU trên Colab, integration và benchmark không được chạy ở module này.
