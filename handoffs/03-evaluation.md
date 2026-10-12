# Handoff module 03 — Evaluation / Table 6 / Figure 3

Status: READY

Ngày bàn giao: 11/10/2026, Asia/Saigon. READY áp dụng implementation, reference transcription/audit và kiểm tra offline của module 03. Chưa có benchmark training/LLM được chạy hoặc báo thành công trong module này. Không tạo chat/subagent, nhắn chat khác, sửa file module khác, stage hoặc commit Git.

## Outputs và allowlist READY

Repo root: `F:/DPL project/neu-esc-reproduction/`. Các đường dẫn bên dưới tương đối với root. Module 04 được stage **đúng 21 file** trong allowlist sau, sau khi đối chiếu diff/hash. Không dùng `git add .`.

```text
src/neu_esc/evaluation/__init__.py
src/neu_esc/evaluation/evaluate.py
src/neu_esc/evaluation/aggregate.py
src/neu_esc/evaluation/plotting.py
src/neu_esc/evaluation/audit_reference.py
tests/evaluation/test_evaluate.py
tests/evaluation/test_aggregate.py
tests/evaluation/test_reference.py
tests/evaluation/create_smoke.py
tests/evaluation/check_data_boundary.py
reports/reference/table6.csv
reports/reference/table6_metadata.json
reports/reference/figure3_sentiment_counts.csv
reports/reference/figure3_topic_counts.csv
reports/reference/figure3_metadata.json
reports/reference/figure3_audit.json
reports/reference/figure3_audit.md
reports/reference/source-page-18.png
reports/reference/source-page-19.png
docs/evaluation.md
handoffs/03-evaluation.md
```

Evidence outputs local (không nằm trong release allowlist trên):

- `runs/smoke/evaluation/verification.json`: runtime, 30-test result, checks, allowlist và SHA-256 từng implementation/test/reference/doc file. Handoff không tự hash chính nó; module 04 kiểm tra diff handoff trực tiếp.
- `runs/smoke/evaluation/unittest-final.log`: đầy đủ tên 30 bài test và PASS.
- `runs/smoke/evaluation/offline-v2/`: fixture inputs, encoder/invalid-LLM predictions, provenance, metrics JSON/CSV, per-class CSV, counts/normalized CSV/PNG, 10 seed-metric JSON, ST 5-seed summary JSON/CSV, negative coverage input và `checks.json`. **4 mẫu synthetic/task**, không có model hoặc API call.
- `runs/smoke/evaluation/real_data_boundary_check-v2.json`: loader thật đọc validation/test rồi evaluator từ chối empty predictions; không tạo predictions/benchmark metrics.
- `runs/smoke/evaluation/offline-v1/` và `real_data_boundary_check.json`: bằng chứng lượt trước được giữ nguyên. Dùng v2 cho trạng thái bàn giao; không ghép hai manifest digest.

## Source / revision / hashes / config

- Paper trực tiếp: `F:/DPL project/references/NEU-ESC-2506.23524.pdf`, https://arxiv.org/abs/2506.23524. SHA-256 `bdc37ae84e6653ca60b325ccb54f7f655fb4cd1ee606d0f9a8a5a15772e7189c`. Trang 18/19 là **1-based PDF page**. Đã dùng pypdf đọc text và Poppler render hai trang ngay từ nguồn; không lấy giá trị README.
- Table 6: 21 rows × 6 metrics, toàn bộ 126 ô số đã đối chiếu render và khớp chính xác text PDF. Năm hàng ST pair, 12 hàng MTL, 4 hàng LLM. Bold/star giữ riêng như typography, không tự sửa thành maxima hay diễn giải thành significance.
- Figure 3: 4×4 và 10×10, toàn bộ 116 ô đã đọc trên hình; không có ô chưa xác minh. SHA-256 bản chép được khóa trong `figure3_metadata.json`; audit từ chối claim verified nếu bytes CSV hoặc nguồn PDF đổi.
- Revision data boundary: GitHub `hung20gg/multi-task-bert` commit `b5637cc0a8a7b5e88e794d180c11902405548eb5`, qua module 01. Digest manifest được kiểm tra ở v2: `ea127f55e840991e53569379a9b3e8ac00bc6798108d6472f8ab300a109f8f7d`. Raw/processed hashes và hashes `data/dataset.py`, `data/common.py` tại lần kiểm tra nằm trong boundary JSON. HF/paper equivalence vẫn **unverified alternative revision**.
- Manifest ở lượt boundary trước có digest `885546b5a21e80ac76cfa7ec128a742ac455d6623a5bf18155262dbbf6925a01`; module 01 cập nhật manifest trong cây dùng chung nên đã chạy lại boundary với output v2, không overwrite bằng chứng cũ. Tại lúc tạo verification, inputs hiện tại khớp v2 (`inputs_still_match_boundary=true`).
- API/schema evaluation: NEU-ESC contract v1, `schema_version="1.0"`; labels 4/10, zero_division=0; seed CM comparison predeclared 42; sample std ddof=1; one-seed std=null. Cohort hash do training/campaign tạo; module 03 kiểm tra cohort ID và metadata nhưng không đoán hash từ metrics.
- Môi trường chạy thật: Windows, bundled Python **3.12.14**, Pillow **12.3.0**, pypdf **6.10.0**, Poppler **26.07.0**. Python executable `C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`.
- Không tạo commit module 03. Snapshot thực thi/packaging do module 04 chốt sau bàn giao. Không claim cây đang được các module sửa là checkout reproducible.

## Lệnh đã chạy thật và kết quả

Tất cả lệnh Python dưới đây chạy tại repo root bằng executable bundled nêu trên, đặt `$env:PYTHONPATH=(Join-Path (Get-Location) 'src')`; các lần kiểm tra cuối đặt `PYTHONDONTWRITEBYTECODE=1`.

1. `pdftoppm -f 18 -l 19 -scale-to 2400 -png 'F:/DPL project/references/NEU-ESC-2506.23524.pdf' 'reports/reference/source-page'` — exit 0; mở và đọc trực tiếp cả hai PNG; Table 6/CM rõ.
2. `python -m neu_esc.evaluation.audit_reference --reference-dir reports/reference --pdf '../references/NEU-ESC-2506.23524.pdf'` — exit 0; Table 6 exact-match 126 ô. Sentiment N=6613/trace=3619/54.72554060184485%; topic N=6613/trace=2074/31.362467866323907%.
3. `python -m unittest discover -s tests/evaluation -p 'test_*.py' -v` — **30 tests PASS**, exit 0, cuối cùng chạy trong 0.499s; log `unittest-final.log`. Các subcases integrity cũng đều qua.
4. `python tests/evaluation/create_smoke.py --output runs/smoke/evaluation/offline-v2` — exit 0. Real data API/real subprocess CLI trên synthetic fixture: encoder exit 0, LLM invalid exit 0 với `--allow-invalid`, missing-record negative exit 2, không tạo metrics. Commands/stdout/stderr từng subprocess được lưu nguyên trong `offline-v2/checks.json`.
5. `python tests/evaluation/check_data_boundary.py --output runs/smoke/evaluation/real_data_boundary_check-v2.json` — exit 0. Data API kiểm chứng và trả validation 3305/test 6613; evaluator với predictions rỗng từ chối thiếu 6610/13226 cặp ID-task. Không tính bất kỳ benchmark score nào.
6. Đã mở kiểm tra PNG counts topic với INVALID và row-normalized sentiment; sau điều chỉnh độ rộng cell đã đọc lại PNG topic v2, nhãn/ô không bị cắt. Unit test xác nhận PNG đọc được và có đủ bốn ảnh/task-pair.

## Điều đã xác minh

- API `evaluate_records(records, expected_records, tasks, allow_invalid=False)` là nguồn dùng chung cho accuracy, macro/weighted F1, per-class P/R/F1/support, CM raw/row-normalized. API không ghi file. Evaluator join `(sample_id,task)` với labels authoritative, không tin thứ tự hoặc `y_true` của predictions.
- Duplicate/missing/extra ID, wrong split/task, class sai, `y_true` sai, duplicate expected ID và invalid encoder đều fail rõ. Shuffle predictions và expected records vẫn giữ metrics/CM như cũ.
- Hand-calculated fixture accuracy=1/2, macro-F1=7/24, weighted-F1=11/24; absent class/all-one-class giữ toàn K labels. Orientation đúng rows=true/columns=predicted.
- LLM invalid giữ N, tạo INVALID column, tăng FN đúng class, report valid coverage riêng. All-invalid có N đầy đủ và metrics 0. Thiếu record không biến thành invalid và không bị drop.
- `metrics.json` fractions; CSV report dùng %, delta dùng pp. Hash predictions lấy trên cùng byte snapshot đã parse, thay đổi file trong lúc evaluation bị từ chối. Counts tổng bằng N, trace/N bằng accuracy trước rounding.
- Aggregation giữ từng seed, mean và sample std ddof=1, không chọn seed theo test. Không trộn cohort/method/manifest/split/task/N/supports. Predeclared seed coverage thiếu/dư bị từ chối. CM seed 42 thiếu thì báo thiếu, không thay seed. CM mỗi seed được giữ; synthetic test N=6613 × 5 cho aggregate N=33065/task.
- ST rows ghép hai cohorts đúng task/backbone/seed, hai runs riêng mỗi seed, giữ cohort ID mỗi task và pairing declaration. Report CSV có mean/std/paper/delta/method/cohort/status.
- Figure 3 sentiment accuracy **54.72554060%**, topic **31.36246787%**, so Table 6 ViSoBERT MTL+SMART **83.58%/79.80%**: delta Figure-minus-Table **−28.85445940/−48.43753213 pp**. Chỉ xác minh bất nhất, không chứng minh nguyên nhân.

## Giả thuyết chưa kiểm chứng

Lệch thứ tự predictions/labels, nhầm checkpoint, hoặc khác evaluation data có thể giải thích Figure 3 nhưng chưa có bằng chứng để chọn một nguyên nhân. Module này không audit gradient SMART, không xác nhận repo reference commit tạo kết quả paper và không chứng minh revision GitHub/HF/paper tương đương.

## Giả định và tham số không được paper xác nhận

Quy tắc seeds 42–46, seed 42 cho CM, std ddof=1, fixed labels/zero_division=0 và cohort isolation là protocol nhóm/contract v1; không gọi chúng là paper protocol công bố. Table alias `mbert` chỉ theo planning matrix: bảng ghi BERT_base và không tự đủ để xác định checkpoint multilingual/cased/uncased. `claude4` giữ nhãn Claude 4 trong bảng, không chứng minh model snapshot Sonnet. Bold/star chỉ giữ quan sát typography. Không thêm giả định training/checkpoint selection trong module này.

## Sai khác với contract và yêu cầu tích hợp

Không đổi API/schema bắt buộc. Các trường CM/coverage và metadata backbone/variant/method được bổ sung để dùng report/aggregation. CLI có flags bổ sung và từ chối infer task set chỉ từ predictions; lệnh bốn flags chuẩn đọc task declaration từ resolved config/provenance mà trainer tạo. Standalone predictions phải khai báo `--tasks` để tránh bỏ sót toàn bộ một task.

API không phụ thuộc sklearn/NumPy. Renderer dùng **Pillow >=10.1**; CLI đọc YAML cần **PyYAML**; source audit cần **pypdf** và Poppler khi re-render. Extra `eval` hiện có matplotlib nên Pillow thường được kéo transitively; đề nghị module 04 khóa **Pillow trực tiếp** và pypdf nếu release có chạy source audit. Không sửa packaging ngoài phạm vi sở hữu.

Standalone evaluation thiếu run metadata giữ keys null và không thể aggregate; full run cần provenance/run ID/cohort/seed theo contract. ST aggregation cần metrics bổ sung `variant`, `backbone`; CLI tự lấy từ provenance/config khi có. Caller API của trainer nên bổ sung cùng metadata khi persist metrics để module 08 aggregate được.

## Blockers và bước tiếp theo

Không còn blocker cho phần module 03 được giao. Chưa chạy evaluation benchmark vì chưa nhận predictions encoder/LLM thật; chưa train, chưa dùng Colab/GPU, chưa gọi paid API. Nguồn dataset vừa kiểm tra là alternative revision và HF/paper equivalence chưa chứng minh; giới hạn này phải theo manifest vào báo cáo.

- **04:** dùng allowlist READY ở trên, kiểm tra hashes trong verification và diff, chốt packaging/Pillow/pypdf cùng commit snapshot. Không stage smoke/namespace khác theo allowlist này.
- **02/05:** validation/test gọi cùng API; predictions giữ đủ error/status/ID/split/task/y_true/y_pred. Encoder default không invalid; LLM sau bounded retries ghi invalid theo contract và dùng allow_invalid=True.
- **06:** integration trên snapshot cuối với actual data API; kiểm tra lệnh CLI bốn flags, metadata/provenance và trainer selection dùng metric chung. Nếu model/data/code thay đổi thì đánh giá lại đúng cohort mới.
- **07:** run seed 42 đã khai báo, sau đó đủ seeds; giữ predictions/CM mọi seed. Input cần cho evaluation là stable predictions JSONL, manifest/digest đúng run, resolved config và provenance.
- **08:** dùng paper CSV/audit ngay; chỉ dùng summary từ cohort hợp lệ cho reproduction. ST cung cấp pairing ID đã công bố; Figure 3 so seed 42 N=6613, CM cộng 5 seeds ghi N=33065/task. Không lấy fixture scores làm kết quả benchmark.

Các mục trên là bàn giao để người dùng điều phối; không có chat/message được tự gửi đi.
