# Module 01 — data/source handoff

Status: READY

**READY cho input GitHub thay thế đã khóa và pipeline kỹ thuật. Chưa chứng minh exact HF/paper equivalence; không được trình bày full training từ input này như tái lập đúng revision paper.** Ngày: 2026-10-11 (Asia/Saigon). Không tạo/nhắn chat, không gọi subagent, không sửa model/evaluator/packaging/notebook hay Git index/commit.

## Outputs

Repo root: `F:/DPL project/neu-esc-reproduction`.

- Ba JSONL thật: `data/processed/train.jsonl`, `validation.jsonl`, `test.jsonl` (32966 mẫu tổng, cache local; không phân phối qua Git).
- Manifest: `data/manifests/dataset.json`; mapping: `data/manifests/label_maps.json`.
- Config: `configs/data/dataset.yaml`; pinned segmentation: `configs/data/segmentation.lock.json`.
- CLI/API ở `src/neu_esc/data/`; `prepare`, `resources`, `neu_esc.data.dataset.load_split(manifest_path, split, text_variant)`.
- Hai audit docs: `docs/source_audit.md`, `docs/data_audit.md`.
- Source/scope ledgers và statistics/evidence: `reports/source/`; smoke receipts/logs: `runs/smoke/data/`.
- Tests CPU: `tests/data/test_pipeline.py`; full real-row verification: `tests/data/verify_real.py`.

## Version / hashes / giao input cho 02,03,05,06,07

- PDF SHA-256: `bdc37ae84e6653ca60b325ccb54f7f655fb4cd1ee606d0f9a8a5a15772e7189c`.
- GitHub data/code source commit: `b5637cc0a8a7b5e88e794d180c11902405548eb5`, 2024-07-18.
- Observed HF revision: `daf543ad1992153cd2be9fec3cb59aa0fc714147`; gated auto, README 200, CSV 401 unauthenticated.
- Preprocessing `neu-esc-data-v1.0.0`; identity cleaning; VnCoreNLP/RDRSegmenter 1.2 at `62bbc58fe5d113c898eae112656be97dcf50b3a0`. JAR + dictionary + rule + license/README hashes trong segmentation lock và manifest.
- Manifest digest **`ea127f55e840991e53569379a9b3e8ac00bc6798108d6472f8ab300a109f8f7d`**.
- Config canonical SHA-256 **`d7dba535773651d135881c9ac5dfccb3e91a4e041aeb127e1a9f8a01fdd409fe`**.
- Config file bytes SHA-256 `ac75920646981f37a232df04ebef032a464457e20356f939ed6842b4e60d7d81` (canonical parsed-YAML hash ở trên mới là hash dùng trong manifest).

| Split | Rows | Raw SHA-256 | Prepared JSONL SHA-256 |
|---|---:|---|---|
| train | 23048 | `8a18a73ca21105e0b424829cc41395719be98dd7f01bf3d962eca86e7a1501ec` | `6e0299a16444d22ccc5973da90a13d112645dbd0fd6e112f874b1e2473aee7e8` |
| validation | 3305 | `0b2567fe5374cf9b8d22de92ce71d3f9bfa036dffac46fa6f7a5650e85c81169` | `0212da01f7a5289b2472f6fca3bd28cc4b3a09ff4bb69ecbeae4d35af7d0c83a` |
| test | 6613 | `cdd81b2c7b96752f9f45d05e231682fcd6c380f40273dac521425ceff71ab312` | `0000e8d418bd591027bc6df2ed274870e306f78b34a91b3b0c3ed4e86bdea2c4` |

ID algorithm: `{split}:{source_row:06d}:{raw_sha256[:12]}`. First IDs: `train:000000:8a18a73ca211`, `validation:000000:0b2567fe5374`, `test:000000:cdd81b2c7b96`. Source row zero-based. Duplicate text vẫn phải giữ ID riêng.

Text variants: `raw` chọn unsegmented `text` đã curated của CSV; `segmented` chọn VnCoreNLP `text_segmented`. Loader trả list records, `text` là variant đã chọn, `text_raw` vẫn giữ text unsegmented; `sample_id,split,source_row,sentiment,topic,text_segmented` giữ nguyên. Training không clean lại; PhoBERT dùng segmented và thiếu variant sẽ fail. Manifest paths tương đối với thư mục manifest; không suy từ cwd.

## Lệnh và kiểm tra đã chạy thật

Tại repo root trên Windows, CPU; Python system 3.10 và môi trường installed editable `.venv` của 04 đều được kiểm tra. Java 22.0.2. Khi chạy segmentation đã set `NEU_ESC_JAVA=C:/Program Files/Java/jdk-22/bin/java.exe`.

```text
python -m neu_esc.data.resources --config configs/data/dataset.yaml --offline
python -m neu_esc.data.prepare --config configs/data/dataset.yaml --output data/processed --offline
python -m unittest discover -s tests/data -v
.venv/Scripts/python.exe -m pytest tests/data -q -o cache_dir=runs/smoke/data/pytest_cache
python tests/data/verify_real.py
.venv/Scripts/python.exe -m neu_esc.data.resources --config configs/data/dataset.yaml --offline
.venv/Scripts/python.exe tests/data/verify_real.py --output runs/smoke/data/verification_installed.json
```

- Nguồn GitHub được download thật qua HTTPS pinned URLs, băm bytes; ba hash raw khớp yêu cầu. CLI resource verifier xác nhận 5 file locked.
- Full final prepare chạy **hai lần cùng final code/config**: cùng digest ở trên, cùng processed hashes/IDs/counts; raw source không mất/đổi. Logs: `final_prepare.log`, `final_restart_prepare.log`; receipt: `final_receipt.json`.
- Unit/pytest: **15 tests pass, 23 subtests pass**. Fixture chỉ phục vụ failure gates; không có model/pretrained download hay benchmark giả.
- Real verifier: **PASS 32966 unique IDs**, đọc tất cả split ở cả hai variants, so từng row/labels/IDs/order với CSV gốc. Receipts `verification.json`, `verification_installed.json`.
- Checked manifest canonical digest, config canonical hash và implementation byte hashes; mã Python đã LF theo `.gitattributes` để tránh hash đổi khi checkout Linux. Read-only `git check-ignore` xác nhận raw/processed/resources và cache nguyên văn bị ignore. Không stage/commit.
- Initial manifests trước sửa mô tả trạng thái HF/normalization line endings được giữ trong `runs/smoke/data/initial_prepare_manifest.json` và `provenance_before_lf_normalization.json`, chỉ là archival receipts, không phải manifest dùng để train. Không training nào dùng chúng; raw và processed bytes không bị đổi.

## Đã xác minh

1. Split counts/schema/mapping 4 sentiment và 10 topic hợp lệ; topic `3=Other` theo Table 2. Null/empty/invalid labels 0 ở real data.
2. Không có exact duplicate hoặc duplicate sau NFC+collapse whitespace trong/giữa split. Không dedup/drop/resplit/relabel. Near duplicates chưa kiểm tra.
3. Topic tổng khớp mọi count Table 2. Sentiment Negative **5200** trong CSV vs **5250** Table 1; tổng Table 1 là 33016 vs Table 3/CSV 32966. Dataset giữ nguyên.
4. Test marginals của cả hai task khớp row totals Figure 3 theo module 03. Không coi đây là bằng chứng cùng text revision.
5. Author `_processed` đủ rows và label pairs aligned, tool/dictionary tạo chúng không tìm thấy trong pinned source. Không xử lý chúng hai lần.
6. Segmentation của pipeline đủ 32966 samples; khác author processed ở train 6723, validation 1005, test 1902 (9630 tổng). Bỏ underscore/collapse whitespace vẫn có 223/26/68 raw-vs-author mismatch. Các khác biệt đều công bố, không che bằng raw fallback.
7. Scope ledger đủ 21 Table 6 rows; 22 encoder runs cho một seed, campaign 70 encoder + 4 LLM. BERT-CNN và external datasets thiếu protocol/target riêng được ghi rõ.

## Giả thuyết chưa kiểm chứng / giả định

- Figure 3 lệch Table 6 là fact từ audit 03; lệch thứ tự predictions/labels, checkpoint khác hoặc dataset khác chỉ là giả thuyết. Chưa test gradient SMART ở module 01.
- GitHub cùng counts/marginals/HF README mô tả không chứng minh sample/text equality với gated HF hay revision dùng trong paper.
- CSV đã curated: cleaning identity để không double-clean. Không có dictionary viết tắt gốc; không tự dựng dictionary hay khôi phục scrape text.
- VnCoreNLP 1.2 chọn theo PhoBERT upstream, không phải version paper công bố. Segmented text là alternative preprocessing cần giữ trong provenance/cohort.
- Seeds 42–46, CM seed 42 và campaign 70 là lựa chọn dự án; paper không công bố multi-seed protocol tương ứng.
- Chưa thiết lập quyền phân phối bản GitHub; HF metadata Apache-2.0 chưa chứng minh quyền cho bản GitHub. Dataset/binary/dictionaries chỉ cache local và không có trong release allowlist.

## Contract / blockers / input cần tiếp

Không có breaking interface change. Additive CLI `resources`, `--offline`, `--manifest-path`; API thêm `text_raw`. Đổi cohort phải dùng output và manifest path mới; target đã tồn tại khác config/code/hash bị từ chối. Java bridge chỉ sửa resource file lookup `%20`, không đổi algorithm/model.

- **HF/paper provenance blocker**: cần quyền HF và exact dataset revision paper, hoặc xác nhận tác giả; cần tool/dictionary/cleaning gốc nếu muốn tuyên bố exact preprocessing. Pipeline kỹ thuật READY dưới notice `unverified_alternative_revision`. Không đăng lại CSV/JSONL vào repo.
- **04**: cài/pin PyYAML và JDK >=11 trong recipe Windows/Colab. Bridge đã embed trong `.py`, không cần chỉnh package-data. Linux example JDK17; phải xác minh runtime rồi chạy `python -m neu_esc.data.resources --config configs/data/dataset.yaml` (online) trước prepare. Do not use downloader unpinned của py_vncorenlp. Source CLI online tải raw CSV theo hash khi cache chưa có.
- **02/03/05**: dùng manifest/digest/API trên, giữ IDs tới predictions; không clean lại. Nếu muốn original author `_processed` làm diagnostic `reference_code`, cần config/manifest/cohort riêng và coordinated interface decision; hiện pipeline không cung cấp variant thứ ba ngầm.
- **06**: kiểm data gate kỹ thuật với notice revision thay thế, kiểm installed package API và snapshot; không đánh dấu exact paper-equivalence gate pass. Synthetic tests không thay benchmark.
- **07**: full GPU training chỉ trên checkout/commit release của 04, sau integration; ghi manifest digest/source notice trong cohort/provenance. Chưa kết nối/xác minh Colab, GPU/Drive/quota; module 01 không chạy full training/LLM evaluation.
- **08**: source/count/preprocessing findings có thể báo sớm; nguyên nhân khác điểm chưa chứng minh.

## Allowlist READY cho module 04

Chỉ stage các file sau sau khi kiểm diff; không `git add .`, không stage cache dữ liệu/tài nguyên/runs hay file module khác. Không xóa caches của người dùng. Metadata dưới đây không chứa sample text; fixture tests chỉ synthetic. Nếu code/config bị sửa sau bàn giao thì invalidate digest/cohort và rerun prepare/tests trước snapshot.

- `data/.gitignore`
- `data/manifests/dataset.json`
- `data/manifests/label_maps.json`
- `configs/data/dataset.yaml`
- `configs/data/segmentation.lock.json`
- `src/neu_esc/data/__init__.py`
- `src/neu_esc/data/_java_bridge.py`
- `src/neu_esc/data/common.py`
- `src/neu_esc/data/audit.py`
- `src/neu_esc/data/dataset.py`
- `src/neu_esc/data/prepare.py`
- `src/neu_esc/data/resources.py`
- `src/neu_esc/data/segmentation.py`
- `docs/source_audit.md`
- `docs/data_audit.md`
- `tests/data/test_pipeline.py`
- `tests/data/verify_real.py`
- `reports/source/.gitignore`
- `reports/source/data_statistics.json`
- `reports/source/duplicate_groups.jsonl`
- `reports/source/download_ledger.json`
- `reports/source/github_tree.json`
- `reports/source/hf_access.json`
- `reports/source/hf_csv_access.json`
- `reports/source/hf_dataset.json`
- `reports/source/paper_data_comparison.json`
- `reports/source/phobert_guidance_provenance.json`
- `reports/source/raw_processed_alignment.json`
- `reports/source/scope_ledger.json`
- `reports/source/source_ledger.json`
- `handoffs/01-data.md`
