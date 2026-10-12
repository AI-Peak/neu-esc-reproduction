# Module 06 — integration / release gates

Status: PARTIAL

Ngày: **12/10/2026, Asia/Saigon**. P0: **CPU_PASS/GPU_PENDING**. Full Colab release: **BLOCKED** (I06-01). Validation reporting: **FAIL** (I06-02). Không full train, không sửa file 01–04, không stage/commit hoặc tạo/nhắn chat/subagent.

Người dùng đã loại benchmark LLM: **LLM OUT_OF_SCOPE / NOT_APPLICABLE**, không chờ module 05 và không dùng thiếu LLM như blocker. Ưu tiên ViSoBERT 2-task + SMART (`paper_method`). Campaign encoder nếu triển khai toàn bộ vẫn 22 configs/70 successful runs; historical Table 6 reference giữ nguyên.

## Output / allowlist READY của 06

Repo: `F:/DPL project/neu-esc-reproduction`.

- `scripts/integration/audit_pipeline.py`: bounded fresh-process E2E sáu variants, interrupt/resume và negative ID/labels/checkpoint checks.
- `scripts/integration/clean_install.py`: isolated wheel/venv CPU install + P0 smoke, nguồn bytes hiện tại chứ không tự tạo release commit.
- `scripts/integration/report_gates.py`: tạo gates từ receipts thật, input hashes và Git HEAD; chưa là generator chung cho các đợt sau sửa I06-02.
- `scripts/integration/reproduce_validation_metadata.py`: reproduction I06-02, exit 2 hiện tại; run mới sau sửa phải exit 0.
- `tests/integration/test_boundaries.py`: 22 config/hash/manifest checks, continuous SMART gradient thật, real test IDs/permutation/negative truth.
- `docs/integration.md`: commands, evidence, limits và lệnh Colab/P0 chuẩn bị.
- `reports/integration/gates.json`: per-gate statuses, inputs/hash/commit, tests/errors/GPU_PENDING.
- `reports/integration/input_snapshot.json`: SHA-256 executable/config/test/lock/manifest bytes và files chưa nằm trong HEAD.
- `reports/integration/campaign_ledger.json`: 22 config/70 seed slots theo scope encoder, actual `runs/encoder` scan, 0 successful unique cohort/seed có benchmark artifacts; không tính smoke/retry.
- `reports/integration/fix-prompts.md`: prompts cho 02/04 và đầu vào cần thực thi của 07.
- `handoffs/06-integration.md`: file này.

Đây là **allowlist file READY của 06 để 04 review**, không có nghĩa toàn pipeline/release READY. Chỉ stage các file này sau review diff; không stage `runs/` hoặc venv/wheel/cache. Không có artifact benchmark.

## Source / revision / hash / config

- Local HEAD: `1bac0e991ecf200e1356debe9ff79d01f331c2ad`; gồm **24 file infrastructure**. `git show HEAD:src/neu_esc/training/train.py` thực tế trả fatal: file chỉ ở disk, chưa trong HEAD. Implementation/configs 01–03 untracked. Không có full release SHA.
- Reference GitHub/data: `b5637cc0a8a7b5e88e794d180c11902405548eb5` qua manifest 01.
- Manifest: `ea127f55e840991e53569379a9b3e8ac00bc6798108d6472f8ab300a109f8f7d`. Prepared/raw hashes trong manifest được real verifier đọc lại.
- P0: `configs/experiments/visobert/mtl_smart.yaml`; model/tokenizer `196a62afad9cbe4f52a54aabad828b13f0eec59a`; batch8/accum16/effective128, length128, fp32, LR2e-5, epochs20, SMART weight0.02 như resolved config 02. Mọi config/handoff/inventory được đọc từ disk; all 22 config byte hashes khớp inventory 02.
- Handoff 01/02/03 READY là readiness module; 04 PARTIAL/infrastructure READY, chưa release. Không suy ra whole pipeline đã sẵn sàng từ các nhãn READY đó.
- CPU stack: Python3.10.11, torch2.14.1+cpu/transformers4.57.6/numpy2.2.6/PyYAML6.0.3/Pillow12.3.0/pytest9.1.1 theo môi trường installed 04; fresh venv dùng hash-locked CPU requirements. CUDA không có. Không dùng CPU lock cho Colab.
- Input byte hashes đầy đủ: `reports/integration/input_snapshot.json`. Code change sau audit invalidate các gates phụ thuộc; cần recheck đúng input/commit.

## Commands/checks đã chạy thật

Lệnh đầy đủ tại `docs/integration.md`; receipts ignored local trong `runs/smoke/integration/`.

| Check | Actual result |
|---|---|
| `pytest tests/data tests/training tests/evaluation scripts/environment/tests` | **79 passed, 8 skipped GPU, 85 subtests passed**; `module-tests.xml` |
| `pytest tests/integration` | **3 passed**; `integration-tests.xml` |
| `tests/data/verify_real.py --output .../real-data-verification.json` | **32966 unique IDs**, all source rows/text/labels/raw+segmented/split/hash aligned |
| `audit_pipeline.py --output .../end-to-end-001` | **6 variants PASS**, real package CLI train→checkpoint→fresh predict→evaluator; fixture-only tiny BERT |
| P0 no-test-file train/resume | PASS; train uses train/validation; prediction before COMPLETED rejected |
| Negative CLI ID/truth/duplicate/missing task | All exit 2, no output report |
| Stale seed/revision/manifest resume + missing best COMPLETE | All rejected; checkpoints restored after controlled negative probes |
| CPU resume / integrity module tests | Model/Adam/scheduler/scaler/Torch RNG bitwise exact across six variants; incomplete/corrupt checkpoint rejected |
| Continuous SMART integration | Noise gradient >0, encoder/two heads gradient >0, finite, special/pad mask=0, projected radius respected |
| Clean build/install | New isolated venv, hash-locked deps, wheel from copied current source, site-packages imports, `pip check`, four CLI helps, P0 E2E all PASS |
| Local pinned ViSoBERT and XLM-R tokenizer loaders | PASS on locked local tokenizer bytes with `use_fast=False`; no pretrained weight load |
| Validation artifact recheck | All six saved predictions reproduce metrics and selection score; `end-to-end-001/validation-artifact-checks.json` |
| P0 report regeneration | All **12 JSON/CSV/PNG files byte-identical**; `regeneration-check.json`; topic PNG viewed and labels/cells verified |
| Real test permutation diagnostic | N=6613/task, 13226 ID-task pairs, truth join and permutation invariant; oracle labels only, no benchmark exported |
| Single/five seed statistics | 03 tests verify ddof=1/null singleton; six E2E single-seed test aggregates work |
| Full pretrained / Colab / notebook / Drive | **NOT RUN / GPU_PENDING**; notebook 07 absent |

Main receipts: `end-to-end-001/receipt.json`, `clean-install-001/receipt.json`, `real-data-verification.json`, `regeneration-check.json`, JUnit XMLs. Synthetic test N=10/task; ST has10 lines, MTL20 lines. These are not dataset benchmark scores.

## Errors / chủ module / expected vs actual

**I06-01 — 04 — P1 — BLOCKS Colab release.** Expected: clean pinned commit đủ data/train/evaluator/configs/integration. Actual: HEAD infrastructure chỉ24files, training source chưa trong commit. Clone trên Colab không có module `neu_esc.training`. Reproduce: `git show HEAD:src/neu_esc/training/train.py`; tracked tree và missing executable files được lưu trong input snapshot/gates. 04 cần chốt candidate từ READY allowlists 01/02/03/06, không chờ05; 06 recheck clean checkout rồi07 dùng SHA đó.

**I06-02 — 02 — P2 — FAIL validation reporting.** Expected: validation metrics persist metadata `method_mode,backbone,variant` cùng config/provenance để aggregate được. Actual: cả sáu variants thiếu ba fields; `aggregate_cohort` báo `aggregation requires nonempty method_mode`. Reproduce: `python scripts/integration/reproduce_validation_metadata.py --run-dir runs/smoke/integration/end-to-end-001/mtl_smart` → exit2. Test metrics qua evaluator CLI có đủ metadata. Lỗi không làm đổi số metric/selection score đã kiểm, không chặn test CLI metrics; nên sửa trước release để artifacts đầy đủ và tránh cohort đổi sau training.

Prompts copy sang chat phụ trách: **`reports/integration/fix-prompts.md`**. Không sửa hộ model/data/evaluator/packaging.

## Điều chưa được chứng minh / giả thuyết

- Tiny initialized BERT CPU không chứng minh pretrained ViSoBERT GPU forward/backward, batch8/length128/accum16 VRAM fit, runtime throughput, GPU determinism/resume hay performance.
- Chưa có `notebooks/colab_reproduce.ipynb` hoặc output phiên Colab; cell order/commit pin/secrets/Drive checksum/completion/queue/resume chưa thể chứng nhận. Đây là input chờ **07**, không phải notebook bug đã chứng minh.
- Dataset là pinned GitHub **alternative revision**. Full-source labels/IDs hợp lệ không chứng minh exact HF/paper content/preprocessing equivalence. Đây là limitation tái lập, không biến thành LLM blocker hoặc che như đã verified.
- Figure3 khác Table6 và giả thuyết sai thứ tự/checkpoint/dataset vẫn chưa chứng minh nguyên nhân. Permutation test ở pipeline hiện tại không giải thích được hình paper cũ.
- Chưa có benchmark encoder run/predictions/report, successful benchmark count **0/70**; chưa có final artifacts của07/08 để final audit. Smoke/fixtures/oracle/retries không tính vào70.

## Assumptions / contract deviations

Không đổi shared interface hoặc sửa file ngoài ownership. Các defaults/physical batch/continuous SMART/normalization/checkpoint rule/5seeds là assumptions đã ghi config02, không được nhận như protocol paper đủ xác minh. Smoke config chủ ý dùng mode=smoke, length12, tiny model và fixture manifest, không thay config full gốc. Clean-wheel source là current bytes; không claim nó là committed release. Source audit/tablepaper thuộc01/03 chỉ đọc.

LLM gate chuyển thành NOT_APPLICABLE theo **quyết định người dùng**, ưu tiên hơn prompt/contract cũ yêu cầu chờ05. Không gọi API/LLM runner, không tạo pseudo LLM results.

## Blockers / bước tiếp theo / inputs cần

1. **02:** sửa I06-02 trong ownership, regression/CPU tests + fresh CLI smoke, inventory/allowlist/handoff mới. Gửi output về người dùng để06 recheck.
2. **04:** dùng allowlists READY mới, commit release candidate đầy đủ; bàn giao SHA + clean checkout + CI/install receipts. Không stage unready/shared files, không chờ05. 06 cần verify candidate execution bytes và CPU gates theo SHA đó.
3. **07:** notebook/Colab GPU runtime logs, pretrained ViSoBERT P0 smoke đúng physical batch/length/accumulation, optimizer-boundary GPU resume và Drive sync/restore/checksum markers. Default GPU tests hiện có chỉ batch2/length24 ST/MLM+SMART; chưa thay được exact P0 smoke. Xem prompt07 trong fix-prompts.
4. Khi release/GPU/Drive gates đạt mới full seed42 rồi campaign seeds đã khóa; **08** nhận stable predictions/manifest/cohort để final metrics/CM/report audit. Không cần retrain để recheck saved prediction reports.

Phần CPU/data/config/artifact/clean-install đã xong và evidence bảo toàn; phần clean committed release và Colab/GPU/Drive/final benchmark chưa chạy. Không tạo giấy chứng nhận READY toàn dự án khi những đầu vào này chưa có.
