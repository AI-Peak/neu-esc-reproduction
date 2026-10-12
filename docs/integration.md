# Integration 06 — trạng thái 12/10/2026

Ưu tiên ViSoBERT `mtl_smart`, sentiment + topic, `paper_method`, Google Colab GPU. LLM **OUT_OF_SCOPE** theo quyết định người dùng; thiếu module 05 không chặn gate. Bảng paper gốc vẫn giữ nguyên như reference. Campaign encoder đầy đủ vẫn gồm 22 config/70 lượt thành công; module 06 không full train.

Kết luận: **CPU_PASS/GPU_PENDING**, release **BLOCKED**, validation reporting có lỗi metadata **I06-02**. Xem `reports/integration/gates.json`, `handoffs/06-integration.md` và prompts tại `reports/integration/fix-prompts.md`.

## Các lệnh đã chạy thật

Từ `F:/DPL project/neu-esc-reproduction`, interpreter `.venv/Scripts/python.exe` trên Windows CPU:

```powershell
.venv/Scripts/python.exe -m pytest tests/data tests/training tests/evaluation scripts/environment/tests -q --junitxml=runs/smoke/integration/module-tests.xml -o cache_dir=runs/smoke/integration/pytest-cache
.venv/Scripts/python.exe -m pytest tests/integration -q --junitxml=runs/smoke/integration/integration-tests.xml -o cache_dir=runs/smoke/integration/pytest-cache
.venv/Scripts/python.exe tests/data/verify_real.py --output runs/smoke/integration/real-data-verification.json
.venv/Scripts/python.exe scripts/integration/audit_pipeline.py --output runs/smoke/integration/end-to-end-001
.venv/Scripts/python.exe scripts/integration/clean_install.py --output runs/smoke/integration/clean-install-001
.venv/Scripts/python.exe scripts/integration/reproduce_validation_metadata.py --run-dir runs/smoke/integration/end-to-end-001/mtl_smart
.venv/Scripts/python.exe scripts/integration/report_gates.py
```

Các output directory smoke là exclusive; lần chạy mới phải dùng tên mới. `report_gates.py` nhận `--e2e` và `--install` nếu dùng receipt mới; script này dành chốt audit hiện tại và sẽ yêu cầu cập nhật kỳ vọng I06-02 sau khi module 02 sửa. Không overwrite evidence hoặc cố tình tái tạo lỗi sau sửa.

Module tests: **79 passed, 8 GPU skipped, 85 subtests passed**. Integration: **3 passed**. Đối chiếu source thật: **32966 unique IDs**, cả raw/segmented variants với CSV; đúng 23048/3305/6613. Loader tokenizer ViSoBERT local pinned đã chạy được, không tải pretrained weights.

Audit E2E chạy sáu variants qua CLI trong tiến trình mới; P0 interrupt/resume ở step 2. Với fixture test N=10, ST xuất 10 prediction rows, MTL xuất 20; mỗi CM có N=10, shape 4×4/10×10 và trace/N bằng accuracy. Không trình bày scores này như benchmark. Tiny model là **random BERT**, dù config fixture lưu ID ViSoBERT theo contract; không nhận đã train/load full ViSoBERT.

Train/resume chạy được khi test JSONL bị giữ ngoài đường dẫn manifest; predict trước COMPLETED bị từ chối. CLI sai ID/y_true, duplicate/missing task đều exit 2 trước tạo report. Resume sai seed/revision/manifest, checkpoint thiếu COMPLETE đều fail. Các tests 02 còn xác nhận model/optimizer/scheduler/scaler/RNG bitwise bằng run liên tục ở cả sáu variants, checkpoint checksum corruption bị từ chối.

SMART integration thực thi autograd, kiểm inner noise gradients dương, encoder/hai heads có gradient; pad/special noise bằng 0 và projection không vượt epsilon. Bộ tests 02 kiểm `.long()` ngắt gradient và MLM masking/loss denominator/no-target. Không chỉ suy từ tên tests; đã đọc bodies và chạy assertion thật.

Fresh clean install build wheel từ bản sao bytes của source hiện tại trong namespace smoke, tạo venv riêng, cài `requirements-cpu.lock` có hash, wheel, `pip check`, imports/CLI help, rồi P0 E2E. Import trỏ vào site-packages venv mới; không dùng editable của môi trường cũ. Đây là kiểm packaging CPU trên **uncommitted bytes**, chưa là clean committed release.

## Tái tạo artifact từ predictions lưu

```powershell
.venv/Scripts/python.exe -m neu_esc.evaluation.evaluate --predictions runs/smoke/integration/end-to-end-001/mtl_smart/predictions/test.jsonl --dataset-manifest runs/smoke/integration/end-to-end-001/dataset.json --output runs/smoke/integration/regenerated-p0-test-001
```

Đã so bytes/hashes **12 artifacts** (JSON/CSV/PNG counts + normalized + per-class), tất cả giống lần đầu. Receipt `runs/smoke/integration/regeneration-check.json`. Validation cả sáu variants đã tái tính bằng API từ saved predictions; tasks metrics và selection score đều khớp, receipt `validation-artifact-checks.json` trong E2E folder. PNG topic P0 cũng đã mở kiểm tra, đủ nhãn/ô.

Real-test identity diagnostic giữ 13226 ID-task pairs và N=6613/CM, permutation không đổi kết quả; predictions diagnostic dùng nhãn nguồn làm oracle **chỉ để kiểm ID/denominator**, không xuất báo cáo benchmark. Sample std ddof=1 và single-seed std=null được kiểm bằng tests 03; E2E single-seed aggregate test metrics cũng thành công.

## Lệnh Colab sau khi giải quyết release gate

Đây là lệnh chuẩn bị cho 07, **chưa thực thi trên Colab**. Trước hết cần SHA release đủ implementation từ 04 và clean checkout được 06 xác nhận. Không dùng HEAD `1bac0e991ecf200e1356debe9ff79d01f331c2ad`: commit này chỉ có infrastructure.

```bash
cd /content/neu-esc-reproduction
git checkout --detach <VERIFIED_FULL_RELEASE_SHA>
python scripts/environment/capture.py --output runs/environment/colab-preflight/environment.json --require-cuda --require-clean
python scripts/environment/colab_install.py --evidence-dir runs/environment/colab-plan
python scripts/environment/colab_install.py --evidence-dir runs/environment/colab-install --apply
# Restart session if required by dependency changes, then repeat preflight.
python scripts/environment/offline_tests.py
python -m neu_esc.data.resources --config configs/data/dataset.yaml
python -m neu_esc.data.prepare --config configs/data/dataset.yaml --output data/processed
python tests/data/verify_real.py --output runs/environment/colab-real-data.json
python -m pytest tests/integration -q
NEU_ESC_RUN_GPU_SMOKE=1 python -m pytest tests/training/test_gpu_backbones.py -q -k visobert
```

Nguồn prepared bytes/digest trên Linux phải khớp manifest đã khóa. Nếu không khớp, dừng và giao evidence về 01; không sửa digest/config để resume run cũ. Resources cần Java/JDK phù hợp; không đưa dataset/resources vào Git.

GPU tests hiện hữu lọc `visobert` chỉ kiểm ST và MLM+SMART với batch2/length24. 07 còn phải kiểm P0 `mtl_smart` với model pretrained thật và batch8/length128/accum16, peak VRAM/throughput, checkpoint/GPU resume và Drive sync. Chưa có notebook để review cell order, Secrets, runtime detection hay queue idempotency.

Sau release/GPU smoke/Drive gates, full run từ notebook 07 gọi đúng CLI (không chạy các lệnh full này trong module 06):

```bash
python -m neu_esc.training.train --config configs/experiments/visobert/mtl_smart.yaml --seed 42 --run-dir runs/encoder/visobert-mtl-smart-seed42-UNIQUE
python -m neu_esc.training.predict --run-dir runs/encoder/visobert-mtl-smart-seed42-UNIQUE --split test
python -m neu_esc.evaluation.evaluate --predictions runs/encoder/visobert-mtl-smart-seed42-UNIQUE/predictions/test.jsonl --dataset-manifest data/manifests/dataset.json --output runs/encoder/visobert-mtl-smart-seed42-UNIQUE/evaluation/test
```

Full MTL test: 13226 prediction rows, **6613 samples ở từng CM**, không cộng hai task thành N của một CM. Seed42 CM được khai báo trước. Resume cùng seed/cohort/config/software/GPU topology và latest đầy đủ; không resume incomplete Drive copies. Runtime mới khác môi trường cần restart seed đúng ledger. Không chuyển full training sang CPU.

## Thứ tự bàn giao

1. 02 sửa metadata validation và giao inventory/allowlist READY mới.
2. 04 review allowlist 01/02/03/06, commit release đầy đủ tuần tự; 06 kiểm clean checkout trên SHA cụ thể. Module 05 không nằm trong chuỗi này.
3. 07 tạo notebook, chạy GPU smoke/resume/sync trên runtime thật, giao log cho 06.
4. Sau gates mới chạy full seed42 rồi seeds đã khai báo. Module 08 chỉ dùng predictions benchmark đã xác minh; không dùng fixture, oracle diagnostics hay retries để đếm 70 runs.
