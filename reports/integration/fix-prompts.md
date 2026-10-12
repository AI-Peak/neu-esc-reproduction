# Prompts chuyển về module phụ trách — 12/10/2026

Phạm vi đã quyết định: benchmark LLM bị loại; không chờ module 05. Ưu tiên ViSoBERT `mtl_smart`, hai task sentiment/topic, `paper_method`, Colab GPU. Không thay các số paper gốc trong `reports/reference/table6.csv`. Các mẫu/metrics trong `runs/smoke/integration/` chỉ kiểm pipeline, không phải benchmark.

## 1. Module 02 — sửa I06-02 trước khi chốt snapshot

```text
Tiếp tục module 02 tại F:/DPL project/neu-esc-reproduction. Module 06 đã kiểm đầu ra thật và phát hiện I06-02 (P2): trainer persist evaluation/validation/metrics.json thiếu method_mode, backbone và variant. aggregate_cohort([metrics], expected_seeds=[42]) từ chối với "aggregation requires nonempty method_mode". Test metrics qua evaluator CLI có đủ các trường này. Lỗi không thay giá trị metric hoặc validation checkpoint selection đã kiểm.

Bằng chứng:
- runs/smoke/integration/end-to-end-001/receipt.json: cả sáu variants đều có validation_aggregation fail cùng lỗi.
- runs/smoke/integration/end-to-end-001/mtl_smart/evaluation/validation/metrics.json.
- runs/smoke/integration/end-to-end-001/validation-artifact-checks.json: metric từ saved validation predictions và selection score vẫn khớp, cả sáu variants.
- reports/integration/gates.json, handoffs/06-integration.md.

Reproduce từ repo root sau install:
python scripts/integration/reproduce_validation_metadata.py --run-dir runs/smoke/integration/end-to-end-001/mtl_smart
Actual: exit 2, missing_fields=[method_mode,backbone,variant]. Expected với run mới sau sửa: exit 0.

Chỉ sửa file thuộc module 02. Bổ sung backbone/variant vào provenance và persist method_mode/backbone/variant từ resolved config/provenance vào validation metrics, nhất quán evaluator CLI. Không đổi selection rule, formulas, weights, SMART/MLM, data hay evaluator. Không vá thủ công artifact cũ, không tự đổi schema/interface module khác. Nếu cần thay helper nội bộ publish_validation thì cập nhật các caller trong phạm vi 02.

Thêm regression tests trong tests/training: train tiny -> đọc validation metrics đã ghi -> aggregate_cohort thành công và kiểm metadata bằng resolved config; bao phủ ST sentiment/topic và MTL. Đối chiếu metrics validation tái tạo qua API/CLI với predictions đã lưu, và checkpoint score không đổi. Chạy tests/training CPU và bounded fresh-process CLI smoke; không full train.

Cập nhật delivery_inventory.json/02-ready-allowlist.txt, handoff 02 và hash bằng chứng của output mới; đưa module 06 kết quả để recheck trước release. Implementation hash thay đổi phải thuộc snapshot/cohort mới. Chưa có benchmark run nào cần bảo toàn (module 06 chỉ tạo smoke).

Không sửa/stage file module khác, không tạo/nhắn chat/subagent. LLM ngoài phạm vi, không chờ 05. Bạn không làm việc một mình trong cây dùng chung; giữ nguyên sửa đổi của các module khác.
```

## 2. Module 04 — khép I06-01 và cung cấp release candidate đầy đủ

```text
Tiếp tục module 04 tại F:/DPL project/neu-esc-reproduction. Quyết định người dùng: bỏ benchmark LLM, không chờ module 05 hoặc coi thiếu LLM là blocker. Ưu tiên release ViSoBERT 2-task + SMART chạy trên Google Colab GPU.

Module 06 đã kiểm thực tế: HEAD=1bac0e991ecf200e1356debe9ff79d01f331c2ad chỉ có 24 file infrastructure. Lệnh "git show HEAD:src/neu_esc/training/train.py" trả fatal (path chỉ tồn tại ở disk); data/training/evaluation/configs là untracked. I06-01 (P1) chặn Colab release: clone/checkout commit này không có pipeline thật. Đây là phần release còn thiếu, không phải lỗi thuật toán.

Bằng chứng: reports/integration/gates.json và input_snapshot.json, handoffs/06-integration.md. CPU kiểm thực tế của 06: 79 module tests passed, 8 GPU skipped, 85 subtests; 3 integration tests passed; cả sáu variants fresh-process tiny train/predict/evaluate pass; fresh isolated venv + hash-locked CPU dependencies + built wheel + P0 smoke pass. Chưa chứng minh Colab GPU.

Sau khi 02 sửa I06-02 và giao READY/hashes mới, đọc lại handoffs 01/02/03/06, đối chiếu allowlist/hash mới và review diff. Chỉ stage file READY theo allowlist, tuần tự; không git add ., không stage dữ liệu/raw/processed/resources/checkpoints, smoke outputs hoặc WIP. Module 05 không nằm trong dependency/release gate. Giữ historical Table 6 gốc, nhưng docs/campaign mới chỉ encoder: 22 configs/70 successful runs, LLM OUT_OF_SCOPE.

Làm release candidate đủ data API + model/loss/training + evaluator + pinned configs/manifests + tests/integration + scripts/integration. Kiểm Linux/Windows CPU CI với implementation thật; CI infrastructure 7 tests cũ không đủ. Dùng checkout sạch cố định để test install/CLI và pipeline; không gọi shared worktree dirty là reproducible. Giao SHA đầy đủ để 06 verify clean checkout, hashes, imports/CLI và CPU smoke trước khi kết luận release READY; docs-only cập nhật sau đó phải xác minh các execution bytes không đổi.

Giữ recipe Colab editable với constraints bảo vệ torch/CUDA stack; runtime Python/PyTorch/Transformers phải hỗ trợ tải pinned ViSoBERT pytorch_model.bin thực, không chỉ tiny initialized model. Khóa Pillow>=10.1 trực tiếp cho renderer nếu chốt dependency contract; protobuf nếu tokenizer/runtime đã chứng minh cần, không tự coi thiếu package này là blocker P0 vì loader ViSoBERT local hiện đã pass. Không thay CUDA stack mù hoặc dùng CPU lock trên Colab. Pip check/fresh-process import/CUDA tensor phải qua trên runtime thật của 07 trước full run.

Không full train, không sửa model/data/evaluator/notebook của module khác, không tự tạo/nhắn chat/subagent. Cập nhật handoff 04 với full SHA, allowlist, dependency receipts, CI actual và trạng thái release candidate/GPU_PENDING; không giữ yêu cầu chờ 05.
```

## 3. Module 07 — đầu vào còn thiếu, chưa phải lỗi notebook đã chứng minh

```text
Thực thi module 07 theo ownership notebook/campaign/queue/sync đã giao. Phạm vi mới loại hoàn toàn benchmark LLM; không chờ 05. Ưu tiên ViSoBERT mtl_smart, sentiment+topic, paper_method; seeds 42-46 khi các gate cho phép. Giữ 22 configs/70 successful encoder runs cho campaign encoder nếu triển khai toàn bộ; không tính smoke/retry.

Đọc handoffs 01-04/06 và reports/integration/gates.json. Hiện chưa có notebooks/colab_reproduce.ipynb, runtime GPU/Drive output hoặc full release candidate được 06 verify. Không nhận đã chạy Colab và không full train từ HEAD infrastructure hoặc shared worktree dirty.

Sau khi có SHA đầy đủ từ 04 và gate CPU của 06: tạo notebook chỉ gọi package CLI, checkout detached SHA ở /content/neu-esc-reproduction, kiểm sạch và relative paths; runtime CUDA/VRAM/disk/Python/GPU capability, constrained install của 04, fresh-process imports/CUDA tensor, dataset resources/prepare/verify và manifest digest đúng config. Người dùng tự bật GPU/mount Drive/quản lý secrets.

Chạy GPU smoke hữu hạn cho ViSoBERT thật theo SHA đã khóa. Có thể bắt đầu bằng NEU_ESC_RUN_GPU_SMOKE=1 python -m pytest tests/training/test_gpu_backbones.py -q -k visobert. Hai case hiện hữu là ST và MLM+SMART, batch2/length24; CHƯA phải smoke chính xác P0. Bổ sung kiểm ViSoBERT mtl_smart trực tiếp, hai heads 4/10 và continuous SMART gradients hữu hạn/khác0; batch8, length128, accum16, fp32 như config thực; đo peak VRAM/throughput và ghi resolved settings.

Throughput CLI có thể ngắt bằng --stop-after-steps 2 trên full config, run-dir mới; đó là INTERRUPTED diagnostic, không benchmark thành công. Khi ngắt trước validation đầu tiên, chỉ latest có thể tồn tại, không giả định có best hay cho predict test. Không âm thầm đổi epochs/batch/precision/hash để làm smoke chạy được. GPU resume phải kiểm đúng ranh giới optimizer và giữ RNG/optimizer/scheduler/scaler/config/environment; GPU/stack đổi thì restart seed bằng run ID/cohort/ledger đúng.

Sync metadata/logs/tokenizer/best/latest về Drive theo checksum và completion marker, phân biệt best với latest; test bản thiếu marker/checksum sai/stale config/manifest/revision bị từ chối và restore về disk local. Queue idempotent, một job mỗi runtime; không bỏ test samples, không chọn seed theo test.

Giao notebook/logs/runtime receipts/GPU smoke/resume/sync receipts về người dùng để module 06 review. Chỉ bắt đầu full seed42 khi release + GPU smoke + Drive safeguards đã kiểm; không fallback full CPU, không mua quota/né giới hạn. Không sửa implementation/config gốc, không tự tạo/nhắn chat/subagent.
```
