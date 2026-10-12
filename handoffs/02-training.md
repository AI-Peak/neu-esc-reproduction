# Module 02 - encoder, SMART/MLM, training và prediction

Status: READY

P0_READY: YES - ViSoBERT MTL+SMART implementation/config/tokenizer/access audit/CPU tests đã bàn giao.

FULL_VARIANTS_READY: YES - đủ 22 config một seed, ST và cả bốn biến thể MTL; đã test sáu variants trên tiny CPU.

GPU_EXECUTION: PENDING - đây là readiness của module implementation cho 04/06 integration, không phải chứng nhận Colab/GPU/full pretrained backward hay benchmark đã hoàn thành.

Ngày cập nhật bàn giao: 12/10/2026, Asia/Saigon (bản gốc 11/10/2026). Không tạo chat/subagent, không gửi message cho module khác, không sửa root/data/evaluator/notebook/queue hoặc stage/commit. LLM đã loại khỏi phạm vi, không chờ module 05.

## Sửa I06-02 ngày 12/10/2026 - READY cho 04/06 recheck

**I06-02_FIXED_LOCAL / MODULE_06_RECHECK_PENDING.** Đã đọc `handoffs/06-integration.md`, `reports/integration/gates.json` và mục 1 của `reports/integration/fix-prompts.md`. Không sửa gates/handoff/script của 06. Full release và Colab vẫn cần các gates riêng của 04/06/07.

Thay đổi production chỉ ở **hai file**:

- `src/neu_esc/training/train.py`: thêm `backbone`, `variant` từ resolved config vào provenance; truyền config cho validation publisher.
- `src/neu_esc/training/runtime.py`: persist `method_mode`, `backbone`, `variant` trong validation metrics; cùng precedence provenance/fallback config và conflict rejection với evaluator CLI.

Các file hỗ trợ sửa/cập nhật: `tests/training/test_cli.py`, `reports/method/cli_smoke.py`, `docs/training.md`, handoff này, inventory và allowlist. File mới: `tests/training/test_validation_metadata.py`, `reports/method/verify_validation_metadata_fix.py`, `reports/method/i06-02-baseline.json`, `reports/method/i06-02-verification.json`.

Không đổi công thức metrics, checkpoint selection hoặc engine, model, SMART/MLM, 22 full configs, manifest/dữ liệu hoặc evaluator. SHA-256 của model/loss/config/data/evaluator files được so với baseline và giữ nguyên. Entire old run `runs/smoke/integration/end-to-end-001/mtl_smart` cũng giữ nguyên file set + mọi byte hash; old validation metrics vẫn thiếu ba fields như bằng chứng 06 đã ghi. Không dùng sửa artifacts cũ để làm kiểm tra đạt.

Lệnh thực chạy từ repo root với `.venv/Scripts/python.exe` của 04 và `PYTHONPATH=src`:

| Check | Kết quả |
|---|---|
| `python scripts/integration/reproduce_validation_metadata.py --run-dir runs/smoke/integration/end-to-end-001/mtl_smart` | Reproduce cũ **exit 2**, thiếu cả ba fields và `aggregation requires nonempty method_mode`; receipt baseline lưu stdout/exit thật. |
| `python -m pytest tests/training -q -o cache_dir=runs/smoke/training/i06-02-pytest-cache --junitxml=runs/smoke/training/i06-02-tests.xml` | **34 passed, 8 GPU skipped**, 87.86s. Bao phủ CLI training/persisted validation/aggregate cho ST sentiment, ST topic, MTL+SMART, MTL+MLM+SMART; unit fallback và conflict của cả ba metadata fields. Six-variant model/optimizer/RNG resume tests vẫn pass. |
| `python reports/method/verify_validation_metadata_fix.py` | **PASS**, ba fresh CPU CLI runs ST sentiment/topic/MTL+SMART; mỗi run train -> interrupt step 2 -> resume process mới -> predict -> evaluator CLI test và validation regeneration vào directory mới. |
| Reproduction script 06 trên mỗi fresh run bên dưới | Cả ba **exit 0 / PASS**. `aggregate_cohort([validation_metrics], expected_seeds=[42])` thành công; metadata bằng config/provenance/evaluator CLI; task metrics khớp API từ saved predictions và selection score vẫn bằng validation macro-F1 theo rule cũ. |

Fresh run directories (chỉ smoke, tiny initialized BERT, benchmark=false):

- `runs/smoke/training/i06-02-b391a317407a/st_sentiment/run`
- `runs/smoke/training/i06-02-b391a317407a/st_topic/run`
- `runs/smoke/training/i06-02-b391a317407a/mtl_smart/run`

Lệnh để 06 recheck P0 ngay:

```text
python scripts/integration/reproduce_validation_metadata.py --run-dir runs/smoke/training/i06-02-b391a317407a/mtl_smart/run
```

Receipt mới `reports/method/i06-02-verification.json`, SHA-256 **`c20f24dbe2f95bdfe0539e2901eaba3a7fb1f97178dc1d5214a3c54570c249d4`**. JUnit mới `runs/smoke/training/i06-02-tests.xml`, SHA-256 **`77982989ba8efa2373ced586dd99a69d13997a915800b3a37475ec64d72e38ba`**. Receipt chứa hashes validation metrics, exact commands/outputs của ba run mới; baseline giữ inventory trước sửa và byte hashes evidence cũ. Chi tiết SHA từng source/support file ở inventory READY mới.

Đầu vào chuyển **04**: dùng `reports/method/02-ready-allowlist.txt` và `delivery_inventory.json` mới, kiểm diff/hash rồi stage tuần tự file READY cùng modules khác; không stage `runs/`, không cần module 05. Code hash đã đổi nên tạo snapshot/cohort mới; 22 config hyperparameters/model/tokenizer/data revisions giữ nguyên. Không resume smoke/cohort cũ bằng code mới.

Đầu vào chuyển **06**: receipt mới, JUnit, ba run dirs và reproduction command trên; recheck validation aggregation/CLI/clean checkout theo SHA release mới của 04 rồi cập nhật gate thuộc 06. Chưa chỉnh hoặc nhận gate FAIL cũ đã được 06 đóng. Không full train/GPU/LLM, không thêm dependency cho metadata fix này.

Các sections dưới đây giữ audit và receipts bàn giao gốc; số 27 tests là lịch sử ngày 11/10, đã được kết quả 34 tests ở trên thay thế cho lần sửa này.

## Outputs READY và allowlist cho module 04

Đường dẫn dưới đây tương đối với `F:/DPL project/neu-esc-reproduction`. File cụ thể và SHA-256 nằm trong `reports/method/delivery_inventory.json`; allowlist cụ thể trong `reports/method/02-ready-allowlist.txt`. Chỉ stage những file có trong allowlist sau kiểm tra hash/diff; không dùng `git add .`.

- `src/neu_esc/models/{__init__,encoder}.py`: pretrained factory, native ST mỗi task riêng, shared MTL Linear/SiLU/LayerNorm và hai heads, auxiliary random untied MLM head theo vocab thực.
- `src/neu_esc/losses/{__init__,classification,mlm,smart}.py`: explicit weighted mean CE; masking 30%/80-10-10 và selected-token normalization; differentiable continuous SMART và public-helper diagnostic riêng.
- `src/neu_esc/training/{__init__,config,checkpoint,engine,runtime,train,predict}.py`: resolved config/hash/gates, checkpoint best/latest có completion/checksum, đủ RNG/optimizer/scheduler/scaler/config/tokenizer/sampler/order/batch state; CLI train/resume/predict.
- `configs/models/{mbert,velectra,xlm_r,visobert,phobert}.yaml`: immutable model/tokenizer revisions và evidence hashes.
- `configs/experiments/<backbone>/<variant>.yaml`: 22 file. mbert/velectra có hai ST; xlm_r/visobert/phobert có hai ST + mtl/mtl_smart/mtl_mlm/mtl_mlm_smart. Không tự thêm CNN/variants ngoài Table 6.
- `tests/training/`: fixture tiny config; objective/checkpoint/resume/CLI tests; opt-in GPU pretrained tests.
- `docs/method_audit.md`, `docs/training.md`: paper/code/implementation/assumptions, commands, resume boundary và sync protocol.
- `reports/method/`: locked read-only reference sources, backbone/config/tokenizer evidence, weight access/LFS metadata, executable audit/config/smoke scripts, gradient/CLI/real-data smoke JSON, delivery inventory và allowlist.
- `handoffs/02-training.md`: handoff này. Hash của handoff không tự đưa vào inventory để tránh vòng tham chiếu; path có trong allowlist.

Ignored local evidence trong `runs/smoke/training/`: pytest XML, venv thử nghiệm, fixture CLI run và real-data tiny run. Không stage hoặc sync venv; không dùng checkpoint/metrics smoke làm benchmark.

## Nguồn / revision / hash / config

- Paper mục 4, trang 13-18; đã xem trực tiếp phương trình trang 15. PDF SHA-256 `bdc37ae84e6653ca60b325ccb54f7f655fb4cd1ee606d0f9a8a5a15772e7189c`.
- Reference code commit `b5637cc0a8a7b5e88e794d180c11902405548eb5`; chín source files byte nguyên gốc trong `reports/method/reference-source.zip`, upstream hashes trong `reference/upstream_hashes.json`; Python copies trong `reference/` chuẩn hóa LF, không sửa logic. Hash checkout ở inventory. Script train/read-only cho IDs/hyperparameters; không đưa commented credentials của upstream vào source snapshots.
- Model/tokenizer SHA: mbert `7cbf9a625e29989f6b9c6c2fa68234c304f7e38f`; velectra `f594fee501933d92d8c13285c969495681d0cc00`; xlm_r `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`; visobert `196a62afad9cbe4f52a54aabad828b13f0eec59a`; phobert `86cd7fd4c148980922ac11a2cf5e257f2ba639e1`.
- Dataset manifest khóa hiện tại từ handoff READY cuối của 01: `ea127f55e840991e53569379a9b3e8ac00bc6798108d6472f8ab300a109f8f7d`; nguồn GitHub commit trên, đúng 23048/3305/6613; HF/paper equivalence chưa chứng minh. Mọi config full chứa digest này và assumption nguồn thay thế.
- Seeds template=42, campaign seeds chính42-46 do07 quản lý. Config hash và cohort digest được writer tính canonical, manifest/model/tokenizer/implementation change tạo cohort mới. Config file hashes ở inventory.
- P0 config: `configs/experiments/visobert/mtl_smart.yaml`, length128, epochs20, LR2e-5, AdamW eps5e-9, decay0.01, betas0.9/0.999, epoch-linear end_factor0.1, physical batch8/accum16/effective128, fp32, task CE weights1/1, SMART weight0.02/radius1e-3/noise std1e-5/1 ascent step/step_size1e-3/gradient_eps1e-6/batchmean symmetric KL. Tất cả thêm trong YAML, không tuning bằng test.

## Lệnh và kiểm tra đã chạy thật

Runtime thử nghiệm 02: `runs/smoke/training/.venv/Scripts/python.exe`, CPython3.10, torch2.5.1+cpu, transformers4.46.3, numpy1.26.4, PyYAML6.0.2, pytest8.3.4, sentencepiece0.2.0, protobuf5.29.6, Pillow11.0.0. `PYTHONPATH=src`; chưa tự sửa package root.

| Lệnh/check | Kết quả thực tế |
|---|---|
| `python reports/method/audit_backbones.py` | Cả năm tokenizer thực thi thành công; model config/metadata và special IDs/vocab hashes đã ghi. |
| `python reports/method/probe_weights.py` | Cả năm pinned weight files HTTP206, consume16bytes/file. LFS SHA-256 và size được ghi; không tải/load toàn bộ weights. |
| `python reports/method/build_configs.py` | 22 YAML, model/tokenizer SHA khóa, manifest digest khóa. |
| `python reports/method/gradient_audit.py` | `.long()` mất grad_fn; allow_unused=True trả None; strict autograd báo lỗi; helper nguyên trạng chạy được trên fixture; continuous gradients hữu hạn/khác0 và special/pad noise=0. |
| `python reports/method/cli_smoke.py` | PASS: CLI train->interrupt step2->resume trong process mới->predict test->evaluator thật, đủ16(ID,task) của fixture8test samples. Evidence `reports/method/cli_smoke.json`. |
| `python reports/method/real_data_smoke.py` | PASS: initialized tiny model trên8mẫu/split từ manifest thật, train và fresh-process predict. Evidence `reports/method/real_data_smoke.json`. Không benchmark; evaluator CLI đủ split không dùng cho subset này. |
| Data API real integration | Đọc train23048, validation3305 raw và validation3305 segmented; segmented không rỗng, IDs hai text variants bằng nhau. Không preprocessing lại. |
| `python -m pytest tests/training -q --junitxml=runs/smoke/training/pytest.xml` | **27 passed, 8 skipped** (Colab GPU opt-in), 17.16s ở lượt cuối trước bàn giao. |
| Cùng tests trên `.venv` stack 04, XML `pytest-release-stack.xml` | **27 passed, 8 skipped**, 27.33s; torch2.14.1+cpu/transformers4.57.6/numpy2.2.6/PyYAML6.0.3/pytest9.1.1/Pillow12.3.0. Không sửa môi trường04. |
| Full CLI trên CPU với config P0 | Từ chối trước tạo run/download: `Full training requires CUDA GPU on Colab`; run-dir `should-not-exist-full-cpu` không được tạo. Đây là kiểm tra negative gate, không run training lỗi trong campaign. |

Cài torch CPU lần đầu thất bại do pip23 dùng index PyTorch cho dependency build `flit_core`; đã dùng pip26 và wheel2.5.1+cpu từ extra-index cùng PyPI. Không thay lock/root của04.

Một real-data smoke ban đầu bị **từ chối trước train** vì module01 hoàn thiện manifest khiến digest cũ không còn khớp. Giữ evidence failed trong namespace smoke, khóa lại configs với digest mới rồi tạo smoke run mới PASS. Không tiếp tục/reuse cohort cũ. Lượt pass cuối và manifest hash mới nằm trong inventory.

## Những điều đã xác minh

- Native ST sentiment/topic độc lập; MTL đúng4/10 logits, encoder/shared/two-head gradients tồn tại. Không joint-head, reweighting hay hướng cải tiến.
- SMART token-ID `.long()` ngắt inner noise gradient; None fallback của code không đồng nghĩa adversarial optimization hợp lệ. Diagnostic adapter match helper gốc bitwise cùng RNG seed. Helper không lỗi runtime ở fixture đã kiểm.
- Continuous SMART giữ vị trí RoBERTa/XLM-R, loại pad/special, gradient noise/model hữu hạn/khác0. Config radius và numerical epsilon tách rõ.
- MLM selected labels đúng cả nhánh mask/random/unchanged, non-target=-100, distribution30%/80-10-10 theo statistical fixture, zero-target skip encoder, loss chuẩn hóa một lần.
- Six-variant CPU tiny resume bitwise model+Adam+scheduler+scaler+RNG so với run liên tục; hỗ trợ optimizer boundary/num_workers0/cùng device-software. Checkpoint corruption/mismatch/incomplete bị từ chối, tie chọn epoch sớm.
- Trainer chỉ train/validation, predictor giữ ID cùng batch và evaluator authoritative checks; process mới load checkpoint/config/tokenizer offline đủ để predict.
- 22 config có resolved assumptions/revisions/manifest; public checkpoint/tokenizer access đã probe. Chưa có full pretrained loading evidence.

## Giả thuyết chưa kiểm chứng và assumptions

Không chứng minh Figure3 mâu thuẫn do SMART, order, checkpoint hay dataset. Không xem source access probe/tiny accuracy là Table6 reproduction.

Paper không công bố đủ hyperparameters/revisions/pooling/activation/checkpoint selection. Mbert multilingual uncased đến từ repo; CNN không có row định danh trong Table6. Current HF model SHAs không được chứng minh là paper-time SHAs. Chi tiết tất cả quyết định trong `docs/method_audit.md` và assumptions của từng YAML.

Khác code công khai có chủ ý: continuous SMART và special masking; batchmean final KL thay sum; classification1/1 thay0.5/0.5; correct MLM selected-token normalization thay meanCE/avgmask; dynamic MLM vocab thay64001; no patience early stopping; fixed actual epochs; microbatch8/accum16 thay physical128. Những lựa chọn này làm baseline theo paper có giả định rõ, không khẳng định tái tạo bitwise historical code.

## Sai khác contract / giới hạn

- Không đổi API contract data/evaluator hoặc tên22config/CLI. YAML thêm `dataset_manifest_digest` để gate khóa data, cùng fields chi tiết; không breaking contract.
- `reference_code` chỉ cung cấp exact public two-task SMART diagnostic, không full historical trainer; chỉ variantmtl_smart được chấp nhận, tách cohort và không nằm70run.
- CPU tiny smoke subset không đủ fullsplit predictions để evaluator CLI benchmark. Synthetic fixture CLI smoke dùng đầy đủ fixture split và evaluator thật; production full runs giữ đủ samples.
- Checkpoint `best/latest` là directories với versioned state file+COMPLETE.json, không tên cố định `model.pt`. Module07 phải copy file marker trỏ tới và verify SHA; rootmetadata cũng phải sync. Latest mặc định50steps, epochend và interrupt boundary. Crash giữa best/latest publication có thể cần restore prior coherent sync hoặc restartseed, không giả resume.
- Offline readiness không bảo đảm full config fit GPU; mặc địnhfp32 và batch8 chưa profile Colab. Không có GPU thật hoặc full pretrained backward/resume đã chạy.

## Blockers và inputs/bước tiếp theo

1. **04**: dependency request cho runtime/package train/eval: torch CUDA, `transformers>=4.46.3,<5`, numpy, PyYAML, sentencepiece, **protobuf**, **Pillow>=11** (shared writer dùng load_default(size=...)), huggingface-hub/safetensors/requests qua Transformers; pytest cho tests. Source/code đã thử cả stack02 torch2.5.1/Transformers4.46.3 và stack04 torch2.14.1/Transformers4.57.6/numpy2.2.6, đều27pass. Stack04 hiện không có protobuf; tokenizer tiny tests không kiểm thiếu dependency này, cần thêm trước pinned-tokenizer Colab smoke. Không cần wandb/pandas/sklearn trong training02; evaluator/01 quản lý dependencies của họ.
2. **04**: tuần tự stage allowlist02 sau xác minh inventory, cùng modulesREADY khác; tạo snapshot commit đầy đủ. Không train cây đang được chat khác sửa. Module02 không commit/stage.
3. **06**: kiểm clean checkout; data/evaluator full integration, checkpoint restored metadata và CLI, CPU tests theo lock04; opt-in8GPU checks trên Colab. `NEU_ESC_RUN_GPU_SMOKE=1 python -m pytest tests/training/test_gpu_backbones.py -q` chỉ trên CUDA thật; defaultskip.
4. **07**: người dùng mở Colab GPU/mountDrive/secrets; notebook do07 quản lý gọi CLI. VerifyGPU/VRAM/disk, full pretrained loading/backward, exactGPU resume và throughput/memory đúng configP0. Sync marker/checksum theo docs/training.md. Chưa có phiên Colab kết nối, không nhận đã chạy.
5. Nếu GPU memory/precision cần đổi:02 chuẩn bị config/cohort mới theo evidence; không vá script gốc trong queue. Sau gates chạyP0seed42, đủ22configsseed42 và70 lượt theo plan; seed42 được tính vào5 nếu cohort không đổi.
6. **01/08**: tiếp tục báo HF/paper equivalence chưa chứng minh; manifest đổi phải khóa configs/cohort mới, không trộn kết quả nguồn khác. Tác giả chưa được liên hệ trong chat này.
