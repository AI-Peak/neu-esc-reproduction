# Training và predict - module 02

Implementation ST/MTL/SMART/MLM sẵn cho integration. Full train/predict bắt buộc CUDA GPU; Windows chỉ dùng tiny initialized CPU smoke. Chưa có full training, GPU benchmark hoặc Table 6 score từ module này.

## Cài đặt và lệnh

Module 04 sở hữu packaging. Runtime CPU 02 đã kiểm tra: Python 3.10, torch 2.5.1+cpu, transformers 4.46.3, numpy 1.26.4, PyYAML 6.0.2, sentencepiece 0.2.0, protobuf 5.29.6, pytest 8.3.4, Pillow 11.0.0. Cùng 27 CPU tests cũng pass trên stack 04: torch 2.14.1+cpu, transformers 4.57.6, numpy 2.2.6, PyYAML 6.0.3, pytest 9.1.1, Pillow 12.3.0. Handoff yêu cầu Pillow cho validation PNG và protobuf cho pinned tokenizer (protobuf chưa có trong stack 04). Colab dùng torch CUDA phù hợp với runtime, không cài wheel `+cpu`.

Chạy từ repo root sau install package, trên snapshot do module 04 commit và 06 kiểm tra:

```bash
python -m neu_esc.training.train --config configs/experiments/visobert/mtl_smart.yaml --seed 42 --run-dir runs/encoder/visobert-mtl-smart-seed42-UNIQUE
python -m neu_esc.training.predict --run-dir runs/encoder/visobert-mtl-smart-seed42-UNIQUE --split test
python -m neu_esc.evaluation.evaluate --predictions runs/encoder/visobert-mtl-smart-seed42-UNIQUE/predictions/test.jsonl --dataset-manifest data/manifests/dataset.json --output runs/encoder/visobert-mtl-smart-seed42-UNIQUE/evaluation/test
```

Thay UNIQUE bằng run ID mới. Run tồn tại bị từ chối nếu thiếu `--resume`. Predictions/evaluation output không được ghi đè. Predictor load `checkpoints/best`, cấu hình/model spec/tokenizer đầy đủ; không cần download pretrained lại. `--device cuda` có thể chỉ định rõ. `--device cpu` chỉ chấp nhận tiny smoke run.

Train chỉ gọi `load_split` với train/validation. Test chỉ được đọc khi predict. `evaluate_records(records, expected_records, tasks, allow_invalid=False)` của module 03 là nguồn metrics duy nhất, dùng cho validation selection và kiểm tra toàn vẹn trước xuất predictions. Selected validation predictions và metrics/CSV/PNG được xuất khi training hoàn thành; không phải logits/labels ghép từ hai vòng dataloader khác nhau.

## Resume và checkpoint

```bash
python -m neu_esc.training.train --config configs/experiments/visobert/mtl_smart.yaml --seed 42 --run-dir runs/encoder/visobert-mtl-smart-seed42-UNIQUE --resume
```

`--stop-after-steps N` ngắt ở global optimizer step tuyệt đối N, ghi status INTERRUPTED và latest đủ state; không tính run đó là hoàn thành. Dùng cho benchmark throughput hữu hạn trước full run hoặc test resume. Khi resume, N phải lớn hơn global step đã lưu nếu tiếp tục dùng option này.

Mỗi `checkpoints/{best,latest}/` chứa `COMPLETE.json` và đúng một `state-<uuid>.pt` được marker trỏ tới. Writer ghi state temp, fsync, rename, tính SHA-256, rồi publish marker JSON qua atomic replace. Chỉ prune state cũ sau publish thành công. State dang dở, thiếu marker, checksum sai hoặc identity mismatch đều bị từ chối.

Payload chứa model specification và weights; resolved config; tokenizer bytes; optimizer/scheduler/scaler; Python/NumPy/Torch/all-CUDA RNG; epoch/global_step; sampler permutation `order`; vị trí microbatch kế tiếp `batch_position`; validation selection; runtime identity. Best lưu cả training state ở ranh giới epoch; trainer CLI chỉ resume từ latest. `selection.json` bên ngoài ghi rule, best_epoch/best_step/score và SHA-256 của best.

Ranh giới hỗ trợ là **sau toàn bộ gradient accumulation group và optimizer update**, gradients đã zero. Latest theo chu kỳ 50 optimizer steps và cuối mỗi epoch; trước step đầu cũng có latest. Không giữ half-computed microbatch/gradient graph. Khi crash giữa group, restore latest rồi tính lại nhóm chưa commit; dropout, masking, adversary noise và sampler dùng RNG đã lưu. Log sau saved global step được giữ làm diagnostics rồi loại khỏi log chính khi resume.

Đã test resume tại giữa epoch cho cả sáu variants, gồm final accumulation group ngắn, dropout và stochastic MLM/SMART. Model tensors, Adam state, scheduler, scaler và Torch RNG **bitwise bằng** run liên tục trên CPU tiny, cùng software/device/thread/deterministic config. Chưa xác minh tương đương GPU hoặc khác hardware. Resume yêu cầu cùng software/device/precision/GPU topology và environmental hash; khác loại GPU/version phải restart seed thành run mới. Num_workers=0, manual batches/permutation thay DataLoader worker randomness. Không đổi batch/accumulation/epoch budget khi resume.

Module 07 sync toàn bộ hai checkpoint directories **cùng config.resolved.yaml, provenance.json, environment.json, selection.json, tokenizer và logs**. Copy state đang được COMPLETE trỏ tới, kiểm SHA sau copy, rồi publish completion marker trên Drive. Local atomic replace không bảo đảm Drive sync nguyên tử. Restore về disk local trước `--resume`; trainer không đoán/check token Drive hoặc chọn một bản dang dở.

Nếu crash sau best được publish nhưng trước latest tương ứng, selection trong latest có thể không khớp best; trainer từ chối để tránh chọn checkpoint không tương ứng. Giữ bản sync hoàn chỉnh trước đó hoặc restart seed với ledger. Crash khi final metrics writer mới ghi một phần output cũng cần module 07 khôi phục snapshot hoặc xử lý ledger; không tự ghi đè artifact không khớp.

## Artifacts và khóa run

Run có config.resolved.yaml, provenance.json, environment.json, logs/steps.jsonl, logs/validation-epoch-XXXX.json, checkpoints/best+latest, selection.json, predictions/validation.jsonl, evaluation/validation/. Predict riêng tạo predictions/test.jsonl; evaluator CLI riêng tạo evaluation/test/. Log step chứa loss, số mẫu thực, optimizer step, lr, thời gian và peak VRAM; dùng đo throughput trước ETA, không có ETA GPU giả định.

Provenance có run/cohort/seed, code commit/clean status, config/manifest/implementation hashes, revisions, môi trường CUDA/GPU/packages, thời gian start/end, run_status, training_mode, model_initialization và benchmark_eligible. Mọi tiny smoke luôn benchmark_eligible=false. Model loading info được ghi riêng khi full pretrained loader chạy. Full training fail sớm trên CPU trước mọi data/model download.

Từ sửa I06-02 ngày 12/10/2026, provenance và `evaluation/validation/metrics.json` đều ghi `method_mode`, `backbone`, `variant`. Validation publisher dùng provenance trước, fallback sang resolved config và từ chối conflict, giống evaluator CLI. Chỉ bổ sung metadata; không đổi task metrics hoặc score/rule chọn checkpoint. Artifact cũ thiếu metadata được giữ nguyên để bảo toàn bằng chứng. Source hash thay đổi nên snapshot/cohort mới cần được 04/06 kiểm lại; không resume run cũ bằng implementation mới.

Dataset config khóa manifest của module 01. Nếu manifest thay đổi, không sửa hash để tiếp tục cùng run: audit dữ liệu mới, tạo cohort và run ID mới. Gate đã thực tế từ chối real-data smoke khi manifest bị module 01 hoàn thiện trong lúc config vẫn chứa digest trước đó; sau khóa lại config, chạy smoke mới. Đây là fail-safe mong muốn, không phải benchmark thất bại.

## Kiểm tra đã có và kiểm tra Colab còn lại

```bash
python -m pytest tests/training -q
python reports/method/gradient_audit.py
python reports/method/cli_smoke.py
python reports/method/real_data_smoke.py
```

CPU tests: logits 4/10 và task loss đúng; native ST của BERT/ELECTRA/RoBERTa/XLM-R; gradients encoder/shared/two heads; cast/allow_unused và helper gốc; continuous SMART, masking pad/special và RoBERTa position IDs; MLM distribution/loss/no-target; checkpoint completeness/hash/roundtrip; resume determinism; tie selection; CPU full gate; CLI mới train/resume/predict/evaluate qua data/evaluator thật trên synthetic fixture.

`cli_smoke.py` tự tạo manifest synthetic trong namespace smoke, mỗi run một UUID; nhận `--variant` và `--output` (directory mới). Ngoài train/interrupt/resume/predict, nó đọc validation metrics đã persist, gọi `aggregate_cohort`, tái tạo validation report qua evaluator CLI vào directory riêng, so metadata/task metrics/selection score. Receipt nằm ở `<output>/commands.json`; không ghi đè receipt cũ trong reports. `real_data_smoke.py` dùng tiny initialized BERT trên tối đa 8 mẫu mỗi split của dataset thật, không nhận là ViSoBERT trained result. Evaluator CLI đủ split **không** chấp nhận file test subset này; predictor kiểm bằng API với expected subset. Không đưa các smoke artifacts vào bảng paper.

Lệnh smoke/recheck metadata hữu hạn (directory đích phải mới):

```bash
python reports/method/cli_smoke.py --variant mtl_smart --output runs/smoke/training/NEW-P0-METADATA-SMOKE
python scripts/integration/reproduce_validation_metadata.py --run-dir runs/smoke/training/NEW-P0-METADATA-SMOKE/run
```

Regression CPU bao phủ ST sentiment/topic, MTL+SMART và MTL+MLM+SMART. Receipt sửa I06-02 ở `reports/method/i06-02-verification.json` gồm ba fresh CLI runs ST sentiment/topic/MTL+SMART và kết quả reproduction script 06. LLM đã loại khỏi scope; metadata fix không phụ thuộc module 05.

Colab còn cần kiểm trên GPU thực (tests mặc định skip):

```bash
NEU_ESC_RUN_GPU_SMOKE=1 python -m pytest tests/training/test_gpu_backbones.py -q
```

Test opt-in tải full pretrained weights theo SHA, forward/backward ST cả năm backbones và MLM+SMART ba MTL backbones, batch=2/length=24. Sau đó module 06/07 cần memory/throughput smoke đúng batch8/length128/accum16, checkpoint/resume trên GPU, bf16 capability nếu định dùng bf16. Test nhỏ không chứng minh config full fit VRAM. Không có phiên Colab được kết nối trong chat này; người dùng mở GPU/mount Drive, module 07 vận hành notebook/queue từ snapshot.
