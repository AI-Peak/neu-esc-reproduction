# Hợp đồng giao diện NEU-ESC v1

Đây là đặc tả thiết kế, chưa phải API đã được implement. Repo tương lai: `F:/DPL project/neu-esc-reproduction/`. Mọi đường dẫn bên dưới tương đối với repo, trừ nguồn PDF. Các module giữ đúng tên/schema này; thay đổi phải được phối hợp qua handoff và tăng version nếu breaking.

## 1. Nguồn và mức tái lập

- Paper: `F:/DPL project/references/NEU-ESC-2506.23524.pdf`; https://arxiv.org/abs/2506.23524.
- Tham khảo: https://github.com/hung20gg/multi-task-bert, commit `b5637cc0a8a7b5e88e794d180c11902405548eb5`.
- Dataset HF: https://huggingface.co/datasets/hung20gg/NEU-ESC. Quyền/revision phải xác minh.
- `method_mode=paper_method` là baseline chính; `reference_code` chỉ dành tái hiện code công khai, tách run và report. Không trộn hai mode trong thống kê.

## 2. Dữ liệu

Giữ split gốc train=23048, validation=3305, test=6613. File UTF-8 JSONL `data/processed/{train,validation,test}.jsonl`; mỗi dòng:

```json
{"sample_id":"test:000000:<raw_sha_prefix>","split":"test","source_row":0,"text":"...","text_segmented":"... hoặc null","sentiment":0,"topic":3}
```

`source_row` zero-based, không tính header. ID tạo từ source split + row + raw file hash, không dựa riêng trên text; duplicate text vẫn có ID riêng. Giữ ID qua preprocessing, tokenization, batching và prediction. Không tự drop/dedup hay chia split lại; audit trùng lặp được báo riêng.

`data/manifests/dataset.json`: `schema_version`, `source_urls`, `source_revision`, `raw_sha256`, `processed_sha256`, `row_counts`, `id_algorithm`, `label_maps`, `preprocessing` (steps, config hash, tool/version, segmentation), `hf_equivalence_status`, `assumptions`, `manifest_digest`. Digest tính từ canonical JSON của manifest bỏ trường digest; UTF-8, sort keys, separators cố định. Hash cả raw và processed bytes; không có đường dẫn máy cá nhân trong digest nếu có thể dùng path tương đối.

Label IDs khóa sau audit: sentiment 0 Neutral,1 Positive,2 Negative,3 Toxic; topic 0 Spam,1 News,2 Academic,3 Other,4 Service,5 Jobs & Recruitment,6 Personal Affairs,7 Social Affairs,8 Help & Share,9 Club & Events. Tên cột nguồn `classification` chuyển thành `topic`. Nếu thực tế khác, dừng gate dữ liệu và báo; không ép mapping cho khớp.

API `load_split(manifest_path, split, text_variant)` trong `neu_esc.data.dataset`; trả records có ID, selected text, hai nhãn. `text_variant` là `raw` hoặc `segmented`; ở đây raw nghĩa là trường text chưa word-segment của bản prepared, mọi bước cleaning phải được ghi trong manifest và nguyên văn nguồn vẫn giữ ở data/raw. Training không tự làm sạch lại. Thiếu segmented cho PhoBERT phải báo lỗi, không âm thầm dùng raw.

## 3. Cấu hình và seed

YAML resolved config có `schema_version`, `experiment_id`, `backbone`, `model_id`, `model_revision`, `tokenizer_id`, `tokenizer_revision`, `variant`, `tasks`, `method_mode`, `dataset_manifest`, `text_variant`, `max_length`, `seed`, `training`, `smart`, `mlm`, `selection`, `assumptions`.

`variant` thuộc `st_sentiment,st_topic,mtl,mtl_smart,mtl_mlm,mtl_mlm_smart`; `tasks` khớp variant. ID model/revision được audit và khóa trước full run; không dùng alias `main` như revision cố định. BERT multilingual uncased trong code chỉ là ứng viên cần xác minh, không mặc định paper nêu đủ checkpoint.

Mọi learning rate, batch size, effective batch, epochs/max_steps, optimizer/scheduler/warmup, weight decay, precision, clipping, dropout thực dùng, loss reductions, SMART epsilon/steps/weight/noise, MLM denominator phải hiện trong config. Thông số không biết ghi giả định và lý do; không mượn mặc định Transformers mà không lưu resolved value.

Nếu không tìm thấy quy tắc checkpoint gốc: giả định được công bố là max validation macro-F1 cho ST, max trung bình hai validation macro-F1 cho MTL, tie chọn epoch sớm hơn. Không gọi đây là protocol được paper công bố. Không dùng test để chọn. Tuning nếu có phải có ngân sách/config độc lập, không tính trong 70 lượt xác nhận.

Set Python/NumPy/PyTorch/CUDA/dataloader seeds, log deterministic flags và nondeterministic ops. Seeds chính 42–46; std `ddof=1`, một seed thì std=null. Định danh cohort bằng hash cấu hình trừ seed/run paths + data/model/tokenizer revisions + implementation/version; mọi thay đổi ảnh hưởng kết quả tạo cohort mới.

## 4. CLI và đầu ra run

Lệnh đích phải hoạt động sau khi package được cài; đây là giao diện các module phải implement:

```powershell
python -m neu_esc.data.prepare --config configs/data/dataset.yaml --output data/processed
python -m neu_esc.training.train --config configs/experiments/visobert/mtl_smart.yaml --seed 42 --run-dir runs/encoder/<run_id>
python -m neu_esc.training.predict --run-dir runs/encoder/<run_id> --split test
python -m neu_esc.evaluation.evaluate --predictions runs/encoder/<run_id>/predictions/test.jsonl --dataset-manifest data/manifests/dataset.json --output runs/encoder/<run_id>/evaluation/test
python -m neu_esc.llm.run --config configs/llm/<model_setting>.yaml --run-dir runs/llm/<run_id> --dry-run
```

Train chỉ đọc train/validation. Predict load checkpoint đã chọn và xuất predictions; không cập nhật weights. Full train bắt buộc GPU; smoke tiny dùng CPU được. `run_id` duy nhất, không ghi đè run cũ. Resume phải so hash/config/cohort, RNG/optimizer/scheduler state; không được giả vờ resume bằng training mới vào cùng thư mục.

Mỗi run: `config.resolved.yaml`, `provenance.json`, `environment.json`, `logs/`, `checkpoints/best` (encoder), `selection.json`, `predictions/{validation,test}.jsonl`, `evaluation/{validation,test}/`. Provenance chứa run/cohort ID, seed, code commit, clean status, manifest/config hashes, revisions, package/CUDA/GPU versions, start/end, run status. Run operator phải dùng snapshot/checkout cố định; không train trên cây code đang bị chat khác sửa. Không stage/commit file của chat khác để làm sạch worktree.

`selection.json`: quy tắc validation, best epoch/step, score, checkpoint hash. Checkpoint phải đủ để predict trong tiến trình mới; giữ best checkpoint đến hết kiểm tra cuối. Không chỉ lưu logits rồi xóa toàn bộ weights quá sớm.

## 5. Google Colab là môi trường full training

Windows workspace dùng phát triển và phân tích. Colab clone commit cố định ở `/content/neu-esc-reproduction`; config dùng relative paths để chạy trên Windows/Linux. Root Drive đề xuất `/content/drive/MyDrive/NEU-ESC-reproduction`, tham số hóa và xác minh quyền/dung lượng khi chạy. Chưa có phiên Colab/GPU được xác minh trong lượt kế hoạch.

Module 04 sở hữu packaging/install recipe Colab; module 07 sở hữu `notebooks/colab_reproduce.ipynb`, queue và sync Drive. Notebook chỉ gọi CLI package. Module 02 implement checkpoint/resume đầy đủ (weights, optimizer, scheduler, scaler, RNG, epoch/step, sampler và batch position khi cần). Bổ sung `checkpoints/latest` bên cạnh `best`. Resume không đủ trạng thái phải bị từ chối hoặc restart seed với ledger rõ ràng.

Module 07 sync metadata/logs/best/latest về Drive theo chu kỳ với checksum + completion marker; bản dang dở không được resume. Không chỉ lưu lúc kết thúc run. Train trên disk local, sync về Drive, restore về local trước resume. Mỗi runtime một job mặc định. Kiểm tra CUDA/VRAM/disk, đo throughput trước ETA. Không có GPU thì dừng full train; CPU smoke vẫn được. Chỉ bật bf16 khi runtime và model smoke xác nhận hỗ trợ; không giả định loại GPU hoặc quota cụ thể.

Secrets lấy qua Colab Secrets/environment, không ghi vào code/output/logs. Mount Drive, mở GPU và đăng nhập là thao tác của người dùng. Nếu không có công cụ truy cập Colab, bàn giao notebook + lệnh và chờ output người dùng; không tuyên bố đã chạy. Không mua gói, tăng ngân sách hoặc né giới hạn phiên tự động.

## 6. Predictions và metrics

JSONL dạng dài; mỗi dòng là một cặp `(sample_id,task)`:

```json
{"sample_id":"test:000000:<raw_sha_prefix>","split":"test","task":"sentiment","y_true":0,"y_pred":0,"status":"ok","error":null}
```

`task` là `sentiment|topic`; logits/probabilities là trường optional có thứ tự label rõ. Một ST run có 6613 dòng test; MTL/LLM có 13226 dòng, nhưng **mỗi confusion matrix có 6613 mẫu**. Label thật do join ID với manifest/split gắn vào; evaluator kiểm tra lại với dữ liệu gốc, không tin mù `y_true` trong predictions. Không nối labels/preds từ hai vòng dataloader độc lập.

Encoder: mọi nhãn phải hợp lệ, mỗi `(ID,task)` xuất hiện đúng một lần, đủ split, không invalid; nếu sai thì fail. LLM: `y_pred=null,status=invalid,error=...` khi không có nhãn sau retry giới hạn; vẫn giữ mẫu. Nhãn thật không nằm trong API prompt/request.

API Python trong `neu_esc.evaluation.evaluate`: `evaluate_records(records, expected_records, tasks, allow_invalid=False)` trả dict metrics và count matrices theo task, không tự ghi file. `expected_records` là records từ data API của đúng split; `tasks` là danh sách sentiment/topic. CLI dùng API này để ghi metrics.json/CSV/PNG. Trainer dùng cùng API cho validation, không tạo công thức metrics riêng.

Evaluator là nguồn duy nhất của Accuracy, macro-F1, weighted-F1, per-class P/R/F1/support và CM. Accuracy tính đúng trên toàn mẫu. F1 dùng tập true labels cố định 4/10, `zero_division=0`; invalid LLM được coi là abstention, làm false negative cho nhãn thật, không bị bỏ khỏi denominator. CM LLM có thể K×(K+1) với cột `INVALID`; đường chéo chỉ tính K nhãn hợp lệ. Encoder CM chuẩn K×K. Report invalid/coverage riêng.

`metrics.json`: `schema_version,run_id,cohort_id,seed,split,predictions_sha256,dataset_manifest_digest,tasks`. Mỗi task có `n_samples,accuracy,macro_f1,weighted_f1,per_class,label_order,invalid_count`. Số trong JSON là [0,1]; bảng báo cáo dùng %, delta là điểm phần trăm (`pp`). `confusion_<task>.csv` chứa counts với nhãn hàng/cột; ảnh counts và row-normalized tách riêng. Assert tổng counts=N và trace/N=accuracy với tolerance <=1e-12 trước làm tròn.

Bảng paper 21 rows lưu `reports/reference/table6.csv`: `row_id,backbone,variant,sentiment_accuracy_pct,sentiment_macro_f1_pct,sentiment_weighted_f1_pct,topic_accuracy_pct,topic_macro_f1_pct,topic_weighted_f1_pct,source_page`. Một row ST ghép metrics từ hai run riêng cùng seed. Bảng tái lập có n_seeds, mean, std, paper, delta_pp, method_mode/cohort và status. Không gộp logits giữa seeds rồi gọi đó là mean metrics.

CM đối chiếu Figure 3 chọn seed 42 từ trước; xuất đủ các seed còn lại. Nếu gộp 5 CM thì ghi rõ N=33065 mỗi task, không so trực tiếp counts với Figure 3 N=6613.

## 7. Hợp tác và quyền sở hữu

Module 01 data/source; 02 models/losses/training/configs encoder; 03 evaluation/reference tables; 04 packaging/root/GitHub; 05 LLM; 06 integration; 07 campaign/GPU outputs; 08 analysis/report. Từng module có namespace test, smoke run và handoff riêng trong PLAN.md. Không ghi đè/revert/stage file ngoài phạm vi; phát hiện thay đổi bất ngờ trong file sở hữu thì dừng sửa file đó và báo đầu mối.

06 báo lỗi về chủ module thay vì sửa implementation của họ. 07 chỉ thực thi snapshot, không vá model/config gốc để cải thiện test. 08 chỉ đề xuất đối chứng; 07 chạy sau khi cấu hình được chủ module chuẩn bị và ngân sách cho phép. Không chat nào tự tạo chat/subagent hoặc nhắn chat khác nếu người dùng chưa chỉ định.

Git index/commit được thao tác tuần tự bởi 04 để tránh index lock giữa các chat. Ngoại lệ cho release: 04 được stage danh sách file READY đã được từng chủ module bàn giao rõ trong handoff, kiểm tra diff chỉ chứa output đó rồi tạo commit snapshot. Không stage phần WIP, thay đổi người dùng hoặc file không có trong allowlist. Nếu còn file ảnh hưởng execution đang dirty, chưa được gọi snapshot reproducible. 06 kiểm tra commit/gates, 07 chỉ checkout commit này trên Colab; các module tiếp tục phát triển local không làm đổi run đang chạy.

Handoff: Status READY/PARTIAL/BLOCKED; files; source/commit/hash/config; commands/tests actually run; verified facts; hypotheses; assumptions/deviations; blockers; next action. Thiếu GPU/quyền/API thì hoàn thành phần offline độc lập và dừng đúng phần phụ thuộc, không giả lập kết quả để tuyên bố hoàn thành.
