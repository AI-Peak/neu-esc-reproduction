# Evaluator NEU-ESC v1

Module 03 sở hữu evaluator, Table 6 và audit Figure 3. Không có kết quả training/LLM benchmark trong các output smoke của module này.

## API dùng chung

```python
from neu_esc.data.dataset import load_split
from neu_esc.evaluation.evaluate import evaluate_records

expected = load_split("data/manifests/dataset.json", "test", "raw")
result = evaluate_records(prediction_records, expected,
                          tasks=["sentiment", "topic"], allow_invalid=False)
score = result["tasks"]["sentiment"]["macro_f1"]
```

`evaluate_records(records, expected_records, tasks, allow_invalid=False)` nhận iterable của mappings và không ghi file. `expected_records` là records rộng từ data API; mỗi record có `sample_id`, `split`, và nhãn của các task yêu cầu. Hàm không dùng text, segmentation hoặc thứ tự hàng. Người gọi API chịu trách nhiệm lấy split đã kiểm chứng từ manifest; CLI thực hiện việc này qua `load_split`.

Predictions là JSONL dài theo hợp đồng v1, một `(sample_id, task)` mỗi dòng; bắt buộc `split`, `y_true`, `y_pred`, `status`, `error`. ST gọi một task, MTL/LLM gọi hai task. `y_true` phải khớp nhãn authoritative theo ID. Duplicate ID trong expected, duplicate cặp ID/task, ID dư/thiếu, task dư, split sai, nhãn sai/ngoài miền, status không hợp lệ đều ném `EvaluationError(ValueError)`. Không sửa bằng vị trí, không tự drop/dedup. Mẫu text trùng nhau vẫn có ID khác.

Nhãn sentiment cố định: `0 Neutral, 1 Positive, 2 Negative, 3 Toxic`. Nhãn topic cố định: `0 Spam, 1 News, 2 Academic, 3 Other, 4 Service, 5 Jobs & Recruitment, 6 Personal Affairs, 7 Social Affairs, 8 Help & Share, 9 Club & Events`.

Kết quả API có `schema_version="1.0"`, `split`, `tasks`. Mỗi task có `n_samples`, `accuracy`, `macro_f1`, `weighted_f1`, `per_class`, `label_order` (integer IDs), `label_names`, `invalid_count`, `valid_count`, `record_coverage`, `valid_prediction_coverage`, `confusion_matrix`, `confusion_matrix_normalized`, `confusion_columns`, `confusion_orientation`. CLI bổ sung metadata run theo contract. JSON giữ số đo dưới dạng fractions; không round trước khi ghi.

## Metric và CM

CM có **hàng=true, cột=predicted**, thứ tự label 4/10 cố định. `sum(CM)=N`, `trace(CM)/N=accuracy` được assert với tolerance `<=1e-12`. Một CM test có N=6.613 dù file MTL chứa 13.226 dòng cho hai task. Row-normalized chia cho tổng hàng, gồm cột INVALID nếu có; hàng class vắng là toàn 0.

Với class i: `TP=CM[i,i]`, `FP=sum(column i)-TP`, `FN=sum(row i)-TP`. Precision=`TP/(TP+FP)`; recall=`TP/(TP+FN)`; F1=`2TP/(2TP+FP+FN)`. Mẫu số 0 trả 0 (`zero_division=0`). Macro-F1 lấy trung bình trên **toàn bộ K=4/10 nhãn**, kể cả class vắng. Weighted-F1=`sum(support_i*F1_i)/N`. Accuracy=`sum(TP_i)/N`.

LLM sau bounded retries dùng `y_pred=null,status="invalid",error="lý do cụ thể"`, đánh giá với `allow_invalid=True`. CM khi đó luôn K×(K+1), cột cuối `INVALID`, kể cả khi một seed không invalid để giữ shape đồng nhất trong cohort. INVALID đóng góp FN cho nhãn thật và giữ nguyên N. `record_coverage=1.0` khi tất cả cặp đã có bản ghi; `valid_prediction_coverage=(N-invalid_count)/N`. Bản ghi vắng luôn fail coverage; `status="timeout"` phải được adapter chuyển thành `invalid` sau retry, không dùng trực tiếp. Encoder dùng `allow_invalid=False`.

## CLI

```powershell
python -m neu_esc.evaluation.evaluate --predictions runs/encoder/<run_id>/predictions/test.jsonl --dataset-manifest data/manifests/dataset.json --output runs/encoder/<run_id>/evaluation/test
```

Tasks lấy từ `provenance.json` hoặc `config.resolved.yaml` tại run directory; `--tasks sentiment topic` khai báo khi dùng predictions standalone. CLI từ chối suy tasks chỉ từ dòng predictions vì mất toàn bộ một task có thể bị hiểu nhầm là ST. Nếu có cả declaration trong config/provenance hoặc explicit flag thì chúng phải khớp. Lệnh contract chuẩn hoạt động với resolved config mà trainer tạo.

Các flag bổ sung: `--split validation|test|train`, `--allow-invalid`, `--no-png`, `--run-id`, `--cohort-id`, `--seed`. Metadata explicit không được mâu thuẫn provenance/config. Split mặc định lấy từ predictions và phải nhất quán với split authoritative. Manifest digest được tính canonical JSON UTF-8, `ensure_ascii=False`, `sort_keys=True`, separators `(',', ':')`, bỏ trường `manifest_digest`. `load_split(..., "raw")` kiểm chứng hashes/IDs/labels; text variant raw không làm đổi nhãn evaluation. Predictions SHA-256 tính trên cùng byte snapshot đã parse; predictions/manifest đổi trong lúc đánh giá sẽ fail.

Outputs: `metrics.json`, `metrics.csv`, `per_class_<task>.csv`, `confusion_<task>.csv/.png`, `confusion_<task>_normalized.csv/.png`. CSV báo `%`; counts CSV là integers; normalized CSV là fractions. PNG counts/normalized riêng, có N, tên nhãn, hướng CM và cột INVALID. Output directory phải mới hoặc rỗng; không overwrite run trước. Validation/coverage lỗi exit 2 và không ghi metrics.

Metadata JSON bắt buộc theo tên contract: `run_id`, `cohort_id`, `seed`, `split`, `predictions_sha256`, `dataset_manifest_digest`, `tasks`, `schema_version`. `method_mode`, `backbone`, `variant` bổ sung để phục vụ aggregation/ST. Thiếu metadata run vẫn có key null khi evaluation standalone; kết quả này không đủ để aggregate cohort. Không đoán cohort ID từ tên đường dẫn. Thiếu declaration task sẽ fail.

Python API/aggregation dùng standard library. CLI PNG cần **Pillow >=10.1**; đọc resolved YAML cần **PyYAML**. `--no-png` cho kiểm tra không có Pillow. Audit nguồn cần **pypdf** và Poppler để render khi đối chiếu. Module 04 sở hữu packaging và lock; module này không sửa root dependencies. Matplotlib trong extra `eval` hiện kéo Pillow transitively, nhưng nên khóa Pillow trực tiếp.

## Tổng hợp seeds và hàng ST

```python
import csv
from neu_esc.evaluation.aggregate import (
    aggregate_cohort, combine_st_runs, paper_metrics_from_row, write_summary_csv,
)

with open("reports/reference/table6.csv", encoding="utf-8", newline="") as f:
    row = next(r for r in csv.DictReader(f)
               if r["backbone"] == "visobert" and r["variant"] == "mtl_smart")
paper = paper_metrics_from_row(row)  # percentages, not fractions
summary = aggregate_cohort(seed_metrics, paper_metrics=paper,
                           expected_seeds=[42,43,44,45,46])
write_summary_csv(summary, "new-report.csv", row_id=row["row_id"],
                  backbone=row["backbone"], variant=row["variant"])
```

`aggregate_cohort` yêu cầu cùng `cohort_id`, `method_mode`, manifest digest, split, task set, N, true class supports và CM columns. Mỗi run ID/seed duy nhất; metric phải khớp counts trước khi tính trung bình. Chủ training/campaign chịu trách nhiệm tạo cohort hash từ resolved config bỏ seed/run paths, data/model/tokenizer revisions và code/version. Evaluator kiểm tra cohort ID khai báo, không tự suy hash từ metrics.

`expected_seeds` là tập predeclared chính xác; thiếu/dư seed bị từ chối, không filter để chọn seed có test tốt. Không truyền tập thì status `observed_seeds_only` thay vì claim đủ campaign. Kết quả có per-seed metrics, mean, sample std `ddof=1`, n; một seed std=null. Fraction mean/std và bản `%` cùng có sẵn; `delta_pp = mean*100 - paper_pct`. `paper_metrics` phải là `{task: {accuracy, macro_f1, weighted_f1}}` đơn vị %. `write_summary_csv` xuất long-form một dòng/task/metric kèm n, mean/std/paper/delta, method, cohort, status, và từ chối overwrite.

ST dùng `combine_st_runs(sentiment_runs, topic_runs, backbone=..., pairing_id=..., paper_metrics=..., expected_seeds=...)`. Hàm kiểm tra đúng `st_sentiment`/`st_topic`, backbone, cùng seed set/split/method/manifest/N, và hai run ID riêng. Hai task config có cohort khác nhau; giữ `cohort_ids` theo task, không giả một cohort hash chung. `pairing_id` là quyết định ghép hai config được campaign công bố trước khi xem test. Cùng seed được ghép qua ID seed, không qua thứ tự list.

CM seed 42 được chọn **trước** cho đối chiếu Figure 3. `per_seed_confusion_matrices` giữ mọi seed; nếu seed 42 chưa có thì status `not_available`, không thay bằng seed tốt nhất. `aggregate_confusion` chỉ cộng counts và ghi N riêng; 5 seeds×6.613 = **33.065/task**. Mean metrics vẫn lấy từ các scores từng seed, không lấy metric của CM cộng và không gộp logits.

## Bảng paper và Figure 3

`reports/reference/table6.csv`: đúng 21 hàng và 126 ô số ở trang PDF 18, gồm 5 ST pair, 12 MTL, 4 LLM. Variant ST pair là tên hàng báo cáo, không là variant trainer. Alias backbone bám `EXPERIMENT_MATRIX.csv` với `claude4` giữ tên bảng. `table6_metadata.json` giữ tên nguyên văn và bold/star; alias mbert không chứng minh checkpoint multilingual/cased/uncased, Claude 4 không chứng minh API snapshot.

Figure 3 trang PDF 19 có 16+100 ô integer trong hai counts CSV. `figure3_metadata.json` khóa SHA-256 bản chép đã đọc hình. Audit không điều chỉnh số để khớp tổng kỳ vọng. Mỗi CM N=6.613; sentiment trace=3.619, accuracy=54,72554060%; topic trace=2.074, accuracy=31,36246787%. Table 6 ViSoBERT MTL+SMART ghi 83,58% và 79,80%. Chỉ kết luận **bất nhất**, chưa chứng minh lệch order/checkpoint/data. Hai ảnh `source-page-18.png` và `source-page-19.png` render trực tiếp từ PDF gốc có hash khóa trong audit.

```powershell
python -m neu_esc.evaluation.audit_reference --reference-dir reports/reference --pdf ../references/NEU-ESC-2506.23524.pdf
```

## Kiểm tra thực thi

Tại repo, đặt `PYTHONPATH=src` nếu package chưa cài. Windows PowerShell: `$env:PYTHONPATH=(Join-Path (Get-Location) 'src')`. Bundled Python runtime đã được dùng cho các kiểm tra trong handoff.

```powershell
python -m unittest discover -s tests/evaluation -p 'test_*.py' -v
python tests/evaluation/create_smoke.py --output runs/smoke/evaluation/<new-fixture-id>
python tests/evaluation/check_data_boundary.py --dataset-manifest data/manifests/dataset.json --output runs/smoke/evaluation/<new-boundary-check>.json
```

`create_smoke.py` chạy real data API + real CLI với **4 mẫu synthetic**, kiểm tra encoder, invalid LLM, shuffle, 5-seed/ST summary và coverage failure; không có model hay API paid. `check_data_boundary.py` chỉ đọc validation/test thật đã chuẩn bị rồi kiểm tra empty predictions bị từ chối; không tạo predictions/benchmark metrics. Kết quả dataset boundary chỉ chứng minh interface trên revision được manifest khai báo, không chứng minh tương đương paper/HF. Handoff và `runs/smoke/evaluation/verification.json` ghi bằng chứng/lệnh thực chạy; full training/LLM evaluation thuộc module vận hành.
