# NEU-ESC reproduction

**Dành cho thành viên mới:** [Tiến độ và trạng thái bằng chứng](docs/PROGRESS.md) · [13 notebook tham khảo chỉ đọc](notebooks/reference/README.md). Các đoạn release/gates bên dưới là hồ sơ ngày 12/10; xem trang tiến độ để phân biệt trạng thái mới với kế hoạch cũ.

Repo tái lập baseline gốc của [NEU-ESC](https://arxiv.org/abs/2506.23524), theo [hợp đồng v1](docs/contracts.md). Tham khảo code `hung20gg/multi-task-bert` tại commit `b5637cc0a8a7b5e88e794d180c11902405548eb5`. `paper_method` là baseline chính; `reference_code` là nhánh đối chiếu riêng.

**Audit ngày 20/10/2026 đã xác minh metrics từ saved predictions của 7 run lịch sử; chưa có bằng chứng tái lập chính xác Table 6.** Release encoder gồm implementation data/model/loss/training/evaluation, 22 configs, manifests và CPU integration. Windows dùng phát triển, audit và smoke CPU; full training dùng Google Colab GPU sau gate tích hợp. GPU/Drive/quota là PENDING tại mốc release này; không dùng trạng thái lịch sử đó làm telemetry hiện tại. Benchmark LLM đã được người dùng loại khỏi phạm vi ngày 12/10/2026; không chờ module 05.

CI CPU kiểm imports/CLI thật, tests data/training/evaluation/integration và fresh-process MTL+SMART train → interrupt/resume → predict → evaluate → validation aggregation. Tiny initialized BERT và dữ liệu synthetic chỉ phục vụ smoke; không phải pretrained ViSoBERT hoặc benchmark paper. Xem [handoff 04](handoffs/04-infrastructure.md) để lấy release SHA và receipts/CI đúng phiên bản; commits `573a4b4`/`1bac0e9` cũ chỉ là infrastructure và không dùng để clone pipeline trên Colab.

## Quickstart Windows / Linux CPU

Dùng CPython 3.10–3.13 trên Windows/Linux x86_64 (Linux glibc >=2.28). Bộ CPU khóa riêng giữ PyTorch CPU và không tải CUDA stack. Lệnh PowerShell:

```powershell
py -3.10 -m venv .venv
.venv/Scripts/python -m pip install --require-hashes -r requirements-cpu.lock
.venv/Scripts/python -m pip install --no-deps --no-build-isolation -e .
.venv/Scripts/python -m pip check
.venv/Scripts/python scripts/environment/check_pipeline.py
.venv/Scripts/python scripts/environment/offline_tests.py
.venv/Scripts/python scripts/integration/audit_pipeline.py --output runs/smoke/local/p0 --variants mtl_smart
.venv/Scripts/python scripts/integration/reproduce_validation_metadata.py --run-dir runs/smoke/local/p0/mtl_smart
.venv/Scripts/python scripts/environment/capture.py --output runs/environment/windows/environment.json
```

Linux dùng `python3 -m venv .venv`, thay executable bằng `.venv/bin/python`. Installation cần mạng/package cache; chạy fixtures sau cài không yêu cầu model download, dataset thật, GPU hoặc API secrets. Chạy capture với đường dẫn mới mỗi lần; không ghi đè bằng chứng run cũ.

Extras và dependency groups: `data`, `train`, `eval`, `llm`, `dev`. `llm` giữ tên group rỗng để tương thích contract; không nằm trong release gate hoặc benchmark hiện tại. `uv.lock` khóa resolution các nhóm trên nhiều nền tảng; `requirements-cpu.lock` là môi trường CPU. Không áp dụng lock CPU hoặc `uv sync --all-extras` vào runtime Colab đang có CUDA. Xem [môi trường và recipe Colab](docs/environment.md).

## Phạm vi tái lập

Table 6 lịch sử có 21 hàng: 5 ST (hai run riêng mỗi hàng), 12 MTL, 4 LLM; giữ nguyên reference CSV và mọi số paper. Campaign hiện tại chỉ encoder: 17 hàng kết quả, 22 configs cho một seed, 70 lượt thành công. ViSoBERT/PhoBERT-base-v2 dùng seeds 42–46 cho mọi ST/MTL variant; các backbone khác chạy seed 42 trước. Smoke, tuning, run lỗi và đối chứng không tính vào 70 lượt. Không chọn seed/backbone theo test; config/data/code ảnh hưởng kết quả thay đổi phải tạo cohort mới.

Định danh mẫu, dataset/config hashes, model/tokenizer revisions, commit sạch, seeds, checkpoint selection và môi trường phải đi cùng từng run. Figure 3 có tổng 6.613 mỗi ma trận nhưng trace không khớp Table 6; nguyên nhân chưa được chứng minh. Không gọi fixture là benchmark.

## Gates đang chờ

| Gate | Điều kiện trước full run |
|---|---|
| Data (01) | Audit quyền/revision HF, manifest/hash, label map, segmentation |
| Encoder (02) | Config/revision khóa, SMART gradient, MLM, checkpoint/resume |
| Evaluation (03) | Join ID, metrics/count matrices, reference table |
| Integration (06) | Gate CPU/tiny smoke từ commit snapshot cụ thể |
| Colab (07) | CUDA/VRAM/Drive kiểm tra thật, throughput và sync/checksum |
| LLM (05) | OUT_OF_SCOPE; không chặn release encoder |

## Dữ liệu, Git và cộng tác

Dataset tải và audit theo module 01; GitHub là alternative revision, chưa chứng minh bằng đúng HF/paper. Không đưa raw/processed data, segmentation resources, downloaded tokenizer caches, weights, checkpoints, run artifacts hoặc secrets lên GitHub. Manifests và fixtures synthetic được phép review. Downloader data/resource dùng URLs/revisions/hashes khóa trong `configs/data/`; model/tokenizer tải theo revisions trong configs model/experiment. Không sửa PDF/proposal/slide gốc. Chưa thêm license cho code nguồn có quyền chưa rõ.

Chỉ module 04 quản lý index/commit/remote. Snapshot chỉ chứa file READY có allowlist và hash chủ module bàn giao; không dùng `git add .`. Cây shared còn WIP không phải snapshot training. Module 07 clone và checkout commit cố định, module 06 xác minh gates trên chính commit đó. Xem [quy trình snapshot và GitHub](docs/github.md) và [handoff 04](handoffs/04-infrastructure.md).
