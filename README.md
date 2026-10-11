# NEU-ESC reproduction

Repo tái lập baseline gốc của [NEU-ESC](https://arxiv.org/abs/2506.23524), theo [hợp đồng v1](docs/contracts.md). Tham khảo code `hung20gg/multi-task-bert` tại commit `b5637cc0a8a7b5e88e794d180c11902405548eb5`. `paper_method` là baseline chính; `reference_code` là nhánh đối chiếu riêng.

**Chưa có kết quả full training được xác minh.** Windows dùng phát triển, audit và smoke CPU; full training dùng Google Colab GPU sau gate tích hợp. Recipe Colab là PREPARED; chưa xác nhận runtime GPU, quota hoặc Drive. Các module triển khai pipeline đồng thời; các CLI trong hợp đồng là giao diện đích và chỉ được coi là hoạt động khi module chủ bàn giao cùng kiểm tra tích hợp.

Hạ tầng 04 đã cài/import được trên Windows và pass [CI CPU](https://github.com/AI-Peak/neu-esc-reproduction/actions/runs/38148094332): 7 tests mỗi job trên Windows/Ubuntu, Python 3.10/3.12. Commit nền tảng chưa chứa implementation WIP của các module khác.

## Quickstart Windows / Linux CPU

Dùng CPython 3.10–3.13 trên Windows/Linux x86_64 (Linux glibc >=2.28). Bộ CPU khóa riêng giữ PyTorch CPU và không tải CUDA stack. Lệnh PowerShell:

```powershell
py -3.10 -m venv .venv
.venv/Scripts/python -m pip install --require-hashes -r requirements-cpu.lock
.venv/Scripts/python -m pip install --no-deps --no-build-isolation -e .
.venv/Scripts/python -m pip check
.venv/Scripts/python scripts/environment/offline_tests.py
.venv/Scripts/python scripts/environment/capture.py --output runs/environment/windows/environment.json
```

Linux dùng `python3 -m venv .venv`, thay executable bằng `.venv/bin/python`. Installation cần mạng/package cache; chạy fixtures sau cài không yêu cầu model download, dataset thật, GPU hoặc API secrets. Chạy capture với đường dẫn mới mỗi lần; không ghi đè bằng chứng run cũ.

Extras và dependency groups: `data`, `train`, `eval`, `llm`, `dev`. `llm` chỉ bổ sung SDK khi module 05 có imports/dependency requests; group rỗng không chứng minh API đã hoạt động. `uv.lock` khóa resolution các nhóm trên nhiều nền tảng; `requirements-cpu.lock` là môi trường CPU. Không áp dụng lock CPU hoặc `uv sync --all-extras` vào runtime Colab đang có CUDA. Xem [môi trường và recipe Colab](docs/environment.md).

## Phạm vi tái lập

Table 6 có 21 hàng: 5 ST (hai run riêng mỗi hàng), 12 MTL, 4 LLM. Một seed encoder cần 22 lượt train; kế hoạch chính gồm 70 lượt encoder thành công và 4 lượt LLM evaluation. ViSoBERT/PhoBERT-base-v2 dùng seeds 42–46 cho mọi ST/MTL variant; các backbone khác chạy seed 42 trước. Smoke, tuning, run lỗi và đối chứng không tính vào 70 lượt. Không chọn seed/backbone theo test; config/data/code ảnh hưởng kết quả thay đổi phải tạo cohort mới.

Định danh mẫu, dataset/config hashes, model/tokenizer revisions, commit sạch, seeds, checkpoint selection và môi trường phải đi cùng từng run. Figure 3 có tổng 6.613 mỗi ma trận nhưng trace không khớp Table 6; nguyên nhân chưa được chứng minh. Không gọi fixture là benchmark.

## Gates đang chờ

| Gate | Điều kiện trước full run |
|---|---|
| Data (01) | Audit quyền/revision HF, manifest/hash, label map, segmentation |
| Encoder (02) | Config/revision khóa, SMART gradient, MLM, checkpoint/resume |
| Evaluation (03) | Join ID, metrics/count matrices, reference table |
| Integration (06) | Gate CPU/tiny smoke từ commit snapshot cụ thể |
| Colab (07) | CUDA/VRAM/Drive kiểm tra thật, throughput và sync/checksum |
| LLM (05) | SDK/protocol/model snapshot/quyền/ngân sách được xác minh |

## Dữ liệu, Git và cộng tác

Dataset tải và audit theo module 01; không tự đưa data raw/processed, weights, checkpoints, run artifacts hoặc secrets lên GitHub. Manifests và fixtures synthetic được phép review. Không sửa PDF/proposal/slide gốc. Chưa thêm license cho code nguồn có quyền chưa rõ.

Chỉ module 04 quản lý index/commit/remote. Snapshot chỉ chứa file READY có allowlist và hash chủ module bàn giao; không dùng `git add .`. Cây shared còn WIP không phải snapshot training. Module 07 clone và checkout commit cố định, module 06 xác minh gates trên chính commit đó. Xem [quy trình snapshot và GitHub](docs/github.md) và [handoff 04](handoffs/04-infrastructure.md).
