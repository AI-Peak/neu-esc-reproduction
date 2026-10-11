# Môi trường Windows và Google Colab

Module 04 quản lý packaging và recipe; module 07 quản lý notebook/Drive/queue. Không có full training trong bàn giao này. Các phiên bản lock là lựa chọn phát triển được resolver kiểm tra, không phải môi trường tác giả paper. Windows local đã cài editable và pass CPU fixtures; CUDA unavailable trong bộ PyTorch CPU này.

## Dependencies và locks

`pyproject.toml` có extras và dependency groups `data/train/eval/llm/dev`; base import `neu_esc` dùng stdlib. Audit imports ở code tham khảo commit `b5637cc0a8a7b5e88e794d180c11902405548eb5` thấy `torch`, `transformers`, `numpy`, `pandas`, `sklearn`, `tqdm`, `wandb`; hashes và imports được lưu ở `scripts/environment/dependency-audit.json`. Implementation local dùng `PyYAML` cho config và `matplotlib` cho ảnh metrics. Không tự kéo `wandb` vì implementation chưa yêu cầu tracking dịch vụ; module 05 phải bàn giao imports/dependencies trước khi bổ sung SDK vào `llm`. `transformers<5` là giả định tương thích API 4.x, không phải version paper công bố. Tokenizer `sentencepiece` phục vụ tokenizers dự kiến; verify checkpoint/tokenizer thật thuộc 01/02.

| File | Mục đích | Giới hạn |
|---|---|---|
| `uv.lock` | Resolution universal có metadata và hashes | Không coi resolver success là GPU verification; không sync toàn bộ lên Colab |
| `requirements-cpu.in` | Build tooling + URLs CPU wheels lấy từ PyTorch index | CPython 3.10–3.13, Windows/Linux x86_64; Linux glibc >=2.28 |
| `requirements-cpu.lock` | Pin/hash dependencies CPU; markers theo Python/platform | Windows local 3.10 và CI Windows/Ubuntu 3.10/3.12 đã pass; Colab GPU chưa chạy |
| `runs/environment/.../environment.json` | Bằng chứng thực tế package/Python/GPU/commit | Ignored, phải sync cùng run metadata |
| `requirements.verified.txt` trong Colab evidence | Exact pins sau install/import/CUDA probe | Chỉ áp dụng runtime tương thích; không mang sang Windows hoặc GPU stack khác |

CPU wheel URLs tách riêng để PyTorch CPU index không trở thành nguồn của các dependency khác. Không tắt `--require-hashes` khi download mismatch. Tạo lại locks sau dependency request đã review:

```powershell
uv lock
python scripts/environment/lock_cpu.py
```

Script resolve metadata CPU từ registry, rồi emit đúng URLs/hash đã audit của tám wheels Windows/Linux; tránh phải tải cả tám wheels chỉ để đọc metadata. Không tự upgrade `torch` pin/URLs; xác minh releases và wheels thật, kiểm tra CPU fixtures rồi cập nhật audit/handoff. Các lựa chọn phiên bản ảnh hưởng kết quả full run phải đổi cohort.

Implementation data 01 dùng VnCoreNLP/RDRSegmenter qua Java source launcher: cần **JDK 11+**, bridge `.java` phải đi trong wheel (package-data đã cấu hình). Không cần Python VnCoreNLP wrapper nếu imports không yêu cầu. Module 01 khóa jars/models/resources và xác minh preprocessing; capture ghi version Java nếu có, không tự cài JDK/resources hay suy segmentation tương đương paper.

## Windows development và CPU offline checks

```powershell
py -3.10 -m venv .venv
.venv/Scripts/python -m pip install --require-hashes -r requirements-cpu.lock
.venv/Scripts/python -m pip install --no-deps --no-build-isolation -e .
.venv/Scripts/python -m pip check
.venv/Scripts/python scripts/environment/offline_tests.py
.venv/Scripts/python scripts/environment/capture.py --output runs/environment/windows-001/environment.json
```

Linux dùng `.venv/bin/python`. Nhóm test có đánh dấu `gpu/network/paid_api/dataset` bị bỏ trước import bởi runner; fixtures CPU còn lại được collect và chạy. Chủ module phải giữ import/conftest không có network, training hoặc API side effects, và đánh dấu test cần tài nguyên ngoài. CPU tests dùng fixtures synthetic, không đại diện benchmark. Các fixture/manifest ví dụ tại `scripts/environment/fixtures/` chỉ minh họa; module 01 sở hữu manifest thật, quyền download và equivalence audit. Không có script download tự động vượt quyền dữ liệu.

## Recipe cung cấp cho notebook 07 — PREPARED

Notebook thuộc `notebooks/colab_reproduce.ipynb`; module 04 không sửa file đó. Chưa kết nối runtime Colab. Người dùng chọn GPU và mount Drive; không mua gói/tăng ngân sách. Mỗi runtime mặc định một job.

1. Người dùng mở notebook và chọn GPU. Kiểm tra Python/PyTorch/CUDA đang có trước khi cài thêm package:

```python
import sys, torch
print(sys.version)
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Full training cần Colab CUDA GPU; hãy chọn GPU runtime")
print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)
print("bf16_supported", torch.cuda.is_bf16_supported())
```

Không có GPU thì dừng full train. Không tự cài CUDA/PyTorch mới để giả lập GPU. T4 dùng fp16 khi đã test; bf16 chỉ bật khi runtime và model smoke xác nhận hỗ trợ. Không suy loại GPU/quota từ tên Colab.

2. Clone URL thật, checkout **full 40-character commit** mà 04 phát hành và 06 đã xác minh. Public repo hiện tại không cần token:

```bash
git clone https://github.com/AI-Peak/neu-esc-reproduction.git /content/neu-esc-reproduction
cd /content/neu-esc-reproduction
git checkout --detach <FULL_COMMIT_FROM_READY_RELEASE>
git rev-parse HEAD
git status --porcelain --untracked-files=all
```

Commit chỉ có nền tảng 04 chưa phải release train. Không dùng alias `main`, editable từ cây shared đang thay đổi, hoặc snapshot thiếu implementation. Nếu repo đã tồn tại, kiểm tra URL/commit trước; không xóa/reset cây chưa rõ nguồn. Repo private nếu thay visibility sau này: người dùng quản lý Secrets/auth, không chèn token vào URL hay output.

3. Python cần nằm trong 3.10–3.13. Dùng chính interpreter GPU hiện có, không tạo virtualenv khiến mất torch GPU. `setuptools>=68` cần có cho `--no-build-isolation`; nếu thiếu, cài build tooling rồi kiểm tra lại. Capture và lập kế hoạch cài:

```bash
cd /content/neu-esc-reproduction
python scripts/environment/capture.py --output runs/environment/colab-preflight/environment.json --require-cuda --require-clean
python scripts/environment/colab_install.py --evidence-dir runs/environment/colab-plan
```

Script ghi pins mọi package torch/torchvision/torchaudio/triton/nvidia-* đang có, dùng pip resolver với constraints, và từ chối thay GPU stack. Nếu resolver thất bại, bàn giao plan và phiên bản về 04/06; không bỏ constraints hoặc downgrade GPU stack mù. Script chỉ lưu name/version trong plan, không lưu credential-bearing URLs. Xem các proposed changes trước khi cài:

```bash
python scripts/environment/colab_install.py --evidence-dir runs/environment/colab-install --apply
```

Script chạy `pip check`, probes imports và CUDA tensor trong process mới, rồi ghi exact package pins và environment before/after. Nó **không** xác minh accuracy/model/tokenizer/dataset. Nếu runtime vốn có package conflicts, báo blocker và review tương thích; không che lỗi pip check. Nếu package bị thay khi đã import vào notebook, chọn **Runtime → Restart session**, chạy lại import/preflight bằng evidence path mới. Restart disconnects kernel; mount Drive/variables/secrets cần thiết lập lại theo notebook. Thay torch/CUDA cố ý chỉ sau kế hoạch tương thích và restart bắt buộc, không nằm trong recipe tự động này.

4. Trước full run, chạy CPU fixtures offline và smoke GPU được 06/02 chỉ định trên commit đó. Người dùng mount Drive; 07 kiểm tra quyền/dung lượng, throughput, copy best/latest và metadata với checksum/completion marker. Chạy capture cho mỗi run:

```bash
python scripts/environment/offline_tests.py
python scripts/environment/capture.py --output runs/environment/<RUN_ID>/preflight.json --require-cuda --require-clean
python -m neu_esc.training.train --config configs/experiments/visobert/mtl_smart.yaml --seed 42 --run-dir runs/encoder/<RUN_ID>
```

Đây là lệnh đích contract, chưa khẳng định trainer đã pass integration. Trainer 02 tạo run-dir mới và tự ghi `environment.json`; preflight nằm ngoài run-dir để không ghi đè hoặc làm trainer từ chối thư mục đã tồn tại. Module 02 vẫn phải log seeds/RNG/deterministic flags thực dùng trong process training, vì preflight process không chứng minh các flags trong trainer. CUDA capture trước/sau không thay thế config/provenance/manifest/checkpoint/selection.

Sau verify, sync evidence/locks cùng mỗi run về Drive. Trên runtime mới, capture stack trước; chỉ dùng pins cũ khi Python/platform/GPU stack tương thích. Dataset/model/tokenizer revisions, implementation commit và dependency changes phải được khóa trong cohort.
