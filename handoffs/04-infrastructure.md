# Module 04 — Repo, môi trường và GitHub

Status: PARTIAL

Local Status: READY

Org/repo Status: READY

Colab recipe Status: PREPARED — chưa chạy trên runtime Colab GPU

Training release Status: BLOCKED — cần allowlist READY của chủ module và gate 06 trên commit đầy đủ

Ngày bàn giao: 11/10/2026, Asia/Saigon. Không tạo chat, gọi subagent, nhắn chat khác, gọi paid API hoặc full training.

## Outputs thuộc 04

- Root: `README.md`, `pyproject.toml`, `uv.lock`, `requirements-cpu.in`, `requirements-cpu.lock`, `.gitignore`, `.gitattributes`.
- `src/neu_esc/__init__.py`: package/version 0.1.0; namespace discovery cho implementation modules.
- `docs/contracts.md`: copy nguyên hợp đồng v1 từ prompt; `docs/environment.md`: Windows/Colab recipe, limits và provenance; `docs/github.md`: account/org/repo thật và READY snapshot procedure.
- `.github/workflows/cpu.yml`: CPU fixtures trên Windows/Linux, Python 3.10/3.12; không yêu cầu GPU/dataset/API secrets.
- `scripts/environment/capture.py`, `colab_install.py`, `offline_tests.py`, `lock_cpu.py`.
- `scripts/environment/dependency-audit.json`: reference imports + byte hashes tại commit khóa và observed local imports WIP.
- `scripts/environment/verification.json`: commands/checks/versions/hashes đã thực thi thật.
- `scripts/environment/tests/test_infrastructure.py`: 7 CPU tests về provenance, overwrite, ignores, offline collection và bảo vệ GPU stack.
- `scripts/environment/fixtures/{synthetic_raw.csv,synthetic_test.jsonl,manifest.example.json}`: hai mẫu synthetic, `benchmark_eligible=false`; không dùng như manifest data thật.
- `scripts/environment/04-allowlist.txt`: file cụ thể READY của module 04 cho infrastructure commit; không chứa implementation của module khác.
- `handoffs/04-infrastructure.md`: handoff này.
- Local ignored evidence: `runs/environment/windows-001/environment.json`, `runs/environment/windows-002/environment.json`; build verification trong `dist/` (wheel/sdist chỉ local, có WIP, không phát hành).

Tất cả outputs 04 trong allowlist là **READY infrastructure** để stage/commit. Không có allowlist READY của module khác trong infrastructure snapshot. File họ tạo vẫn giữ nguyên, chưa stage. Đây chưa phải snapshot chạy benchmark.

## Sources, revisions, hashes và config

- Paper: `F:/DPL project/references/NEU-ESC-2506.23524.pdf`, https://arxiv.org/abs/2506.23524; hash tham khảo kế hoạch `bdc37ae84e6653ca60b325ccb54f7f655fb4cd1ee606d0f9a8a5a15772e7189c` (04 không sửa/không đọc lại PDF).
- Reference: https://github.com/hung20gg/multi-task-bert, commit `b5637cc0a8a7b5e88e794d180c11902405548eb5`. Tám Python files đã fetch read-only để audit AST imports; từng byte hash lưu trong dependency audit. Không vendor code hoặc thêm license chưa rõ.
- HF: https://huggingface.co/datasets/hung20gg/NEU-ESC; revision/quyền/equivalence do module 01 khóa, chưa được 04 chứng minh.
- Contract v1 SHA-256: `d9abd45e16f1fdea1e5a590a459eefab574ab5eef415cccb25565aa8dd134b15`. Giữ interfaces nguyên bản, không đổi schema.
- Packaging/lock hashes và installed versions: `scripts/environment/verification.json`.
- CPU dev stack thật: CPython 3.10.11, torch 2.14.1+cpu, transformers 4.57.6, numpy 2.2.6, pandas 2.3.3, PyYAML 6.0.3, scikit-learn 1.7.2, matplotlib 3.10.9, pytest 9.1.1, setuptools 84.0.0. Java/JDK 22.0.2 đã kiểm tra local.
- `uv.lock` là universal resolver lock; CPU lock dùng tám URL/hash wheels thật lấy từ official PyTorch CPU index. Không bịa version paper hoặc dùng CPU lock làm CUDA environment lock.
- Git commit infrastructure đã push: **`573a4b4d0adce02edf4dae9d0c7643c0faa12f45`**, đúng 24 file 04 allowlist. Handoff này được cập nhật ở commit tài liệu sau đó; full training release SHA chưa có.

## Commands/checks thực tế

| Check | Kết quả |
|---|---|
| Workspace/AGENTS | Cha `F:/DPL project` không là Git repo; repo con được tạo bởi modules cùng lúc. Không có AGENTS áp dụng được tìm thấy lúc kiểm tra. Chỉ `git init -b main` tại repo con. |
| `gh auth status` + membership APIs | Active `MarcusCyborg`; AI-Peak membership `active/admin`. |
| `gh repo create AI-Peak/neu-esc-reproduction --public ...` + view | Repo public thật đã tạo, origin HTTPS đúng. User cung cấp org đã có và chọn repo/public. |
| `uv lock --check` | 87 packages resolved; lock hợp lệ/current. |
| `python scripts/environment/lock_cpu.py` | Resolve thành công và emit CPU wheel pins/hashes. |
| `uv pip install ... --require-hashes -r requirements-cpu.lock` | 52 packages installed với hash verification. |
| `ensurepip`, editable install, `pip check` | Installed package 0.1.0; không có broken requirements. |
| Package/dependency imports | `neu_esc`, torch, transformers, yaml, numpy, pandas, sklearn, matplotlib, sentencepiece import được; `torch.cuda.is_available()=false`. |
| `offline_tests.py` trên shared tree lúc kiểm tra | 61 tests collected; **61 passed + 60 subtests passed**, 33.82s. Code modules khác còn thay đổi, đây không phải release gate. |
| `pytest scripts/environment/tests -q` sau cập nhật packaging/provenance | **7 passed**, 4.29s. |
| Clone sạch commit `573a4b4...` với package source của snapshot | **7 collected, 7 passed**, 5.25s; Git status clean. Chỉ xác minh infrastructure, không train pipeline. |
| GitHub Actions run `38148094332` trên commit `573a4b4...` | **SUCCESS**, đủ 4 jobs Windows/Ubuntu × Python 3.10/3.12; mỗi job 7 collected/7 passed và pip check pass. Logs/API đã đọc lại. |
| `python -m build --no-isolation` + inspect wheel | sdist/wheel built; `neu_esc/data/WsegBridge.java` có trong wheel. Không upload dist chứa WIP. |
| capture windows-002 | Lưu package/Python/CPU/Java/Git provenance, không thu thập env secrets. |
| colab_install trên Windows CPU | Từ chối trước install vì chưa có Linux CUDA verified, không tạo evidence-dir. Không phải Colab test. |
| Ignore checks | Raw/processed data, run artifacts, secrets/checkpoints bị ignore; manifests/synthetic fixtures/locks vẫn review được. |

Lần cài lock đầu gặp hash mismatch ở MarkupSafe do CPU index được dùng cho dependency không thuộc torch. Đã giữ hash verification, sửa nguồn CPU thành URL/hash riêng cho torch, resolve các package còn lại từ PyPI rồi cài lại thành công. Lần resolve tám direct wheel metadata ban đầu chậm và được dừng; recipe generator hiện resolve registry metadata trước khi emit audited wheel URLs.

## Đã xác minh và chưa xác minh

Đã xác minh packaging editable, local import/collection/CPU tests, build package-data Java bridge, locks resolver/download, ignores, org/account membership, public repo URL/remote/push và CI CPU trên Windows/Ubuntu Python 3.10/3.12. Không thay đổi dataset/proposal/PDF/slide hoặc implementation/test của module khác.

Chưa xác minh Colab GPU execution, CUDA stack phù hợp model thật, Drive/quota/VRAM, full model/tokenizer revisions, real dataset equivalence hay benchmark. Linux CPU đã được GitHub CI thực thi; không suy ra Colab GPU readiness. Fixture counts/metrics không được báo như paper reproduction.

Giả thuyết Figure 3 lệch do order/checkpoint/dataset chưa kiểm chứng; 04 không thêm kết luận nguyên nhân. SMART audit thuộc 02; hai method_mode không được gộp.

## Giả định và sai khác

- `transformers<5` là compatibility bound cho API 4.x; các version chọn bởi resolver là dev choice, không phải versions paper.
- `sentencepiece` là tokenizer support dependency, chưa chứng minh checkpoint thật cần/đủ chỉ bằng local import.
- `llm` group hiện rỗng: chờ SDK/import request module 05; không đồng nghĩa LLM pipeline READY.
- Native segmentation yêu cầu JDK 11+ và resources khóa của 01; package-data `.java` được thêm theo requirement thấy trong implementation local.
- CPU lock hỗ trợ CPython 3.10–3.13 Windows/Linux x86_64, Linux glibc >=2.28. Windows 3.10 test local và Windows/Ubuntu 3.10/3.12 test CI; 3.11/3.13 chỉ resolver, chưa chạy tests trực tiếp trong bàn giao này.
- Không đổi interface contract. Colab preflight lưu ngoài encoder run-dir vì trainer 02 tự tạo run-dir mới và ghi environment.json.

## GitHub và blockers

Org: https://github.com/AI-Peak (AI Peak). Repo: https://github.com/AI-Peak/neu-esc-reproduction — PUBLIC. User là owner và sẽ tự add thành viên; **không gửi invitations**, không mua plan/cấp owner khác. Org/repo scope đã hoàn tất.

Phần Colab dừng ở PREPARED vì chưa có runtime/output thật. Full training release dừng ở dependent gate: không stage source/config/tests/reports WIP, không nhận shared tree sạch. Module 04 có thể commit/push outputs của mình; 07 chỉ được train từ release đủ files READY và gate 06.

## Bước tiếp theo / inputs cần

1. Module 01/02/03/05 giao dependency requests + file READY cụ thể/hashes trong handoff; 04 review/bổ sung SDK nếu có rồi stage allowlist tuần tự. Không nhắn chat khác nếu user chưa chỉ định.
2. Module 06 kiểm tra clean release checkout đầy đủ, tests/gates và CLI interfaces trên commit SHA đó.
3. Module 07 dùng recipe `docs/environment.md` trong notebook của mình: checkout commit cố định, user mở GPU/mount Drive, dry-run constrained install, verify/restart runtime khi cần, smoke và sync evidence. Gửi Python/PyTorch/CUDA/GPU/pip-check output, không gửi token.
4. 04 xác minh lock GPU từ runtime thật và compatibility requests; 07 chạy full training sau gate, không fallback CPU. Không giả định T4/quota/bf16.
