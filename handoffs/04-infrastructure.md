# Module 04 — encoder release candidate, 12/10/2026

Status: PARTIAL

Local/package Status: READY

Org/repo Status: READY

Release candidate Status: CPU_AND_CI_VERIFIED — full encoder commit đã push; READY_FOR_MODULE_06_RECHECK

GPU_EXECUTION: PENDING

MODULE_06_RECHECK: PENDING — giữ các gates/handoff 06 cũ như evidence lịch sử, không sửa thay chủ module

LLM: OUT_OF_SCOPE — không chờ 05

## I06-02 prerequisite đã kiểm tra trước chốt release

Handoff 02 mới READY; receipt `reports/method/i06-02-verification.json` hash `c20f24dbe2f95bdfe0539e2901eaba3a7fb1f97178dc1d5214a3c54570c249d4` và JUnit `77982989ba8efa2373ced586dd99a69d13997a915800b3a37475ec64d72e38ba` khớp. Ba fresh smoke ST sentiment/ST topic/MTL+SMART PASS, metadata/aggregate/metrics/selection khớp. Đã kiểm protected model/loss/config/data/evaluator bytes và toàn bộ artifacts cũ không đổi.

04 đã thực chạy lại reproduction P0 → exit 0/PASS, regression metadata unit → 4 pass, và toàn bộ `pytest tests/training` → **34 pass, 8 GPU skipped**, 92.92s. Không sửa implementation/test của modules 01/02/03/06. Những checks này là smoke/CPU, không benchmark.

## Allowlist/review và outputs mới

- `scripts/environment/release-inventory-20261012.json`: 151 READY files của 01/02/03/06, hashes gốc khi owner/receipt có và hashes review 04 cho paths READY chưa có checksum riêng. Hash mismatch được reject; READY paths không đồng nghĩa cho phép ship cache.
- `scripts/environment/release-allowlist.txt`: file-by-file stage list gồm READY module paths và đúng updates thuộc 04. Không glob/directory hoặc `git add .`.
- `scripts/environment/review_release.py`: read-only ownership/hash/syntax/metadata/archive/credential review; không stage hoặc sửa source owner.
- `scripts/environment/verify_release.py`: clean committed checkout → fresh wheel/hash-locked CPU venv → imports/configs/CLI/tests → sáu fresh-process variants và explicit validation aggregation assertions. Chỉ tạo ignored local evidence.
- `scripts/environment/check_pipeline.py`: imports data/model/loss/training/evaluator thật, 22 full configs, 5 CLI helps, Transformers torch bin-loader compatibility guard; không download model.
- Packaging/locks: `torch>=2.6`, `Pillow>=10.1`, `pypdf`; CPU lock resolved/installed có hash verification, PyYAML/JDK/Colab constraints giữ đúng phạm vi.
- CI: module/integration CPU tests + P0 train→interrupt/resume→predict→evaluate→persisted validation aggregation. Runner giữ CPU cases trong file có case-level dataset/GPU marker.
- `.gitattributes` giữ nguyên hai file CRLF đã được 03 bàn giao (`figure3_audit.json/.md`), tránh đổi checksum khi checkout. Các source/config/test files READY khác đã LF; không normalize hoặc sửa implementation owner.
- 17 downloaded tokenizer cache/resource files dù có trong allowlist 02 vẫn **không ship**. Hashes/revisions/evidence config/metadata còn trong release; full run tải qua HF pinned revision. Raw/processed/resources/checkpoints/secrets/runs và WIP đều excluded.
- Reference ZIP được kiểm trực tiếp: 9 Python source members, không có dataset/weights/checkpoints. Historical Table 6 giữ nguyên đủ 21 paper rows; campaign hiện tại chỉ 22 encoder configs/70 successful runs, benchmark actual 0.

## Release checks và bàn giao SHA

**Tested/pushed encoder SHA: `caaa76228bde261fd33bba15e0c742d77967fdb2`**. Public origin: https://github.com/AI-Peak/neu-esc-reproduction. Commit này có implementation/configs/manifests/tests/integration scripts; 168 paths đã stage file-by-file, 151 owner READY hashes match, 17 tokenizer caches excluded. `.gitattributes` có ngoại lệ whitespace hẹp cho exact upstream copies/CRLF reports và blank EOF evaluator init; không thay bytes của owner.

Clean detached clone: `F:/DPL project/tmp/04-encoder-release-caaa762`. Lệnh thực chạy từ clone:

```powershell
& 'F:/DPL project/neu-esc-reproduction/.venv/Scripts/python.exe' scripts/environment/verify_release.py --output runs/environment/release-04-001
```

Verifier build wheel từ committed clone, tạo venv mới, cài CPU lock với hashes, cài wheel (không editable), bỏ inherited PYTHONPATH/PYTHONHOME và xác nhận imports tới site-packages của venv. Wheel gồm embedded Java bridge và mọi namespace pipeline. pip check PASS; 10 imports/22 configs/5 CLI helps/torch bin-loader guard PASS. **88 CPU tests + 85 subtests PASS; 9 GPU/dataset cases deselected**. Sáu variants ST sentiment/ST topic/MTL/MTL+SMART/MTL+MLM/MTL+MLM+SMART: 28 fresh-process CLI checks PASS, validation metadata/aggregation PASS cả sáu; P0 interrupt/resume và negative rejects PASS. Checkout vẫn sạch sau chạy. Chỉ tiny initialized BERT + synthetic data, không pretrained/full training hoặc benchmark.

Local receipt/logs/wheel/venv/predictions/checkpoints giữ trong ignored `runs/environment/release-04-001` của clone. `scripts/environment/release-verification-20261012.json` là summary packaging/check status/hash, không chứa smoke artifacts hoặc logs. Nó lưu SHA-256 của 93 execution/config/manifest/test/fixture/lock/workflow files để kiểm docs-only follow-up không thay bytes đã test. Historical release allowlist giữ nguyên ứng với tested commit; evidence follow-up dùng allowlist riêng.

CI thật của tested SHA: [run 38214475551](https://github.com/AI-Peak/neu-esc-reproduction/actions/runs/38214475551) **SUCCESS đủ bốn jobs Windows/Ubuntu × Python 3.10/3.12**. Đã đọc từng job logs: mỗi job **88 tests + 85 subtests PASS, 9 deselected**, implementation/config/CLI PASS, fresh-process P0 PASS, persisted validation metadata/aggregation PASS và pip check PASS. GitHub tree API xác nhận cả 180 blobs match local tested commit. Logs CI chỉ local ignored; summary lưu status/URLs/checks/hashes. Bản docs/evidence tiếp theo stage đúng bốn paths ở `release-evidence-allowlist.txt`; 93 execution/config/manifest/test/fixture/lock/workflow bytes khớp clean tested SHA, không đổi source owner. Docs-only commit bỏ CI lặp lại bằng `[skip ci]`; CI chứng minh luôn gắn với tested SHA ở trên.

Colab dùng SHA cố định:

```bash
git clone https://github.com/AI-Peak/neu-esc-reproduction.git /content/neu-esc-reproduction
cd /content/neu-esc-reproduction
git checkout --detach caaa76228bde261fd33bba15e0c742d77967fdb2
git rev-parse HEAD
```

Module 06 kiểm lại gates trên SHA này; module 07 kiểm runtime GPU thật theo recipe môi trường trước full training. `reports/integration/gates.json/input_snapshot.json` và handoff 06 trong release là audit cũ trước fix, không phải release certification mới.

## Blockers / inputs tiếp theo

- I06-01: CLOSED trong phạm vi 04 — commit đầy đủ đã push và kiểm clean wheel/CPU; chờ 06 xác nhận gate trên SHA cụ thể.
- Colab GPU/JDK/Drive/quota/VRAM/pretrained bin load/P0 backward/resume/sync: **PENDING**, không được suy ra từ CPU guard/fixtures. 07 cung cấp runtime output; 06 review gates thật trước full training.
- Dataset manifest digest `ea127f55e840991e53569379a9b3e8ac00bc6798108d6472f8ab300a109f8f7d`: pinned GitHub alternative revision, chưa chứng minh exact HF/paper equivalence. Không thay manifests/22 configs để che giới hạn.
- Không chờ LLM, không full train, không tạo/nhắn chat/subagent, không sửa gates của 06.

---

## Historical infrastructure handoff — 11/10/2026

Các mục bên dưới lưu kết quả thời điểm chỉ có infrastructure, không phải trạng thái release mới.

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
4. 04 xác minh lock GPU từ runtime thật và compatibility requests; 07 chạy full training sau gate, không fallback CPU. Không giả định loại GPU, quota hoặc hỗ trợ bf16.
