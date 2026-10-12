# NEU-ESC: kiểm toán nguồn và phạm vi

Ngày kiểm tra: 2026-10-11, Asia/Saigon. Chủ module: 01. Các số đo model và bảng paper do module 03 sở hữu; tài liệu này chỉ khóa nguồn, phạm vi và giới hạn tái lập.

## Nguồn đã khóa

| Nguồn | Revision/hash | Bằng chứng và giới hạn |
|---|---|---|
| Paper [arXiv 2506.23524](https://arxiv.org/abs/2506.23524) | PDF SHA-256 `bdc37ae84e6653ca60b325ccb54f7f655fb4cd1ee606d0f9a8a5a15772e7189c` | Đọc PDF local 24 trang; kiểm tra trực quan trang 18–19. Số trang ở đây là số trang PDF, bắt đầu từ 1. Chưa so byte với bản arXiv hiện tại. Không sửa PDF. |
| [GitHub tác giả](https://github.com/hung20gg/multi-task-bert/tree/b5637cc0a8a7b5e88e794d180c11902405548eb5) | `b5637cc0a8a7b5e88e794d180c11902405548eb5`, commit 2024-07-18 | Lưu cây Git, README, logic đọc CSV và cache nguồn để audit. Commit trước paper; chưa xác minh là code tạo Table 6. README có kết quả khác Table 6. |
| [HF dataset](https://huggingface.co/datasets/hung20gg/NEU-ESC) | Revision quan sát `daf543ad1992153cd2be9fec3cb59aa0fc714147` | API public báo `gated=auto`, license metadata `apache-2.0`. README pinned đọc không xác thực: HTTP 200. CSV train pinned: HTTP 401. Không dùng token, không thử cách vượt gate, chưa tải/so nội dung HF. |
| [Hướng dẫn PhoBERT](https://github.com/VinAIResearch/PhoBERT/blob/7b4a3e38cd316c14d4006fe0328d3e2d6f1edada/README.md) | `7b4a3e38cd316c14d4006fe0328d3e2d6f1edada` | Khuyến nghị RDRSegmenter/VnCoreNLP cho input chưa word-segment. Đây là căn cứ lựa chọn hiện tại, không chứng minh tool của paper NEU-ESC. |
| [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP/tree/62bbc58fe5d113c898eae112656be97dcf50b3a0) | JAR 1.2 và tài nguyên tại `62bbc58fe5d113c898eae112656be97dcf50b3a0` | JAR, dictionary, rule model, README/license khóa SHA-256 trong `configs/data/segmentation.lock.json`. License upstream GPL-3.0-or-later. Chỉ dùng annotator `wseg`. |

Source ledger máy đọc được: `reports/source/source_ledger.json`. Metadata HF và kết quả truy cập nằm trong `hf_dataset.json`, `hf_access.json`, `hf_csv_access.json`. Những file evidence lớn/nguyên văn được giữ local và loại khỏi danh sách release.

### Hash CSV của GitHub

| Split | Số mẫu | SHA-256 bytes nguồn |
|---|---:|---|
| train | 23048 | `8a18a73ca21105e0b424829cc41395719be98dd7f01bf3d962eca86e7a1501ec` |
| validation (`val_set.csv`) | 3305 | `0b2567fe5374cf9b8d22de92ce71d3f9bfa036dffac46fa6f7a5650e85c81169` |
| test | 6613 | `cdd81b2c7b96752f9f45d05e231682fcd6c380f40273dac521425ceff71ab312` |

Cả ba khớp hash người dùng đã quan sát. Hash sáu CSV, kể cả bản tác giả `_processed`, nằm trong config, manifest và source ledger. Tổng 32966 khớp Table 3, nhưng số mẫu bằng nhau không chứng minh cùng revision hay cùng nội dung.

### License và phân phối

Không thấy LICENSE trong toàn bộ cây Git tại commit đã khóa, README không cấp quyền phân phối dataset. License Apache-2.0 của HF là metadata của bản HF gated; chưa xác lập nó áp dụng cho bản GitHub. Vì vậy `redistribution_status=not_authorized_for_github_copy`. Cache raw, JSONL và binary/dictionary chỉ lưu local, bị ignore bởi `data/.gitignore`; module 04 không đưa chúng vào GitHub organization/release. Recipe tải nguồn public có checksum, config/manifest/audit/code có thể bàn giao để tái tạo input local. Không coi việc đọc được URL public là quyền đăng lại dữ liệu. Cache code tác giả không được đưa vào release, trong đó có nội dung trông như credential cũ; không sao chép giá trị vào báo cáo/log.

## Sổ phạm vi

Danh sách từng hàng và seed trong `reports/source/scope_ledger.json`; không lưu metrics paper tại đây.

| Nhóm | Phần tái lập được đưa vào kế hoạch | Phần chưa xác minh |
|---|---|---|
| Table 6, trang 18: 5 ST rows | BERTbase, vELECTRA, XLM-Rbase, ViSoBERT, PhoBERT-base-v2; mỗi hàng gồm hai run `st_sentiment` và `st_topic` | Table ghi BERTbase; checkpoint multilingual uncased là ứng viên từ code/plan, chưa phải định danh đầy đủ của paper. |
| Table 6: 12 MTL rows | XLM-Rbase/ViSoBERT/PhoBERT-base-v2 × `mtl`, `mtl_smart`, `mtl_mlm`, `mtl_mlm_smart` | Revisions model/tokenizer, hyperparameters và checkpoint selection gốc thiếu; SMART/MLM cần audit module 02. |
| Table 6: 4 LLM rows | GPT 4o và Claude 4, mỗi model zero-shot/few-shot; few-shot 10 mẫu train (trang 15–16) | Exact API snapshot, prompt, IDs ví dụ và xử lý lỗi chưa công bố đủ; module 05 không được suy từ tên model sang snapshot. |
| Figure 3, trang 19 | Hai CM cho ViSoBERT MTL+SMART, N=6613 mỗi task; seed 42 chọn trước theo plan | Seed paper không được xác định. Số ô CM mâu thuẫn với Table 6; chưa có nguyên nhân chứng minh. Module 03 sở hữu CSV/count audit. |
| BERT-CNN, trang 6–7, 13–14 | Đã đưa vào scope ledger như mô hình được mô tả | Không có hàng CNN riêng trong Table 6, thiếu filter/architecture/target đủ; không tự nhân đôi campaign. |
| UIT-VSFC, ViHSD, ViCTSD, trang 11–13 và 21 | Audit các so sánh mô tả Tables 4–5 và lời kết luận benchmark | Không có target table/protocol cross-dataset đầy đủ để tái lập thêm run; CSV/code lẻ trong GitHub không thay thế protocol. |
| VLSP 2016/2018, Stanford MOOC Posts, trang 5 | Background/related work | Không phải campaign NEU-ESC được đặc tả. |
| Thu thập/gán nhãn, trang 8–9 | Ghi provenance mô tả của paper | Không rescrape Facebook/Voz/XamVN, không tự tạo dictionary viết tắt hay relabel. |
| Cải tiến proposal/future work | Ghi ngoài phạm vi baseline | Không triển khai Factorized Joint MTL/PaTGrad/PCGrad/ALICE/adapters hoặc fine-tune Qwen/LLaMA trong module này. |

Một seed phủ encoder: `5×2 + 3×4 = 22` run. Theo yêu cầu dự án, mọi ST/MTL variant của ViSoBERT/PhoBERT dùng [42,43,44,45,46], các backbone còn lại seed 42 trước: 26 ST + 44 MTL = **70** encoder run thành công, cộng 4 LLM evaluation. Đây là kế hoạch của dự án, không phải protocol multi-seed paper đã công bố. Không chọn seed theo test; smoke/tuning/run lỗi không tính vào campaign. `paper_method` và `reference_code` tách cohort/report.

## Bằng chứng so với paper

`reports/source/paper_data_comparison.json` ghi các đối chiếu sau:

- Topic counts toàn bộ CSV khớp đầy đủ 10 counts Table 2. Mapping Table 2 xác nhận `3=Other`; câu mô tả “nine”/“Neutral” trong prose trang 9–10 không được dùng để thay mapping.
- Sentiment CSV: [22773,4148,5200,845]. Table 1 ghi Negative=5250, làm tổng counts Table 1 thành 33016, hơn Table 3 đúng 50. Không sửa dataset để khớp số in.
- True-label marginals test của cả hai task khớp tất cả row totals Figure 3 từ audit module 03. Điều này hỗ trợ thứ tự label và counts; không chứng minh các text/IDs/checkpoint giống paper.
- Figure 3 cho trace/N khoảng 54.72554% sentiment và 31.36247% topic, khác Table 6 83.58%/79.80%; tham chiếu `reports/reference/figure3_audit.json`. Labels/predictions lệch thứ tự, checkpoint khác hoặc dữ liệu khác vẫn là giả thuyết. Không quy kết `shuffle=True` tự nó gây lỗi.
- Public loader tạo `sent_index` nhưng TensorDataset không trả index, và có shuffle; đó là rủi ro provenance, không phải chứng minh lỗi Figure 3. Pipeline mới giữ `sample_id` xuyên suốt API.

## Những giả định cần giữ trong báo cáo/full training

Input hiện tại là **GitHub alternative revision, HF/paper equivalence unverified**; manifest/config lưu đúng notice này. Raw ở API nghĩa là text của CSV curated chưa word-segment, không phải raw scrape. Không lặp lại cleaning mô tả trong paper, vì source đã clean, tool/dictionary nguyên gốc không có. Nhánh PhoBERT dùng VnCoreNLP đã khóa nhưng vẫn là preprocessing thay thế (chi tiết trong data audit).

`utils/loss_function.py` public có ép `.long()` sau cộng nhiễu và trainer truyền token IDs; module 01 chỉ đọc nguồn, chưa kiểm thử gradient. Module 02 phải audit động và phân biệt `reference_code`/`paper_method`. MLM denominator, selection rule, seeds, optimizer và các tham số thiếu không được coi là đã xác minh bằng default thư viện.

Để khép provenance cần quyền HF và dataset revision thực sự dùng cho paper, cleaning/acronym dictionary/segmenter gốc hoặc xác nhận từ tác giả. Không gửi yêu cầu/liên hệ tự động. Thiếu những thông tin này không cản chuẩn bị input thay thế có thể kiểm toán; nó giới hạn tuyên bố tái lập chính xác.
