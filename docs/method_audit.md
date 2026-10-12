# Audit phương pháp encoder NEU-ESC - module 02

Ngày audit: 11/10/2026. Baseline triển khai là `paper_method`. Không có kết quả benchmark trong module này.

Nguồn paper: `references/NEU-ESC-2506.23524.pdf` trong workspace, SHA-256 `bdc37ae84e6653ca60b325ccb54f7f655fb4cd1ee606d0f9a8a5a15772e7189c`. Đã đọc mục 4, trang 13-18 và xem trực tiếp trang 15 để xác nhận vị trí mẫu số trong phương trình. Nguồn code đọc tại commit `b5637cc0a8a7b5e88e794d180c11902405548eb5` của [multi-task-bert](https://github.com/hung20gg/multi-task-bert/tree/b5637cc0a8a7b5e88e794d180c11902405548eb5). Byte gốc của chín file liên quan được lưu trong `reports/method/reference-source.zip`, hash upstream ở `reports/method/reference/upstream_hashes.json`. Bản Python dùng để đọc/chạy audit trong `reports/method/reference/` được chuẩn hóa line endings LF cho Git checkout; hash checkout ở `reports/method/delivery_inventory.json`. Không sửa logic upstream.

## Backbone và tokenizer

| Slug | Checkpoint/tokenizer ID đã audit | Revision khóa | Vocab | Family |
|---|---|---|---:|---|
| mbert | google-bert/bert-base-multilingual-uncased | 7cbf9a625e29989f6b9c6c2fa68234c304f7e38f | 105879 | BERT |
| velectra | FPTAI/velectra-base-discriminator-cased | f594fee501933d92d8c13285c969495681d0cc00 | 32054 | ELECTRA |
| xlm_r | FacebookAI/xlm-roberta-base | e73636d4f797dec63c3081bb6ed5c7b0bb3f2089 | 250002 | XLM-R |
| visobert | uitnlp/visobert | 196a62afad9cbe4f52a54aabad828b13f0eec59a | 15004 | XLM-R |
| phobert | vinai/phobert-base-v2 | 86cd7fd4c148980922ac11a2cf5e257f2ba639e1 | 64001 | RoBERTa |

Paper gọi BERT-base nhưng không xác định cased/uncased hoặc checkpoint/revision đầy đủ. Multilingual **uncased** là thông tin từ `train.py` của repo, không phải chi tiết được paper nêu. ID `xlm-roberta-base` trong repo được khóa dưới namespace hiện tại `FacebookAI/xlm-roberta-base`. Không có bằng chứng các SHA trên là revision tạo Table 6; đây là các revision công khai được khóa cho cohort tái lập hiện tại.

Tất cả có hidden size 768, 12 layers, hidden/attention dropout 0.1. Dùng `use_fast=False` như dataloader gốc. Đã tải và thực thi tokenizer cả năm checkpoint, kiểm tra ID token không vượt vocab. Đã GET Range 16 byte của file weights tại từng SHA: cả năm trả HTTP 206; metadata LFS SHA-256/size trong `reports/method/weights_access.json`. **Chưa tải/load toàn bộ pretrained weights hoặc chạy GPU backward**; phép probe không thay thế test đó. Config JSON, byte hash và tokenizer snapshots trong `reports/method/backbones/`, registry trong `configs/models/`.

PhoBERT đọc trường segmented từ API data; thiếu trường này phải fail. Training không cleaning/segmentation lại. Dataset hiện khóa là nguồn GitHub thay thế có số mẫu đúng, chưa chứng minh tương đương HF/paper. Nội dung notice và segmentation assumptions của module 01 đi cùng provenance/config cohort.

## Paper / code / implementation / assumption

| Quyết định | Paper | Code công khai | Implementation | Giả định hoặc khác biệt |
|---|---|---|---|---|
| ST | Fine-tune sentiment và topic riêng | `trainer/base_trainer.py`: AutoModelForSequenceClassification | Hai lần train riêng, native head của từng family, logits 4 hoặc 10 | Không dùng projection MTL cho ST. Native head dropout 0.1; vị trí/số lần dropout tùy family |
| MTL | Encoder, linear, layer norm, hai FC heads | `BertLinear2HEAD`: first hidden token, Linear/SiLU/LayerNorm/Linear(4,10) | Cùng đường tính; encoder trainable; heads độc lập | SiLU và first-token pooling lấy từ code. LayerNorm của projection eps=1e-5 |
| Dropout MTL | Không cho tỷ lệ | Hai Dropout(0.2) được khai báo nhưng không gọi trong forward | Extra head dropout thực dùng=0.0; native encoder dropout giữ 0.1 | Không tự thêm dropout 0.2 vào head |
| Pretrained loading | Fine-tuning BERT | AutoModel.from_pretrained; ST dùng native sequence classifier | SHA khóa, `trust_remote_code=False`; encoder-body missing/mismatched weights bị từ chối; log loading info | Pooler/classifier mới có thể được khởi tạo theo native loader; pooler MTL không dùng |
| Classification loss | Cộng loss các head | 2-head trainer dùng 0.5/0.5; 3-head dùng 1/1 | CE `mean` mỗi task rồi cộng với 1/1 | Theo phương trình paper, không class weighting; khác scaling code 2-head |
| CNN | Nhắc BERT-CNN ở mục 4.1 | Có class CNN, nhưng head trainer chọn linear | Không thêm CNN vào 22 cấu hình | Table 6 không định danh hàng CNN riêng; không đủ căn cứ nhân đôi ma trận |
| MLM head | Head dự đoán token phụ trợ | 3-head: Linear/SiLU/LayerNorm/Linear(64001), encoder forward hai lần | Untied random head, đầu ra bằng vocab checkpoint; forward sạch cho classification, forward masked riêng cho MLM | Head pretrained MLM không được dùng. Sửa vocab hardcode là cần để hỗ trợ ViSoBERT/XLM-R |
| Masking | 30%, selected 80/10/10 | Sampling vị trí có replacement, có thể trùng/special; số đếm trước dedup; random ID hardcode 64000 | Bernoulli 30% không trùng vị trí; special/pad bị loại; random replacement trong vocab và loại special | Paper không chỉ rõ chọn chính xác floor(0.3N) hay Bernoulli. Không ép một token mask ở batch rỗng |
| MLM sigma | Dòng đầu: average tokens/sentence; đoạn sau: average masked tokens/sentence | `mean` CE rồi chia `total_mask / batch_size` lần nữa | Sum token CE / selected token count | Chọn mô tả masked tokens và chuẩn hóa đúng một lần; không gọi là công thức duy nhất được paper xác định |
| SMART | Smooth predictions dưới perturbation nhỏ, mỗi task cộng lambda R | Token IDs + noise, `.long()`, inner KL batchmean, final sym-KL sum; lambda=0.02 | Perturb word embeddings liên tục, adversary riêng từng task, batchmean symmetric KL | Chỉ smoothness regularizer mà NEU-ESC mô tả; không nhận là toàn bộ thuật toán SMART gốc gồm proximal/Bregman optimization |
| Selection | Không xác định đủ protocol | ST mặc định validation macro-F1, patience=4; MTL lưu nhiều epoch theo ngưỡng/epoch | Validation macro-F1, ST hoặc mean hai task, tie epoch sớm | Fixed 15/20 epochs, không early stopping; test không tham gia |

## SMART: bằng chứng thực thi và giới hạn

`Trainer/head_trainer.py` truyền `b_input_ids` trực tiếp cho SMART. Helper tại commit khóa cộng noise float vào các ID nguyên, rồi gọi `eval_fn(embed_perturbed.long(), input_mask)`. `.long()` có `requires_grad=False` và không có `grad_fn`. Với tiny initialized transformer và đường lookup embeddings thật, `torch.autograd.grad(loss, noise, allow_unused=True)` trả **None**. `allow_unused=False` báo tensor không được dùng trong graph. Không thể diễn giải None là gradient adversarial bằng 0 đã được tính đúng.

Helper gốc xử lý None bằng `step=noise`, rồi vẫn normalize và chạy final forward. Nó **chạy được** trên fixture đã thử, có final loss; không thấy lỗi runtime trong fixture này. Gradient model vẫn có thể xuất hiện từ final KL dù gradient noise bị ngắt. `.long()` sau nhiễu âm nhỏ có thể đổi ID dương thành ID liền trước (phụ thuộc độ chính xác float và độ lớn ID), gồm cả special tokens; không chỉ là một perturbation nhỏ trong không gian embeddings. Không tự sửa helper gốc rồi gọi đó là hành vi nguyên trạng.

Mode `reference_code` chỉ chấp nhận variant `mtl_smart` để chẩn đoán **đường SMART hai task**. `reference_smart_loss` giữ cast/None fallback/normalization/sum reduction và không sửa masking/dropout. Test so sánh kết quả bitwise với helper gốc dùng cùng seed. Các runtime errors của helper không bị nuốt. Mode này không hứa tái tạo toàn bộ historical trainer, save thresholds hoặc Table 6; không nằm trong 70 run baseline và không trộn thống kê với `paper_method`.

Mode `paper_method`:

- Word embeddings float, giữ vị trí token gốc cho RoBERTa/XLM-R qua explicit `position_ids`.
- Chỉ perturb vị trí attention=1 và token không thuộc `all_special_ids`.
- Noise Gaussian std=1e-5; một ascent step, step_size=1e-3; normalize gradient L-infinity từng token với epsilon số học 1e-6; project radius L-infinity=1e-3.
- `epsilon` trong config là **radius**; `gradient_eps` là ổn định số học. Helper repo gọi epsilon=1e-6 cho mẫu số normalization, không định nghĩa radius như config mới.
- Lambda=0.02 lấy từ repo; final symmetric KL `batchmean` mỗi task, targets detach theo mỗi chiều, rồi cộng hai task. Repo final reduction `sum` khác batchmean theo batch size; giữ 0.02 với normalization mới là giả định công khai.
- Clean/adversarial SMART forwards chạy eval mode để không đo hai dropout samples khác nhau; supervised và MLM forwards vẫn train mode. Adversarial path FP32 ngay cả khi supervised dùng AMP.
- Inner noise gradient phải tồn tại và hữu hạn; final model gradients được test hữu hạn/khác 0 ở encoder/shared/two heads. Batch không có eligible tokens nhận regularizer 0 hữu hạn.

Bằng chứng: `reports/method/gradient_audit.json`, `tests/training/test_objectives.py`. Trên fixture seed 42, inner noise gradient norms sentiment/topic khoảng `1.576e-8` / `1.511e-8`; số này chỉ dùng kiểm tra differentiability. **Không chứng minh SMART là nguyên nhân Figure 3/Table 6 mâu thuẫn.**

## MLM: định nghĩa loss được khóa

Gọi B là số câu trong microbatch, M là số vị trí được chọn thực tế. Định nghĩa raw MLM loss theo sum token loss trung bình mỗi câu: `L_raw = sum(CE_selected) / B`. Chọn `sigma = M/B` theo đoạn văn cuối trang 15. Khi M>0, `L_raw/sigma = sum(CE_selected)/M`. Code dùng CE reduction=sum rồi chia M đúng một lần. Đây cũng bằng mean CE trên các selected tokens, không chia tiếp cho M/B. Định nghĩa raw loss/sigma là giả định có căn cứ từ cách chuẩn hóa theo câu; paper không định nghĩa reduction của L_MLM đủ rõ.

Labels ở mọi vị trí selected giữ **ID gốc**, kể cả nhánh random và unchanged; non-target=-100. Nhánh batch M=0 bỏ lượt encoder MLM, đóng góp loss 0; helper trực tiếp cũng trả 0 có đường gradient. Classification luôn dùng IDs sạch. Two-forward three-head architecture theo file `architecture/bert3head_mlm/model.py`; file single-task-plus-MLM `bert2head_mlm/model.py` có một forward khác nhưng không phải variant Table 6 đang implement.

## Hyperparameters/cohort

22 resolved YAML độc lập: hai ST cho cả năm backbone; bốn MTL cho XLM-R, ViSoBERT, PhoBERT. Seed mặc định 42. Chạy ViSoBERT/PhoBERT seeds 42-46 theo campaign 07, không chọn seed theo test. Cấu hình không chứa matrix chạy lại hay queue.

LR=2e-5; ST 15 epochs, Adam epsilon=1e-8, epoch-linear end_factor=0.2. MTL 20 epochs, Adam epsilon=5e-9, end_factor=0.1. Betas=0.9/0.999, weight_decay=0.01, warmup=0, clipping=None, fp32, length=128. Physical batch=8, accumulation=16, effective=128; eval batch=32. Đây là lựa chọn bộ nhớ Colab, khác dropout/microbatch MLM normalization so với physical batch=128 của script repo. Nhóm accumulation cuối epoch được chuẩn hóa theo số mẫu thực tế; MLM/SMART mean từng microbatch được weighted theo số mẫu. Không nhận tương đương bitwise với một batch 128 cho objectives này.

Cohort hash dùng config bỏ seed/experiment_id/manifest path, manifest digest và implementation digest. Implementation digest gồm các file Python package thực thi, bao gồm data/evaluator; code changes tạo cohort mới. Config hash vẫn bao gồm seed và mọi thông số/path cấu hình. Full run yêu cầu checkout sạch có commit; resume full từ chối đổi commit, config/cohort/manifest/implementation/environment. Thay precision/batch/assumption sau GPU smoke phải tạo config/cohort mới trước lượt xác nhận.
