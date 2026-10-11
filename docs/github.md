# GitHub và snapshot

Đã kiểm tra ngày **11/10/2026, Asia/Saigon** bằng read-only `gh auth status`, `gh api user`, `gh api orgs/AI-Peak`, `gh api user/memberships/orgs/AI-Peak` và `gh repo view`.

| Mục | Trạng thái thực tế |
|---|---|
| Account | `MarcusCyborg`, active account, HTTPS Git |
| Organization | [AI-Peak — AI Peak](https://github.com/AI-Peak), org có sẵn do người dùng chỉ định |
| Account membership | `state=active`, `role=admin` do GitHub API trả; người dùng xác nhận là owner |
| Repository | [AI-Peak/neu-esc-reproduction](https://github.com/AI-Peak/neu-esc-reproduction), PUBLIC do người dùng chọn |
| Local origin | `https://github.com/AI-Peak/neu-esc-reproduction.git` |
| Invitations | Không gửi invitation; người dùng sẽ tự thêm thành viên sau |
| Plan/billing | Không mua hoặc thay đổi |

`gh repo create AI-Peak/neu-esc-reproduction --public --description ...` đã thực thi và URL/visibility kiểm tra lại. Không tạo org mới; không suy tên thành viên từ proposal; không cấp owner hoặc quyền team khác. Không lưu token vào repo/docs/logs. Token scopes đã kiểm tra nhưng quyền cụ thể dùng membership/API, không suy quyền chỉ từ auth.

## Snapshot theo allowlist READY

Module 04 sở hữu Git index/commit/remote. Chủ module giao handoff với status READY và danh sách **file cụ thể**, SHA-256/config và checks thật. Status PARTIAL không cấp phép stage toàn namespace; chỉ output có READY rõ ràng mới đủ điều kiện bàn giao. Module 04 kiểm tra file READY chưa đổi từ hash đã bàn giao; thay đổi bất ngờ phải dừng file đó. Không stage dataset/checkpoints/secrets dù file tồn tại.

1. Đọc handoffs, kiểm tra `git status --short`, `git diff`, `git diff --cached`, ignore rules và hashes. Index có thay đổi không thuộc snapshot phải dừng và phối hợp, không tự reset/revert chúng.
2. Viết allowlist vào `scripts/environment/release-allowlist.txt` khi mọi chủ module đã giao READY. Mỗi dòng là repo-relative filename đã review; không dùng glob, directory hay `.`. File này do 04 tạo cho lần release cụ thể, chưa có release toàn pipeline trong bàn giao nền tảng.
3. Stage tuần tự chỉ filenames và kiểm tra diff **trước commit**:

```powershell
$taskReleasePaths = Get-Content -LiteralPath scripts/environment/release-allowlist.txt
git add -- $taskReleasePaths
git diff --cached --name-status
git diff --cached --check
git diff --cached
# Chỉ tiếp tục khi cached diff chứa đúng file READY đã review và không có data/secrets.
git commit -m 'Release audited NEU-ESC reproduction snapshot'
git rev-parse HEAD
```

4. Kiểm tra commit bằng checkout/worktree riêng và tests/gates 06. Nếu worktree shared có execution files WIP, không gọi nó clean/reproducible. Nếu snapshot thiếu file runtime phụ thuộc, chưa được train. Một commit chỉ chứa nền tảng 04 là infrastructure snapshot, không phải release benchmark.
5. Push commit READY lên origin khi kiểm tra nội dung và phạm vi công khai hoàn tất; 07 clone và checkout full commit SHA, không `main` chuyển động. Nếu file lớn hoặc quyền phân phối chưa xác minh, giữ local/Drive, chỉ version manifest hướng dẫn download.

Các thay đổi module khác vẫn nằm local chưa stage cho đến khi chủ module bàn giao READY. Không tự thêm giấy phép cho source chưa rõ quyền. Nếu cần mời thành viên sau này, người dùng cung cấp GitHub usernames và roles chính xác; xác minh API từng username và theo dõi invited/pending/active riêng.
