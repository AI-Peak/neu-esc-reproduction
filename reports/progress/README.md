# Progress evidence index

Trang đọc chính: [Tiến độ NEU-ESC](../../docs/PROGRESS.md).

- [run-status-20261020.csv](run-status-20261020.csv): 27 dòng được giữ từ audit nguồn, gồm benchmark, unknown và diagnostic. Chỉ 6 dòng `PASS_RECOMPUTED`; không coi 27 dòng là 27 benchmark thành công.
- [notebook-catalog-20261021.csv](notebook-catalog-20261021.csv): 21 notebook đại diện (13 release candidates, 4 legacy trong repo local, 4 root). Catalog không chứa source hay dữ liệu đăng nhập; cột public_reference_path dẫn tới 13 bản notebook chỉ đọc khi có.
- [source-provenance.json](source-provenance.json): hashes tài liệu nguồn và giới hạn evidence. Audit nguồn đầy đủ còn local; đây là summary có truy vết, không là bộ tái thực thi độc lập.

Status là mốc audit 20/10/2026, không phải live GPU/Drive telemetry. Bản CSV bỏ cột đường dẫn cá nhân và locator evidence nội bộ của run audit; giữ các kết luận và giới hạn. Original notebook paths trong catalog là workspace-relative để maintainer tra cứu; chúng không phải repo download links.
