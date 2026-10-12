"""Recompute the paper audit from transcribed CSVs; never fit to expected totals."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from .evaluate import LABELS, METRICS, _require, metrics_from_counts


def audit_reference(reference_dir, pdf_path):
    from pypdf import PdfReader

    reference_dir, pdf_path = Path(reference_dir), Path(pdf_path)
    metadata = json.loads((reference_dir / "table6_metadata.json").read_text(encoding="utf-8"))
    figure_metadata = json.loads((reference_dir / "figure3_metadata.json").read_text(encoding="utf-8"))
    pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    _require(pdf_hash == metadata["source"]["sha256"], "source PDF changed; reverify all cells")
    _require(pdf_hash == figure_metadata["source"]["pdf_sha256"], "Figure 3 source PDF changed")
    reader = PdfReader(pdf_path)
    table_text = reader.pages[17].extract_text()
    extracted = [re.findall(r"\d+\.\d+\*?", line) for line in table_text.splitlines()]
    extracted = [[value.rstrip("*") for value in row] for row in extracted if len(row) == 6]
    with (reference_dir / "table6.csv").open(encoding="utf-8", newline="") as stream:
        table = list(csv.DictReader(stream))
    columns = metadata["metric_column_order"]
    _require(len(table) == len(extracted) == 21, "Table 6 must contain 21 rows")
    _require([[row[c] for c in columns] for row in table] == extracted,
             "Table 6 CSV differs from direct PDF numeric extraction")
    _require(all(row["source_page"] == "18" for row in table), "wrong Table 6 source page")
    selected = next(row for row in table if row["backbone"] == "visobert" and row["variant"] == "mtl_smart")
    tasks = {}
    for task, labels in LABELS.items():
        path = reference_dir / f"figure3_{task}_counts.csv"
        _require(hashlib.sha256(path.read_bytes()).hexdigest() ==
                 figure_metadata["verified_transcription_sha256"][path.name],
                 f"Figure 3 {task} transcription changed; visually reverify cells first")
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        _require(rows[0][1:] == list(labels), f"wrong Figure 3 {task} column labels")
        _require([row[0] for row in rows[1:]] == list(labels), f"wrong Figure 3 {task} row labels")
        matrix = [[int(value) for value in row[1:]] for row in rows[1:]]
        values = metrics_from_counts(matrix, task)
        trace = sum(matrix[i][i] for i in range(len(labels)))
        paper_accuracy = float(selected[f"{task}_accuracy_pct"])
        tasks[task] = dict(
            source_page=19, source_figure="Figure 3", source_csv=path.name,
            csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            shape=[len(matrix), len(matrix[0])], orientation="rows=true, columns=predicted",
            visually_verified_cells=len(labels)**2, unverified_cells=[],
            n_samples=values["n_samples"], trace=trace, accuracy=values["accuracy"],
            accuracy_pct=values["accuracy"]*100,
            row_totals=[sum(row) for row in matrix],
            column_totals=[sum(row[c] for row in matrix) for c in range(len(labels))],
            table6_accuracy_pct=paper_accuracy,
            figure_minus_table6_pp=values["accuracy"]*100-paper_accuracy,
            inconsistent_with_table6=abs(values["accuracy"]*100-paper_accuracy) > 0.005,
        )
    audit = dict(
        schema_version="1.0", source=dict(url=metadata["source"]["url"],
            pdf_sha256=pdf_hash, page_numbering="1-based PDF pages", table6_page=18, figure3_page=19),
        verification=dict(table6_rows=21, table6_numeric_cells=126,
            table6_pdf_text_exact_match=True, table6_visually_verified=True,
            figure3_visually_verified=True, figure3_numeric_cells=116,
            method="Direct PDF text extraction for Table 6 plus visual inspection of freshly rendered pages 18/19; Figure 3 manually transcribed"),
        artifacts_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            reference_dir/"table6.csv", reference_dir/"table6_metadata.json", reference_dir/"figure3_metadata.json",
            reference_dir/"source-page-18.png", reference_dir/"source-page-19.png"]},
        tasks=tasks,
        conclusion="Figure 3 counts are inconsistent with Table 6 for ViSoBERT MTL+SMART. Cause is not established.",
        unverified_hypotheses=["predictions/labels order mismatch", "different checkpoint", "different evaluation data"],
        reproduction_results=False,
    )
    (reference_dir / "figure3_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Audit Table 6 và Figure 3", "",
        "Nguồn: paper NEU-ESC, arXiv:2506.23524, trang PDF 18 (Table 6) và 19 (Figure 3).",
        f"SHA-256 PDF: `{pdf_hash}`.", "",
        "Đã render trực tiếp hai trang từ PDF bằng Poppler và đọc hình. 126 ô số Table 6 đã đối chiếu hình và khớp chính xác text trích xuất từ PDF; 116 ô Figure 3 đã đọc trực tiếp. Không có ô chưa xác minh. CSV dùng nhãn hàng thật và nhãn cột dự đoán theo hình.", "",
        "| Task | N | Trace | Accuracy từ Figure 3 (%) | Table 6 (%) | Figure 3 − Table 6 (pp) |",
        "|---|---:|---:|---:|---:|---:|"]
    for task, v in tasks.items():
        lines.append(f"| {task} | {v['n_samples']} | {v['trace']} | {v['accuracy_pct']:.8f} | {v['table6_accuracy_pct']:.2f} | {v['figure_minus_table6_pp']:.8f} |")
    lines += ["", "Mỗi ma trận chứa 6.613 mẫu. Không cộng hai task để báo N của một ma trận. Accuracy tính bằng trace/N trước khi làm tròn.", "",
        "**Kết luận:** số đếm Figure 3 bất nhất với accuracy Table 6 của ViSoBERT 2-task & SMART. Chưa chứng minh nguyên nhân. Sai thứ tự predictions/labels, nhầm checkpoint và khác dữ liệu chỉ là giả thuyết; không dùng các giả thuyết để sửa số paper.", "",
        "Các dấu in đậm và dấu * của Table 6 được lưu riêng trong `table6_metadata.json`, không lẫn vào số. Alias `mbert` bám ma trận kế hoạch nhưng Table 6 chỉ ghi BERT_base; alias `claude4` giữ nguyên Claude 4 như bảng. Module này không xác nhận checkpoint hoặc API snapshot.", "",
        "Seed 42 được chọn trước cho CM tái lập so với Figure 3. CM mọi seed phải giữ lại; nếu cộng 5 seed đủ test thì N=33.065 mỗi task. Chưa có predictions benchmark trong module này.", "",
        "Tái tạo phép tính: `python -m neu_esc.evaluation.audit_reference --reference-dir reports/reference --pdf ../references/NEU-ESC-2506.23524.pdf` (cần pypdf)."]
    (reference_dir / "figure3_audit.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--pdf", required=True, type=Path)
    args = parser.parse_args()
    result = audit_reference(args.reference_dir, args.pdf)
    print(json.dumps({t: {k:v[k] for k in ("n_samples", "trace", "accuracy_pct")}
                      for t,v in result["tasks"].items()}))


if __name__ == "__main__":
    main()
