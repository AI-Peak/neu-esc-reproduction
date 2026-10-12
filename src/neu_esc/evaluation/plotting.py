"""Portable Pillow rendering of counts and row-normalized matrices."""

from pathlib import Path


def render_confusion_png(matrix, row_names, column_names, path, title, normalized=False):
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
        heading = ImageFont.truetype("DejaVuSans.ttf", 23)
    except OSError:
        # Pillow's bundled font is available on Windows and Linux.
        font = ImageFont.load_default(size=17)
        heading = ImageFont.load_default(size=23)
    cols, rows = len(column_names), len(row_names)
    cell_w, cell_h = 120, 60
    left, top, bottom = 246, 210, 62
    canvas = Image.new("RGB", (left + cols*cell_w + 28, top + rows*cell_h + bottom), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 22), title, fill="#172b4d", font=heading)
    draw.text((24, 61), "Rows: true label | Columns: predicted label", fill="#334155", font=font)
    maximum = 1.0 if normalized else max(max(row) for row in matrix)
    for c, name in enumerate(column_names):
        # Wrap category names into at most three lines; never truncate labels.
        lines, current = [], ""
        for word in name.split():
            trial = (current + " " + word).strip()
            if current and draw.textlength(trial, font=font) > cell_w - 8:
                lines.append(current)
                current = word
            else:
                current = trial
        lines.append(current)
        x = left + c*cell_w + cell_w/2
        for j, line in enumerate(lines):
            draw.text((x, top-18-(len(lines)-j)*24), line, fill="#172b4d", font=font, anchor="mm")
    for r, (name, row) in enumerate(zip(row_names, matrix)):
        y = top + r*cell_h
        draw.text((left-12, y+cell_h/2), name, fill="#172b4d", font=font, anchor="rm")
        for c, value in enumerate(row):
            x = left + c*cell_w
            scale = value/maximum if maximum else 0
            color = tuple(round(a+(b-a)*scale) for a,b in zip((239,246,255),(30,64,175)))
            draw.rectangle((x,y,x+cell_w,y+cell_h), fill=color, outline="#d5deeb", width=1)
            label = f"{value:.3f}" if normalized else str(value)
            draw.text((x+cell_w/2,y+cell_h/2), label,
                      fill="white" if scale > 0.55 else "#172b4d", font=font, anchor="mm")
    draw.text((24, top+rows*cell_h+22),
              "Absent true classes have a zero row." if normalized else "Each sample contributes once per task.",
              fill="#334155", font=font)
    canvas.save(Path(path), format="PNG")
