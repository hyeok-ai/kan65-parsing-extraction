"""파싱 결과 육안 확인용: PDF 페이지 이미지 위에 블록 bbox를 겹쳐 그린다."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .schemas import Block

TYPE_COLORS = {
    "heading": (142, 68, 173),
    "text": (41, 128, 185),
    "list_item": (22, 160, 133),
    "table": (230, 126, 34),
    "caption": (127, 140, 141),
    "other": (127, 140, 141),
}
HIGHLIGHT = (231, 76, 60)

_FONT_CANDIDATES = [r"C:\Windows\Fonts\malgunbd.ttf", r"C:\Windows\Fonts\malgun.ttf",
                    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
                    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"]


def _font(size: int):
    for f in _FONT_CANDIDATES:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def render_pages(path: str | Path, scale: float = 1.5) -> list[Image.Image]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        return [pdf[i].render(scale=scale).to_pil().convert("RGB") for i in range(len(pdf))]
    finally:
        pdf.close()


def draw_overlay(
    page_img: Image.Image,
    blocks: list[Block],
    scale: float,
    highlight: set[str] | None = None,
    dim_others: bool = False,
) -> Image.Image:
    highlight = highlight or set()
    img = page_img.copy()
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = _font(max(11, int(9 * scale)))
    for b in blocks:
        if not b.bbox:
            continue
        hl = b.block_id in highlight
        if dim_others and highlight and not hl:
            continue
        color = HIGHLIGHT if hl else TYPE_COLORS.get(b.type, TYPE_COLORS["other"])
        x0, y0, x1, y1 = (v * scale for v in b.bbox)
        l, t, r, btm = min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
        d.rectangle([l, t, r, btm], outline=color + (255,), fill=color + (60 if hl else 28,),
                    width=4 if hl else 2)
        # 라벨은 오른쪽 위 바깥에 붙여 원문 글자를 가리지 않게 한다.
        tag = f"{b.block_id} {b.type}"
        tw = d.textlength(tag, font=font)
        tx = max(0, r - tw - 8)
        ty = max(0, t - font.size - 4)
        d.rectangle([tx, ty, tx + tw + 8, ty + font.size + 4], fill=color + (210,))
        d.text((tx + 4, ty + 1), tag, fill=(255, 255, 255, 255), font=font)
    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
