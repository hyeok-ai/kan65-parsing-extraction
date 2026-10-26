"""bbox 좌표 뒤집힘 회귀 테스트 (ValueError: y1 must be greater than or equal to y0)."""

from __future__ import annotations

from PIL import Image

from extractor.parsing import _norm_bbox
from extractor.schemas import Block
from extractor.visualize import draw_overlay


def test_norm_bbox_orders_coordinates():
    assert _norm_bbox(200, 250, 50, 100) == [50, 100, 200, 250]
    assert _norm_bbox(10, 20, 30, 40) == [10, 20, 30, 40]


def test_overlay_accepts_inverted_bbox():
    img = Image.new("RGB", (300, 300), "white")
    blocks = [Block(block_id="B1", page=1, type="text", text="x", bbox=[200, 250, 50, 100])]
    out = draw_overlay(img, blocks, scale=1.0)
    assert out.size == img.size
