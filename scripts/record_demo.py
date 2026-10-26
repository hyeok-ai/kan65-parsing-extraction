"""README용 데모 GIF 녹화 (docs/images/).

  uv run python scripts/record_demo.py
  uv run python scripts/record_demo.py --exp 20261026-163329_baseline-luna --only playground

Windows 기본 Edge를 headless로 쓴다(브라우저 추가 설치 없음). 뷰어가 8517에서 안 떠 있으면 직접 띄운다.
뷰어 장면에서 02·03을 한 번 실행하므로 LLM 비용이 약 $0.001 든다.
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
BASE = "http://localhost:8517"
VIEWPORT = {"width": 1280, "height": 800}
GIF_WIDTH = 960
FONT = r"C:\Windows\Fonts\malgunbd.ttf"


class Recorder:
    def __init__(self, page: Page):
        self.page = page
        self.frames: list[tuple[Image.Image, int]] = []

    def wait_idle(self, extra_ms: int = 800, timeout_ms: int = 240_000):
        """Streamlit의 '실행 중' 표시가 사라질 때까지 기다린다."""
        self.page.wait_for_timeout(400)
        self.page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')",
                                    timeout=timeout_ms)
        self.page.wait_for_timeout(extra_ms)

    def hide_chrome(self):
        self.page.add_style_tag(content='[data-testid="stToolbar"], [data-testid="stDecoration"] {display: none !important}')

    def scroll_to(self, locator, offset: int = -20):
        """대상을 화면 맨 위에 맞춘 뒤 offset만큼 더 내린다(음수면 위쪽 여백)."""
        locator.first.evaluate("e => e.scrollIntoView({block: 'start'})")
        self.page.evaluate("(dy) => document.querySelector('[data-testid=\"stMain\"]')?.scrollBy(0, dy)", offset)
        self.page.wait_for_timeout(700)

    def snap(self, caption: str, hold: float = 2.0):
        im = Image.open(io.BytesIO(self.page.screenshot())).convert("RGB")
        im = im.resize((GIF_WIDTH, round(im.height * GIF_WIDTH / im.width)), Image.LANCZOS)
        d = ImageDraw.Draw(im, "RGBA")
        font = ImageFont.truetype(FONT, 20) if Path(FONT).exists() else ImageFont.load_default()
        bar = 44
        d.rectangle([0, im.height - bar, im.width, im.height], fill=(20, 24, 32, 225))
        d.text((16, im.height - bar + 10), caption, fill=(255, 255, 255, 255), font=font)
        self.frames.append((im, int(hold * 1000)))
        print(f"  frame {len(self.frames)}: {caption}")

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # 디더링 없이 양자화해야 글자가 깨끗하다. MEDIANCUT은 범례처럼 작은 색 조각을 다른 색으로 뭉개서
        # 차트 색을 보존하는 FASTOCTREE를 쓴다.
        pal = [im.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
               for im, _ in self.frames]
        pal[0].save(path, save_all=True, append_images=pal[1:], duration=[d for _, d in self.frames],
                    loop=0, optimize=True, disposal=1)
        print(f"저장: {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.1f} MB, {len(pal)} frames)")


def visible(page: Page, selector: str):
    return page.locator(f"{selector} >> visible=true")


def record_viewer(page: Page) -> Recorder:
    r = Recorder(page)
    page.goto(BASE + "/", wait_until="networkidle")
    page.get_by_text("선택된 전략").wait_for(timeout=240_000)
    r.hide_chrome()
    r.wait_idle()
    r.snap("① 문서 파싱 — Docling 결과를 공통 JSON으로, 품질 검사 결과와 함께", 2.4)

    r.scroll_to(visible(page, '[data-testid="stImage"]'), -60)
    r.snap("블록 위치(bbox)를 원본 페이지 위에 겹쳐 그려 육안으로 확인", 2.4)

    # 드롭다운이 가끔 첫 클릭에 안 열려서 두 번까지 시도하고, 그래도 안 되면 이 장면은 건너뛴다.
    ms = page.locator('[data-testid="stMultiSelect"]').filter(has_text="강조할 블록")
    option = page.get_by_role("option", name="B6", exact=True)
    for _ in range(2):
        ms.locator("input").click()
        try:
            option.wait_for(timeout=5000)
            option.click()
            break
        except Exception:
            page.keyboard.press("Escape")
    page.keyboard.press("Escape")
    r.wait_idle()
    if "B6" in ms.inner_text():
        r.scroll_to(visible(page, '[data-testid="stImage"]'), -60)
        r.snap("블록을 고르면 페이지와 목록에서 함께 강조 (표 B6)", 2.4)

    page.get_by_role("tab", name="② 파서 비교").click()
    page.get_by_role("button", name="두 전략으로 파싱해 비교").click()
    r.wait_idle()
    r.snap("② 파서 비교 — Docling vs pdfplumber 시간·블록·표·포함률", 2.2)
    r.scroll_to(visible(page, '[data-testid="stImage"]'), -80)
    r.snap("같은 페이지를 두 파서가 어떻게 나눴는지 나란히 비교", 2.2)

    page.get_by_role("tab", name="③ 클래스·속성 추출").click()
    page.get_by_role("button", name="02 클래스 매핑 + 03 속성 추출 실행").click()
    page.get_by_text("다음 노드").wait_for(timeout=240_000)
    r.wait_idle(1500)
    r.snap("③ gpt-6-luna로 클래스 매핑(02) → 속성 추출(03), LangGraph 실행", 2.4)
    r.scroll_to(visible(page, "h3").filter(has_text="03 속성 추출"))
    r.snap("속성마다 원문 값·근거 블록·코드 검증 결과", 2.6)
    r.scroll_to(visible(page, "h3").filter(has_text="근거 위치"), -10)
    r.snap("근거로 쓴 블록을 원본 페이지에서 빨간색으로 확인", 2.6)
    return r


def record_playground(page: Page, exp: str | None) -> Recorder:
    r = Recorder(page)
    page.goto(BASE + "/lab" + (f"?exp={exp}" if exp else ""), wait_until="networkidle")
    page.get_by_text("실험 플레이그라운드").first.wait_for(timeout=120_000)
    r.hide_chrome()
    r.wait_idle()
    r.snap("실험 플레이그라운드 — 문서·방법·모델·effort 조합을 골라 실행", 2.4)

    page.get_by_role("tab", name="② 결과 비교").click()
    r.wait_idle()
    r.scroll_to(visible(page, "h3").filter(has_text="방법별 요약"))
    r.snap("② 방법 7가지 비교 — 정확도·자동통과 오류·토큰·비용", 2.6)
    r.scroll_to(visible(page, "h3").filter(has_text="정확도 대비 비용"))
    r.snap("정확도 대비 비용 — 코드 우선 + 실패 시 에이전트(C·D)", 2.6)
    r.scroll_to(visible(page, "h3").filter(has_text="문서 × 방법"))
    r.snap("문서별로 어떤 방법이 통하는지 (이미지로 붙은 표는 에이전트만 복구)", 2.6)

    page.get_by_role("tab", name="③ 속도").click()
    r.wait_idle()
    r.scroll_to(visible(page, "h3").filter(has_text="단계별 평균 시간"))
    r.snap("③ 속도 — 파싱·LLM·도구·MCP 기동 시간으로 분해, p50·p95", 2.6)

    page.get_by_role("tab", name="④ 속성·일관성").click()
    r.wait_idle()
    r.scroll_to(visible(page, "h3").filter(has_text="속성별 정답률"))
    r.snap("④ 어떤 속성을 어느 방법이 자주 놓치는지", 2.6)

    page.get_by_role("tab", name="⑥ 정답 관리").click()
    r.wait_idle()
    r.snap("⑥ 정답 관리 — 문서를 추가하고 실행 결과로 미리 채워 평가셋을 늘린다", 2.6)
    return r


def ensure_server() -> subprocess.Popen | None:
    try:
        urllib.request.urlopen(BASE, timeout=3)
        return None
    except OSError:
        pass
    print("뷰어 시작…")
    proc = subprocess.Popen(["uv", "run", "streamlit", "run", "app.py", "--server.headless", "true",
                             "--server.port", "8517", "--server.address", "localhost"],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            urllib.request.urlopen(BASE, timeout=2)
            return proc
        except OSError:
            time.sleep(1)
    proc.terminate()
    sys.exit("뷰어가 뜨지 않았습니다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=None, help="플레이그라운드에 보여 줄 실험 ID (기본: 가장 최근)")
    ap.add_argument("--only", choices=["viewer", "playground"], default=None)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    proc = ensure_server()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1, locale="ko-KR")
            if a.only in (None, "viewer"):
                print("뷰어 녹화")
                record_viewer(page).save(OUT / "demo_viewer.gif")
            if a.only in (None, "playground"):
                print("플레이그라운드 녹화")
                record_playground(page, a.exp).save(OUT / "demo_playground.gif")
            browser.close()
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
