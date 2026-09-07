"""
Generates BSP_presentation.pptx from BSP_presentation_outline.md and the figures in results/.

The outline .md is the single source of truth: this script parses it (title, on-slide
bullets, image reference(s), speaker notes per slide) rather than hard-coding slide content,
so re-running it after editing the .md regenerates the deck.

Usage: python build_pptx.py   (run from the presentation/ directory, or anywhere -- paths
below are resolved relative to this file.)
"""
import re
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
OUTLINE_MD = HERE / "BSP_presentation_outline.md"
OUTPUT_PPTX = HERE / "BSP_presentation.pptx"

# ---------------------------------------------------------------------------
# Design system -- lifted directly from src/plotting.py's own palette, so the
# deck's chrome and the embedded result figures read as one visual system.
# ---------------------------------------------------------------------------
SLATE = RGBColor(0x2C, 0x3E, 0x50)      # SA color everywhere in the actual figures
TEAL = RGBColor(0x16, 0xA0, 0x85)       # ICA color everywhere in the actual figures
RED = RGBColor(0xE7, 0x4C, 0x3C)        # sparing highlight (matches detected/error markers)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
OFFWHITE = RGBColor(0xF8, 0xF9, 0xFA)   # matches axes.facecolor in plotting.py
LIGHT_GRAY = RGBColor(0xDD, 0xE1, 0xE5)
TEXT_GRAY = RGBColor(0x5A, 0x6B, 0x78)
FONT = "Calibri"

SLIDE_W_IN = 13.333
SLIDE_H_IN = 7.5
MARGIN_IN = 0.55
TITLE_TOP_IN = 0.35
TITLE_H_IN = 0.85
CONTENT_TOP_IN = 1.35
FOOTER_TOP_IN = 7.05
CONTENT_W_IN = SLIDE_W_IN - 2 * MARGIN_IN
CONTENT_BOTTOM_IN = FOOTER_TOP_IN - 0.12
CONTENT_H_IN = CONTENT_BOTTOM_IN - CONTENT_TOP_IN

DECK_SHORT_TITLE = "Robust Fetal ECG Detection -- Sequential Analysis vs. ICA"

_figure_counter = [0]


def next_figure_number():
    _figure_counter[0] += 1
    return _figure_counter[0]


# ---------------------------------------------------------------------------
# Typography helpers: no compound word may ever be split across a line-wrap,
# and wrapped bullet lines must hang-indent under the text, not the glyph.
# Also: a character-width heuristic for estimating wrapped line counts, since
# there is no live text-layout engine available inside the generator.
# ---------------------------------------------------------------------------
AVG_CHAR_WIDTH_FACTOR = 0.52  # rough average glyph width for Calibri, as a fraction of font size
BULLET_INDENT_IN = 0.3


def nbhyphen(text):
    """Replace a hyphen directly between two word characters with a non-breaking
    hyphen (U+2011), so a compound word (heart-rate-dependent, non-invasive, ...)
    can never be split across a PowerPoint line-wrap."""
    return re.sub(r"(?<=\w)-(?=\w)", "‑", text)


def estimate_line_count(text, box_width_in, font_pt, indent_in=0.0):
    plain = re.sub(r"\*\*|\*", "", text)
    avg_char_w_in = font_pt * AVG_CHAR_WIDTH_FACTOR / 72
    usable_w = max(0.5, box_width_in - indent_in)
    chars_per_line = max(8, int(usable_w / avg_char_w_in))
    return max(1, -(-len(plain) // chars_per_line))  # ceil division


def estimate_block_height_in(bullets, box_width_in, font_pt, space_after_pt=12, indent_in=0.0):
    line_h_in = font_pt * 1.22 / 72
    total = 0.0
    for b in bullets:
        n_lines = estimate_line_count(b, box_width_in, font_pt, indent_in)
        total += n_lines * line_h_in + space_after_pt / 72
    return total


def centered_top(block_h_in, top_min, bottom_max, max_fraction=0.4):
    avail = bottom_max - top_min
    offset = max(0.0, (avail - block_h_in) / 2)
    offset = min(offset, avail * max_fraction)
    return top_min + offset


def set_hanging_indent(paragraph, indent_in=BULLET_INDENT_IN):
    pPr = paragraph._p.get_or_add_pPr()
    marL = int(indent_in * 914400)
    pPr.set("marL", str(marL))
    pPr.set("indent", str(-marL))


# ---------------------------------------------------------------------------
# Outline parsing
# ---------------------------------------------------------------------------
def parse_outline(md_path):
    text = md_path.read_text(encoding="utf-8")
    header_re = re.compile(r"^## Slide (\d+) . (.+)$", re.MULTILINE)
    matches = list(header_re.finditer(text))
    slides = []
    for i, m in enumerate(matches):
        num = int(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        bullets_m = re.search(
            r"\*\*On-slide content[^\n]*\*\*:?\s*\n(.*?)\n\s*\n\*\*Image:", block, re.DOTALL
        )
        bullet_lines = []
        if bullets_m:
            for line in bullets_m.group(1).split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    bullet_lines.append(line[2:].strip())
                elif line.startswith("`") or (line and not line.startswith("-")):
                    # non-bulleted content line (e.g. Slide 5's inline diagram text)
                    bullet_lines.append(line)

        image_m = re.search(r"\*\*Image:\*\*\s*(.+?)\n", block)
        image_line = image_m.group(1).strip() if image_m else ""

        notes_m = re.search(
            r"\*\*Speaker notes:\*\*\s*(.+?)(?:\n\s*\n---|\Z)", block, re.DOTALL
        )
        notes = notes_m.group(1).strip() if notes_m else ""
        notes = re.sub(r"\s+", " ", notes)

        image_paths = re.findall(r"`(results/[^`]+)`", image_line)
        crop = None
        if "right half" in image_line.lower():
            # The source figure has a full-width suptitle shared by both panels; a
            # plain left=0.5 crop slices that title in half too. crop_top removes
            # the truncated suptitle band (measured via PIL: the right panel's own
            # content starts at 8.5% of the image height) while leaving the panel's
            # own subtitle untouched.
            crop = {"left": 0.5, "top": 0.06}

        slides.append(
            {
                "num": num,
                "title": title,
                "bullets": bullet_lines,
                "image_line": image_line,
                "image_paths": [REPO_ROOT / p for p in image_paths],
                "crop": crop,
                "notes": notes,
            }
        )
    return slides


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def add_markdown_runs(paragraph, text, size=18, color=SLATE, bold_color=None, font=FONT):
    bold_color = bold_color or color
    tokens = re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text)
    for tok in tokens:
        if not tok:
            continue
        run = paragraph.add_run()
        if tok.startswith("**") and tok.endswith("**"):
            run.text = nbhyphen(tok[2:-2])
            run.font.bold = True
            run.font.color.rgb = bold_color
        elif tok.startswith("*") and tok.endswith("*"):
            run.text = nbhyphen(tok[1:-1])
            run.font.italic = True
            run.font.color.rgb = color
        else:
            run.text = nbhyphen(tok)
            run.font.color.rgb = color
        run.font.size = Pt(size)
        run.font.name = font


def set_notes(slide, text):
    if not text:
        return
    notes_tf = slide.notes_slide.notes_text_frame
    notes_tf.text = text


# ---------------------------------------------------------------------------
# Slide chrome (title bar, footer, page number)
# ---------------------------------------------------------------------------
def add_blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_background(slide):
    rect = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, Inches(SLIDE_W_IN), Inches(SLIDE_H_IN)
    )
    rect.fill.solid()
    rect.fill.fore_color.rgb = WHITE
    rect.line.fill.background()
    rect.shadow.inherit = False
    # send to back
    spTree = slide.shapes._spTree
    spTree.remove(rect._element)
    spTree.insert(2, rect._element)
    return rect


def add_title(slide, title_text, accent=SLATE):
    tb = slide.shapes.add_textbox(
        Inches(MARGIN_IN), Inches(TITLE_TOP_IN), Inches(CONTENT_W_IN), Inches(TITLE_H_IN)
    )
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = nbhyphen(title_text)
    run.font.size = Pt(28)
    run.font.bold = True
    run.font.name = FONT
    run.font.color.rgb = accent

    rule_top = TITLE_TOP_IN + TITLE_H_IN - 0.12
    rule = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(MARGIN_IN),
        Inches(rule_top),
        Inches(1.1),
        Pt(3),
    )
    rule.fill.solid()
    rule.fill.fore_color.rgb = accent
    rule.line.fill.background()
    rule.shadow.inherit = False


def add_footer(slide, page_num):
    tb = slide.shapes.add_textbox(
        Inches(MARGIN_IN), Inches(FOOTER_TOP_IN), Inches(CONTENT_W_IN - 0.6), Inches(0.35)
    )
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = nbhyphen(DECK_SHORT_TITLE)
    run.font.size = Pt(9)
    run.font.name = FONT
    run.font.color.rgb = TEXT_GRAY
    run.font.italic = True

    tb2 = slide.shapes.add_textbox(
        Inches(SLIDE_W_IN - MARGIN_IN - 0.6), Inches(FOOTER_TOP_IN), Inches(0.6), Inches(0.35)
    )
    tf2 = tb2.text_frame
    tf2.paragraphs[0].alignment = PP_ALIGN.RIGHT
    run2 = tf2.paragraphs[0].add_run()
    run2.text = str(page_num)
    run2.font.size = Pt(9)
    run2.font.name = FONT
    run2.font.color.rgb = TEXT_GRAY


# ---------------------------------------------------------------------------
# Image placement (fit-to-box, preserve aspect ratio, optional crop, framed,
# auto-numbered "Figure N." caption)
# ---------------------------------------------------------------------------
def image_native_ratio(path, crop=None):
    with Image.open(path) as im:
        w, h = im.size
    ratio = w / h
    if crop:
        lr = crop.get("left", 0.0) + crop.get("right", 0.0)
        tb = crop.get("top", 0.0) + crop.get("bottom", 0.0)
        ratio *= (1 - lr) / (1 - tb)
    return ratio


def place_image(slide, path, box_left, box_top, box_w, box_h, crop=None, caption=None):
    ratio = image_native_ratio(path, crop)
    box_ratio = box_w / box_h
    if ratio > box_ratio:
        disp_w = box_w
        disp_h = box_w / ratio
    else:
        disp_h = box_h
        disp_w = box_h * ratio
    left = box_left + (box_w - disp_w) / 2
    top = box_top + (box_h - disp_h) / 2

    frame_pad = 0.06
    frame = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(left - frame_pad),
        Inches(top - frame_pad),
        Inches(disp_w + 2 * frame_pad),
        Inches(disp_h + 2 * frame_pad),
    )
    frame.fill.solid()
    frame.fill.fore_color.rgb = WHITE
    frame.line.color.rgb = LIGHT_GRAY
    frame.line.width = Pt(0.75)
    frame.shadow.inherit = False

    pic = slide.shapes.add_picture(
        str(path), Inches(left), Inches(top), width=Inches(disp_w), height=Inches(disp_h)
    )
    if crop:
        for side, frac in crop.items():
            setattr(pic, f"crop_{side}", frac)

    cap_top = top + disp_h + frame_pad + 0.04
    if caption:
        n = next_figure_number()
        cap_box = slide.shapes.add_textbox(
            Inches(box_left), Inches(cap_top), Inches(box_w), Inches(0.3)
        )
        cap_tf = cap_box.text_frame
        cap_p = cap_tf.paragraphs[0]
        cap_p.alignment = PP_ALIGN.CENTER
        r = cap_p.add_run()
        r.text = f"Figure {n}. {caption}"
        r.font.size = Pt(11)
        r.font.italic = True
        r.font.name = FONT
        r.font.color.rgb = TEXT_GRAY
    return left, top, disp_w, disp_h


def add_bullets(slide, left, top, width, height, bullets, size=18, color=SLATE, valign="top"):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP if valign == "top" else MSO_ANCHOR.MIDDLE
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(12)
        p.level = 0
        set_hanging_indent(p, BULLET_INDENT_IN)
        bullet_run = p.add_run()
        bullet_run.text = "▪  "  # small square bullet, teal
        bullet_run.font.size = Pt(size)
        bullet_run.font.name = FONT
        bullet_run.font.color.rgb = TEAL
        add_markdown_runs(p, b, size=size, color=color)
    return tb


# ---------------------------------------------------------------------------
# Slide builders
# ---------------------------------------------------------------------------
def build_title_slide(prs, s):
    slide = add_blank_slide(prs)
    add_background(slide)

    band = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, Inches(0), Inches(0.18), Inches(SLIDE_H_IN)
    )
    band.fill.solid()
    band.fill.fore_color.rgb = SLATE
    band.line.fill.background()
    band.shadow.inherit = False

    bullets = s["bullets"]
    main_title = bullets[0] if len(bullets) > 0 else s["title"]
    subtitle = bullets[1] if len(bullets) > 1 else ""
    author = bullets[2] if len(bullets) > 2 else ""
    course = bullets[3] if len(bullets) > 3 else ""

    title_box_w = 11.3
    title_top = 2.3
    tb = slide.shapes.add_textbox(Inches(1.0), Inches(title_top), Inches(title_box_w), Inches(1.8))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = nbhyphen(main_title)
    r.font.size = Pt(40)
    r.font.bold = True
    r.font.name = FONT
    r.font.color.rgb = SLATE

    # Rule/subtitle position must clear however many lines the title actually
    # wraps to -- a fixed offset here previously struck through a 2-line title.
    title_lines = estimate_line_count(main_title, title_box_w, 40)
    title_line_h = 40 * 1.15 / 72
    rule_top = title_top + title_lines * title_line_h + 0.15
    subtitle_top = rule_top + 0.2

    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.05), Inches(rule_top), Inches(1.6), Pt(4))
    rule.fill.solid()
    rule.fill.fore_color.rgb = TEAL
    rule.line.fill.background()
    rule.shadow.inherit = False

    tb2 = slide.shapes.add_textbox(Inches(1.0), Inches(subtitle_top), Inches(11.3), Inches(0.7))
    tb2.text_frame.word_wrap = True
    p2 = tb2.text_frame.paragraphs[0]
    r2 = p2.add_run()
    r2.text = nbhyphen(subtitle)
    r2.font.size = Pt(19)
    r2.font.name = FONT
    r2.font.color.rgb = TEXT_GRAY
    r2.font.italic = True

    tb3 = slide.shapes.add_textbox(Inches(1.0), Inches(6.15), Inches(11.3), Inches(1.0))
    tf3 = tb3.text_frame
    tf3.word_wrap = True
    p3 = tf3.paragraphs[0]
    r3 = p3.add_run()
    r3.text = nbhyphen(author)
    r3.font.size = Pt(15)
    r3.font.bold = True
    r3.font.name = FONT
    r3.font.color.rgb = SLATE
    p4 = tf3.add_paragraph()
    r4 = p4.add_run()
    r4.text = nbhyphen(course)
    r4.font.size = Pt(13)
    r4.font.name = FONT
    r4.font.color.rgb = TEXT_GRAY
    set_notes(slide, s["notes"])
    return slide


def build_equations_slide(prs, s, page_num):
    slide = add_blank_slide(prs)
    add_background(slide)
    add_title(slide, s["title"])

    n = len(s["bullets"])
    gap = 0.18
    card_h = (CONTENT_H_IN - gap * (n - 1)) / n
    top = CONTENT_TOP_IN
    for b in s["bullets"]:
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(MARGIN_IN), Inches(top), Inches(CONTENT_W_IN), Inches(card_h)
        )
        card.fill.solid()
        card.fill.fore_color.rgb = OFFWHITE
        card.line.color.rgb = LIGHT_GRAY
        card.line.width = Pt(0.75)
        card.shadow.inherit = False
        card.adjustments[0] = 0.12
        tf = card.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = Inches(0.3)
        tf.margin_right = Inches(0.3)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT
        add_markdown_runs(p, b, size=19, color=SLATE)
        top += card_h + gap

    add_footer(slide, page_num)
    set_notes(slide, s["notes"])
    return slide


def build_text_only_slide(prs, s, page_num):
    slide = add_blank_slide(prs)
    add_background(slide)
    add_title(slide, s["title"])
    block_h = estimate_block_height_in(s["bullets"], CONTENT_W_IN, 20, indent_in=BULLET_INDENT_IN)
    top = centered_top(block_h, CONTENT_TOP_IN + 0.3, CONTENT_BOTTOM_IN)
    add_bullets(slide, MARGIN_IN, top, CONTENT_W_IN, CONTENT_BOTTOM_IN - top, s["bullets"], size=20)
    add_footer(slide, page_num)
    set_notes(slide, s["notes"])
    return slide


def caption_from_title(title):
    # strip a leading "Results N: " / "Stage X in practice: " prefix and a trailing
    # " (...)" parenthetical, for a cleaner figure caption
    t = re.sub(r"^(Results [IVX]+:\s*|Stage \d.*?:\s*)", "", title).strip()
    t = re.sub(r"\s*\([^)]*\)\s*$", "", t).strip()
    return t


def build_image_bullets_slide(prs, s, page_num):
    slide = add_blank_slide(prs)
    add_background(slide)
    add_title(slide, s["title"])

    paths = s["image_paths"]
    primary = paths[0] if paths else None
    secondary = paths[1] if len(paths) > 1 else None
    bullets = s["bullets"]

    if primary is None:
        add_bullets(slide, MARGIN_IN, CONTENT_TOP_IN + 0.3, CONTENT_W_IN, CONTENT_H_IN, bullets, size=20)
        add_footer(slide, page_num)
        set_notes(slide, s["notes"])
        return slide

    ratio = image_native_ratio(primary, s["crop"])
    wide = ratio > 1.8

    if wide:
        bullet_h = 1.55
        add_bullets(slide, MARGIN_IN, CONTENT_TOP_IN, CONTENT_W_IN, bullet_h, bullets, size=17)
        img_top = CONTENT_TOP_IN + bullet_h + 0.1
        img_h = CONTENT_BOTTOM_IN - img_top - 0.35
        if secondary:
            # Non-overlapping side-by-side split: the backup image gets its own
            # reserved strip, carved out before the primary is sized, so the two
            # can never collide regardless of aspect ratio.
            gap, sec_w = 0.3, 2.9
            prim_w = CONTENT_W_IN - sec_w - gap
            place_image(
                slide, primary, MARGIN_IN, img_top, prim_w, img_h,
                crop=s["crop"], caption=caption_from_title(s["title"]),
            )
            place_image(slide, secondary, MARGIN_IN + prim_w + gap, img_top, sec_w, img_h)
        else:
            place_image(
                slide, primary, MARGIN_IN, img_top, CONTENT_W_IN, img_h,
                crop=s["crop"], caption=caption_from_title(s["title"]),
            )
    else:
        bullet_w = CONTENT_W_IN * 0.38
        img_w = CONTENT_W_IN - bullet_w - 0.35
        add_bullets(
            slide, MARGIN_IN, CONTENT_TOP_IN, bullet_w, CONTENT_H_IN, bullets, size=17, valign="middle"
        )
        img_left = MARGIN_IN + bullet_w + 0.35
        img_h = CONTENT_H_IN - 0.35
        if secondary:
            gap, sec_w = 0.25, 2.5
            prim_w = img_w - sec_w - gap
            place_image(
                slide, primary, img_left, CONTENT_TOP_IN, prim_w, img_h,
                crop=s["crop"], caption=caption_from_title(s["title"]),
            )
            place_image(slide, secondary, img_left + prim_w + gap, CONTENT_TOP_IN, sec_w, img_h)
        else:
            place_image(
                slide, primary, img_left, CONTENT_TOP_IN, img_w, img_h,
                crop=s["crop"], caption=caption_from_title(s["title"]),
            )

    add_footer(slide, page_num)
    set_notes(slide, s["notes"])
    return slide


# ---- Native diagrams (Slides 4 and 5 -- no source PNG exists for either) ----
def _box(slide, left, top, w, h, text, fill, text_color=WHITE, size=13, bold=True):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(w), Inches(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    shp.shadow.inherit = False
    shp.adjustments[0] = 0.08
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.08)
    tf.margin_right = Inches(0.08)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = nbhyphen(text)
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = FONT
    r.font.color.rgb = text_color
    return shp


def _pill(slide, cx, cy, w, h, text, outline_color):
    left, top = cx - w / 2, cy - h / 2
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(w), Inches(h))
    shp.adjustments[0] = 0.5
    shp.fill.solid()
    shp.fill.fore_color.rgb = WHITE
    shp.line.color.rgb = outline_color
    shp.line.width = Pt(1.5)
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.name = FONT
    r.font.color.rgb = outline_color
    return shp


def _connector_line(slide, x1, y, x2, color=TEXT_GRAY):
    # Plain connecting line, no arrowhead: it's sent behind the pills/boxes as a
    # background connector (flow direction is already obvious from box order),
    # and a decorated tail-end would poke out past the last shape's edge there.
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y), Inches(x2), Inches(y))
    conn.line.color.rgb = color
    conn.line.width = Pt(1.5)
    return conn


def build_diagram_slide_paradigms(prs, s, page_num):
    slide = add_blank_slide(prs)
    add_background(slide)
    add_title(slide, s["title"])

    box_w, box_h = 4.9, 3.0
    block_h = box_h + 0.35 + 0.4  # boxes + gap-to-thesis + thesis line
    top = centered_top(block_h, CONTENT_TOP_IN, CONTENT_BOTTOM_IN)
    left1 = MARGIN_IN
    left2 = SLIDE_W_IN - MARGIN_IN - box_w

    b1 = _box(slide, left1, top, box_w, 0.6, "Blind (ICA / BSS)", TEAL, size=20)
    b2 = _box(slide, left2, top, box_w, 0.6, "Non-blind (Sequential Analysis)", SLATE, size=20)

    def _detail_list(left, lines):
        box = slide.shapes.add_textbox(Inches(left + 0.25), Inches(top + 0.85), Inches(box_w - 0.5), Inches(box_h - 0.9))
        tf = box.text_frame
        tf.word_wrap = True
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(10)
            set_hanging_indent(p, 0.22)
            r = p.add_run()
            r.text = "▪  " + nbhyphen(line)
            r.font.size = Pt(15)
            r.font.name = FONT
            r.font.color.rgb = SLATE
        return box

    _detail_list(left1, [
        "No prior knowledge assumed",
        "Relies purely on statistical independence of sources",
        "As many independent sources as channels required",
    ])
    _detail_list(left2, [
        "Exploits a priori knowledge of each interferer",
        "Uses periodicity, morphology, frequency content",
        "Paper's thesis: more robust, especially at low SNR",
    ])

    for shp in (b1, b2):
        shp2 = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left1 if shp is b1 else left2), Inches(top), Inches(box_w), Inches(box_h)
        )
        shp2.fill.background()
        shp2.line.color.rgb = TEAL if shp is b1 else SLATE
        shp2.line.width = Pt(1.5)
        shp2.shadow.inherit = False
        shp2.adjustments[0] = 0.04

    vs = slide.shapes.add_textbox(Inches(SLIDE_W_IN / 2 - 0.4), Inches(top + box_h / 2 - 0.3), Inches(0.8), Inches(0.6))
    p = vs.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "vs."
    r.font.size = Pt(18)
    r.font.italic = True
    r.font.bold = True
    r.font.name = FONT
    r.font.color.rgb = TEXT_GRAY

    thesis = slide.shapes.add_textbox(Inches(MARGIN_IN), Inches(top + box_h + 0.35), Inches(CONTENT_W_IN), Inches(0.9))
    tf3 = thesis.text_frame
    tf3.word_wrap = True
    p3 = tf3.paragraphs[0]
    p3.alignment = PP_ALIGN.CENTER
    add_markdown_runs(p3, "→ This is what we test on real data in this presentation", size=17, color=RED)

    add_footer(slide, page_num)
    set_notes(slide, s["notes"])
    return slide


def build_diagram_slide_pipeline(prs, s, page_num):
    slide = add_blank_slide(prs)
    add_background(slide)
    add_title(slide, s["title"])

    stages = ["S1", "S2", "S3", "S4", "S5", "S6"]
    processes = [
        ("Baseline Wander\nRemover", None),
        ("PLI\nCanceller", None),
        ("Sampling-Rate\nIncreaser", None),
        ("MECG\nCanceller", "[Maternal QRS Detector]"),
        ("FECG\nExtractor", "[Fetal QRS Detector]"),
    ]

    n_boxes = len(processes)
    block_h = 0.42 + 0.7 + 1.3 + 0.6  # sub-caption + box row + gap-to-note + note line
    row_y = centered_top(block_h, CONTENT_TOP_IN, CONTENT_BOTTOM_IN) + 0.42
    total_w = CONTENT_W_IN
    box_w = 1.85
    pill_w = 0.65
    n_pills = n_boxes + 1
    gap = (total_w - n_boxes * box_w - n_pills * pill_w) / (n_boxes + n_pills - 1)

    x = MARGIN_IN
    centers = []
    for i in range(n_boxes):
        pill_cx = x + pill_w / 2
        centers.append(pill_cx)
        _pill(slide, pill_cx, row_y + 0.35, pill_w, 0.5, stages[i], SLATE)
        x += pill_w + gap
        box_left = x
        label, sub = processes[i]
        if sub:
            sub_box = slide.shapes.add_textbox(Inches(box_left - 0.3), Inches(row_y - 0.42), Inches(box_w + 0.6), Inches(0.35))
            sp = sub_box.text_frame.paragraphs[0]
            sp.alignment = PP_ALIGN.CENTER
            r = sp.add_run()
            r.text = nbhyphen(sub)
            r.font.size = Pt(10)
            r.font.italic = True
            r.font.name = FONT
            r.font.color.rgb = TEXT_GRAY
        _box(slide, box_left, row_y, box_w, 0.7, label, TEAL if i % 2 else SLATE, size=12)
        x += box_w + gap
    last_pill_cx = x + pill_w / 2
    _pill(slide, last_pill_cx, row_y + 0.35, pill_w + 0.35, 0.5, "S6\n+FHR", RED)
    centers.append(last_pill_cx)

    # Connecting line, sent behind the pills/boxes (z-order) so it only shows in
    # the gaps between them instead of visibly crossing through the solid shapes.
    line = _connector_line(slide, MARGIN_IN, row_y + 0.35, x + pill_w + 0.35, color=LIGHT_GRAY)
    spTree = slide.shapes._spTree
    spTree.remove(line._element)
    spTree.insert(3, line._element)

    note = s["bullets"][-1] if s["bullets"] else ""
    note_box = slide.shapes.add_textbox(Inches(MARGIN_IN), Inches(row_y + 1.3), Inches(CONTENT_W_IN), Inches(0.6))
    ntf = note_box.text_frame
    ntf.word_wrap = True
    np_ = ntf.paragraphs[0]
    np_.alignment = PP_ALIGN.CENTER
    add_markdown_runs(np_, note, size=16, color=SLATE)

    add_footer(slide, page_num)
    set_notes(slide, s["notes"])
    return slide


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
NATIVE_DIAGRAM_BUILDERS = {
    4: build_diagram_slide_paradigms,
    5: build_diagram_slide_pipeline,
}
TEXT_ONLY_NUMS = {2, 3, 18, 19, 20}
EQUATIONS_NUM = 12


def main():
    slides_data = parse_outline(OUTLINE_MD)
    assert len(slides_data) == 20, f"Expected 20 slides, parsed {len(slides_data)}"

    prs = Presentation()
    prs.slide_width = Emu(int(SLIDE_W_IN * 914400))
    prs.slide_height = Emu(int(SLIDE_H_IN * 914400))

    for s in slides_data:
        num = s["num"]
        if num == 1:
            build_title_slide(prs, s)
        elif num in NATIVE_DIAGRAM_BUILDERS:
            NATIVE_DIAGRAM_BUILDERS[num](prs, s, num)
        elif num == EQUATIONS_NUM:
            build_equations_slide(prs, s, num)
        elif num in TEXT_ONLY_NUMS or not s["image_paths"]:
            build_text_only_slide(prs, s, num)
        else:
            missing = [p for p in s["image_paths"] if not p.exists()]
            if missing:
                raise FileNotFoundError(f"Slide {num}: missing image(s) {missing}")
            build_image_bullets_slide(prs, s, num)

    prs.save(str(OUTPUT_PPTX))
    print(f"Wrote {OUTPUT_PPTX} ({len(slides_data)} slides)")


if __name__ == "__main__":
    main()
