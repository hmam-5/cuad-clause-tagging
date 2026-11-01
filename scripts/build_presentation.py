"""Build an evidence-based draft CUAD project presentation."""

import csv
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "CUAD_project_presentation.pptx"

NAVY = RGBColor(17, 34, 64)
BLUE = RGBColor(44, 104, 186)
TEAL = RGBColor(33, 155, 151)
PALE = RGBColor(239, 244, 250)
INK = RGBColor(39, 51, 68)
MUTED = RGBColor(94, 110, 128)
WHITE = RGBColor(255, 255, 255)
ORANGE = RGBColor(218, 133, 54)


def _textbox(slide, x, y, width, height, text, size, color, bold=False):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.03)
    frame.margin_right = Inches(0.03)
    frame.margin_top = Inches(0.02)
    frame.margin_bottom = Inches(0.02)
    for index, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(size)
        paragraph.font.bold = bold
        paragraph.font.color.rgb = color
        paragraph.space_after = Pt(7)
    return shape


def _base_slide(prs, title, section):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = WHITE
    band = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(0.12)
    )
    band.fill.solid()
    band.fill.fore_color.rgb = TEAL
    band.line.fill.background()
    _textbox(slide, 0.65, 0.42, 12, 0.3, section.upper(), 10, TEAL, True)
    _textbox(slide, 0.65, 0.78, 12, 0.62, title, 26, NAVY, True)
    _textbox(slide, 0.65, 7.12, 11.9, 0.2, "CUAD clause tagging  |  Research project draft", 9, MUTED)
    return slide


def _card(slide, x, y, width, label, value, note, color=BLUE):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(x),
        Inches(y),
        Inches(width),
        Inches(1.48),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = PALE
    shape.line.color.rgb = PALE
    _textbox(slide, x + 0.18, y + 0.13, width - 0.36, 0.24, label, 10, MUTED, True)
    _textbox(slide, x + 0.18, y + 0.43, width - 0.36, 0.48, value, 25, color, True)
    _textbox(slide, x + 0.18, y + 1.01, width - 0.36, 0.36, note, 10, INK)


def _bullets(slide, items, x=0.9, y=1.75, width=11.4, height=4.8, size=20):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.05)
    frame.margin_right = Inches(0.05)
    for index, item in enumerate(items):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = item
        paragraph.level = 0
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = INK
        paragraph.space_after = Pt(17)
    return shape


def _load():
    stats = json.loads((ROOT / "data" / "processed" / "stats.json").read_text(encoding="utf-8"))
    with (RESULTS / "metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    return stats, {row["run_id"]: row for row in metrics}


def _fmt(value, digits=3):
    return f"{float(value):.{digits}f}"


def build():
    stats, metrics = _load()
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = NAVY
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.2), prs.slide_height
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = TEAL
    _textbox(slide, 0.95, 1.0, 11.2, 0.35, "NLP PROJECT  |  CUAD V1", 13, TEAL, True)
    _textbox(slide, 0.95, 1.75, 11.4, 1.45, "Tagging legal clauses\nat scale", 36, WHITE, True)
    _textbox(
        slide,
        0.98,
        3.55,
        10.6,
        0.9,
        "An auditable pipeline for 41 contract clause categories",
        21,
        RGBColor(207, 220, 237),
    )
    _textbox(
        slide,
        0.98,
        5.85,
        11.0,
        0.65,
        "Measured benchmarks are complete on one contract split; variance remains under study.",
        14,
        RGBColor(255, 207, 137),
        True,
    )
    _textbox(slide, 0.98, 6.85, 10, 0.25, "CUAD dataset  |  Reproducible research, not legal advice", 10, WHITE)

    slide = _base_slide(prs, "A span-labeling problem, not ordinary NER", "The task")
    _bullets(
        slide,
        [
            "Input: the full text of a commercial contract.",
            "Output: token-level BIO tags for 41 lawyer-defined clause categories (83 labels including O).",
            "A clause can be a short date or a long, multi-sentence provision; some tokens belong to overlapping annotations.",
            "Goal: locate and classify spans reliably, not make legal judgments.",
        ],
        size=19,
    )
    _card(slide, 1.0, 5.65, 3.45, "CONTRACTS", f"{stats['contracts']:,}", "CUAD v1")
    _card(slide, 4.92, 5.65, 3.45, "ANSWER SPANS", f"{stats['raw_spans']:,}", "non-empty annotations", TEAL)
    _card(slide, 8.84, 5.65, 3.45, "WORD TOKENS", f"{stats['token_count'] / 1e6:.2f}M", "after tokenization", ORANGE)

    slide = _base_slide(prs, "Offsets are validated before labels are created", "Data integrity")
    _card(slide, 0.9, 1.75, 3.55, "EXACT OFFSETS", f"{stats['kept_spans']:,}", "answers verified against source text", TEAL)
    _card(slide, 4.9, 1.75, 3.55, "CORRECTED / DROPPED", f"{stats['corrected_spans']} / {stats['dropped_spans']}", "no silent span edits", BLUE)
    _card(slide, 8.9, 1.75, 3.55, "ROUND-TRIP AGREEMENT", f"{stats['roundtrip_span_agreement']:.0%}", "token spans map back to annotations", ORANGE)
    _bullets(
        slide,
        [
            "Whitespace cleanup retains a character-to-source offset map.",
            "Offset repair is allowed only for a unique exact nearby match; every repair or drop is auditable.",
            "All 510 contracts are split before windowing, preventing a contract's windows from crossing splits.",
        ],
        y=3.72,
        height=2.7,
        size=17,
    )

    slide = _base_slide(prs, "Two data properties shape the modeling", "Challenges")
    _card(slide, 0.9, 1.75, 3.55, "OUTSIDE TOKENS", f"{stats['o_token_share']:.1%}", "severe class imbalance", ORANGE)
    _card(slide, 4.9, 1.75, 3.55, "MULTI-CATEGORY", f"{stats['multi_category_token_share']:.2%}", "tokens carry >1 category", TEAL)
    _card(slide, 8.9, 1.75, 3.55, "OVERLAP PAIRS", f"{stats['overlapping_annotation_pairs']:,}", "cross-category answer intersections", BLUE)
    _bullets(
        slide,
        [
            "The multi-label representation retains all overlapping category assignments.",
            "The single-label BIO baseline resolves collisions using a documented rare-category priority.",
            "Three categories have fewer than five test spans; those per-category scores will be uncertain.",
        ],
        y=3.75,
        height=2.65,
        size=17,
    )

    slide = _base_slide(prs, "A reproducible, contract-level pipeline", "Method")
    steps = [
        ("01", "Official CUAD JSON", "Original contexts + answer offsets"),
        ("02", "Normalize & validate", "Preserve offsets; report changes"),
        ("03", "Tokenize & label", "BIO + sparse multi-label targets"),
        ("04", "Split by contract", "408 train / 51 validation / 51 test"),
        ("05", "Window & evaluate", "512 words; stride 128; stitch predictions"),
    ]
    for index, (number, heading, detail) in enumerate(steps):
        y = 1.62 + index * 0.93
        circle = slide.shapes.add_shape(
            MSO_SHAPE.OVAL, Inches(0.95), Inches(y), Inches(0.52), Inches(0.52)
        )
        circle.fill.solid()
        circle.fill.fore_color.rgb = TEAL if index != 4 else BLUE
        circle.line.fill.background()
        frame = circle.text_frame
        frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        frame.paragraphs[0].text = number
        frame.paragraphs[0].font.size = Pt(12)
        frame.paragraphs[0].font.bold = True
        frame.paragraphs[0].font.color.rgb = WHITE
        _textbox(slide, 1.75, y - 0.01, 3.0, 0.35, heading, 17, NAVY, True)
        _textbox(slide, 4.55, y, 7.4, 0.42, detail, 15, INK)

    slide = _base_slide(prs, "Measured model results on the fixed test split", "Benchmark")
    run_rows = [
        ("O-only baseline", "o-only-seed42"),
        ("Most-frequent word", "most-frequent-word-seed42"),
        ("Feature CRF", "crf-full-seed42-test"),
        ("BiLSTM-CRF (100d random embeddings)", "bilstm-crf-seed42"),
        ("BERT cased", "bert-base-cased-full-seed42"),
        ("BERT cased", "bert-base-cased-full-seed43"),
        ("BERT uncased", "bert-base-uncased-full-seed42"),
        ("LegalBERT seed 42", "nlpaueb_legal-bert-base-uncased-full-seed42"),
        ("LegalBERT seed 43", "nlpaueb_legal-bert-base-uncased-full-seed43"),
        ("RoBERTa", "roberta-base-full-seed42"),
        ("DeBERTa", "microsoft_deberta-v3-base-full-seed42"),
    ]
    rows = [
        ("Model", "Seed", "Token macro F1", "Strict span F1", "Partial span F1"),
    ]
    for label, run_id in run_rows:
        result = metrics[run_id]
        rows.append(
            (
                label,
                result["seed"],
                _fmt(result["token_macro_f1"]),
                _fmt(result["span_strict_micro_f1"]),
                _fmt(result["span_partial_micro_f1"]),
            )
        )
    table = slide.shapes.add_table(
        len(rows), 5, Inches(0.55), Inches(1.48), Inches(12.25), Inches(4.75)
    ).table
    widths = [4.25, 0.75, 2.3, 2.3, 2.65]
    for column, width in enumerate(widths):
        table.columns[column].width = Inches(width)
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            cell = table.cell(row_index, column_index)
            cell.text = value
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if row_index == 0 else (PALE if row_index % 2 else WHITE)
            cell.margin_left = Inches(0.1)
            cell.margin_right = Inches(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            paragraph = cell.text_frame.paragraphs[0]
            paragraph.font.name = "Aptos"
            paragraph.font.size = Pt(10 if row_index == 0 else 10.5)
            paragraph.font.bold = row_index == 0
            paragraph.font.color.rgb = WHITE if row_index == 0 else INK
    _bullets(
        slide,
        [
            "DeBERTa seed 42 has the highest individual test token macro F1 (0.5402); LegalBERT seed 43 is a separate seed-specific result.",
            "BiLSTM-CRF partial span F1 (0.4236) exceeds the individual transformer runs, despite lower token macro F1.",
            "Bootstrap intervals measure test-contract resampling, not seed or split variance.",
        ],
        y=6.28,
        height=0.75,
        size=11,
    )

    slide = _base_slide(
        prs, "A second split provides an initial sensitivity check", "Split variance"
    )
    _card(
        slide,
        0.9,
        1.8,
        3.55,
        "TEST TOKEN MACRO F1",
        "0.5383",
        "LegalBERT seed 42",
        BLUE,
    )
    _card(
        slide,
        4.9,
        1.8,
        3.55,
        "STRICT SPAN MICRO F1",
        "0.2257",
        "52 alternate-split test contracts",
        TEAL,
    )
    _card(
        slide,
        8.9,
        1.8,
        3.55,
        "PARTIAL SPAN MICRO F1",
        "0.4281",
        "95% token-macro CI: 0.4749–0.5612",
        ORANGE,
    )
    _bullets(
        slide,
        [
            "The alternate contract split contains 407 train, 51 validation, and 52 test contracts; 153 of 510 assignments differ from the original split.",
            "This is a separate split-specific result, not a paired comparison: both training and test contracts differ.",
            "One model on one alternate split is an initial sensitivity check, not a general estimate of split variance.",
        ],
        y=3.85,
        height=2.5,
        size=17,
    )

    slide = _base_slide(prs, "Evaluation rewards the right kind of correctness", "Evaluation")
    _card(slide, 0.9, 1.85, 3.55, "TOKEN MACRO F1", "41 classes", "outside class excluded", BLUE)
    _card(slide, 4.9, 1.85, 3.55, "STRICT SPAN F1", "Exact match", "category + both boundaries", TEAL)
    _card(slide, 8.9, 1.85, 3.55, "PARTIAL SPAN F1", "Overlap match", "same category; one-to-one pairing", ORANGE)
    _bullets(
        slide,
        [
            "Contract-level bootstrap confidence intervals capture variation across documents rather than tokens.",
            "Paired bootstrap compares systems on the same held-out contracts.",
            "Use validation data for checkpoint selection; report repeated seeds and test-set uncertainty.",
        ],
        y=4.0,
        height=2.45,
        size=18,
    )

    slide = _base_slide(prs, "Per-category scores reveal uneven coverage", "Error analysis")
    category_path = RESULTS / "per_category_f1.csv"
    if category_path.is_file():
        with category_path.open(encoding="utf-8-sig", newline="") as source:
            category_rows = list(csv.DictReader(source))
        category_rows.sort(key=lambda row: float(row["token_f1"]))
        lowest = category_rows[:5]
        highest = list(reversed(category_rows[-5:]))
        for x, heading, rows in (
            (0.9, "Lowest observed token F1", lowest),
            (6.85, "Highest observed token F1", highest),
        ):
            _textbox(slide, x, 1.75, 5.4, 0.38, heading, 17, NAVY, True)
            text = "\n".join(
                f"{row['category']}  |  F1 {float(row['token_f1']):.3f}  |  "
                f"{int(float(row['test_spans']))} test spans"
                for row in rows
            )
            _textbox(slide, x, 2.3, 5.55, 2.25, text, 14, INK)
    else:
        _bullets(
            slide,
            ["Per-category test predictions are not yet available."],
            y=2.0,
            size=18,
        )
    _textbox(
        slide,
        0.95,
        5.2,
        11.4,
        0.9,
        "These are token-level scores from the best validation-selected full-data run currently recorded. "
        "A single seed is exploratory; categories with few test spans have unstable estimates.",
        15,
        MUTED,
    )
    _textbox(
        slide,
        0.95,
        6.15,
        11.4,
        0.5,
        "See results/per_category_f1.csv for all 41 categories and explanations.",
        12,
        BLUE,
    )

    slide = _base_slide(prs, "Measured results, with variance still to assess", "Status & limitations")
    _bullets(
        slide,
        [
            "Completed: source acquisition, offset-safe preprocessing, split/window generation, analysis, baseline and neural benchmarks, scoring, and audits.",
            "Current coverage: most model runs use the original split; one LegalBERT seed-42 run uses an alternate split.",
            "LegalBERT and cased BERT have two seeds; other completed models have one.",
            "Known limitations: CRF did not converge by 100 iterations; three classes have fewer than five gold test spans; duplicate screening is approximate.",
            "Next: additional model seeds and at least one independently prepared contract split; report individual results, not pooled estimates.",
        ],
        y=1.75,
        height=3.9,
        size=17,
    )
    banner = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.95), Inches(5.85), Inches(11.3), Inches(0.65)
    )
    banner.fill.solid()
    banner.fill.fore_color.rgb = NAVY
    banner.line.fill.background()
    _textbox(slide, 1.2, 5.98, 10.8, 0.3, "Highest individual test token macro F1: DeBERTa seed 42 = 0.5402.", 15, WHITE, True)

    slide = _base_slide(prs, "Takeaway: trustworthy evaluation starts with trustworthy labels", "Conclusion")
    _bullets(
        slide,
        [
            "CUAD offers 510 contracts, 41 clause categories, and 13,823 answer spans.",
            "Every original answer offset matched; contract-level splits and offset-preserving transformations support reproducibility.",
            "The dataset is highly imbalanced and overlapping, so exact spans, partial spans, and uncertainty all matter.",
            "DeBERTa leads individual token macro F1; the ensemble leads strict and partial span F1 in this single-split benchmark.",
        ],
        y=1.65,
        height=3.5,
        size=18,
    )
    _textbox(
        slide,
        0.95,
        5.62,
        11.2,
        0.8,
        "Hendrycks et al. (2021), CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review, NeurIPS.\nDataset: The Atticus Project CUAD v1 (CC BY 4.0).",
        12,
        MUTED,
    )
    _textbox(slide, 0.95, 6.52, 11.2, 0.3,     "Single-seed and single-split coverage limits generalization claims; this is not legal advice.", 12, ORANGE, True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUTPUT)
    print(f"Saved presentation to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    build()
