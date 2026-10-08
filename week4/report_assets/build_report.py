"""Build my first-person Week 4 PDF from measured results and live UI captures.

Run: python report_assets/build_report.py --desktop-copy
Only reportlab and Pillow are required for rebuilding the PDF.
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import shutil

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent
RESULTS = json.loads((ROOT / "evaluation-results.json").read_text(encoding="utf-8"))
OUTPUT = ROOT / "Petr_Vorozhtsov_Week4_Report.pdf"
DESKTOP = Path.home() / "Desktop" / "Petr_Vorozhtsov_Week4_Final_Report.pdf"
W, H = A4
MARGIN = 46
WIDTH = W - 2 * MARGIN
NAVY = colors.HexColor("#163b5c")
BLUE = colors.HexColor("#1769aa")
INK = colors.HexColor("#243b53")
MUTED = colors.HexColor("#526d85")
PALE = colors.HexColor("#edf4fa")
LINE = colors.HexColor("#d9e2ec")
GREEN = colors.HexColor("#23775d")
RED = colors.HexColor("#9f4949")


base = getSampleStyleSheet()
styles = {
    "title": ParagraphStyle("MyTitle", parent=base["Title"], fontName="Helvetica-Bold",
        fontSize=19, leading=23, textColor=NAVY, spaceAfter=7, alignment=0),
    "h1": ParagraphStyle("MyH1", parent=base["Heading1"], fontName="Helvetica-Bold",
        fontSize=13, leading=17, textColor=NAVY, spaceBefore=2, spaceAfter=8),
    "h2": ParagraphStyle("MyH2", parent=base["Heading2"], fontName="Helvetica-Bold",
        fontSize=10.5, leading=14, textColor=NAVY, spaceBefore=9, spaceAfter=5),
    "body": ParagraphStyle("MyBody", parent=base["BodyText"], fontName="Helvetica",
        fontSize=9.1, leading=13.1, textColor=INK, spaceAfter=7),
    "small": ParagraphStyle("MySmall", parent=base["BodyText"], fontName="Helvetica",
        fontSize=8.1, leading=11.4, textColor=INK, spaceAfter=5),
    "caption": ParagraphStyle("MyCaption", parent=base["BodyText"], fontName="Helvetica",
        fontSize=7.8, leading=10.5, textColor=MUTED, spaceAfter=8),
    "cell": ParagraphStyle("MyCell", parent=base["BodyText"], fontName="Helvetica",
        fontSize=8, leading=10.6, textColor=INK),
    "cellhead": ParagraphStyle("MyCellHead", parent=base["BodyText"], fontName="Helvetica-Bold",
        fontSize=8, leading=10.6, textColor=NAVY),
    "mono": ParagraphStyle("MyMono", parent=base["BodyText"], fontName="Courier",
        fontSize=8.2, leading=12, textColor=INK, leftIndent=9, spaceAfter=4),
}


def p(text: str, kind: str = "body") -> Paragraph:
    return Paragraph(text, styles[kind])


def table(rows: list[list[str]], widths: list[float], *, first_col_bold: bool = False) -> Table:
    formatted = []
    for row_index, row in enumerate(rows):
        formatted.append([
            p(str(value), "cellhead" if row_index == 0 or (first_col_bold and col == 0) else "cell")
            for col, value in enumerate(row)
        ])
    result = Table(formatted, colWidths=widths, hAlign="LEFT", repeatRows=1)
    result.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), PALE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fbfd")]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.35, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return result


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def count(value: int) -> str:
    return f"{value:,}"


def screenshot(name: str, caption: str, width: float = WIDTH) -> list:
    image_path = ASSETS / name
    with PILImage.open(image_path) as source:
        px_width, px_height = source.size
    figure = Image(str(image_path), width=width, height=width * px_height / px_width)
    figure.hAlign = "CENTER"
    return [figure, Spacer(1, 3), p(caption, "caption")]


def metric_table(measure: dict) -> Table:
    return table([
        ["N (invoices)", "nA", "nB", "nAB", "Support", "Confidence", "Lift"],
        [count(measure["N"]), count(measure["nA"]), count(measure["nB"]),
         count(measure["nAB"]), pct(measure["support"]),
         pct(measure["confidence"]), f'{measure["lift"]:.2f}x'],
    ], [73, 56, 56, 56, 79, 91, 92])


def frame(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(MARGIN, 42, W - MARGIN, 42)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.7)
    canvas.drawString(MARGIN, 29, "Petr Vorozhtsov | My Week 4 association-rules report")
    canvas.drawRightString(W - MARGIN, 29, f"Page {doc.page}")
    canvas.restoreState()


def make_story() -> list:
    source = RESULTS["source"]
    data = RESULTS["data"]
    good = RESULTS["examples"]["promising"]
    bad = RESULTS["examples"]["rejected"]
    full_good = good["full"]
    full_bad = bad["full"]
    reverse = RESULTS["reversePromising"]
    stats = RESULTS["defaultRun"]
    story = []

    # Page 1 - my implementation, data, and threshold choices.
    story += [p("My Week 4 Association Rules Report", "title"),
        p("Petr Vorozhtsov | Recommender Systems | 8 October 2026", "small"),
        p("I rebuilt the Week 4 example around the lecture's Online Retail assignment. "
          "My project now converts invoices into baskets, mines frequent itemsets with Apriori, "
          "and lets me inspect directional rules with their underlying counts. "
          "The interactive page and this report are in "
          "<link href='https://github.com/PetrVorozhtsov/RecSys-LLMs/tree/main/week4' color='#1769aa'>"
          "my Week 4 project in the fork</link> (implementation commit <b>7bb475b</b>)."),
        p("Project link: <link href='https://github.com/PetrVorozhtsov/RecSys-LLMs/tree/main/week4' "
          "color='#1769aa'>https://github.com/PetrVorozhtsov/RecSys-LLMs/tree/main/week4</link>", "small"),
        p("What I used", "h2"),
        table([
            ["Dataset", "My preparation"],
            ["UCI Online Retail; 541,909 original rows; December 2010 to December 2011",
             f"{count(data['kept_rows'])} retained rows; {count(data['basket_count'])} invoices; "
             f"{count(data['item_count'])} distinct StockCodes"],
            ["InvoiceNo = transaction; StockCode = item; Description = label",
             "I collapsed repeated StockCodes within each invoice to binary presence."],
        ], [205, WIDTH - 205]),
        Spacer(1, 7),
        p("I excluded C-prefixed cancellations, nonpositive quantity or price, and rows missing "
          "a product code, description, or date. The recorded exclusions are "
          f"{count(data['cancelled_or_missing_invoice'])} cancellation/missing-invoice rows, "
          f"{count(data['missing_product_or_name'])} missing-product/name rows, "
          f"{count(data['nonpositive_quantity'])} nonpositive-quantity rows, and "
          f"{count(data['nonpositive_price'])} nonpositive-price rows. "
          "I kept one-product invoices in the denominator."),
        p("How I mined and filtered rules", "h2"),
        p("I generated Apriori candidates from frequent subsets and counted basket membership exactly. "
          "At my default 1% support threshold, a rule needs at least 200 joint invoices. "
          "I chose 30% minimum confidence to require B in a substantial share of A baskets, "
          "then kept rules with lift above 1 for further analysis. These are practical starting "
          "thresholds, not universal constants."),
        p("support = nAB / N; confidence(A -&gt; B) = nAB / nA; "
          "lift = confidence / (nB / N)", "mono"),
        p(f"My default run found <b>{count(stats['frequentItemsets'])} frequent itemsets</b> "
          f"and <b>{count(stats['retainedRules'])} retained rules</b>. "
          "The rejected-candidates view shows rules before the confidence/lift filter; "
          "that lets me explain why a plausible-looking association should not be used."),
        p("Key code changes", "h2"),
        p("I added a reproducible UCI workbook converter, a browser/Node Apriori module, "
          "six unit tests, a measured threshold and time-split evaluation script, and an "
          "interactive interface with rule search, counts, formulas, and five live self-checks. "
          "I removed the unrelated MovieLens Two-Tower files from this Week 4 folder."),
        p("Source: Chen, D. (2015), <i>Online Retail</i>, UCI Machine Learning Repository, "
          "<link href='https://doi.org/10.24432/C5BW33' color='#1769aa'>"
          "doi:10.24432/C5BW33</link>; CC BY 4.0. "
          f"Original workbook SHA-256: {escape(source['workbook_sha256'])}.", "small"),
        p("To reproduce my results, I serve the week4 folder over HTTP, then run "
          "<b>node apriori.test.js</b> and <b>node evaluate.js</b>. The committed "
          "basket JSON and evaluation JSON hold the exact counts and source checksums.", "small"),
        PageBreak()]

    # Page 2 - one promising rule and its exact full-data evidence.
    story += [p("A rule I would investigate", "h1"),
        p("I selected <b>JUMBO BAG PINK POLKADOT (22386) -&gt; JUMBO BAG RED RETROSPOT "
          "(85099B)</b>. These are different bag designs that could plausibly be shown together "
          "in a cross-sell placement. This is a candidate for an experiment, not a proven sale."),
        *screenshot("promising_rule.png", "My live page at 1% support and 30% confidence. "
                    "I searched for the exact directed StockCode pair and opened its counts.", width=WIDTH),
        metric_table(full_good),
        Spacer(1, 8),
        p(f"I observed both bags in <b>{count(full_good['nAB'])}</b> invoices. "
          f"The red bag's overall rate is {pct(full_good['baselineConsequence'])}, "
          f"but its rate among invoices with the pink bag is {pct(full_good['confidence'])}; "
          f"lift is {full_good['lift']:.2f}. This clears my support, confidence, and lift filters."),
        p(f"Direction matters: for red bag -&gt; pink bag, confidence is "
          f"{pct(reverse['confidence'])}, while support and lift stay the same. "
          "The antecedent has a different basket count in the reverse rule."),
        PageBreak()]

    # Page 3 - a candidate that fails my confidence criterion.
    story += [p("A candidate I would reject", "h1"),
        p("I also inspected <b>REGENCY CAKESTAND 3 TIER (22423) -&gt; JUMBO BAG RED "
          "RETROSPOT (85099B)</b>. It reaches the 1% joint-support floor and has lift above 1, "
          "but it is a <b>rejected candidate</b> because its confidence is below my 30% minimum."),
        *screenshot("rejected_rule.png", "My rejected-candidates tab at the same 1% / 30% settings. "
                    "The view keeps the unfiltered candidate visible for comparison.", width=WIDTH),
        metric_table(full_bad),
        Spacer(1, 8),
        p(f"Only {count(full_bad['nAB'])} of {count(full_bad['nA'])} cakestand invoices "
          f"also contain the red bag: {pct(full_bad['confidence'])} confidence. "
          f"That is only {pct(full_bad['confidence'] - full_bad['baselineConsequence'])} "
          f"above the bag's {pct(full_bad['baselineConsequence'])} overall basket rate, "
          f"despite lift {full_bad['lift']:.2f}. I would not prioritize a bundle or promotion "
          "from this association."),
        p("Its lift above 1 does not override the confidence threshold, and no association "
          "alone establishes that displaying the bag causes an additional purchase."),
        PageBreak()]

    # Page 4 - parameter sensitivity and cautious next steps.
    story += [p("What changed when I changed the inputs", "h1"),
        *screenshot("high_evidence.png", "My page with the High evidence preset: "
                    "2% minimum support and 50% minimum confidence. The bag rule still passes.", width=WIDTH),
        table([
            ["Setting", "Support", "Confidence", "Frequent sets", "Retained rules"],
            *[[escape(row["name"]), pct(row["minSupport"]), pct(row["minConfidence"]),
               count(row["counts"]["frequentItemsets"]), count(row["counts"]["retainedRules"])]
              for row in RESULTS["thresholdSweep"]],
        ], [135, 70, 85, 100, WIDTH - 390]),
        Spacer(1, 6),
        p("I can see the trade-off: the 0.5% / 20% setting surfaces many rare rules, while "
          "2% / 50% leaves only 62. Stricter thresholds reduce the review workload but may "
          "hide useful niche products."),
        p("My exploratory later-period check", "h2"),
        p(f"I split at {RESULTS['temporalCutoff']}: "
          f"{count(good['early']['N'])} earlier and {count(good['late']['N'])} later invoices. "
          f"For the bag pair, confidence was {pct(good['early']['confidence'])} earlier "
          f"and {pct(good['late']['confidence'])} later; lift was "
          f"{good['early']['lift']:.2f} and {good['late']['lift']:.2f}. "
          f"The rejected rule stayed below 30% confidence "
          f"({pct(bad['early']['confidence'])} and {pct(bad['late']['confidence'])}). "
          "I chose these examples using all periods, so this is a stability check rather "
          "than a blind holdout evaluation."),
        p("What I would test next", "h2"),
        p("I would randomize eligible shoppers into a red-bag suggestion near the pink bag "
          "and a control with the usual page. I would compare incremental attach rate and "
          "gross margin per exposed shopper, while monitoring total conversion. "
          "I would make a sales-impact claim only after that controlled comparison.")]
    return story


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--desktop-copy", action="store_true")
    args = parser.parse_args()
    document = SimpleDocTemplate(str(OUTPUT), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN, topMargin=47, bottomMargin=56,
        title="My Week 4 Association Rules Report",
        author="Petr Vorozhtsov",
        subject="Online Retail association rules and business interpretation")
    document.build(make_story(), onFirstPage=frame, onLaterPages=frame)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
    if args.desktop_copy:
        shutil.copy2(OUTPUT, DESKTOP)
        print(f"Copied {DESKTOP} ({DESKTOP.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
