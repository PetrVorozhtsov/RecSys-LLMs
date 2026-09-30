"""Build the first-person Week 3 PDF from measured results and browser captures.

Run: python report_assets/build_report.py --desktop-copy
Requires reportlab and Pillow only for regenerating the report; the web app does not.
"""

import argparse
import json
import shutil
from io import BytesIO
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent
REPORT = ROOT / "Petr_Vorozhtsov_Week3_Report.pdf"
RESULTS = json.loads((ROOT / "evaluation-results.json").read_text(encoding="utf-8"))
W, H = A4
MARGIN = 48
CONTENT_W = W - 2 * MARGIN
NAVY = colors.HexColor("#163b5c")
BLUE = colors.HexColor("#1769aa")
INK = colors.HexColor("#243b53")
MUTED = colors.HexColor("#526d85")
PALE = colors.HexColor("#edf4fa")
LINE = colors.HexColor("#d9e2ec")


BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=9.5,
                      leading=13.8, textColor=INK, spaceAfter=0)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.5, leading=12)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=11)
CELL_BOLD = ParagraphStyle("cell_bold", parent=CELL, fontName="Helvetica-Bold")


def paragraph(pdf, html, y, style=BODY, gap=8, x=MARGIN, width=CONTENT_W):
    block = Paragraph(html, style)
    _, height = block.wrap(width, H)
    block.drawOn(pdf, x, y - height)
    return y - height - gap


def heading(pdf, title, y):
    y -= 10
    pdf.setFillColor(NAVY)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(MARGIN, y, title)
    return y - 22


def table(pdf, rows, widths, y, header=True):
    formatted = []
    for row_index, row in enumerate(rows):
        style = CELL_BOLD if header and row_index == 0 else CELL
        formatted.append([Paragraph(str(value), style) for value in row])
    widget = Table(formatted, colWidths=widths, hAlign="LEFT")
    rules = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.45, LINE),
    ]
    if header:
        rules.append(("BACKGROUND", (0, 0), (-1, 0), PALE))
    widget.setStyle(TableStyle(rules))
    _, height = widget.wrap(CONTENT_W, H)
    widget.drawOn(pdf, MARGIN, y - height)
    return y - height - 11


def frame(pdf, page):
    pdf.setStrokeColor(LINE)
    pdf.line(MARGIN, 45, W - MARGIN, 45)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 8)
    pdf.drawString(MARGIN, 31, "Petr Vorozhtsov - My Week 3 final report")
    pdf.drawRightString(W - MARGIN, 31, f"Page {page}")


def continuation_header(pdf):
    pdf.setFillColor(NAVY)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(MARGIN, H - 42, "My Week 3 Collaborative Filtering Report")
    pdf.setStrokeColor(LINE)
    pdf.line(MARGIN, H - 52, W - MARGIN, H - 52)


def screenshot(pdf, user_id, y):
    source = Image.open(ASSETS / f"user_{user_id}.png").convert("RGB")
    # Crop the direct browser capture to keep the selector and both Top-5 lists legible.
    crop = source.crop((0, 580, source.width, 2250))
    buffer = BytesIO()
    crop.save(buffer, format="PNG")
    buffer.seek(0)
    height = CONTENT_W * crop.height / crop.width
    pdf.drawImage(ImageReader(buffer), MARGIN, y - height, width=CONTENT_W,
                  height=height, mask="auto")
    return y - height


def page_one(pdf):
    pdf.setFillColor(NAVY)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(MARGIN, H - 69, "My Week 3 Collaborative Filtering")
    pdf.drawString(MARGIN, H - 94, "Movie Recommender")
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(MARGIN, H - 115, "Student report by Petr Vorozhtsov | 30 September 2026")
    y = H - 139
    y = paragraph(pdf, 'My project: <link href="https://github.com/PetrVorozhtsov/RecSys-LLMs" color="#1769aa">github.com/PetrVorozhtsov/RecSys-LLMs</link> | My code revision: b14047a', y)

    y = heading(pdf, "1. What I completed", y - 3)
    y = paragraph(pdf, "I reused MovieLens 100K from Week 2 and built both User-Based and Item-Based collaborative filtering. I kept the two Top-5 lists side by side, added evidence counts, and evaluated predictions against ratings hidden from the model.", y)
    y = table(pdf, [
        ["Users", "Movies", "Ratings", "Observed matrix"],
        ["943", "1,682", "100,000", "6.30%"],
    ], [CONTENT_W / 4] * 4, y)

    y = heading(pdf, "2. My data and sparse-rating strategy", y)
    y = paragraph(pdf, "I verified that <b>u.data</b> and <b>u.item</b> are byte-identical to Week 2. I store missing values as zero because real ratings are 1-5. I chose the assignment's common-rating weighting strategy for both user and item similarities:", y)
    y = paragraph(pdf, "<b>similarity = cosine over shared ratings x n / (n + 10)</b>, where n is the number of common ratings. I require n &gt;= 2; a single shared rating no longer creates perfect similarity. I use the training-set mean as a small prediction prior and require at least two contributing matches. Item candidates also need five community ratings.", y)

    y = heading(pdf, "3. What I changed in my code", y)
    y = table(pdf, [
        ["File", "My change and purpose"],
        ["data.js", "I validated rating rows, preserved movie title encoding, and counted training ratings."],
        ["script.js", "I added overlap weighting, cautious predictions, evidence counts, and reusable item-similarity caching."],
        ["index.html / style.css", "I clarified the predicted-score scale and displayed support without hiding long titles."],
        ["evaluate.js / test", "I added a deterministic hidden-rating comparison, timings, and five behavior checks."],
    ], [125, CONTENT_W - 125], y)
    if y < 62:
        raise RuntimeError(f"Page 1 overflow: {y}")
    frame(pdf, 1)
    pdf.showPage()


def scenario_page(pdf, page, user_id, count, user_top, item_top, note):
    continuation_header(pdf)
    y = H - 83
    y = heading(pdf, f"{page + 2}. My experiment - User {user_id}", y)
    y = paragraph(pdf, f"I selected User {user_id}, who had {count} ratings in MovieLens. The capture below comes directly from my running local page after I clicked Get Recommendations.", y)
    y = screenshot(pdf, user_id, y - 4) - 13
    y = paragraph(pdf, f"<font color='#526d85'>Figure {page - 1}.</font> My live User {user_id} result: two Top-5 lists, predicted scores, and support counts.", y, SMALL, 10)
    y = paragraph(pdf, f"My first User-Based result was <b>{user_top}</b>. My first Item-Based result was <b>{item_top}</b>. {note}", y, BODY)
    y = paragraph(pdf, "I read these numbers as model predictions with different amounts of support, not as proof that one of the recommended films will be liked. I use hidden real ratings for the accuracy comparison on page 4.", y, BODY)
    if y < 62:
        raise RuntimeError(f"Page {page} overflow: {y}")
    frame(pdf, page)
    pdf.showPage()


def page_four(pdf):
    continuation_header(pdf)
    y = H - 83
    y = heading(pdf, "6. My measured comparison", y)
    y = paragraph(pdf, "I hid five actual ratings from each of 50 deterministically selected users (250 ratings total). I rebuilt the matrix, item counts, prior, and caches from the remaining 99,750 ratings. The target movie originally had at least ten ratings, so this checks existing users and items. Lower MAE/RMSE is better; coverage is the share for which I could predict.", y)

    quality = RESULTS["quality"]
    ub = quality["userBased"]
    ib = quality["itemBased"]
    pub = quality["pairedUserBased"]
    pib = quality["pairedItemBased"]
    y = table(pdf, [
        ["Method", "Predicted / 250", "Coverage", "Paired MAE", "Paired RMSE"],
        ["User-Based", str(ub["evaluated"]), f'{ub["coverage"] * 100:.1f}%', f'{pub["mae"]:.3f}', f'{pub["rmse"]:.3f}'],
        ["Item-Based", str(ib["evaluated"]), f'{ib["coverage"] * 100:.1f}%', f'{pib["mae"]:.3f}', f'{pib["rmse"]:.3f}'],
    ], [111, 106, 75, 100, 107], y)
    y = paragraph(pdf, "The paired error columns use the same 248 hidden ratings for both methods. In this sample, Item-Based had lower error. This tests predicted ratings, not whether unobserved Top-5 movies would be liked.", y, SMALL)

    y = heading(pdf, "7. My efficiency check", y)
    timing = RESULTS["timing"]["medianMilliseconds"]
    y = table(pdf, [
        ["Full-data Top-5 method", "Median time across Users 1, 100, 196, 300, 500"],
        ["User-Based", f'{timing["userBased"]:.1f} ms'],
        ["Item-Based, cold cache", f'{timing["itemBasedCold"]:.1f} ms'],
        ["Item-Based, warm cache", f'{timing["itemBasedWarm"]:.1f} ms'],
    ], [203, CONTENT_W - 203], y)
    y = paragraph(pdf, "I measured on Windows with Node 24.19.0. The cold item method was slower here; reusing cached similarities made it faster on a repeated request. Full user-pair work scales roughly with U squared times I, while full item-pair work scales with I squared times U. Here I=1,682 exceeds U=943; when users greatly outnumber items, the item approach can be cheaper. These measured times are machine-dependent.", y, SMALL)

    y = heading(pdf, "8. Trade-offs, limitations, and reproducibility", y)
    y = paragraph(pdf, "I chose <b>common-rating weighting</b> because it lowers trust in tiny overlaps without inventing missing ratings. Mean imputation is easy but can pull tastes toward averages. Matrix factorization can capture latent patterns but needs training and tuning. A new user or movie has no collaborative evidence, and this 250-rating warm-start check does not measure cold-start or full Top-5 ranking quality.", y, SMALL)
    y = paragraph(pdf, "To reproduce my work, I run <b>cd week3</b>, <b>python -m http.server 8000</b>, then open <b>http://localhost:8000/</b>. I run <b>node evaluate.js</b> and <b>node --test recommender.test.js</b> for the results and checks. The saved metrics are in <b>evaluation-results.json</b>; the full browser captures are in <b>report_assets/</b>.", y, SMALL)
    y = heading(pdf, "9. My conclusion", y)
    y = paragraph(pdf, "I completed both CF approaches, tested their predictions on withheld data, and made small-overlap evidence explicit. On this sample Item-Based had lower prediction error, while User-Based was faster on this dataset. I would test ranking relevance and true cold-start users before making a production quality claim.", y, BODY)
    if y < 62:
        raise RuntimeError(f"Page 4 overflow: {y}")
    frame(pdf, 4)
    pdf.showPage()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--desktop-copy", action="store_true")
    args = parser.parse_args()
    pdf = canvas.Canvas(str(REPORT), pagesize=A4, pageCompression=1)
    pdf.setTitle("My Week 3 Collaborative Filtering Movie Recommender")
    pdf.setAuthor("Petr Vorozhtsov")
    page_one(pdf)
    scenario_page(pdf, 2, 1, 272, "Braindead (4.603; 3 matches)",
                  "Magic Hour (4.184; 107 matches)",
                  "The different lists show that user and item evidence do not rank movies identically.")
    scenario_page(pdf, 3, 196, 39, "All About Eve (4.661; 8 matches)",
                  "Dingo (3.969; 14 matches)",
                  "The smaller profile still gives full lists, while the evidence counts show how much supports each score.")
    page_four(pdf)
    pdf.save()
    print(REPORT)
    if args.desktop_copy:
        target = Path.home() / "Desktop" / "Petr_Vorozhtsov_Week3_Final_Report.pdf"
        shutil.copy2(REPORT, target)
        print(target)


if __name__ == "__main__":
    main()
