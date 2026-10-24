from pathlib import Path
import csv, json, re
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = Path(__file__).parent
OUT = ROOT / "report.pdf"
pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleRu", parent=styles["Title"], fontName="Arial-Bold", fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=10))
styles.add(ParagraphStyle(name="H1Ru", parent=styles["Heading1"], fontName="Arial-Bold", fontSize=12, leading=15, spaceBefore=8, spaceAfter=5))
styles.add(ParagraphStyle(name="BodyRu", parent=styles["BodyText"], fontName="Arial", fontSize=8.8, leading=11.2, spaceAfter=4))
styles.add(ParagraphStyle(name="SmallRu", parent=styles["BodyText"], fontName="Arial", fontSize=7, leading=8.5, spaceAfter=2))

def P(text, style="BodyRu"):
    return Paragraph(text, styles[style])

validation=json.loads((ROOT/'outputs/validation_results.json').read_text(encoding='utf-8'))
audit=json.loads((ROOT/'outputs/audit.json').read_text(encoding='utf-8'))
join_audit=audit['join']
examples=json.loads((ROOT/'outputs/examples.json').read_text(encoding='utf-8'))
ids={sid for user in examples.values() for method in user['predictions'].values() for sid in method}
meta={}
for path in (ROOT/'data').glob('export_arh_*.csv'):
    with path.open(encoding='utf-8-sig', newline='', errors='replace') as fh:
        for row in csv.DictReader(fh):
            sid=row.get('tv_show_id','')
            if sid in ids and sid != '0':
                meta.setdefault(sid, (row.get('tv_show_title',''), row.get('tv_show_category','')))

doc=SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=15*mm, leftMargin=15*mm, topMargin=13*mm, bottomMargin=13*mm)
story=[]
story += [P('SWEET.TV TV Program Recommendation', 'TitleRu'), P('Пётр Ворожцов · Kaggle: PetrVorozhtsovLMS', 'SmallRu'), P('Repository: https://github.com/PetrVorozhtsov/RecSys-LLMs/tree/exam-sweettv/exam_sweettv', 'SmallRu')]
story += [P('1. Data and protocol','H1Ru'), P('View logs were joined to labelled schedule rows by equal channel and interval overlap. A view was accepted when the overlap was at least 60 seconds and at least 10% of the scheduled programme duration. Rows with <b>tv_show_id = 0</b> were excluded; when several rows overlapped, the row with the largest overlap was selected. Training used weeks 11–20, validation used weeks 21–30, and the final catalogue used labelled programmes from weeks 11–42. Test-period user behaviour was not used for fitting.', 'BodyRu')]
story += [P(f"Join audit: {join_audit['views']:,} input views; {join_audit['matched']:,} matched labelled interactions; {join_audit['unmatched_or_unlabelled']:,} unmatched or unlabelled; {join_audit.get('filtered_users',0):,} views from users outside the submission template.", 'SmallRu')]
story += [P('2. Methods','H1Ru'), P('The popularity reference ranks programmes by the number of unique training viewers. The content-based model represents category, genres, year bucket, director and actor tokens and ranks candidates by cosine similarity to a duration-weighted user profile. The association model mines programme pairs from training histories and scores candidates using co-view confidence and lift. Every method uses the same temporal validation protocol and excludes programmes already watched by the user; users without usable history use the popularity fallback.', 'BodyRu')]
story += [P('3. Validation result','H1Ru')]
table=[[P('<b>Method</b>','SmallRu'),P('<b>MAP@5</b>','SmallRu')]]
for k,v in validation['map5'].items(): table.append([P(k,'SmallRu'),P(f'{v:.6f}','SmallRu')])
t=Table(table,colWidths=[55*mm,30*mm]); t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.3,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('VALIGN',(0,0),(-1,-1),'TOP')])); story += [t, Spacer(1,4*mm), P(f"The final selected method is <b>{validation['selected']}</b>, selected solely from the temporal validation MAP@5. The association score is close to the popularity reference, while the content model is weak on this sparse, metadata-heavy catalogue.",'BodyRu')]
story += [P('4. Two-user comparison','H1Ru')]
for user,data in examples.items():
    story.append(P(f'<b>User {user}</b>: {len(data["history"])} labelled programmes in the joined history.', 'BodyRu'))
    rows=[[P('<b>Method</b>','SmallRu'),P('<b>Top-5 IDs and programme metadata</b>','SmallRu')]]
    for method,preds in data['predictions'].items():
        bits=[]
        for sid in preds[:5]:
            title,cat=meta.get(sid,('', ''))
            title=re.sub(r'\s+',' ',title).strip()
            bits.append(f'{sid} — {title[:55]} ({cat})')
        rows.append([P(method,'SmallRu'),P('<br/>'.join(bits),'SmallRu')])
    tab=Table(rows,colWidths=[28*mm,145*mm]); tab.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.25,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('VALIGN',(0,0),(-1,-1),'TOP')])); story.append(tab); story.append(Spacer(1,3*mm))
story += [P('5. Strategy and limitations','H1Ru'), P('For this run I would deploy association scoring as a candidate generator with the popularity list as a fallback, then add a lightweight content re-ranker when metadata quality is sufficient. The association list stays close to popularity on validation, but uses the user history to alter ordering. The content model exposes the limitation of the catalogue: most schedule rows have no programme ID and metadata is uneven. The view log records platform behaviour rather than enjoyment, absence from the log does not prove a programme was never seen, and no A/B or revenue claim follows from this offline evaluation.', 'BodyRu')]
story += [P('6. Reproducibility and submission','H1Ru'), P('The implementation is in exam_sweettv/run.py. Generated artefacts include validation_results.json, audit.json, examples.json and submission.csv. The submission contains 1,260 rows and five space-separated IDs per row. Runtime environment: Python standard library, deterministic ranking and no random seed. Kaggle private-score screenshot and the original AI-session export must be attached before final submission.', 'BodyRu')]
doc.build(story)
print(OUT)
