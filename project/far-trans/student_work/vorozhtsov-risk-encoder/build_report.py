"""Build the individual Session 2 PDF directly from the measured run artifacts.

No benchmark numbers or student verification are invented. Run this only after
run_experiment.py has completed; the PDF is a presentation of its saved evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parent
INK = colors.HexColor("#173F45")
MUTED = colors.HexColor("#53646C")
TINT = colors.HexColor("#EDF4F3")
LINE = colors.HexColor("#CCD9DA")
WARN = colors.HexColor("#FFF5DD")
WIDTH = A4[0] - 36 * mm


def read_json(path, required=False):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Complete the experiment before building the report: {path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def text(value, default="not recorded"):
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def clean(value, limit=None):
    result = text(value).replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", " - ")
    if limit and len(result) > limit:
        result = result[:limit].rsplit(" ", 1)[0] + "..."
    return escape(result)


def number(value, digits=4, missing="not recorded"):
    if value is None:
        return missing
    try:
        numeric = float(value)
    except (ValueError, TypeError):
        return clean(value)
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else missing


def percentage(value, digits=1):
    return f"{100 * float(value):.{digits}f}%" if value is not None else "not recorded"


def first(mapping, *keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def local_commit(identity):
    supplied = identity.get("experiment_commit") or identity.get("commit")
    if supplied:
        return str(supplied)
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "not committed"


def register_fonts():
    font_dir = Path("C:/Windows/Fonts")
    font_files = {
        "Report": "arial.ttf", "Report-Bold": "arialbd.ttf",
        "Report-Italic": "ariali.ttf", "Report-BoldItalic": "arialbi.ttf",
    }
    if all((font_dir / file).exists() for file in font_files.values()):
        for name, file in font_files.items():
            pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
        pdfmetrics.registerFontFamily(
            "Report", normal="Report", bold="Report-Bold",
            italic="Report-Italic", boldItalic="Report-BoldItalic",
        )
        return "Report", "Report-Bold", "Report-Italic"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


def make_styles():
    regular, bold, italic = register_fonts()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "ReportTitle", fontName=bold, fontSize=21, leading=25,
        textColor=INK, spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        "ReportBody", fontName=regular, fontSize=10, leading=12.5,
        textColor=colors.HexColor("#222D33"), spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "ReportSmall", parent=styles["ReportBody"], fontSize=8.5,
        leading=10.5, textColor=MUTED, spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        "ReportTiny", parent=styles["ReportSmall"], fontSize=7.7, leading=9.6,
    ))
    styles.add(ParagraphStyle(
        "ReportH1", fontName=bold, fontSize=13.3, leading=16,
        textColor=INK, spaceBefore=10, spaceAfter=6, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        "ReportH2", fontName=bold, fontSize=10.4, leading=13,
        textColor=INK, spaceBefore=7, spaceAfter=4, keepWithNext=True,
    ))
    styles.add(ParagraphStyle(
        "ReportCell", parent=styles["ReportBody"], fontSize=8.3,
        leading=10.4, spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        "ReportCellSmall", parent=styles["ReportCell"], fontSize=7.6, leading=9.5,
    ))
    styles.add(ParagraphStyle(
        "ReportTableHeader", parent=styles["ReportCell"],
        fontName=bold, textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        "ReportReference", parent=styles["ReportBody"], fontSize=8.8,
        leading=11.4, leftIndent=16, firstLineIndent=-16, spaceAfter=10,
    ))
    return styles, regular, bold, italic


def paragraph(content, styles, style="ReportBody"):
    return Paragraph(content, styles[style])


def heading(title, styles):
    return paragraph(title, styles, "ReportH1")


def table(headers, rows, widths, styles, small=False):
    cell_style = "ReportCellSmall" if small else "ReportCell"
    cells = [[paragraph(str(value), styles, "ReportTableHeader") for value in headers]]
    cells.extend([[paragraph(str(value), styles, cell_style) for value in row] for row in rows])
    result = Table(cells, colWidths=widths, hAlign="LEFT", repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TINT]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, -1), (-1, -1), .5, LINE),
    ]))
    return result


def notice(content, styles):
    result = Table([[paragraph(content, styles, "ReportSmall")]], colWidths=[WIDTH])
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WARN),
        ("BOX", (0, 0), (-1, -1), .5, colors.HexColor("#D8BC78")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return result


def check_text(check):
    if isinstance(check, str):
        return clean(check, 400)
    if not isinstance(check, dict):
        return clean(check, 300)
    label = first(check, "name", "check", "title", default="Check")
    detail = first(check, "detail", "description", "outcome", "summary", default="")
    result = first(check, "passed", "result", "status")
    if isinstance(result, bool):
        result = "passed" if result else "failed"
    return f"<b>{clean(label)}:</b> {clean(result)}. {clean(detail, 350)}"


def verification_paragraphs(verification, styles):
    result = []
    summary = first(verification, "summary", "verification_summary")
    if summary:
        result.append(paragraph(clean(summary, 650), styles))
    checks = verification.get("checks", [])
    if isinstance(checks, dict):
        checks = [{"name": key, "result": value} for key, value in checks.items()]
    for check in checks[:3]:
        result.append(paragraph(check_text(check), styles, "ReportSmall"))
    ndcg = verification.get("ndcg_check", {})
    if ndcg:
        vector = first(ndcg, "relevance_vector", "relevance", "hits")
        dcg = first(ndcg, "dcg", "dcg_at_10")
        idcg = first(ndcg, "idcg", "idcg_at_10")
        value = first(ndcg, "ndcg", "ndcg_at_10")
        if dcg is not None and idcg is not None:
            result.append(paragraph(
                f"<b>Anonymous metric example.</b> Relevance vector {clean(vector)}; "
                f"DCG = {number(dcg, 6)}, IDCG = {number(idcg, 6)}, "
                f"nDCG@10 = {number(value, 6)}. The independent arithmetic is saved in "
                "<i>verification.json</i>.", styles, "ReportSmall"))
    feature = verification.get("feature_check", {})
    if feature:
        peak = feature.get("peak_price")
        trough = feature.get("trough_price")
        calculation = ""
        if peak is not None and trough is not None and float(peak) > 0:
            computed_drawdown = 1 - float(trough) / float(peak)
            calculation = (
                f" The explicit drawdown calculation is (1 - {float(trough):.2f} / "
                f"{float(peak):.2f}) x 100 = {100 * computed_drawdown:.6f}%."
            )
        result.append(paragraph(
            "<b>Price-feature check.</b> Transaction "
            f"{clean(feature.get('transaction_id', '8996071'))}: "
            f"{clean(feature.get('price_count'))} prior prices and "
            f"{clean(feature.get('return_count'))} valid changes; "
            f"volatility {percentage(feature.get('volatility'), 2)}, maximum drawdown "
            f"{percentage(feature.get('max_drawdown'), 2)}.{calculation} This is an assistant-performed check, "
            "with the underlying price observations retained for student inspection.",
            styles, "ReportSmall"))
    if not result:
        result.append(notice(
            "The separate verification artifact is not yet available. No personal or "
            "independent verification is claimed in this report.", styles))
    return result


def build_report(output_dir, destination, identity_path=None, student_path=None):
    output_dir = Path(output_dir)
    metrics = read_json(output_dir / "metrics.json", required=True)
    timings = read_json(output_dir / "timings.json", required=True)
    model = read_json(output_dir / "model.json", required=True)
    experiment = read_json(output_dir / "experiment.json", required=True)
    protocol = read_json(output_dir / "protocol.json", required=True)
    verification = read_json(output_dir / "verification.json")
    analysis = read_json(output_dir / "analysis.json")
    format_pilot = read_json(output_dir.parent / "format_pilot" / "validation.json")
    identity = read_json(identity_path or ROOT / "identity.json")
    student = read_json(student_path or ROOT / "student_verification.json")
    methods = metrics["methods"]
    required = ("popularity", "cbf", "latest_buy", "latest_buy_or_sell",
                "user_knn", "statistical_risk", "llm_risk")
    missing = [method for method in required if method not in methods]
    if missing:
        raise ValueError("Missing measured methods: " + ", ".join(missing))
    llm, control = methods["llm_risk"], methods["statistical_risk"]
    knn = methods["user_knn"]
    comparison = analysis.get("comparison", analysis.get("paired_comparison", {}))
    input_summary = analysis.get("input_summary", {})
    collapsed = (
        input_summary.get("uniform_effective_llm_profiles") == llm["users"]
        and comparison.get("llm_knn_identical_top10_users") == llm["users"]
    )
    if metrics.get("llm_attempts") != llm["users"]:
        raise ValueError("A complete report requires one actual model attempt per frozen user")
    for method in required:
        if methods[method]["users"] != llm["users"]:
            raise ValueError("Report comparison has mismatched user counts")
    styles, regular, bold, _ = make_styles()
    name = identity.get("name", "Pyotr Alekseyevich Vorozhtsov")
    team = identity.get("team", "FAR-Trans Lab")
    report_date = identity.get("date", "2026-11-25")
    repository = first(identity, "repository", "github_repository", "github_url")
    email = identity.get("email")
    commit = local_commit(identity)
    confirmed = (student.get("confirmed") is True or student.get("personally_verified") is True) and bool(
        first(student, "statement", "summary", "description")
    )
    failure_count = len(metrics.get("llm_failures", []))
    delta = llm["ndcg_at_10"] - control["ndcg_at_10"]
    delta_nonempty = (llm.get("ndcg_at_10_nonempty") or 0) - (control.get("ndcg_at_10_nonempty") or 0)
    baseline_seconds = timings.get("classical_baselines_whole_run_seconds")
    preparation_seconds = timings.get("risk_preparation_whole_run_seconds")
    generation_seconds = metrics.get("generation_seconds")
    quantization = text(model.get("quantization"), "GGUF")
    short_model = text(model.get("model_repository"), "Qwen/Qwen3-1.7B-GGUF")
    runtime_tag = text(model.get("runtime_tag"))
    constrained_output = bool(model.get("grammar"))
    format_policy = (
        "A fixed JSON schema constrains the seven-by-three shape and numeric bounds. "
        "It does not determine the probabilities or enforce their sums. No semantic repair is used."
        if constrained_output else
        "Generation is unconstrained; no output grammar or semantic repair is used."
    )
    story = []

    # Page 1: the question and its directly relevant prior work.
    story.append(paragraph("Investor Risk Profiles from an LLM:<br/>A Controlled FAR-Trans Experiment", styles, "ReportTitle"))
    identity_rows = [
        [paragraph(f"<b>Student</b> {clean(name)}", styles, "ReportSmall"),
         paragraph(f"<b>Team</b> {clean(team)}", styles, "ReportSmall")],
        [paragraph(f"<b>Assignment</b> Week 8 / Session 2", styles, "ReportSmall"),
         paragraph(f"<b>Date</b> {clean(report_date)}", styles, "ReportSmall")],
        [paragraph(f"<b>Email</b> {clean(text(email, 'not supplied'))}", styles, "ReportSmall"),
         paragraph(f"<b>Student repository</b> {clean(text(repository, 'not supplied'))}", styles, "ReportSmall")],
    ]
    info = Table(identity_rows, colWidths=[WIDTH * .57, WIDTH * .43], hAlign="LEFT")
    info.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                             ("LEFTPADDING", (0, 0), (-1, -1), 0),
                             ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                             ("TOPPADDING", (0, 0), (-1, -1), 1),
                             ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    story.append(info)
    story.append(paragraph(f"<b>Experiment commit</b> {clean(commit)}", styles, "ReportTiny"))
    pending = []
    if not email:
        pending.append("email")
    if not repository:
        pending.append("student GitHub repository")
    if not confirmed:
        pending.append("student's personal verification")
    if pending:
        story.append(notice("<b>Student confirmation pending:</b> " + ", ".join(pending) + ".", styles))
    story.append(heading("Abstract", styles))
    abstract_result = (
        f"On {llm['users']} FAR-Trans investors, the LLM nDCG@10 is {llm['ndcg_at_10']:.4f}, "
        f"versus {control['ndcg_at_10']:.4f} for the statistical control and "
        f"{knn['ndcg_at_10']:.4f} for original user-kNN, with {failure_count} invalid-output fallbacks. "
    )
    abstract_takeaway = (
        f"All {llm['users']} effective LLM profiles are near-uniform and their Top-10 lists match "
        "user-kNN, so this run found no incremental ranking benefit from the LLM Encoder."
        if collapsed else
        "The result measures purchase prediction under a small teaching protocol and must be "
        "interpreted alongside candidate coverage, output validity, and the separate nonempty-truth average."
    )
    abstract = (
        "Observed asset purchases may reveal patterns in price risk that help predict later purchases. "
        "We compare a zero-shot LLM investor Encoder with a recency-weighted statistical profile, using "
        "the same seven historical risk measures, user-kNN candidates, and deterministic scoring rule. "
        + abstract_result + abstract_takeaway
    )
    words = len(abstract.split())
    if not 50 <= words <= 150:
        raise ValueError(f"Abstract must contain 50-150 words, got {words}")
    story.append(paragraph(abstract, styles))
    story.append(paragraph("<b>Index Terms</b> - FAR-Trans, LLM Encoder, risk features, nDCG@10.", styles, "ReportSmall"))
    story.append(heading("1. Introduction", styles))
    story.append(paragraph(
        "The course task ranks ten eligible assets for each investor and compares them with later purchases [1], [5]. "
        "Our question is whether an LLM-built profile improves mean nDCG@10 over a recency-weighted "
        "frequency profile when both receive the same purchase ages and seven price-risk bands. "
        "A purchase of an asset with deep past drawdowns is evidence of an observed choice; it does not "
        "establish the investor's psychological risk tolerance.", styles))
    story.append(paragraph(
        "The contribution is a matched-input Encoder comparison with point-in-time risk features and "
        "the instructor's unchanged four classical baselines. A product owner can use the measured gain, "
        "failure rate, and inference time to assess whether this extra representation step is useful.", styles))
    story.append(heading("2. Related Work", styles))
    story.append(table(
        ["Study", "Method and evaluation", "Connection and limit"],
        [
            ["Sanz-Cruzado et al. (2024) [1]", "FAR-Trans; eleven algorithms, nDCG@10 and ROI.",
             "Establishes the data and task. Its six-month, non-repeat evaluation differs from HW2."],
            ["Dai, Zhou, Joachims (2024) [2]", "LLM profile Encoder and LLM Decoder on MovieLens.",
             "Motivates user-side profiles. Our downstream scorer is fixed and uses numeric risk bands."],
            ["Spadea, Seneviratne (2025) [3]", "FLARKO: transaction/market graphs, Qwen fine-tuning; 180-day metrics.",
             "Closest financial LLM study; combines changes that our controlled test holds separate."],
        ], [WIDTH * .24, WIDTH * .36, WIDTH * .40], styles, small=True))
    story.append(paragraph(
        "These works motivate the design but do not establish that an LLM is better than a frequency "
        "profile on this seven-feature, fixed-candidate teaching comparison. We test that narrower question.",
        styles, "ReportSmall"))
    story.append(PageBreak())

    # Page 2: full specification, including all seven requested feature formulas.
    story.append(heading("3. Method", styles))
    story.append(paragraph(
        "The official FAR-Trans CSVs [1] are processed using the vendored, unmodified instructor starter [5]. "
        "Eligibility requires asset metadata and a positive close strictly before the prediction date. "
        "Reconstructed negative-ID transactions and exact duplicates are excluded. Only earlier Buy events "
        "build histories; questionnaire profiles are omitted from model inputs.", styles))
    story.append(paragraph("3.1. Point-in-time asset risk", styles, "ReportH2"))
    story.append(paragraph(
        "For each date t, use positive closes P in [t - 90 days, t). Let r be log returns between consecutive "
        "observations at most seven days apart, s their sample standard deviation, and "
        "D<sub>i</sub> = 1 - P<sub>i</sub> / max(P<sub>1</sub>, ..., P<sub>i</sub>). "
        "Require at least 40 valid returns and a last close at most seven days old. Constant or "
        "insufficient series are unknown.", styles))
    feature_rows = [
        ["Volatility", "s(r) = sqrt(sum((r - mean(r))<super>2</super>) / (n - 1))"],
        ["Maximum drawdown", "max(D<sub>i</sub>)"],
        ["Downside volatility", "sqrt(sum(min(r, 0)<super>2</super>) / n)"],
        ["Worst five-interval return", "min(P<sub>i</sub> / P<sub>i-5</sub> - 1), only if all five gaps are at most seven days"],
        ["Current drawdown", "D<sub>N</sub>"],
        ["Share in drawdown &gt; 5%", "count(D<sub>i</sub> &gt; 0.05) / N"],
        ["Change in volatility", "s(last 20 valid returns) - s(all valid returns)"],
    ]
    story.append(table(["Measure", "Calculation"], feature_rows, [WIDTH * .33, WIDTH * .67], styles))
    story.append(paragraph(
        "Each measure is assigned low/middle/high bands by the 33rd and 67th percentiles among eligible "
        "assets of the same category on that date. Quantile ties go to the lower band. Reverse the sign "
        "of the worst return for banding, so a larger loss means higher measured risk. Historical purchases "
        "use bands computed before their own dates; candidate bands use the prediction cutoff.", styles))
    story.append(paragraph("3.2. Matched profiles and shared ranking", styles, "ReportH2"))
    story.append(paragraph(
        "Use at most the 40 most recent purchases. The statistical control adds weight "
        "2<super>-d/90</super> to each observed band, where d is purchase age in days, then normalizes "
        "each of seven triples. The LLM sees exactly the same sequence of ages and bands, in chronological "
        "order, and returns seven probability triples as JSON. It receives no customer ID, ISIN, asset name, "
        "questionnaire, or future labels. A measure with no observations is uniform in both arms. "
        "Invalid JSON, shape, bounds, or sums outside 1 +/- 0.02 trigger the statistical fallback; "
        "accepted triples are normalized. " + format_policy, styles))
    story.append(paragraph(
        "Binary purchase cosine selects 50 nearest other investors. Sum neighbors' similarities per "
        "asset and retain up to 50 candidates; popularity fills only a pool shorter than ten. Both arms "
        "reuse these candidates. For zero-based candidate rank j among m assets, "
        "R = 1 - j/(m - 1). Risk match M is the mean of seven profile weights at the candidate's bands; "
        "an unknown band contributes 1/3. Final score S = 0.75R + 0.25M; ties preserve kNN order. "
        "The first ten form the output. Original kNN order is an additional control.", styles))
    story.append(paragraph("3.3. Model and execution", styles, "ReportH2"))
    story.append(paragraph(
        f"<b>{clean(short_model)}</b>, {clean(quantization)} quantization [4]; "
        f"revision <font size='8'>{clean(model.get('model_revision'))}</font>. "
        f"Local llama.cpp {clean(runtime_tag)} [6], {clean(model.get('backend'))}; "
        f"{clean(model.get('gpu'))}. Greedy generation, temperature {clean(model.get('temperature', 0))}, "
        f"seed {clean(model.get('seed', 42))}, context {clean(model.get('context_limit', 8192))}, "
        "maximum 256 generated tokens; thinking disabled. One sequential local call per investor. "
        "Python " + clean(model.get("python_version", experiment.get("python", "not recorded")), 80) +
        "; model/runtime digests, prompts, settings, and source versions are retained in the artifacts. "
        "The GGUF backend is a resource-driven change from the HW1 planned Transformers runtime.",
        styles, "ReportSmall"))
    story.append(PageBreak())

    # Page 3: measured comparisons and independent AI checks, never student claims.
    story.append(heading("4. Experiments", styles))
    story.append(paragraph(
        f"The frozen instructor sample contains {llm['users']} investors with prior purchases; "
        f"{llm['nonempty_truth_users']} have eligible future buys and {llm['empty_truth_users']} have empty truth. "
        f"The cutoff is {clean(protocol.get('cutoff'))}, and the evaluation ends before "
        f"{clean(protocol.get('end_exclusive'))}. Repeat purchases are allowed. "
        "All methods share users, eligibility, labels, and the starter's binary nDCG@10. "
        "Empty truth contributes zero to the primary all-user average. The nonempty average is reported separately.", styles))
    labels = {
        "popularity": "Popularity", "cbf": "CBF", "latest_buy": "Latest buy",
        "latest_buy_or_sell": "Latest buy or sell", "user_knn": "User-kNN",
        "statistical_risk": "Statistical risk", "llm_risk": "LLM risk + fallback",
    }
    rows = []
    for key in required:
        row = methods[key]
        runtime = "shared A" if key in required[:4] else ("shared B" if key != "llm_risk" else "C")
        rows.append([
            labels[key], str(row["users"]), str(row["nonempty_truth_users"]),
            number(row["ndcg_at_10"]), number(row.get("ndcg_at_10_nonempty")),
            runtime, str(failure_count if key == "llm_risk" else 0),
        ])
    story.append(table(
        ["Method", "Users", "Future<br/>buyers", "nDCG@10<br/>all", "nDCG@10<br/>nonempty", "Runtime<br/>scope", "Failed<br/>cases"],
        rows, [WIDTH * fraction for fraction in (.245, .075, .09, .14, .15, .115, .085)],
        styles, small=True))
    story.append(Spacer(1, 6))
    story.append(paragraph(
        f"<b>A:</b> {number(baseline_seconds, 2)} s for the single subprocess containing loading and all "
        f"four provided baselines. <b>B:</b> {number(preparation_seconds, 2)} s shared preparation "
        "for point-in-time features, kNN retrieval, statistical profiles, and prompts; no per-method "
        f"preparation timings were fabricated. <b>C:</b> {number(generation_seconds, 2)} s measured model requests "
        f"plus {number(metrics.get('llm_scoring_seconds'), 3)} s downstream LLM-profile scoring, "
        "excluding the shared preparation and model download/server startup.", styles, "ReportSmall"))
    story.append(paragraph(
        f"<b>Matched comparison.</b> LLM minus statistical-control nDCG@10 is {delta:+.4f} "
        f"over all users and {delta_nonempty:+.4f} over future buyers. "
        f"There were {metrics['llm_attempts']} actual model attempts and {failure_count} invalid outputs "
        f"({failure_count / llm['users']:.1%}); the latter remain in the cohort using statistical fallback. "
        "Thus the LLM row measures the fallback-inclusive system. "
        f"Recorded tokens: {clean(metrics.get('input_tokens'))} input / "
        f"{clean(metrics.get('output_tokens'))} output. Local API charges are zero; "
        "electricity and hardware costs were not measured.", styles))
    if collapsed:
        story.append(paragraph(
            f"<b>No gain over retrieval.</b> All {llm['users']} LLM Top-10 lists are identical to "
            "original user-kNN. Every effective LLM profile is within 0.001 of uniform, even though "
            f"{clean(input_summary.get('users_with_any_known_historical_risk_band'))} investors have "
            "known historical bands. The difference versus the statistical control therefore does "
            "not demonstrate useful LLM risk encoding.", styles))
    story.append(paragraph(
        f"<b>Candidate ceiling.</b> The shared kNN lists contain "
        f"{clean(metrics.get('candidate_truth_hits'))} of {clean(metrics.get('total_truth_assets'))} "
        f"held-out user-asset purchases: micro recall {percentage(metrics.get('candidate_recall_micro'))}. "
        "Profile scoring cannot recover purchases outside those lists. The four instructor baselines "
        "retain their own specified retrieval rules.", styles))
    if comparison:
        improved = first(comparison, "improved_users", "better_users")
        tied = first(comparison, "tied_users", "unchanged_users")
        worse = first(comparison, "worse_users", "degraded_users")
        changed = first(comparison, "changed_top10_users", "changed_rankings")
        story.append(paragraph(
            f"Per-user paired outcomes: improved {clean(improved)}, tied {clean(tied)}, "
            f"worse {clean(worse)}; Top-10 lists changed for {clean(changed)} investors. "
            f"With only {llm['nonempty_truth_users']} future buyers in the teaching sample, these are descriptive results.",
            styles, "ReportSmall"))
    story.append(paragraph("4.1. Verification and evidence", styles, "ReportH2"))
    story.extend(verification_paragraphs(verification, styles))
    story.append(paragraph(
        "All checks above were performed by Codex unless the student confirmation in Section 6 states otherwise. "
        "Protocol hashes and full-precision metrics remain in the saved files, rather than the main table.",
        styles, "ReportSmall"))
    story.append(PageBreak())

    # Page 4: real failure analysis, documented historical team context, disclosure.
    story.append(heading("5. Discussion", styles))
    story.append(paragraph("5.1. Observed result and failure analysis", styles, "ReportH2"))
    interpretation = first(analysis, "summary", "interpretation")
    if interpretation:
        story.append(paragraph(clean(interpretation, 850), styles))
    else:
        direction = "higher" if delta > 0 else ("lower" if delta < 0 else "equal")
        story.append(paragraph(
            f"The LLM system is {direction} than the matched statistical control on the measured primary "
            f"average (difference {delta:+.4f}). This estimate describes this specific quantized model, "
            "prompt, candidate policy, and teaching cohort. It does not establish general LLM superiority "
            "or investment suitability.", styles))
    failure = analysis.get("failure_case", {})
    if failure:
        for label, key in (("Observed case", "description"), ("Root cause", "root_cause"),
                           ("Handling or fix", "fix"), ("Verification", "verification")):
            if failure.get(key):
                story.append(paragraph(f"<b>{label}.</b> {clean(failure[key], 450)}", styles))
    elif failure_count:
        example = metrics["llm_failures"][0]
        story.append(paragraph(
            f"<b>Observed failure.</b> A model output failed validation: {clean(example.get('error'), 250)}. "
            "The validator rejected it and substituted the recorded statistical profile. This behavior "
            "preserves the user in the comparison; it does not turn an invalid generation into valid LLM evidence. "
            "Raw responses and per-user fallback decisions remain in the audit artifacts.", styles))
    else:
        story.append(paragraph(
        "No invalid-generation case occurred in the completed main run. A valid response can still rank poorly: "
            "candidate misses and changes in ordering must be examined using the saved per-user results. "
            "A specific numeric failure analysis is pending in analysis.json.", styles))
    pilot_cases = format_pilot.get("validation", [])
    if pilot_cases:
        pilot_invalid = sum(case.get("valid") is False for case in pilot_cases)
        pilot_total = len(pilot_cases)
        story.append(paragraph(
            f"<b>Format failure and correction.</b> An actual {pilot_total}-user pilot produced "
            f"{pilot_invalid} invalid unconstrained outputs (wrong shape or non-JSON/truncated text). "
            "The instruction alone did not enforce seven triples within 256 tokens. Before the main run, "
            "a shape-and-range schema was frozen with the same model, prompt, token budget, and greedy "
            "settings, using format validity rather than labels or custom scores. Original pilot outputs "
            "remain separate from the main table.", styles, "ReportSmall"))
        if constrained_output:
            story.append(paragraph(
                f"<b>Format confirmation.</b> Python validation of all {metrics['llm_attempts']} main "
                f"outputs found {failure_count} requiring statistical fallback. This confirms the "
                "format correction, while the uniform profiles demonstrate the unresolved semantic failure.",
                styles, "ReportSmall"))
    story.append(paragraph("5.2. Team discussion and individual contribution", styles, "ReportH2"))
    story.append(paragraph(
        "The documented 11 November team milestone established chronological evaluation, user-kNN retrieval, "
        "fixed candidate lists, and a profile-aware Ranker study [7]. The subsequent individual HW1 study "
        "selected an Encoder comparison using seven measured price-risk features. No new team discussion "
        "for this implementation is documented. The two-profile design and insistence on all seven "
        "features originated in the student's earlier work; the instructor supplied the baseline/evaluation "
        "code. Codex implemented and executed the new feature, profile, inference, and reporting code. "
        "Student confirmation and independent verification are reported separately below.", styles))
    story.append(paragraph("5.3. Limitations and next experiment", styles, "ReportH2"))
    story.append(paragraph(
        "Only a small fraction of the cohort purchases during the test window, so most primary-metric "
        "contributions are zero. Price coverage and unknown bands constrain profile information; "
        "fixed retrieval constrains achievable relevance. Purchases reflect behavior at one bank rather "
        "than a complete statement of preferences. The quantized model and greedy prompt were not "
        "fine-tuned. Keeping asset identities and questionnaire fields out of prompts reduces direct "
        "identity dependence and avoids using possibly unavailable assessments.", styles))
    next_experiment = analysis.get("next_experiment") or (
        "A next controlled ablation would retain only volatility and maximum drawdown while holding "
        "users, candidates, generation settings, and the scoring mixture fixed, to test the value of the other five measures."
    )
    story.append(paragraph(clean(next_experiment, 550), styles))
    story.append(heading("6. AI Usage Disclosure", styles))
    story.append(paragraph(
        "OpenAI Codex (desktop; CLI 0.171.0-alpha.13) read requirements and primary sources, generated "
        "and debugged code, ran the experiment and numerical audits, and drafted this report. "
        "Qwen is the experimental model. The session export includes tool calls, results, and delegated "
        "implementation work. The original collection of bank records and training of pretrained "
        "weights were accepted from the public releases and were not independently audited.", styles))
    if confirmed:
        statement = first(student, "statement", "summary", "description", default="A student confirmation was supplied.")
        outcome = first(student, "outcome", "result", default="")
        story.append(paragraph(
            f"<b>Student-reported personal verification.</b> {clean(statement, 550)} "
            f"{clean(outcome, 250) if outcome else ''}", styles))
    else:
        story.append(notice(
            "<b>Personal verification pending.</b> The numerical checks in this report were performed by "
            "AI. The student has not yet supplied a personal hand-worked check or their own explanation "
            "of its result; neither is claimed as completed. Before submission, independently inspect "
            "the saved example and confirm the calculation and interpretation.", styles))
    story.append(PageBreak())

    # Page 5 contains references only, outside the five-page body cap.
    story.append(heading("References", styles))
    access = clean(report_date)
    references = [
        '[1] J. Sanz-Cruzado, N. Droukas, and R. McCreadie, "FAR-Trans: An Investment Dataset for Financial Asset Recommendation," '
        'in <i>Fin-RecSys Workshop at IJCAI</i>, 2024, arXiv:2407.08692. '
        '<link href="https://arxiv.org/abs/2407.08692" color="#173F45">https://arxiv.org/abs/2407.08692</link>. '
        'Official data: doi:10.5525/gla.researchdata.1658. ',
        '[2] Y. Dai, J. Zhou, and T. Joachims, "Language-Based User Profiles for Recommendation," '
        '<i>arXiv preprint</i>, arXiv:2402.15623, 2024. '
        '<link href="https://arxiv.org/abs/2402.15623" color="#173F45">https://arxiv.org/abs/2402.15623</link>. ',
        '[3] F. Spadea and O. Seneviratne, "Aligning Language Models with Investor and Market Behavior for Financial Recommendations," '
        'in <i>Proc. 6th ACM International Conference on AI in Finance (ICAIF)</i>, 2025, doi:10.1145/3768292.3770399. '
        '<link href="https://arxiv.org/abs/2510.15993" color="#173F45">https://arxiv.org/abs/2510.15993</link>. ',
        f'[4] Qwen Team, <i>Qwen3-1.7B and official GGUF model cards</i>, 2025. '
        f'<link href="https://huggingface.co/{escape(short_model)}" color="#173F45">https://huggingface.co/{escape(short_model)}</link>. '
        f'Revision used: {clean(model.get("model_revision"))}. ',
        '[5] S. Jin, <i>Homework 2 - Implement and Test Your Idea; FAR-Trans Student Starter</i>, HSE University, course repository. '
        '<link href="https://github.com/dryjins/RecSys-LLMs/tree/main/project/far-trans" color="#173F45">github.com/dryjins/RecSys-LLMs/project/far-trans</link>. '
        f'Vendored starter commit: {clean(experiment.get("upstream_commit"))}. ',
        f'[6] G. Gerganov and contributors, <i>llama.cpp</i>, ggml-org, release {clean(runtime_tag)}. '
        '<link href="https://github.com/ggml-org/llama.cpp" color="#173F45">https://github.com/ggml-org/llama.cpp</link>. '
        f'Runtime commit: {clean(model.get("runtime_commit"))}. ',
        '[7] FAR-Trans Lab, <i>Profile-Aware LLM Re-Ranking for Financial Asset Recommendation</i>, '
        'Session 1 team milestone, 11 November 2026. Internal course document supplied with HW1. ',
    ]
    for reference in references:
        story.append(paragraph(reference + f"[Accessed: {access}].", styles, "ReportReference"))

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(destination), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=17 * mm,
        title="Investor Risk Profiles from an LLM: A Controlled FAR-Trans Experiment",
        author=name, subject="LLM4Rec Week 8 / Session 2 individual experiment",
        pageCompression=1,
    )

    def page_frame(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(.55)
        canvas.line(18 * mm, A4[1] - 11 * mm, A4[0] - 18 * mm, A4[1] - 11 * mm)
        canvas.setFont(regular, 7.4)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 10 * mm, "HSE University | LLM4Rec | Session 2")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, str(doc.page))
        canvas.restoreState()

    document.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
    from pypdf import PdfReader
    pdf = PdfReader(destination)
    reference_pages = [i for i, page in enumerate(pdf.pages)
                       if "References" in [line.strip() for line in (page.extract_text() or "").splitlines()[:6]]]
    if not reference_pages:
        raise ValueError("Could not verify the reference-page boundary")
    body_pages = reference_pages[0]
    if body_pages > 5:
        raise ValueError(f"Report body exceeds five pages: {body_pages}")
    print(json.dumps({"pdf": str(destination.resolve()), "pages": len(pdf.pages),
                      "body_pages": body_pages, "abstract_words": words,
                      "student_confirmation_pending": pending}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "main")
    parser.add_argument("--destination", type=Path,
                        default=ROOT / "submission" / "vorozhtsov_petr_session2_report.pdf")
    parser.add_argument("--identity", type=Path, default=None)
    parser.add_argument("--student-verification", type=Path, default=None)
    args = parser.parse_args()
    build_report(args.output_dir, args.destination, args.identity, args.student_verification)


if __name__ == "__main__":
    main()
