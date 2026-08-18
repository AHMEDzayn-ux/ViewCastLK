"""Build the ViewCastLK team briefing PDF for the SRS and Design documents."""
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Spacer, Table, TableStyle, KeepTogether)

OUT = os.path.join(os.path.dirname(__file__), "..", "Deliverables",
                   "ViewCastLK_Team_Brief.pdf")

NAVY = colors.HexColor("#1F3A5F")
RED = colors.HexColor("#A02A2A")
GREY = colors.HexColor("#555555")
LIGHT = colors.HexColor("#EDF1F6")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=13, textColor=NAVY,
                    spaceBefore=11, spaceAfter=5, leading=15)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=10.5, textColor=NAVY,
                    spaceBefore=8, spaceAfter=3, leading=12.5)
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontSize=8.9, leading=12.4,
                      alignment=TA_LEFT, spaceAfter=4)
BULLET = ParagraphStyle("BULLET", parent=BODY, leftIndent=10, bulletIndent=2,
                        spaceAfter=2.5)
WARN = ParagraphStyle("WARN", parent=BODY, textColor=RED, fontSize=8.9)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=7.9, textColor=GREY)
TITLE = ParagraphStyle("TITLE", parent=ss["Title"], fontSize=19, textColor=NAVY,
                       spaceAfter=2, leading=22)

S = []


def h1(t): S.append(Paragraph(t, H1))
def h2(t): S.append(Paragraph(t, H2))
def p(t): S.append(Paragraph(t, BODY))
def warn(t): S.append(Paragraph(t, WARN))
def small(t): S.append(Paragraph(t, SMALL))
def gap(h=4): S.append(Spacer(1, h))
def bullets(items):
    for i in items:
        S.append(Paragraph(i, BULLET, bulletText="•"))


def table(rows, widths, header=True, size=8.2):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONTSIZE", (0, 0), (-1, -1), size),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C8CFD8")),
          ("LEFTPADDING", (0, 0), (-1, -1), 4),
          ("RIGHTPADDING", (0, 0), (-1, -1), 4),
          ("TOPPADDING", (0, 0), (-1, -1), 3),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY),
               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st))
    S.append(t)
    gap(5)


def cell(txt, style=None):
    return Paragraph(txt, style or ParagraphStyle("c", parent=BODY, fontSize=8.2,
                                                  leading=10.4, spaceAfter=0))


# ============================================================ TITLE
S.append(Paragraph("ViewCastLK — Team Briefing", TITLE))
p("<b>Read before writing the SRS or the System Architecture &amp; Design document.</b> "
  "This is the current, verified state of the project. Several earlier documents "
  "describe the system incorrectly; those errors are listed in section 6 so they "
  "are not carried forward.")
small("Prepared 27 July 2026 &nbsp;·&nbsp; Group 2 &nbsp;·&nbsp; Project P07 &nbsp;·&nbsp; "
      "Data current as of 27 July 2026, 20:04 UTC")

# ============================================================ 1
h1("1. Project at a glance")
table([
    [cell("<b>Item</b>"), cell("<b>Value</b>")],
    [cell("System"), cell("ViewCastLK — A Data-Driven Tool for Forecasting Viewership "
                          "Trends of Sri Lankan YouTube Content")],
    [cell("Group / Project"), cell("Group 2 &nbsp;|&nbsp; Project P07")],
    [cell("Team"), cell("AHAMED M.J.S (230023E) · AHAMED M.U.A (230025L) · AHMEDH M.R.R (230027U)")],
    [cell("Mentor"), cell("Dr. Chathuranga Hettiarachchi &nbsp;·&nbsp; TA: Muthumala V.D.W.")],
    [cell("Repository"), cell("github.com/AHMEDzayn-ux/ViewCastLK")],
    [cell("Database"), cell("Supabase PostgreSQL, ap-south-1 (Mumbai)")],
], [32 * mm, 145 * mm])

# ============================================================ 2
h1("2. What the system actually does")
p("ViewCastLK forecasts how a YouTube video will perform <b>before it is published</b>. "
  "A creator enters planned video details — category, duration, intended publication "
  "day and time — and receives a predicted view count at <b>days 7, 14, 21 and 30</b> "
  "after publication, together with publishing recommendations derived from Sri Lankan "
  "publishing patterns.")
p("To make that possible the project runs a continuous data collection pipeline that "
  "tracks Sri Lankan YouTube channels, discovers their new uploads, and records "
  "view/like/comment counts repeatedly over time, building a day-by-day engagement "
  "history for every tracked video.")

# ============================================================ 3
h1("3. The novelty claim — state this correctly")
warn("This is the single most important thing to get right in both documents. "
     "A previous batch built a very similar system; an examiner may open their "
     "repository.")
gap(2)
h2("What is NOT novel")
p("Forecasting YouTube view trajectories for Sri Lankan content, using time-series "
  "engagement data, at 7- and 30-day horizons, with gradient-boosted trees. A prior "
  "batch project (ViewTrendsSL, github.com/View-Rush) already did all of this.")
h2("What IS novel")
bullets([
    "<b>Pre-publication forecasting.</b> The prior system requires the video to already "
    "exist — it takes observed daily view counts from day 1 to day N as input and "
    "forecasts forward. ViewCastLK predicts from planned metadata alone, with no "
    "engagement data, because the video does not yet exist. That is a materially "
    "harder problem and the recognised gap in the literature.",
    "<b>Verified Sri Lankan collection.</b> Channels are confirmed as Sri Lankan via "
    "the channel's declared country before being tracked, rather than assumed from a "
    "region-filtered search. Section 10 shows why this matters.",
    "<b>Four forecast horizons</b> (7/14/21/30) rather than two, at six-hourly "
    "sampling resolution.",
])

# ============================================================ 4
h1("4. System architecture as built")
table([
    [cell("<b>Layer</b>"), cell("<b>How it actually works</b>")],
    [cell("<b>Channel discovery</b>"),
     cell("A curated roster of channel handles/IDs held in <font face='Courier'>channel_handles.txt</font>. "
          "Grown occasionally by broad keyword searches (<font face='Courier'>search.list</font>, "
          "100 units/call) across categories; each candidate is then verified via "
          "<font face='Courier'>channels.list</font> and kept only if its declared country is LK. "
          "<b>This is channel-roster-based discovery, not video-first.</b>")],
    [cell("<b>Upload discovery</b>"),
     cell("For each tracked channel, <font face='Courier'>playlistItems.list</font> (1 unit) walks the "
          "channel's uploads playlist, stopping as soon as it reaches a video older than the "
          "cutoff. A 26-hour look-back window absorbs delayed or skipped runs.")],
    [cell("<b>Statistics collection</b>"),
     cell("<font face='Courier'>videos.list</font> (1 unit per call, batched 50 videos at a time) "
          "returns view/like/comment counts plus metadata. Every video inside its 60-day "
          "tracking window is re-snapshotted on every run.")],
    [cell("<b>Scheduling</b>"),
     cell("GitHub Actions cron, <b>four runs per day at six-hourly intervals</b> — not daily. "
          "Two are full runs (refresh channel stats + discover + snapshot); two are "
          "discovery-only runs that skip the channel refresh to save quota.")],
    [cell("<b>Storage</b>"),
     cell("Supabase PostgreSQL. Five tables: <font face='Courier'>channels</font>, "
          "<font face='Courier'>channel_snapshots</font>, <font face='Courier'>videos</font>, "
          "<font face='Courier'>video_snapshots</font>, <font face='Courier'>video_categories</font>. "
          "Schema managed as versioned migrations via the Supabase CLI.")],
    [cell("<b>Data model</b>"),
     cell("<b>Identity vs snapshot split.</b> Identity tables hold fields that do not change "
          "(title, category, duration) and are written once per entity. Snapshot tables hold "
          "fields that change (view/like/comment/subscriber counts), one row per entity per "
          "run, keyed on (id, captured_at). The snapshot tables are the time series.")],
], [30 * mm, 147 * mm])

# ============================================================ 5
h1("5. Current data status")
table([
    [cell("<b>Metric</b>"), cell("<b>Value</b>")],
    [cell("Collection running since"), cell("17 July 2026, continuously")],
    [cell("Completed poll runs"), cell("30")],
    [cell("Channels tracked"), cell("1,282 (all verified country = LK)")],
    [cell("Videos captured"), cell("8,115")],
    [cell("Video snapshots"), cell("141,074")],
    [cell("Channel snapshots"), cell("29,055")],
    [cell("Videos with a mature day-7 label"), cell("2,265")],
    [cell("Database size"), cell("50 MB of the 500 MB free tier (~4.7 MB/day)")],
    [cell("Daily quota usage"), cell("~8,000 of 10,000 units")],
], [62 * mm, 115 * mm])
p("Day-14 labels begin maturing from early August, day-21 from mid-August and day-30 "
  "from mid-to-late August. Collection is <b>continuous for the duration of the project</b> "
  "— there is no fixed 40-day window.")

# ============================================================ 6
h1("6. Corrections — do not repeat these")
warn("Each of these appeared in an earlier version of the feasibility study and is wrong. "
     "The feasibility study has since been corrected; the SRS and design document must "
     "not reintroduce them.")
gap(2)
table([
    [cell("<b>Incorrect statement</b>"), cell("<b>Correct fact</b>")],
    [cell("“video-first discovery approach”"),
     cell("Channel-roster-based discovery. A verified list of channels is tracked; their "
          "uploads playlists are polled for new videos.")],
    [cell("“daily polling” / “daily snapshots” / “reliable daily execution”"),
     cell("Four times daily, at six-hourly intervals.")],
    [cell("“search.list reserved for one-time <i>video</i> discovery”"),
     cell("search.list is used occasionally for <i>channel</i> discovery. Routine collection "
          "uses only 1-unit endpoints.")],
    [cell("“40-day collection window”"),
     cell("Collection is continuous for the project duration, with model retraining at "
          "checkpoints as labels mature.")],
    [cell("“GitHub Actions free tier provides 2,000 minutes/month”"),
     cell("Workflow minutes are unlimited for public repositories, which this is.")],
    [cell("“the workflow has been running stably since deployment”"),
     cell("Several deployment issues occurred and were resolved. No data was lost, because "
          "writes are idempotent and the 26-hour look-back covers missed runs.")],
    [cell("“channels … which almost completely viewed by Sri Lankan audience”"),
     cell("Unsupportable. The API does not expose viewer country. Collection is scoped to "
          "Sri Lanka–based channels as a <b>documented scoping assumption</b>, not a claim "
          "about who watches.")],
    [cell("“The forecasting engine uses XGBoost”"),
     cell("XGBoost is the primary <i>candidate</i>. It has not yet been benchmarked against "
          "alternatives or the naive baseline.")],
    [cell("“covering all fifteen standard categories”"),
     cell("14 of 15 categories are currently represented. Category coverage depends on what "
          "tracked channels actually publish.")],
    [cell("Dashboard “tracks historical prediction-versus-actual accuracy”"),
     cell("Not achievable as described — forecasts are for planned videos the user never "
          "links back to a real upload. Report <b>backtested accuracy on held-out historical "
          "data</b> instead.")],
    [cell("Made-for-kids as a headline user input"),
     cell("Collected for all videos, but only 0.27% are true. Near-useless as a feature and "
          "odd as a prominent form field.")],
], [62 * mm, 115 * mm])

# ============================================================ 7
h1("7. Model plan — decided vs open")
h2("Decided")
bullets([
    "Targets: view count at days 7, 14, 21, 30, predicted from pre-publication metadata only.",
    "Features: category, duration, planned publish day/time (cyclically encoded), "
    "channel historical performance, lightweight title features.",
    "Primary metric MAPE, supported by R², MAE and RMSE, compared against a "
    "<b>naive category-average baseline</b>.",
    "Prediction intervals via quantile regression (separate lower/median/upper models).",
    "Outlier handling: 3-sigma filtering, log-transformed targets.",
])
h2("Open — do not state these as settled")
bullets([
    "Whether XGBoost, LightGBM or an ensemble of both performs best.",
    "Whether to train separate models for Shorts and long-form video. Prior work did; "
    "we should evaluate it.",
    "Whether to include multimodal features (title embeddings, thumbnail metrics).",
    "Whether pre-training on an external dataset is needed (see section 10).",
])
warn("<b>Expectation-setting:</b> pre-publication forecasting is harder than forecasting "
     "from observed early engagement, and published benchmarks mostly do the latter. Our "
     "MAPE will likely look worse than those numbers. Evaluate against the naive baseline "
     "and say this explicitly, or a good result will read as a poor one.")

# ============================================================ 8
h1("8. Hosting and deployment")
p("Dashboard on <b>Cloudflare Pages</b> (free tier, unlimited requests and bandwidth, no "
  "cold starts). The prediction service is exposed as a lightweight endpoint, delivered "
  "either as a Cloudflare Worker executing an exported model artefact, or as a small "
  "service on an equivalent free-tier platform. Total infrastructure cost remains zero.")

# ============================================================ 9
h1("9. Platform quirks discovered the hard way")
bullets([
    "<b>search.list never returns more than ~500 results</b> for one query, regardless of "
    "the totalResults figure it reports, and costs 100 units per page. Many distinct "
    "queries beat deep pagination.",
    "<b>A channel's declared country is often absent.</b> Reliable when present, but far "
    "from universal — high precision, incomplete recall.",
    "<b>defaultAudioLanguage is not trustworthy</b> — a Sinhala-titled, Sinhala-described "
    "video was tagged “en”. Use Unicode script ranges on the title instead.",
    "<b>Live videos</b> report no usable duration (P0D) and concurrent viewers only while "
    "actually live.",
    "<b>Private, deleted and scheduled videos</b> appear in uploads playlists without a "
    "publish date, and crashed the pipeline once before being handled.",
    "<b>Uploads playlists are newest-first</b>, which is what makes incremental discovery cheap.",
    "<b>Quota resets at midnight US Pacific</b>, not local midnight.",
])

# ============================================================ 10
h1("10. Prior work — what exists, and what we found")
p("A previous batch produced <b>ViewTrendsSL</b> and <b>View-Rush</b> "
  "(github.com/View-Rush), plus public datasets on HuggingFace and Kaggle. "
  "ViewTrendsSL is a working system: LightGBM+XGBoost hybrid models, 7-day and 30-day "
  "horizons, separate models for Shorts and long-form, title and thumbnail features, "
  "confidence bounds, served via FastAPI. It collects hourly, using rotation across "
  "multiple API keys.")
gap(2)
warn("<b>Finding worth reporting.</b> Their public dataset labels every row "
     "<font face='Courier'>country = LK</font>, but that label comes from a region-filtered "
     "search rather than from verifying each channel. Every one of the 41,127 channels in "
     "their dataset was queried against the YouTube API for its declared country:")
gap(2)
table([
    [cell("<b>Declared country of the 41,127 channels</b>"), cell("<b>Count</b>"), cell("<b>Share</b>")],
    [cell("Verified Sri Lanka (LK)"), cell("8,888"), cell("21.6%")],
    [cell("Explicitly another country"), cell("16,768"), cell("40.8%")],
    [cell("&nbsp;&nbsp;&nbsp;of which India"), cell("8,549"), cell("20.8%")],
    [cell("&nbsp;&nbsp;&nbsp;of which United States"), cell("3,222"), cell("7.8%")],
    [cell("No country declared"), cell("12,306"), cell("29.9%")],
    [cell("Deleted or terminated"), cell("3,165"), cell("7.7%")],
], [110 * mm, 33 * mm, 34 * mm])
p("Barely a fifth of a dataset published as Sri Lankan is verifiably Sri Lankan, and "
  "Indian channels are almost as numerous as Sri Lankan ones. This is precisely the "
  "failure mode our proposal's scoping assumption was written to avoid — Sri Lankan "
  "audiences watch a great deal of Indian content, so a region-filtered search returns it. "
  "Our verify-before-tracking approach is therefore a genuine methodological contribution "
  "and should be presented as one.")
gap(2)
h2("What remains usable")
p("Filtering their data to LK-verified channels leaves <b>112,447 videos</b>, which is a "
  "legitimate pre-training source while our own labels mature (Apache 2.0 licence — "
  "attribution required):")
table([
    [cell("<b>Horizon</b>"), cell("<b>Prior dataset, LK-verified</b>"), cell("<b>Our own data today</b>")],
    [cell("Day 7"), cell("19,449"), cell("2,265")],
    [cell("Day 14"), cell("1,145"), cell("~30")],
    [cell("Day 21"), cell("1,038"), cell("0")],
    [cell("Day 30"), cell("1,364"), cell("0")],
], [40 * mm, 70 * mm, 67 * mm])
p("Caveats: their observations are dated September–October 2025, ages are recorded only "
  "to whole days, and there is one row per video so no trajectory shape. It is suitable "
  "for pre-training and for establishing a working model before our own day-14 to day-30 "
  "labels mature — not as the final training set. Our own data stays the held-out "
  "evaluation set because it has precise ages and point-in-time channel features.")

# ============================================================ 11
h1("11. Scope boundaries")
bullets([
    "<b>In scope:</b> publicly available channel and video data from Sri Lanka–verified "
    "channels; pre-publication forecasting; a web dashboard with forecasts and publishing "
    "recommendations.",
    "<b>Out of scope:</b> viewer-level data, audience demographics, private channel "
    "analytics, and any claim about which country a video's views originate from — the API "
    "does not expose it.",
    "<b>Dropped:</b> Social Blade is no longer a data source and should not appear in the "
    "SRS or design document as one.",
])

# ============================================================ 12
h1("12. Ownership and key dates")
table([
    [cell("<b>Area</b>"), cell("<b>Owner</b>")],
    [cell("Data collection pipeline, database, automation, prediction API, deployment, "
          "data cleaning and EDA, testing document"), cell("AHAMED M.J.S (230023E)")],
    [cell("Dashboard front end, wireframes, database backup, architecture document, "
          "usability validation, final report"), cell("AHAMED M.U.A (230025L)")],
    [cell("Feasibility study, SRS, feature engineering, model training and evaluation, "
          "model export, insight and accuracy visualisations, demo video"),
     cell("AHMEDH M.R.R (230027U)")],
], [122 * mm, 55 * mm])
table([
    [cell("<b>Date</b>"), cell("<b>Deliverable</b>")],
    [cell("Thursday this week"), cell("SRS &nbsp;·&nbsp; System Architecture &amp; Design")],
    [cell("21 August 2026"), cell("Mid-evaluation — deployed end-to-end MVP required")],
    [cell("20 September 2026"), cell("Testing &amp; Evaluation document")],
    [cell("2 October 2026"), cell("Final evaluation")],
    [cell("9 October 2026"), cell("Final report")],
], [45 * mm, 132 * mm])

gap(4)
small("Sources: live Supabase database; ViewCastLK repository; github.com/View-Rush; "
      "HuggingFace dataset madhushankhades/Sri-Lankan-YouTube-Channel-Data (Apache 2.0); "
      "YouTube Data API v3 verification queries run 26–27 July 2026.")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(GREY)
    canvas.drawString(18 * mm, 10 * mm, "ViewCastLK — Team Briefing · 27 July 2026")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=15 * mm, bottomMargin=16 * mm,
                      title="ViewCastLK — Team Briefing")
frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])
doc.build(S)
print("saved", os.path.abspath(OUT))
