import os
import datetime as dt
from typing import List, Tuple

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
REPORT_DIR = os.path.join(BASE_DIR, "report")


def ensure_dirs() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load financial summary, FCF history, and DCF scenario tables."""
    fin_path = os.path.join(OUTPUT_DIR, "msft_financial_summary.csv")
    fcf_path = os.path.join(OUTPUT_DIR, "msft_fcf.csv")
    dcf_path = os.path.join(OUTPUT_DIR, "dcf_scenarios.csv")

    if not os.path.exists(fin_path):
        raise FileNotFoundError(f"Missing financial summary at {fin_path}. Run msft_valuation.py first.")
    if not os.path.exists(fcf_path):
        raise FileNotFoundError(f"Missing FCF history at {fcf_path}. Run msft_valuation.py first.")
    if not os.path.exists(dcf_path):
        raise FileNotFoundError(f"Missing DCF scenarios at {dcf_path}. Run msft_valuation.py first.")

    fin_df = pd.read_csv(fin_path)
    fcf_df = pd.read_csv(fcf_path)
    dcf_df = pd.read_csv(dcf_path)
    return fin_df, fcf_df, dcf_df


def load_charts() -> Tuple[str, str, str]:
    """Return paths to revenue, FCF and margins charts."""
    rev = os.path.join(OUTPUT_DIR, "revenue_chart.png")
    fcf = os.path.join(OUTPUT_DIR, "fcf_chart.png")
    margins = os.path.join(OUTPUT_DIR, "margins_chart.png")
    for p in (rev, fcf, margins):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing chart {p}. Run msft_valuation.py first.")
    return rev, fcf, margins


def build_business_overview(styles) -> List:
    story = []
    story.append(Paragraph("Business Overview", styles["Heading1"]))
    story.append(Spacer(1, 0.15 * inch))

    bullets = [
        "Microsoft Corporation is a global technology leader with diversified revenue streams.",
        "Core segments include Productivity and Business Processes (Office, LinkedIn, Dynamics).",
        "Intelligent Cloud segment covers Azure, server products, and enterprise services.",
        "More Personal Computing includes Windows, Surface, gaming (Xbox), and search advertising.",
        "Business model is characterized by recurring subscription revenue and strong free cash flow generation.",
    ]

    for b in bullets:
        story.append(Paragraph(f"&bull; {b}", styles["BodyText"]))
        story.append(Spacer(1, 0.05 * inch))

    story.append(Spacer(1, 0.2 * inch))
    return story


def df_to_table(df: pd.DataFrame, styles, max_rows: int = 10) -> Table:
    """Convert a DataFrame into a styled ReportLab table."""
    df_display = df.head(max_rows).copy()
    data = [list(df_display.columns)] + df_display.values.tolist()
    table = Table(data, hAlign="LEFT")
    style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ]
    )
    table.setStyle(style)
    return table


def valuation_conclusion(dcf_df: pd.DataFrame, current_price: float) -> str:
    """Generate text conclusion based on base scenario upside."""
    base_row = dcf_df[dcf_df["Scenario"] == "Base"]
    if base_row.empty:
        return "Unable to determine valuation conclusion because the Base scenario is missing."

    upside = base_row["UpsidePct"].iloc[0]
    if pd.isna(upside):
        return "Unable to determine valuation conclusion due to missing upside estimate."

    pct = upside * 100

    if pct > 15:
        verdict = "undervalued"
    elif pct < -15:
        verdict = "overvalued"
    else:
        verdict = "fairly valued"

    return (
        f"Based on the Base DCF scenario, Microsoft appears {verdict}. "
        f"The intrinsic value estimate implies approximately {pct:.1f}% "
        f"{'upside' if pct >= 0 else 'downside'} relative to the current market price of "
        f"${current_price:.2f} per share."
    )


def extract_current_price_and_upside(dcf_df: pd.DataFrame) -> float:
    """
    Since the CSV does not store current price, approximate it by backing out
    from the upside of the Base scenario if possible. Otherwise default to NaN.

    upside = (IV / price) - 1 -> price = IV / (1 + upside)
    """
    base_row = dcf_df[dcf_df["Scenario"] == "Base"]
    if base_row.empty:
        return float("nan")

    iv = base_row["IntrinsicValuePerShare"].iloc[0]
    up = base_row["UpsidePct"].iloc[0]
    if pd.isna(iv) or pd.isna(up) or (1 + up) == 0:
        return float("nan")
    return float(iv / (1 + up))


def build_pdf() -> str:
    """Generate the final MSFT valuation PDF report."""
    ensure_dirs()
    fin_df, fcf_df, dcf_df = load_data()
    rev_chart, fcf_chart, margins_chart = load_charts()

    report_path = os.path.join(REPORT_DIR, "MSFT_Valuation_Report_Final.pdf")
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    styles = getSampleStyleSheet()

    story: List = []

    # Title page
    title = "Microsoft Equity Valuation Report"
    subtitle = "Discounted Cash Flow Analysis"
    prepared_by = "Prepared by: [Your Name]"
    date_str = dt.date.today().strftime("%B %d, %Y")
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(subtitle, styles["Heading2"]))
    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph(prepared_by, styles["Heading3"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(f"Date: {date_str}", styles["Heading3"]))
    story.append(Spacer(1, 3 * inch))
    story.append(Paragraph("Generated by Equity-Valuation-Project", styles["BodyText"]))
    story.append(PageBreak())

    # Executive Summary
    story.append(Paragraph("Executive Summary", styles["Heading1"]))
    story.append(Spacer(1, 0.2 * inch))
    current_price = extract_current_price_and_upside(dcf_df)
    base_row = dcf_df[dcf_df["Scenario"] == "Base"]
    if not base_row.empty:
        base_iv = base_row["IntrinsicValuePerShare"].iloc[0]
        base_up = base_row["UpsidePct"].iloc[0]
    else:
        base_iv = float("nan")
        base_up = float("nan")
    summary_lines = []
    if not pd.isna(base_iv):
        summary_lines.append(
            f"The Base-case DCF scenario implies an intrinsic value of approximately "
            f"<b>${base_iv:.2f}</b> per share for Microsoft."
        )
    if not pd.isna(base_up) and not pd.isna(current_price):
        direction = "upside" if base_up >= 0 else "downside"
        summary_lines.append(
            f"This corresponds to an estimated {direction} of about {base_up * 100:.1f}% "
            f"versus the inferred current market price of roughly ${current_price:.2f} per share."
        )
    summary_lines.append(
        "Microsoft continues to generate strong revenue growth and robust free cash flow, "
        "supported by a diversified business model across productivity software, cloud, and personal computing."
    )
    for line in summary_lines:
        story.append(Paragraph(line, styles["BodyText"]))
        story.append(Spacer(1, 0.1 * inch))
    story.append(PageBreak())

    # Financial Performance
    story.append(Paragraph("Financial Performance", styles["Heading1"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Revenue Trend", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Image(rev_chart, width=6 * inch, height=3 * inch))
    story.append(Spacer(1, 0.25 * inch))

    # Free Cash Flow Analysis
    story.append(Paragraph("Free Cash Flow Analysis", styles["Heading1"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Free Cash Flow Trend", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Image(fcf_chart, width=6 * inch, height=3 * inch))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Historical Free Cash Flow (select years)", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(df_to_table(fcf_df, styles, max_rows=10))
    story.append(PageBreak())

    # Profitability Analysis (margins chart + key metrics table)
    story.append(Paragraph("Profitability Analysis", styles["Heading1"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Profitability Margins", styles["Heading2"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Image(margins_chart, width=6 * inch, height=3 * inch))
    story.append(Spacer(1, 0.25 * inch))

    # Key metrics table: use recent years and margin columns
    margin_cols = [c for c in fin_df.columns if "margin" in c.lower()]
    fm = fin_df[["Year"] + margin_cols].copy() if margin_cols else fin_df.copy()
    story.append(Paragraph("Key Profitability Metrics", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(df_to_table(fm, styles, max_rows=8))
    story.append(PageBreak())

    # DCF Assumptions section
    story.append(Paragraph("DCF Assumptions", styles["Heading1"]))
    story.append(Spacer(1, 0.2 * inch))
    dcf_text = (
        "The discounted cash flow (DCF) model projects Microsoft's free cash flows "
        "over a five-year horizon using scenario-specific growth and discount rate "
        "assumptions. A terminal value is calculated using a perpetual growth model "
        "with a 2.5% terminal growth rate. Enterprise value is derived from the "
        "present value of projected free cash flows and terminal value, after which "
        "net cash (cash minus debt) is added to arrive at equity value. Dividing by "
        "shares outstanding yields an intrinsic value per share for each scenario."
    )
    story.append(Paragraph(dcf_text, styles["BodyText"]))
    story.append(PageBreak())

    # Scenario Valuation Table
    story.append(Paragraph("Scenario Valuation Table", styles["Heading1"]))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("DCF Scenario Summary", styles["Heading2"]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(df_to_table(dcf_df, styles, max_rows=10))
    story.append(PageBreak())

    # Intrinsic Value vs Market Price
    story.append(Paragraph("Intrinsic Value vs Market Price", styles["Heading1"]))
    story.append(Spacer(1, 0.2 * inch))

    if not pd.isna(current_price):
        story.append(
            Paragraph(
                f"Approximate current market price (backed out from DCF scenarios): "
                f"<b>${current_price:.2f}</b> per share.",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 0.15 * inch))

    # Summary of Bear/Base/Bull intrinsic values
    summary_rows = []
    for _, row in dcf_df.iterrows():
        scen = row.get("Scenario", "")
        iv = row.get("IntrinsicValuePerShare", float("nan"))
        up = row.get("UpsidePct", float("nan"))
        if pd.isna(iv):
            continue
        up_pct = "" if pd.isna(up) else f"{up * 100:.1f}%"
        summary_rows.append((scen, f"${iv:.2f}", up_pct))

    if summary_rows:
        table = Table(
            [["Scenario", "Intrinsic Value / Share", "Implied Upside"]] + summary_rows,
            hAlign="LEFT",
        )
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 0.25 * inch))

    # Investment Conclusion
    story.append(Paragraph("Investment Conclusion", styles["Heading1"]))
    story.append(Spacer(1, 0.2 * inch))

    if pd.isna(current_price):
        concl_text = (
            "A precise investment conclusion cannot be drawn because the current "
            "market price could not be inferred from the saved outputs. Please "
            "re-run the valuation script to regenerate inputs."
        )
    else:
        concl_text = valuation_conclusion(dcf_df, current_price)
    story.append(Paragraph(concl_text, styles["BodyText"]))

    doc.build(story)
    return report_path


def main() -> None:
    ensure_dirs()
    try:
        path = build_pdf()
        print(f"Generated report at {path}")
    except Exception as exc:
        print(f"Failed to generate report: {exc}")


if __name__ == "__main__":
    main()

