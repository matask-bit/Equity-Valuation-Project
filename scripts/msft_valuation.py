import os
import sys
import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


TICKER = "MSFT"
BENCHMARK = "^GSPC"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")


class DataFetchError(Exception):
    """Custom error for data fetching issues."""


def ensure_output_dir() -> None:
    """Ensure that the outputs directory exists."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_financial_statements(
    ticker: str, max_years: int = 10, fallback_years: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch annual income statement, cash flow, and balance sheet.

    Returns dataframes with years as index (int).
    """
    tk = yf.Ticker(ticker)

    income = tk.income_stmt
    cf = tk.cashflow
    bs = tk.balance_sheet

    if income is None or income.empty:
        raise DataFetchError("Income statement data not available from yfinance.")
    if cf is None or cf.empty:
        raise DataFetchError("Cash flow statement data not available from yfinance.")
    if bs is None or bs.empty:
        raise DataFetchError("Balance sheet data not available from yfinance.")

    def clean(df: pd.DataFrame) -> pd.DataFrame:
        # yfinance returns columns as periods (dates) and index as line items
        df = df.copy()
        # transpose so rows are periods, columns are items
        df = df.T
        # convert index to year
        years = df.index.year
        df.index = years
        # drop duplicate years keeping latest
        df = df[~df.index.duplicated(keep="first")]
        # sort chronologically
        df = df.sort_index()
        # limit to max_years or fallback_years
        n = max_years if df.shape[0] >= max_years else fallback_years
        df = df.tail(n)
        return df

    income_clean = clean(income)
    cf_clean = clean(cf)
    bs_clean = clean(bs)

    return income_clean, cf_clean, bs_clean


def map_first_available(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    """
    Given a dataframe with financial statement items as columns,
    return the first matching column as a Series.
    """
    lower_cols = {c.lower(): c for c in df.columns}
    for label in candidates:
        col = lower_cols.get(label.lower())
        if col and col in df.columns:
            return df[col]
    # also try partial matching
    for label in candidates:
        for col in df.columns:
            if label.lower().replace(" ", "") in col.lower().replace(" ", ""):
                return df[col]
    return None


def build_standardized_financials(
    income: pd.DataFrame, cf: pd.DataFrame, bs: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a standardized financial table with:
    revenue, operating_income, net_income, ocf, capex, fcf, cash, debt.
    Index is year.
    """
    years = sorted(set(income.index) | set(cf.index) | set(bs.index))
    fin = pd.DataFrame(index=years)

    revenue_labels = ["Total Revenue", "TotalRevenue", "Revenue"]
    op_income_labels = ["Operating Income", "OperatingIncome"]
    net_income_labels = [
        "Net Income",
        "NetIncome",
        "Net Income Common Stockholders",
        "Net Income Applicable To Common Shares",
    ]
    ocf_labels = [
        "Total Cash From Operating Activities",
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
    ]
    capex_labels = ["Capital Expenditures", "CapitalExpenditures"]
    cash_labels = [
        "Cash And Cash Equivalents",
        "Cash",
        "Cash And Short Term Investments",
    ]
    debt_labels = [
        "Total Debt",
        "Long Term Debt",
        "Long Term Debt And Capital Lease Obligation",
    ]

    fin["revenue"] = map_first_available(income, revenue_labels)
    fin["operating_income"] = map_first_available(income, op_income_labels)
    fin["net_income"] = map_first_available(income, net_income_labels)
    fin["ocf"] = map_first_available(cf, ocf_labels)
    fin["capex"] = map_first_available(cf, capex_labels)
    # FCF will be extracted directly from the raw cash flow statement via extract_fcf_from_cashflow.
    fin["fcf"] = np.nan

    fin["cash"] = map_first_available(bs, cash_labels)

    # Debt: try to sum multiple possible columns if available
    debt_series_list: List[pd.Series] = []
    lower_cols = {c.lower(): c for c in bs.columns}
    for label in debt_labels:
        col = lower_cols.get(label.lower())
        if col and col in bs.columns:
            debt_series_list.append(bs[col])
    if debt_series_list:
        debt_combined = pd.concat(debt_series_list, axis=1).sum(axis=1)
        # align to years index
        debt_combined.index = bs.index
        debt_combined.index = debt_combined.index.astype(int)
        fin["debt"] = debt_combined
    else:
        fin["debt"] = np.nan

    fin = fin.sort_index()
    return fin


def extract_fcf_from_cashflow(ticker: str) -> pd.Series:
    """
    Extract free cash flow from yfinance cashflow data using robust label matching.

    Uses:
        FCF = Operating Cash Flow - abs(CapEx)

    Ensures:
        - numeric values
        - sorted chronologically
        - roughly 4–10 years of history if available
        - saved to outputs/msft_fcf.csv
    """
    tk = yf.Ticker(ticker)
    cashflow = tk.cashflow
    if cashflow is None or cashflow.empty:
        raise ValueError("yfinance cashflow data is empty")

    print("Cashflow index labels:", list(cashflow.index))

    def find_first_matching(df: pd.DataFrame, labels: List[str]) -> Optional[pd.Series]:
        """Return the first row whose index label matches any of the provided labels."""
        index_labels = list(df.index)
        # Exact match first
        for label in labels:
            if label in df.index:
                return df.loc[label]
        # Fallback: case-insensitive / stripped-space comparison
        norm = {str(idx).replace(" ", "").lower(): idx for idx in index_labels}
        for label in labels:
            key = label.replace(" ", "").lower()
            if key in norm:
                return df.loc[norm[key]]
        return None

    operating_labels = [
        "Total Cash From Operating Activities",
        "Operating Cash Flow",
        "Cash Flow From Continuing Operating Activities",
    ]

    capex_labels = [
        "Capital Expenditures",
        "CapitalExpenditures",
    ]

    operating_cf = find_first_matching(cashflow, operating_labels)
    capex = find_first_matching(cashflow, capex_labels)

    free_cf_labels = ["Free Cash Flow", "FreeCashFlow"]
    free_cf = find_first_matching(cashflow, free_cf_labels)

    if operating_cf is None or capex is None:
        # Fallback: use Free Cash Flow row directly if available
        if free_cf is None:
            raise ValueError("Could not find operating cash flow, capex, or free cash flow in yfinance cashflow")
        fcf = pd.to_numeric(free_cf, errors="coerce")
    else:
        operating_cf = pd.to_numeric(operating_cf, errors="coerce")
        capex = pd.to_numeric(capex, errors="coerce").abs()
        fcf = operating_cf - capex
    fcf = fcf.dropna()
    fcf = fcf.sort_index()

    print("Extracted FCF:")
    print(fcf)

    # Convert to yearly series (sum by calendar year)
    try:
        idx = pd.to_datetime(fcf.index)
    except Exception:
        # If index cannot be parsed as dates, just keep as-is
        idx = fcf.index

    if hasattr(idx, "year"):
        years = pd.Index(getattr(idx, "year"), name="Year")
        fcf_by_year = pd.Series(fcf.values, index=years)
        fcf_by_year = fcf_by_year.groupby(level=0).sum()
    else:
        fcf_by_year = pd.Series(fcf.values, index=fcf.index)

    fcf_by_year = fcf_by_year.sort_index()
    # Keep a reasonable amount of history (up to last 10 years)
    if fcf_by_year.shape[0] > 10:
        fcf_by_year = fcf_by_year.tail(10)

    ensure_output_dir()
    fcf_df = pd.DataFrame({"FCF": fcf_by_year})
    fcf_df.to_csv(os.path.join(OUTPUT_DIR, "msft_fcf.csv"), index_label="Year")

    return fcf_by_year


def fetch_market_data(ticker: str) -> Dict[str, float]:
    """Fetch current price, market cap, shares outstanding, trailing PE and TTM FCF if possible."""
    tk = yf.Ticker(ticker)
    info = tk.info or {}

    price = info.get("currentPrice")
    if price is None:
        hist = tk.history(period="5d")
        if hist is None or hist.empty:
            raise DataFetchError("Unable to obtain current price from yfinance.")
        price = float(hist["Close"].iloc[-1])

    market_cap = info.get("marketCap")
    shares_out = info.get("sharesOutstanding")

    if market_cap is None and shares_out is not None:
        market_cap = price * shares_out

    if shares_out is None and market_cap is not None and price > 0:
        shares_out = market_cap / price

    trailing_pe = info.get("trailingPE")

    ttm_eps = info.get("trailingEps")
    if trailing_pe is None and ttm_eps not in (None, 0):
        trailing_pe = price / ttm_eps

    # Try to approximate TTM FCF using cashflow TTM if available
    ttm_cf = tk.cashflow
    ttm_fcf = None
    if ttm_cf is not None and not ttm_cf.empty:
        ocf = map_first_available(ttm_cf.T, ["Total Cash From Operating Activities", "Operating Cash Flow"])
        capex = map_first_available(ttm_cf.T, ["Capital Expenditures", "CapitalExpenditures"])
        if ocf is not None and capex is not None:
            ttm_fcf = float(ocf.iloc[0] - capex.iloc[0])

    return {
        "price": float(price),
        "market_cap": float(market_cap) if market_cap is not None else np.nan,
        "shares_outstanding": float(shares_out) if shares_out is not None else np.nan,
        "trailing_pe": float(trailing_pe) if trailing_pe is not None else np.nan,
        "ttm_fcf": float(ttm_fcf) if ttm_fcf is not None else np.nan,
    }


def compute_cagr(first_value: float, last_value: float, periods: int) -> Optional[float]:
    """Compute CAGR given first and last value and number of periods."""
    try:
        if first_value <= 0 or last_value <= 0 or periods <= 0:
            return None
        return (last_value / first_value) ** (1 / periods) - 1
    except Exception:
        return None


def compute_financial_metrics(fin: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
    """
    Compute margins and CAGRs.

    Returns:
        metrics_by_year: DataFrame with margins per year.
        summary: dict with revenue_cagr, fcf_cagr.
    """
    fin = fin.copy()
    fin = fin.sort_index()

    metrics = pd.DataFrame(index=fin.index)
    metrics["revenue"] = fin["revenue"]
    metrics["fcf"] = fin["fcf"]
    metrics["operating_margin"] = fin["operating_income"] / fin["revenue"]
    metrics["net_margin"] = fin["net_income"] / fin["revenue"]
    metrics["fcf_margin"] = fin["fcf"] / fin["revenue"]

    first_year, last_year = metrics.index[0], metrics.index[-1]
    periods = last_year - first_year

    revenue_cagr = compute_cagr(
        float(metrics["revenue"].iloc[0]),
        float(metrics["revenue"].iloc[-1]),
        periods,
    )
    fcf_cagr = None
    if not metrics["fcf"].isna().all():
        try:
            fcf_cagr = compute_cagr(
                float(metrics["fcf"].iloc[0]),
                float(metrics["fcf"].iloc[-1]),
                periods,
            )
        except Exception:
            fcf_cagr = None

    summary = {
        "revenue_cagr": revenue_cagr,
        "fcf_cagr": fcf_cagr,
    }

    return metrics, summary


def save_financial_summary(fin: pd.DataFrame, metrics: pd.DataFrame, summary: Dict[str, Optional[float]]) -> str:
    """Save combined financial summary to CSV and return path."""
    out = fin.copy()
    out = out.join(metrics[["operating_margin", "net_margin", "fcf_margin"]])

    ensure_output_dir()
    path = os.path.join(OUTPUT_DIR, "msft_financial_summary.csv")
    out.to_csv(path, index_label="Year")
    return path


def plot_series(
    series: pd.Series,
    title: str,
    ylabel: str,
    filename: str,
) -> str:
    """Generic line plot for a series."""
    ensure_output_dir()
    plt.figure(figsize=(8, 5))
    plt.plot(series.index, series.values, marker="o")
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.close()
    return path


def plot_margins(metrics: pd.DataFrame, filename: str) -> str:
    """Plot operating, net and FCF margins on one chart."""
    ensure_output_dir()
    plt.figure(figsize=(8, 5))
    for col, label in [
        ("operating_margin", "Operating Margin"),
        ("net_margin", "Net Margin"),
        ("fcf_margin", "FCF Margin"),
    ]:
        if col in metrics.columns:
            plt.plot(
                metrics.index,
                metrics[col].values * 100,
                marker="o",
                label=label,
            )
    plt.title("MSFT Profitability Margins")
    plt.xlabel("Year")
    plt.ylabel("Margin (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path)
    plt.close()
    return path


@dataclass
class DCFScenarioResult:
    name: str
    growth_rate: float
    discount_rate: float
    terminal_growth: float
    projected_fcfs: List[float]
    terminal_value: float
    enterprise_value: float
    equity_value: float
    intrinsic_value_per_share: float
    upside_pct: Optional[float] = None


def build_dcf_scenario(
    name: str,
    base_fcf: float,
    growth_rate: float,
    discount_rate: float,
    terminal_growth: float,
    cash: float,
    debt: float,
    shares_outstanding: float,
) -> DCFScenarioResult:
    """Build a single DCF scenario."""
    projected_fcfs: List[float] = []
    for year in range(1, 6):
        projected_fcfs.append(base_fcf * (1 + growth_rate) ** year)

    # discount factors
    dfs = [(1 / ((1 + discount_rate) ** year)) for year in range(1, 6)]
    pv_fcfs = [fcf * df for fcf, df in zip(projected_fcfs, dfs)]

    fcf5 = projected_fcfs[-1]
    if discount_rate <= terminal_growth:
        # avoid division by zero or negative denominator
        terminal_value = np.nan
        pv_terminal = np.nan
    else:
        terminal_value = fcf5 * (1 + terminal_growth) / (discount_rate - terminal_growth)
        pv_terminal = terminal_value * dfs[-1]

    enterprise_value = float(np.nansum(pv_fcfs) + (pv_terminal if not np.isnan(pv_terminal) else 0.0))
    equity_value = enterprise_value + cash - debt

    if shares_outstanding and not np.isnan(shares_outstanding) and shares_outstanding > 0:
        intrinsic = equity_value / shares_outstanding
    else:
        intrinsic = np.nan

    return DCFScenarioResult(
        name=name,
        growth_rate=growth_rate,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        projected_fcfs=projected_fcfs,
        terminal_value=terminal_value,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        intrinsic_value_per_share=intrinsic,
    )


def build_dcf_model(
    fin: pd.DataFrame,
    market_data: Dict[str, float],
    terminal_growth: float = 0.025,
) -> List[DCFScenarioResult]:
    """
    Build DCF model using latest FCF (TTM if available, otherwise latest annual).
    """
    latest_fcf = fin["fcf"].dropna().iloc[-1] if not fin["fcf"].dropna().empty else np.nan
    base_fcf = market_data.get("ttm_fcf")
    if base_fcf is None or np.isnan(base_fcf) or base_fcf == 0:
        base_fcf = float(latest_fcf)

    cash = float(fin["cash"].dropna().iloc[-1]) if not fin["cash"].dropna().empty else 0.0
    debt = float(fin["debt"].dropna().iloc[-1]) if not fin["debt"].dropna().empty else 0.0
    shares = market_data.get("shares_outstanding", np.nan)

    scenarios = [
        ("Bear", 0.04, 0.10),
        ("Base", 0.07, 0.09),
        ("Bull", 0.10, 0.08),
    ]

    results: List[DCFScenarioResult] = []
    for name, g, r in scenarios:
        res = build_dcf_scenario(
            name=name,
            base_fcf=base_fcf,
            growth_rate=g,
            discount_rate=r,
            terminal_growth=terminal_growth,
            cash=cash,
            debt=debt,
            shares_outstanding=shares,
        )
        results.append(res)

    return results


def attach_market_comparison(
    scenarios: List[DCFScenarioResult],
    current_price: float,
) -> None:
    """Compute upside/downside percentage for each scenario in-place."""
    for s in scenarios:
        if np.isnan(s.intrinsic_value_per_share) or s.intrinsic_value_per_share is None:
            s.upside_pct = None
        else:
            s.upside_pct = (s.intrinsic_value_per_share / current_price) - 1.0


def save_dcf_scenarios(
    scenarios: List[DCFScenarioResult],
    filepath: Optional[str] = None,
) -> str:
    """Save DCF scenario table to CSV and return path."""
    ensure_output_dir()
    if filepath is None:
        filepath = os.path.join(OUTPUT_DIR, "dcf_scenarios.csv")

    rows = []
    for s in scenarios:
        rows.append(
            {
                "Scenario": s.name,
                "GrowthRate": s.growth_rate,
                "DiscountRate": s.discount_rate,
                "TerminalGrowth": s.terminal_growth,
                "IntrinsicValuePerShare": s.intrinsic_value_per_share,
                "UpsidePct": s.upside_pct,
                "EnterpriseValue": s.enterprise_value,
                "EquityValue": s.equity_value,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    return filepath


def compute_beta_and_quality_metrics(
    ticker: str,
    benchmark: str,
    market_data: Dict[str, float],
    fin_summary: Dict[str, Optional[float]],
) -> Optional[str]:
    """
    Optional: compute 2y daily beta vs benchmark and save risk/quality metrics.

    Returns CSV path or None.
    """
    try:
        end = dt.date.today()
        start = end - dt.timedelta(days=365 * 2 + 10)
        tk_hist = yf.download(ticker, start=start, end=end, progress=False)
        bm_hist = yf.download(benchmark, start=start, end=end, progress=False)
        if tk_hist.empty or bm_hist.empty:
            return None

        tk_ret = tk_hist["Adj Close"].pct_change().dropna()
        bm_ret = bm_hist["Adj Close"].pct_change().dropna()
        df = pd.concat([tk_ret, bm_ret], axis=1, join="inner")
        df.columns = ["asset", "benchmark"]
        df = df.dropna()
        if df.shape[0] < 30:
            return None

        x = df["benchmark"].values
        y = df["asset"].values
        x = x - np.mean(x)
        y = y - np.mean(y)
        beta, _ = np.polyfit(x, y, 1)

        pe = market_data.get("trailing_pe", np.nan)
        market_cap = market_data.get("market_cap", np.nan)
        ttm_fcf = market_data.get("ttm_fcf", np.nan)
        if market_cap and not np.isnan(market_cap):
            if ttm_fcf and not np.isnan(ttm_fcf):
                fcf_yield = ttm_fcf / market_cap
            else:
                fcf_yield = np.nan
        else:
            fcf_yield = np.nan

        ensure_output_dir()
        path = os.path.join(OUTPUT_DIR, "risk_quality_metrics.csv")
        metrics = {
            "Beta2Y": beta,
            "TrailingPE": pe,
            "FCFYield": fcf_yield,
            "RevenueCAGR": fin_summary.get("revenue_cagr"),
            "FCFCAGR": fin_summary.get("fcf_cagr"),
        }
        pd.DataFrame([metrics]).to_csv(path, index=False)
        return path
    except Exception:
        return None


def main() -> None:
    """Run full MSFT valuation workflow."""
    ensure_output_dir()
    print(f"Running MSFT valuation as of {dt.date.today()}...")

    try:
        income, cf, bs = fetch_financial_statements(TICKER)
        fin = build_standardized_financials(income, cf, bs)
        # Robust free cash flow extraction directly from cash flow statement
        try:
            fcf_by_year = extract_fcf_from_cashflow(TICKER)
            # Align FCF to the standardized financials index (years)
            fin["fcf"] = fcf_by_year.reindex(fin.index)
        except ValueError as exc:
            # If FCF cannot be extracted at all, surface a clear data error
            raise DataFetchError(str(exc))
        metrics_by_year, fin_summary = compute_financial_metrics(fin)

        fin_summary_path = save_financial_summary(fin, metrics_by_year, fin_summary)
        print(f"Saved financial summary to {fin_summary_path}")

        # Charts
        revenue_chart = plot_series(
            fin["revenue"],
            title="MSFT Revenue (Annual)",
            ylabel="Revenue (USD)",
            filename="revenue_chart.png",
        )
        fcf_chart = plot_series(
            fin["fcf"],
            title="MSFT Free Cash Flow (Annual)",
            ylabel="Free Cash Flow (USD)",
            filename="fcf_chart.png",
        )
        margins_chart = plot_margins(metrics_by_year, filename="margins_chart.png")
        print("Saved charts:", revenue_chart, fcf_chart, margins_chart)

        # Market data
        market_data = fetch_market_data(TICKER)
        price = market_data["price"]
        market_cap = market_data["market_cap"]

        # FCF yield
        ttm_fcf = market_data.get("ttm_fcf")
        if ttm_fcf is None or np.isnan(ttm_fcf):
            latest_fcf = fin["fcf"].dropna().iloc[-1] if not fin["fcf"].dropna().empty else np.nan
            ttm_fcf = latest_fcf
        if market_cap and not np.isnan(market_cap) and market_cap != 0 and ttm_fcf and not np.isnan(ttm_fcf):
            fcf_yield = ttm_fcf / market_cap
        else:
            fcf_yield = np.nan

        # trailing PE already handled in fetch_market_data
        trailing_pe = market_data.get("trailing_pe")

        # DCF
        scenarios = build_dcf_model(fin, market_data)
        attach_market_comparison(scenarios, price)
        dcf_path = save_dcf_scenarios(scenarios)

        print(f"Saved DCF scenarios to {dcf_path}")

        # Optional beta & risk metrics
        risk_path = compute_beta_and_quality_metrics(TICKER, BENCHMARK, market_data, fin_summary)
        if risk_path:
            print(f"Saved risk and quality metrics to {risk_path}")
        else:
            print("Risk and quality metrics not available (insufficient data).")

        # Print key results
        print("\n=== Key Results ===")
        print(f"Current price: {price:.2f} USD")
        if trailing_pe and not np.isnan(trailing_pe):
            print(f"Trailing P/E: {trailing_pe:.2f}")
        if not np.isnan(fcf_yield):
            print(f"FCF Yield: {fcf_yield * 100:.2f}%")
        if fin_summary.get("revenue_cagr") is not None:
            print(f"Revenue CAGR: {fin_summary['revenue_cagr'] * 100:.2f}%")
        if fin_summary.get("fcf_cagr") is not None:
            print(f"FCF CAGR: {fin_summary['fcf_cagr'] * 100:.2f}%")

        for s in scenarios:
            iv = s.intrinsic_value_per_share
            up = s.upside_pct
            if iv is None or np.isnan(iv):
                continue
            if up is not None:
                print(
                    f"{s.name} scenario IV/share: {iv:.2f} USD "
                    f"({up * 100:.1f}% vs current price)"
                )
    except DataFetchError as e:
        print(f"Data fetch error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

