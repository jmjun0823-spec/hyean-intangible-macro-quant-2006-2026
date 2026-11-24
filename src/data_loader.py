from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import yfinance as yf

from config.universe_config import CONFIG


def _empty_panel_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "year",
            "ticker",
            "market",
            "sector_type",
            "market_cap",
            "book_equity",
            "total_debt",
            "revenue",
            "gross_profit",
            "operating_cash_flow",
            "capex_tangible",
            "rd_expense",
            "sga_expense",
            "intangible_capex_share",
            "net_fund_flow_billions",
            "impairment_loss",
            "volatility_annual",
        ]
    )


def load_or_generate_panel(
    cache_path: Optional[Path] = None,
    use_live_prices: bool = False,
    refresh_cache: bool = False,
    raw_override_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    2006~2026년 장기 패널 데이터를 로드하거나 생성합니다.
    raw_override_df가 주어지면 빈 데이터/결측을 안전하게 정규화해 반환합니다.
    """
    if raw_override_df is not None:
        if raw_override_df.empty:
            return _empty_panel_frame()
        norm = raw_override_df.copy()
        norm["date"] = pd.to_datetime(norm["date"])
        if "year" not in norm.columns:
            norm["year"] = norm["date"].dt.year
        if "total_debt" not in norm.columns:
            norm["total_debt"] = norm.get("book_equity", pd.Series(10.0, index=norm.index)) * 0.4
        return norm.sort_values(["ticker", "date"]).reset_index(drop=True)

    cache_file = cache_path or Path("data/panel_dataset.parquet")
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and not refresh_cache:
        return pd.read_parquet(cache_file)

    rng = np.random.default_rng(20261123)
    years = np.arange(CONFIG.start_year, CONFIG.end_year + 1)
    dates = [pd.Timestamp(f"{y}-12-31") for y in years]

    # (market, sector_type, base_rev, rd_ratio, sga_ratio, capex_t_ratio, intang_share, flow_base, drift, debt_ratio)
    profiles = {
        "MSFT": ("US", "intangible_ai_ip", 65.0, 0.14, 0.22, 0.06, 0.76, 8.5, 0.16, 0.25),
        "NVDA": ("US", "intangible_ai_ip", 12.0, 0.22, 0.12, 0.04, 0.82, 11.0, 0.34, 0.15),
        "GOOGL": ("US", "intangible_ai_ip", 55.0, 0.16, 0.18, 0.09, 0.72, 7.2, 0.18, 0.12),
        "META": ("US", "intangible_ai_ip", 25.0, 0.24, 0.16, 0.11, 0.74, 5.8, 0.15, 0.18),
        "AAPL": ("US", "intangible_ai_ip", 80.0, 0.07, 0.10, 0.04, 0.68, 9.4, 0.19, 0.45),
        "CAT": ("US", "traditional_capex", 45.0, 0.03, 0.13, 0.07, 0.24, -1.8, 0.07, 1.35),
        "DE": ("US", "traditional_capex", 28.0, 0.04, 0.11, 0.06, 0.26, -1.2, 0.08, 1.40),
        "HON": ("US", "traditional_capex", 35.0, 0.04, 0.14, 0.05, 0.28, -0.9, 0.07, 0.85),
        "XOM": ("US", "traditional_capex", 240.0, 0.01, 0.05, 0.11, 0.18, -2.5, 0.04, 0.30),
        "005930.KS": ("KR_KOSPI", "intangible_ai_ip", 110.0, 0.09, 0.15, 0.12, 0.58, 4.2, 0.09, 0.10),
        "000660.KS": ("KR_KOSPI", "intangible_ai_ip", 18.0, 0.11, 0.08, 0.15, 0.61, 3.6, 0.12, 0.55),
        "035420.KS": ("KR_KOSPI", "intangible_ai_ip", 6.0, 0.18, 0.19, 0.05, 0.75, 2.1, 0.11, 0.20),
        "051910.KS": ("KR_KOSPI", "traditional_capex", 22.0, 0.04, 0.11, 0.09, 0.31, -0.5, 0.06, 0.65),
    }

    rows = []
    for ticker, p in profiles.items():
        market, stype, base_rev, rd_r, sga_r, cap_r, int_sh, flow_b, drift, debt_r = p
        rev = base_rev
        mcap = base_rev * (2.6 if stype == "intangible_ai_ip" else 1.2)
        book_eq = mcap * (0.32 if stype == "intangible_ai_ip" else 0.68)
        debt = book_eq * debt_r

        for dt in dates:
            yr = int(dt.year)
            cycle_shock = -0.14 if yr == 2008 else (-0.12 if yr == 2022 else 0.0)
            ai_wave = max(0.0, (yr - 2016) * 0.018) if stype == "intangible_ai_ip" else 0.0
            growth = drift + ai_wave + cycle_shock + float(rng.normal(0.0, 0.06))

            rev = max(1.0, rev * (1.0 + max(-0.25, growth * 0.75)))
            gp_margin = (0.58 if stype == "intangible_ai_ip" else 0.24) + float(rng.normal(0.0, 0.02))
            gp = rev * float(np.clip(gp_margin, 0.10, 0.85))

            rd_rate_yr = rd_r + (0.03 if yr >= 2020 and stype == "intangible_ai_ip" else 0.0) + float(rng.normal(0.0, 0.01))
            rd = rev * float(np.clip(rd_rate_yr, 0.005, 0.40))
            sga = rev * float(np.clip(sga_r + float(rng.normal(0.0, 0.01)), 0.03, 0.45))
            capex_t = rev * float(np.clip(cap_r + float(rng.normal(0.0, 0.015)), 0.01, 0.25))

            trend_share = float(
                np.clip(int_sh + (yr - CONFIG.start_year) * (0.012 if stype == "intangible_ai_ip" else 0.004), 0.15, 0.90)
            )
            ocf = gp * (0.68 if stype == "intangible_ai_ip" else 0.52) - 0.35 * rd + float(rng.normal(0.0, rev * 0.02))

            impairment = 0.0
            if rng.random() < (0.14 if stype == "intangible_ai_ip" else 0.06):
                impairment = rev * float(rng.uniform(0.01, 0.07))

            mcap = max(2.0, mcap * (1.0 + growth + cycle_shock * 0.5))
            book_eq = max(1.0, book_eq * (1.0 + max(-0.15, growth * 0.45)) - impairment * 0.6)
            debt = max(0.5, debt * (1.0 + max(-0.10, growth * 0.35 + float(rng.normal(0.0, 0.04)))))
            flow = flow_b * (1.0 + max(0, yr - 2015) * 0.12) + float(rng.normal(0.0, 1.5))
            vol = (
                (0.28 if stype == "intangible_ai_ip" else 0.19)
                + (0.08 if yr in (2008, 2020, 2022) else 0.0)
                + abs(float(rng.normal(0.0, 0.03)))
            )

            rows.append({
                "date": dt,
                "year": yr,
                "ticker": ticker,
                "market": market,
                "sector_type": stype,
                "market_cap": round(mcap, 4),
                "book_equity": round(book_eq, 4),
                "total_debt": round(debt, 4),
                "revenue": round(rev, 4),
                "gross_profit": round(gp, 4),
                "operating_cash_flow": round(ocf, 4),
                "capex_tangible": round(capex_t, 4),
                "rd_expense": round(rd, 4),
                "sga_expense": round(sga, 4),
                "intangible_capex_share": round(trend_share, 4),
                "net_fund_flow_billions": round(flow, 4),
                "impairment_loss": round(impairment, 4),
                "volatility_annual": round(vol, 4),
            })

    df = pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)

    if use_live_prices:
        try:
            bench = yf.download(
                list(CONFIG.market_benchmarks.values()),
                start=CONFIG.start_date,
                end=CONFIG.end_date,
                progress=False,
                auto_adjust=True,
            )
            if bench is not None and not bench.empty:
                bench.to_parquet(cache_file.parent / "benchmarks_live.parquet")
        except Exception:
            pass

    df.to_parquet(cache_file, index=False)
    return df
