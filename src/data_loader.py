from pathlib import Path
from typing import Dict, Optional
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
            "data_source",
        ]
    )


def _pick_row(df: Optional[pd.DataFrame], candidates: list[str]) -> pd.Series:
    """재무제표 DataFrame에서 후보 계정명 중 존재하는 첫 행을 연도별 Series로 반환합니다."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    for name in candidates:
        if name in df.index:
            s = pd.to_numeric(df.loc[name], errors="coerce").dropna()
            if not s.empty:
                s.index = pd.to_datetime(s.index).year
                return s.groupby(level=0).last()
    return pd.Series(dtype=float)


def _fetch_live_ticker_history(ticker: str, market: str, stype: str, scale: float = 1e9) -> pd.DataFrame:
    """yfinance를 통해 종목의 실제 주가·변동성·연간 재무제표를 수집합니다."""
    tk = yf.Ticker(ticker)
    hist = tk.history(start=CONFIG.start_date, end=CONFIG.end_date, auto_adjust=True)
    if hist is None or hist.empty:
        return pd.DataFrame()

    hist = hist.copy()
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    hist["year"] = hist.index.year
    hist["ret"] = hist["Close"].pct_change()

    yearly_px = hist.groupby("year").agg(
        close_last=("Close", "last"),
        vol_ann=("ret", lambda s: float(s.std(ddof=1) * np.sqrt(252)) if s.dropna().shape[0] > 10 else 0.25),
    )

    shares = None
    try:
        shares = getattr(tk.fast_info, "shares", None)
    except Exception:
        shares = None
    if not shares:
        try:
            shares = tk.info.get("sharesOutstanding")
        except Exception:
            shares = None

    fin = getattr(tk, "financials", pd.DataFrame())
    bs = getattr(tk, "balance_sheet", pd.DataFrame())
    cf = getattr(tk, "cashflow", pd.DataFrame())

    rev_s = _pick_row(fin, ["Total Revenue", "Operating Revenue"]) / scale
    gp_s = _pick_row(fin, ["Gross Profit"]) / scale
    rd_s = _pick_row(fin, ["Research And Development", "Research Development"]) / scale
    sga_s = _pick_row(fin, ["Selling General And Administration", "Selling General And Administrative"]) / scale
    imp_s = _pick_row(fin, ["Impairment Of Capital Assets", "Write Off", "Other Non Operating Income Expenses"]).abs() / scale

    eq_s = _pick_row(bs, ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"]) / scale
    debt_s = _pick_row(bs, ["Total Debt", "Long Term Debt And Capital Lease Obligation"]) / scale

    ocf_s = _pick_row(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]) / scale
    capex_s = _pick_row(cf, ["Capital Expenditure", "Purchase Of PPE"]).abs() / scale

    years = sorted(yearly_px.index.unique())
    latest_close = float(yearly_px["close_last"].iloc[-1])
    base_mcap_latest = (
        float(shares) * latest_close / scale
        if shares
        else max(10.0, float(rev_s.iloc[-1]) * 2.5 if not rev_s.empty else 50.0)
    )

    ref_rev = float(rev_s.iloc[-1]) if not rev_s.empty else base_mcap_latest * 0.4
    ref_gp_ratio = (
        float((gp_s / rev_s).dropna().iloc[-1])
        if (not gp_s.empty and not rev_s.empty)
        else (0.58 if stype == "intangible_ai_ip" else 0.25)
    )
    ref_rd_ratio = (
        float((rd_s / rev_s).dropna().iloc[-1])
        if (not rd_s.empty and not rev_s.empty)
        else (0.12 if stype == "intangible_ai_ip" else 0.02)
    )
    ref_sga_ratio = (
        float((sga_s / rev_s).dropna().iloc[-1])
        if (not sga_s.empty and not rev_s.empty)
        else 0.14
    )
    ref_ocf_ratio = (
        float((ocf_s / rev_s).dropna().iloc[-1])
        if (not ocf_s.empty and not rev_s.empty)
        else 0.25
    )
    ref_capex_ratio = (
        float((capex_s / rev_s).dropna().iloc[-1])
        if (not capex_s.empty and not rev_s.empty)
        else 0.06
    )
    ref_eq_to_mcap = (
        float((eq_s.iloc[-1] / base_mcap_latest))
        if (not eq_s.empty and base_mcap_latest > 0)
        else (0.25 if stype == "intangible_ai_ip" else 0.55)
    )
    ref_debt_to_eq = (
        float((debt_s.iloc[-1] / eq_s.iloc[-1]))
        if (not debt_s.empty and not eq_s.empty and eq_s.iloc[-1] > 0)
        else 0.35
    )

    prev_closes = yearly_px["close_last"].shift(1).fillna(yearly_px["close_last"])

    rows = []
    for yr in years:
        if yr < CONFIG.start_year or yr > CONFIG.end_year:
            continue
        cur_close = float(yearly_px.loc[yr, "close_last"])
        prev_close = float(prev_closes.loc[yr])
        px_ratio = cur_close / latest_close
        mcap = max(1.0, base_mcap_latest * px_ratio)
        vol = float(yearly_px.loc[yr, "vol_ann"])

        rev = float(rev_s.loc[yr]) if yr in rev_s.index else max(1.0, ref_rev * (px_ratio ** 0.72))
        gp = float(gp_s.loc[yr]) if yr in gp_s.index else rev * float(np.clip(ref_gp_ratio, 0.08, 0.88))
        rd = float(rd_s.loc[yr]) if yr in rd_s.index else rev * float(np.clip(ref_rd_ratio, 0.0, 0.40))
        sga = float(sga_s.loc[yr]) if yr in sga_s.index else rev * float(np.clip(ref_sga_ratio, 0.02, 0.45))
        ocf = float(ocf_s.loc[yr]) if yr in ocf_s.index else rev * ref_ocf_ratio
        capex_t = float(capex_s.loc[yr]) if yr in capex_s.index else rev * float(np.clip(ref_capex_ratio, 0.01, 0.30))
        book_eq = float(eq_s.loc[yr]) if yr in eq_s.index else max(1.0, mcap * float(np.clip(ref_eq_to_mcap, 0.08, 1.20)))
        debt = float(debt_s.loc[yr]) if yr in debt_s.index else max(0.2, book_eq * float(np.clip(ref_debt_to_eq, 0.05, 2.50)))
        imp = float(imp_s.loc[yr]) if yr in imp_s.index else 0.0

        intang_flow = rd * CONFIG.rd_capitalization_rate + sga * CONFIG.sga_intangible_share
        tot_inv = capex_t + intang_flow
        intang_share = float(intang_flow / tot_inv) if tot_inv > 0 else 0.0
        px_ret = (cur_close / prev_close - 1.0) if prev_close > 0 else 0.0
        net_flow_b = mcap * px_ret * 0.05

        source_flag = "yfinance_actual_fin" if yr in rev_s.index else "yfinance_price_scaled"
        rows.append(
            {
                "date": pd.Timestamp(f"{yr}-12-31"),
                "year": int(yr),
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
                "intangible_capex_share": round(intang_share, 4),
                "net_fund_flow_billions": round(net_flow_b, 4),
                "impairment_loss": round(imp, 4),
                "volatility_annual": round(vol, 4),
                "data_source": source_flag,
            }
        )

    return pd.DataFrame(rows)


def load_or_generate_panel(
    cache_path: Optional[Path] = None,
    use_live_prices: bool = True,
    refresh_cache: bool = False,
    raw_override_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    2006~2026년 패널 데이터를 로드합니다.
    use_live_prices=True일 때 Yahoo Finance 실데이터를 우선 수집하고,
    수집 실패 시 오프라인 백업 모델로 보완합니다.
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
        if "data_source" not in norm.columns:
            norm["data_source"] = "override_fixture"
        return norm.sort_values(["ticker", "date"]).reset_index(drop=True)

    cache_file = cache_path or Path("data/panel_dataset.parquet")
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    if cache_file.exists() and not refresh_cache:
        return pd.read_parquet(cache_file)

    sector_map: Dict[str, tuple[str, str, float]] = {
        "MSFT": ("US", "intangible_ai_ip", 1e9),
        "NVDA": ("US", "intangible_ai_ip", 1e9),
        "GOOGL": ("US", "intangible_ai_ip", 1e9),
        "META": ("US", "intangible_ai_ip", 1e9),
        "AAPL": ("US", "intangible_ai_ip", 1e9),
        "CAT": ("US", "traditional_capex", 1e9),
        "DE": ("US", "traditional_capex", 1e9),
        "HON": ("US", "traditional_capex", 1e9),
        "XOM": ("US", "traditional_capex", 1e9),
        "005930.KS": ("KR_KOSPI", "intangible_ai_ip", 1e12),
        "000660.KS": ("KR_KOSPI", "intangible_ai_ip", 1e12),
        "035420.KS": ("KR_KOSPI", "intangible_ai_ip", 1e12),
        "051910.KS": ("KR_KOSPI", "traditional_capex", 1e12),
    }

    collected = []
    if use_live_prices:
        for ticker, (market, stype, scale) in sector_map.items():
            try:
                live_df = _fetch_live_ticker_history(ticker, market, stype, scale=scale)
                if not live_df.empty and live_df["year"].nunique() >= 10:
                    collected.append(live_df)
            except Exception:
                continue

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

    if collected:
        df = pd.concat(collected, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
        df.to_parquet(cache_file, index=False)
        return df

    rng = np.random.default_rng(20261123)
    years = np.arange(CONFIG.start_year, CONFIG.end_year + 1)
    dates = [pd.Timestamp(f"{y}-12-31") for y in years]
    rows = []
    for ticker, (market, stype, _) in sector_map.items():
        rev = 50.0 if stype == "intangible_ai_ip" else 40.0
        mcap = rev * 2.5
        book_eq = mcap * 0.35
        debt = book_eq * 0.3
        for dt in dates:
            yr = int(dt.year)
            growth = (0.15 if stype == "intangible_ai_ip" else 0.06) + float(rng.normal(0, 0.05))
            rev = max(1.0, rev * (1.0 + growth * 0.7))
            gp = rev * (0.58 if stype == "intangible_ai_ip" else 0.24)
            rd = rev * (0.14 if stype == "intangible_ai_ip" else 0.03)
            sga = rev * 0.15
            capex_t = rev * 0.06
            ocf = gp * 0.65 - 0.3 * rd
            mcap = max(2.0, mcap * (1.0 + growth))
            book_eq = max(1.0, book_eq * (1.0 + growth * 0.4))
            debt = max(0.5, debt * 1.03)
            intang_flow = rd * CONFIG.rd_capitalization_rate + sga * CONFIG.sga_intangible_share
            rows.append(
                {
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
                    "intangible_capex_share": round(intang_flow / (capex_t + intang_flow), 4),
                    "net_fund_flow_billions": round(mcap * 0.02, 4),
                    "impairment_loss": 0.0,
                    "volatility_annual": 0.25,
                    "data_source": "offline_fallback",
                }
            )
    df = pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)
    df.to_parquet(cache_file, index=False)
    return df
