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
    base_mcap_latest = (float(shares) * latest_close / scale) if shares else max(10.0, float(rev_s.iloc[-1]) * 2.5 if not rev_s.empty else 50.0)

    # 최근 실공시 연도의 재무 비율 산출 (과거 구간 역산용 기준 비율)
    ref_rev = float(rev_s.iloc[-1]) if not rev_s.empty else base_mcap_latest * 0.4
    ref_gp_ratio = float((gp_s / rev_s).dropna().iloc[-1]) if (not gp_s.empty and not rev_s.empty) else (0.58 if stype == "intangible_ai_ip" else 0.25)
    ref_rd_ratio = float((rd_s / rev_s).dropna().iloc[-1]) if (not rd_s.empty and not rev_s.empty) else (0.12 if stype == "intangible_ai_ip" else 0.02)
    ref_sga_ratio = float((sga_s / rev_s).dropna().iloc[-1]) if (not sga_s.empty and not rev_s.empty) else 0.14
    ref_ocf_ratio = float((ocf_s / rev_s).dropna().iloc[-1]) if (not ocf_s.empty and not rev_s.empty) else 0.25
    ref_capex_ratio = float((capex_s / rev_s).dropna().iloc[-1]) if (not capex_s.empty and not rev_s.empty) else 0.06
    ref_eq_to_mcap = float((eq_s.iloc[-1] / base_mcap_latest)) if (not eq_s.empty and base_mcap_latest > 0) else (0.25 if stype == "intangible_ai_ip" else 0.55)
    ref_debt_to_eq = float((debt_s.iloc[-1] / eq_s.iloc[-1])) if (not debt_s.empty and not eq_s.empty and eq_s.iloc[-1] > 0) else 0.35

    rows = []
    for yr in years:
        if yr < CONFIG.start_year or yr > CONFIG.end_year:
            continue
        px_ratio = float(yearly_px.loc[yr, "close_last"]) / latest_close
        mcap = max(1.0, base_mcap_latest * px_ratio)
        vol = float(yearly_px.loc[yr, "vol_ann"])

        rev = float(rev_s.loc[yr]) if yr in rev_s.index else max(1.0, ref_rev * (px_ratio ** 0.72))
        gp = float(gp_s.loc[yr]) if yr in gp_s.index else rev * np.clip(ref_gp_ratio, 0.08, 0.88)
        rd = float(rd_s.loc[yr]) if yr in rd_s.index else rev * np.clip(ref_rd_ratio, 0.0, 0.40)
        sga = float(sga_s.loc[yr]) if yr in sga_s.index else rev * np.clip(ref_sga_ratio, 0.02, 0.45)
        ocf = float(ocf_s.loc[yr]) if yr in ocf_s.index else rev * ref_ocf_ratio
        capex_t = float(capex_s.loc[yr]) if yr in capex_s.index else rev * np.clip(ref_capex_ratio, 0.01, 0.30)
        book_eq = float(eq_s.loc[yr]) if yr in eq_s.index else max(1.0, mcap * np.clip(ref_eq_to_mcap, 0.08, 1.20))
        debt = float(debt_s.loc[yr]) if yr in debt_s.index else max(0.2, book_eq * np.clip(ref_debt_to_eq, 0.05, 2.50))
        imp = float(imp_s.loc[yr]) if yr in imp_s.index else 0.0

        intang_flow = rd * CONFIG.rd_capitalization_rate + sga * CONFIG.sga_intangible_share
        tot_inv = capex_t + intang_flow
        intang_share = float(intang_flow / tot_inv) if tot_inv > 0 else 0.0
        # 연간 시가총액 순증감 기반 기관/시장 자금 유입 프록시
        net_flow_b = mcap * float(yearly_px.loc[yr, "close_last"] / yearly_px["close_last"].shift(1).fillna(yearly_px["close_last"]).loc[yr] - 1.0) * 0.05

        source_flag = "yfinance_actual_fin" if yr in rev_s.index else "yfinance_price_scaled"
        rows.append({
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
