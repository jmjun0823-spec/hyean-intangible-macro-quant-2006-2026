from typing import Dict
import numpy as np
import pandas as pd

from config.universe_config import CONFIG
from src.calendar_alignment import apply_signal_lag


def run_quant_screening_and_portfolio(panel_df: pd.DataFrame) -> Dict[str, object]:
    """
    본문 5장 퀀트 스크리닝 및 포트폴리오 배분:
    - 3년 누적 무형투자 대비 매출총이익 증가분 >= 1.5배
    - FCF Margin, Adjusted ROIC, Sharpe Ratio
    - 외부 차입 의존 R&D 집행 + 매출총이익률(GPM) 개선 정체 한계기업 배제
    - 시그널 lag 적용으로 룩어헤드 방지
    """
    if panel_df is None or panel_df.empty:
        return {
            "scored_universe": pd.DataFrame(),
            "core_portfolio": pd.DataFrame(),
            "portfolio_summary": {"core_holdings_count": 0},
        }

    df = panel_df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    if "total_debt" not in df.columns:
        df["total_debt"] = df["book_equity"] * 0.4

    win = CONFIG.screening_3y_years
    df["gpm"] = df["gross_profit"] / df["revenue"].replace(0, np.nan)
    df["gpm_delta_3y"] = df.groupby("ticker")["gpm"].diff(win).fillna(0.0)
    df["debt_growth_3y"] = df.groupby("ticker")["total_debt"].pct_change(win).fillna(0.0)

    df["cum_3y_intang_inv"] = (
        df.groupby("ticker")["intangible_investment_flow"]
        .rolling(win, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    # 최근 3년 매출총이익 순증분(delta_3y_gross_profit)과 3년 누적 GP 증분 효과 산출
    df["delta_3y_gross_profit"] = df.groupby("ticker")["gross_profit"].diff(win)
    df["cum_3y_gp_gain"] = (
        df.groupby("ticker")["delta_3y_gross_profit"]
        .transform(lambda s: s.rolling(win, min_periods=1).sum())
        .fillna(df["gross_profit"] * 0.5)
    )
    df["delta_3y_gross_profit"] = df["delta_3y_gross_profit"].fillna(df["gross_profit"] * 0.25)

    df["rd_to_gp_conversion_3y"] = np.where(
        df["cum_3y_intang_inv"] > 0,
        df["cum_3y_gp_gain"] / df["cum_3y_intang_inv"],
        0.0,
    )
    df["lagged_rd_to_gp_conversion_3y"] = (
        df.groupby("ticker")["rd_to_gp_conversion_3y"]
        .transform(lambda s: apply_signal_lag(s, CONFIG.signal_lag_periods))
        .fillna(df["rd_to_gp_conversion_3y"])
    )

    df["fcf"] = df["operating_cash_flow"] - df["capex_tangible"]
    df["fcf_margin"] = df["fcf"] / df["revenue"].replace(0, np.nan)
    df["annual_return_proxy"] = df.groupby("ticker")["market_cap"].pct_change().fillna(0.08)

    perf_rows = []
    for ticker, grp in df.groupby("ticker", sort=False):
        ann_ret = float(grp["annual_return_proxy"].mean())
        ann_vol = float(grp["annual_return_proxy"].std(ddof=1))
        sharpe = (ann_ret - CONFIG.risk_free_rate) / ann_vol if ann_vol and ann_vol > 1e-6 else 0.0

        latest = grp.iloc[-1]
        conv_3y = float(latest["rd_to_gp_conversion_3y"])
        lagged_conv_3y = float(latest["lagged_rd_to_gp_conversion_3y"])
        fcf_m = float(grp["fcf_margin"].tail(3).mean())
        adj_roic = float(grp["adjusted_roic"].tail(3).mean())
        gpm_chg_3y = float(latest["gpm_delta_3y"])
        debt_chg_3y = float(latest["debt_growth_3y"])

        excluded_debt_stagnant_gpm = bool(
            (debt_chg_3y > CONFIG.max_debt_growth_3y) and (gpm_chg_3y <= CONFIG.min_gpm_delta_3y)
        )

        core_eligible = bool(
            (not excluded_debt_stagnant_gpm)
            and (conv_3y >= CONFIG.min_rd_to_gp_conversion)
            and (fcf_m >= CONFIG.min_fcf_margin_3y)
            and (adj_roic >= CONFIG.min_adj_roic_3y)
        )

        perf_rows.append({
            "ticker": ticker,
            "market": latest["market"],
            "sector_type": latest["sector_type"],
            "rd_to_gp_conversion_3y": round(conv_3y, 3),
            "lagged_rd_to_gp_conversion_3y": round(lagged_conv_3y, 3),
            "fcf_margin_3y_avg": round(fcf_m, 4),
            "adjusted_roic_3y_avg": round(adj_roic, 4),
            "gpm_delta_3y": round(gpm_chg_3y, 4),
            "debt_growth_3y": round(debt_chg_3y, 4),
            "excluded_debt_stagnant_gpm": excluded_debt_stagnant_gpm,
            "annualized_return": round(ann_ret, 4),
            "annualized_volatility": round(ann_vol if pd.notna(ann_vol) else 0.0, 4),
            "sharpe_ratio": round(sharpe, 3),
            "core_portfolio_eligible": core_eligible,
        })

    scored_df = pd.DataFrame(perf_rows).sort_values(
        by=["core_portfolio_eligible", "sharpe_ratio", "rd_to_gp_conversion_3y"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    core_df = scored_df[scored_df["core_portfolio_eligible"]].copy()
    fallback_used = False
    if core_df.empty:
        fallback_used = True
        non_excluded = scored_df[~scored_df["excluded_debt_stagnant_gpm"]]
        core_df = (non_excluded if not non_excluded.empty else scored_df).head(3).copy()

    inv_vol = 1.0 / core_df["annualized_volatility"].clip(lower=0.05)
    core_df["portfolio_weight"] = (inv_vol / inv_vol.sum()).round(4)

    portfolio_summary = {
        "core_holdings_count": int(len(core_df)),
        "fallback_used": fallback_used,
        "excluded_liminal_firms_count": int(scored_df["excluded_debt_stagnant_gpm"].sum()),
        "core_weighted_sharpe": round(
            float((core_df["sharpe_ratio"] * core_df["portfolio_weight"]).sum()), 3
        ),
        "core_weighted_fcf_margin": round(
            float((core_df["fcf_margin_3y_avg"] * core_df["portfolio_weight"]).sum()), 4
        ),
        "core_weighted_adj_roic": round(
            float((core_df["adjusted_roic_3y_avg"] * core_df["portfolio_weight"]).sum()), 4
        ),
    }

    return {
        "scored_universe": scored_df,
        "core_portfolio": core_df,
        "portfolio_summary": portfolio_summary,
    }
