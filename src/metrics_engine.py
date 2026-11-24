from typing import Dict
import numpy as np
import pandas as pd
from scipy import stats

from config.universe_config import CONFIG
from src.calendar_alignment import apply_signal_lag


def compute_all_metrics(df: pd.DataFrame) -> Dict[str, object]:
    """
    본문 1~4장의 모든 검증 항목을 산출합니다:
    - 1장: dY_gap = alpha_u * dK_u + dTFP_unmeasured 및 무형자산 투자 집약도
    - 2장: 명목 PBR vs Adjusted PBR, Adjusted ROIC, OCF, 상위 20% vs 하위 20% H0 검정
    - 3장: 무형자산 투자 비중(55% 임계치 돌파 연도) 및 글로벌 수급 상관계수
    - 4장: 진부화(Cliff Effect) 및 손상차손/변동성 테일 리스크
    """
    if df is None or df.empty:
        return {
            "panel": pd.DataFrame(),
            "yearly_macro": pd.DataFrame(),
            "risk_by_sector": pd.DataFrame(),
            "summary": {"empty_input": True},
        }

    d = df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    if "total_debt" not in d.columns:
        d["total_debt"] = d["book_equity"] * 0.4

    # 무형투자 추정치 및 누적 무형/유형 자본 스톡
    d["intangible_investment_flow"] = (
        d["rd_expense"] * CONFIG.rd_capitalization_rate
        + d["sga_expense"] * CONFIG.sga_intangible_share
    )
    d["intangible_intensity_to_rev"] = (
        (d["rd_expense"] + d["sga_expense"] * CONFIG.sga_intangible_share)
        / d["revenue"].replace(0, np.nan)
    )
    d["total_economic_investment"] = d["capex_tangible"] + d["intangible_investment_flow"]
    d["empirical_intangible_share"] = np.where(
        d["total_economic_investment"] > 0,
        d["intangible_investment_flow"] / d["total_economic_investment"],
        0.0,
    )

    cap_intang_stock = []
    cap_tangible_stock = []
    for _, grp in d.groupby("ticker", sort=False):
        k_u = 0.0
        k_t = float(grp["book_equity"].iloc[0]) * 0.5
        for inv_u, inv_t in zip(grp["intangible_investment_flow"], grp["capex_tangible"]):
            k_u = k_u * (1.0 - CONFIG.intangible_amort_rate) + float(inv_u)
            k_t = k_t * (1.0 - CONFIG.tangible_deprec_rate) + float(inv_t)
            cap_intang_stock.append(k_u)
            cap_tangible_stock.append(k_t)

    d["capitalized_intangible_stock"] = cap_intang_stock
    d["capitalized_tangible_stock"] = cap_tangible_stock

    # [1장] 거시 통계 구조적 맹점: dY_gap = alpha_u * dK_u + dTFP_unmeasured
    d["dK_u"] = d.groupby("ticker")["capitalized_intangible_stock"].diff().fillna(0.0)
    d["dTFP_unmeasured"] = (
        d["intangible_investment_flow"] * 0.15
        + np.where(d["sector_type"] == "intangible_ai_ip", d["revenue"] * 0.025, d["revenue"] * 0.005)
    )
    d["delta_y_gap"] = CONFIG.alpha_u * d["dK_u"] + CONFIG.unmeasured_tfp_weight * d["dTFP_unmeasured"]

    # 시그널 지연(t 시점 생성 시그널 -> t+1 시점 적용)
    d["lagged_intangible_share"] = (
        d.groupby("ticker")["empirical_intangible_share"]
        .transform(lambda s: apply_signal_lag(s, CONFIG.signal_lag_periods))
    )

    yearly_macro = (
        d.groupby("year", as_index=False)
        .agg(
            total_revenue=("revenue", "sum"),
            total_intang_flow=("intangible_investment_flow", "sum"),
            mean_intang_share=("empirical_intangible_share", "mean"),
            reported_intangible_capex_share=("intangible_capex_share", "mean"),
            delta_y_gap=("delta_y_gap", "sum"),
            dk_u=("dK_u", "sum"),
            dtfp_unmeasured=("dTFP_unmeasured", "sum"),
            net_fund_flow_b=("net_fund_flow_billions", "sum"),
            total_ocf=("operating_cash_flow", "sum"),
            total_impairment=("impairment_loss", "sum"),
            avg_volatility=("volatility_annual", "mean"),
        )
    )
    yearly_macro["official_gdp_proxy_growth"] = yearly_macro["total_revenue"].pct_change().fillna(0.05)
    yearly_macro["latent_potential_growth_proxy"] = (
        yearly_macro["official_gdp_proxy_growth"]
        + (yearly_macro["delta_y_gap"] / yearly_macro["total_revenue"].replace(0, np.nan)).fillna(0.0) * 0.10
    )

    # [2장] 회계 비용 처리 왜곡 & Adjusted PBR / Adjusted ROIC / H0 검정
    d["nominal_pbr"] = d["market_cap"] / d["book_equity"].replace(0, np.nan)
    d["adjusted_book_equity"] = d["book_equity"] + d["capitalized_intangible_stock"]
    d["adjusted_pbr"] = d["market_cap"] / d["adjusted_book_equity"].replace(0, np.nan)

    d["amort_tax_shield_adj"] = d["rd_expense"] * CONFIG.rd_capitalization_rate - (
        d["capitalized_intangible_stock"] * CONFIG.intangible_amort_rate
    )
    d["adjusted_nopat"] = (d["gross_profit"] - d["sga_expense"] * 0.5 + d["amort_tax_shield_adj"]) * 0.78
    d["invested_capital_adj"] = (d["capitalized_tangible_stock"] + d["capitalized_intangible_stock"]).replace(0, np.nan)
    d["adjusted_roic"] = d["adjusted_nopat"] / d["invested_capital_adj"]

    recent_10y = d[d["year"] >= (d["year"].max() - CONFIG.rolling_10y_years + 1)].copy()
    q80 = recent_10y["intangible_intensity_to_rev"].quantile(1.0 - CONFIG.top_quintile_pct)
    q20 = recent_10y["intangible_intensity_to_rev"].quantile(CONFIG.bottom_quintile_pct)

    top20 = recent_10y[recent_10y["intangible_intensity_to_rev"] >= q80]
    bot20 = recent_10y[recent_10y["intangible_intensity_to_rev"] <= q20]

    def _safe_ttest(a: pd.Series, b: pd.Series) -> float:
        a_clean = a.dropna()
        b_clean = b.dropna()
        if len(a_clean) < 2 or len(b_clean) < 2:
            return 1.0
        _, pval = stats.ttest_ind(a_clean, b_clean, equal_var=False)
        return float(pval) if np.isfinite(pval) else 1.0

    p_val_pbr = _safe_ttest(top20["nominal_pbr"], bot20["nominal_pbr"])
    p_val_adj_pbr = _safe_ttest(top20["adjusted_pbr"], bot20["adjusted_pbr"])
    p_val_roic = _safe_ttest(top20["adjusted_roic"], bot20["adjusted_roic"])

    # [3장] 수급 및 무형자산 투자 비중(55% 임계치)
    if len(yearly_macro) >= 3 and yearly_macro["reported_intangible_capex_share"].nunique() > 1:
        flow_corr_raw, flow_pval_raw = stats.pearsonr(
            yearly_macro["reported_intangible_capex_share"], yearly_macro["net_fund_flow_b"]
        )
        flow_corr = float(flow_corr_raw) if np.isfinite(flow_corr_raw) else 0.0
        flow_pval = float(flow_pval_raw) if np.isfinite(flow_pval_raw) else 1.0
    else:
        flow_corr, flow_pval = 0.0, 1.0

    years_over_55pct = [
        int(y) for y in yearly_macro.loc[yearly_macro["reported_intangible_capex_share"] >= 0.55, "year"]
    ]

    # [4장] 진부화 속도(Cliff Effect) 및 테일 리스크
    d["impairment_to_equity"] = d["impairment_loss"] / d["book_equity"].replace(0, np.nan)
    risk_by_sector = (
        d.groupby("sector_type", as_index=False)
        .agg(
            mean_volatility=("volatility_annual", "mean"),
            p95_volatility=("volatility_annual", lambda s: float(np.nanpercentile(s, 95))),
            mean_impairment_ratio=("impairment_to_equity", "mean"),
            tail_impairment_p95=("impairment_to_equity", lambda s: float(np.nanpercentile(s, 95))),
            mean_adj_roic=("adjusted_roic", "mean"),
        )
    )

    summary = {
        "analysis_period": f"{int(d['year'].min())}-{int(d['year'].max())}",
        "chapter_1_macro_gap": {
            "total_cumulative_delta_y_gap": round(float(yearly_macro["delta_y_gap"].sum()), 2),
            "latest_year_delta_y_gap": round(float(yearly_macro["delta_y_gap"].iloc[-1]), 2),
            "latest_dk_u": round(float(yearly_macro["dk_u"].iloc[-1]), 2),
            "latest_dtfp_unmeasured": round(float(yearly_macro["dtfp_unmeasured"].iloc[-1]), 2),
        },
        "chapter_2_accounting_distortion_10y": {
            "top20_mean_nominal_pbr": round(float(top20["nominal_pbr"].mean()), 2),
            "bot20_mean_nominal_pbr": round(float(bot20["nominal_pbr"].mean()), 2),
            "nominal_pbr_ttest_pval": p_val_pbr,
            "top20_mean_adjusted_pbr": round(float(top20["adjusted_pbr"].mean()), 2),
            "bot20_mean_adjusted_pbr": round(float(bot20["adjusted_pbr"].mean()), 2),
            "adjusted_pbr_ttest_pval": p_val_adj_pbr,
            "top20_mean_adjusted_roic": round(float(top20["adjusted_roic"].mean()), 4),
            "bot20_mean_adjusted_roic": round(float(bot20["adjusted_roic"].mean()), 4),
            "adjusted_roic_ttest_pval": p_val_roic,
            "h0_rejected": bool(p_val_pbr < 0.05 and p_val_roic < 0.05),
        },
        "chapter_3_flows_and_capex_share": {
            "years_intangible_share_gte_55pct": years_over_55pct,
            "latest_intangible_share": round(float(yearly_macro["reported_intangible_capex_share"].iloc[-1]), 4),
            "flow_vs_intangible_share_corr": round(flow_corr, 4),
            "flow_vs_intangible_share_pval": flow_pval,
        },
        "chapter_4_tail_risk_and_obsolescence": risk_by_sector.to_dict(orient="records"),
    }

    return {
        "panel": d,
        "yearly_macro": yearly_macro,
        "risk_by_sector": risk_by_sector,
        "summary": summary,
    }
