from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class AnalysisConfig:
    start_date: str = "2006-01-01"
    end_date: str = "2026-11-23"
    start_year: int = 2006
    end_year: int = 2026
    rolling_10y_years: int = 10
    screening_3y_years: int = 3
    signal_lag_periods: int = 1
    risk_free_rate: float = 0.035

    # 무형자산 자본화 가정치
    rd_capitalization_rate: float = 0.80
    sga_intangible_share: float = 0.25
    intangible_amort_rate: float = 0.20
    tangible_deprec_rate: float = 0.08

    # 거시 성장회계 Gap 파라미터: dY_gap = alpha_u * dK_u + dTFP_unmeasured
    alpha_u: float = 0.35
    unmeasured_tfp_weight: float = 0.50

    # 5장 퀀트 스크리닝 및 한계기업 배제 기준
    min_rd_to_gp_conversion: float = 1.50
    min_fcf_margin_3y: float = 0.08
    min_adj_roic_3y: float = 0.12
    min_gpm_delta_3y: float = 0.00
    max_debt_growth_3y: float = 0.25
    top_quintile_pct: float = 0.20
    bottom_quintile_pct: float = 0.20

    market_benchmarks: Dict[str, str] = field(default_factory=lambda: {
        "SP500": "^GSPC",
        "NASDAQ": "^IXIC",
        "KOSPI": "^KS11",
        "INTANGIBLE_TECH_PROXY": "XLK",
        "TRADITIONAL_CAPEX_PROXY": "XLI",
    })

    sample_tickers: List[str] = field(default_factory=lambda: [
        "MSFT", "NVDA", "GOOGL", "META", "AAPL",
        "CAT", "DE", "HON", "XOM",
        "005930.KS", "000660.KS", "035420.KS", "051910.KS",
    ])


CONFIG = AnalysisConfig()
