import pandas as pd
from src.calendar_alignment import align_market_time_series, apply_signal_lag, get_business_calendar
from src.data_loader import load_or_generate_panel
from src.metrics_engine import compute_all_metrics
from src.portfolio_backtest import run_quant_screening_and_portfolio


def test_us_and_kr_holiday_alignment_and_empty_input():
    us_cal = get_business_calendar("2026-07-02", "2026-07-06", market="US")
    kr_cal = get_business_calendar("2026-07-02", "2026-07-06", market="KR_KOSPI")

    # 2026-07-04는 토요일이지만, 2026-07-03(금) 또는 공휴일 필터 동작 확인
    assert pd.Timestamp("2026-07-04") not in us_cal
    assert pd.Timestamp("2026-07-04") not in kr_cal

    # 빈 응답/빈 DataFrame이 들어와도 예외 없이 정렬
    empty_aligned = align_market_time_series(pd.DataFrame(), us_cal, ["close", "volume"])
    assert len(empty_aligned) == len(us_cal)
    assert (empty_aligned["close"] == 0.0).all()

    # 휴장일/결측 포함 시계열 정렬 및 ffill 검증
    sparse = pd.DataFrame({
        "date": ["2026-07-02", "2026-07-06"],
        "close": [100.0, 105.0],
    })
    aligned = align_market_time_series(sparse, us_cal, ["close"], fill_method="ffill")
    assert aligned.loc[pd.Timestamp("2026-07-03"), "close"] == 100.0
    assert aligned.loc[pd.Timestamp("2026-07-06"), "close"] == 105.0


def test_signal_lag_and_debt_stagnant_gpm_exclusion():
    raw_sig = pd.Series([10.0, 20.0, 30.0], index=[2024, 2025, 2026])
    lagged = apply_signal_lag(raw_sig, lag_periods=1)
    assert pd.isna(lagged.iloc[0])
    assert lagged.iloc[1] == 10.0
    assert lagged.iloc[2] == 20.0

    # 고정된 작은 패널: 한계기업(차입 급증 + GPM 정체) vs 우량 무형자산 기업
    fixture = pd.DataFrame([
        # DISTRESSED_LIM: 부채 10 -> 25 (+150%), GP/Rev는 0.20 -> 0.20 (GPM 정체), 높은 전환율 가짜 시그널
        {"date": "2023-12-31", "year": 2023, "ticker": "DISTRESSED_LIM", "market": "US", "sector_type": "intangible_ai_ip",
         "market_cap": 100.0, "book_equity": 40.0, "total_debt": 10.0, "revenue": 100.0, "gross_profit": 20.0,
         "operating_cash_flow": 15.0, "capex_tangible": 2.0, "rd_expense": 5.0, "sga_expense": 8.0,
         "intangible_capex_share": 0.65, "net_fund_flow_billions": 1.0, "impairment_loss": 0.0, "volatility_annual": 0.25},
        {"date": "2024-12-31", "year": 2024, "ticker": "DISTRESSED_LIM", "market": "US", "sector_type": "intangible_ai_ip",
         "market_cap": 115.0, "book_equity": 42.0, "total_debt": 15.0, "revenue": 110.0, "gross_profit": 22.0,
         "operating_cash_flow": 16.0, "capex_tangible": 2.0, "rd_expense": 5.0, "sga_expense": 8.0,
         "intangible_capex_share": 0.66, "net_fund_flow_billions": 1.2, "impairment_loss": 0.0, "volatility_annual": 0.25},
        {"date": "2025-12-31", "year": 2025, "ticker": "DISTRESSED_LIM", "market": "US", "sector_type": "intangible_ai_ip",
         "market_cap": 125.0, "book_equity": 44.0, "total_debt": 20.0, "revenue": 120.0, "gross_profit": 24.0,
         "operating_cash_flow": 17.0, "capex_tangible": 2.0, "rd_expense": 5.0, "sga_expense": 8.0,
         "intangible_capex_share": 0.67, "net_fund_flow_billions": 1.3, "impairment_loss": 0.0, "volatility_annual": 0.25},
        {"date": "2026-12-31", "year": 2026, "ticker": "DISTRESSED_LIM", "market": "US", "sector_type": "intangible_ai_ip",
         "market_cap": 135.0, "book_equity": 45.0, "total_debt": 25.0, "revenue": 130.0, "gross_profit": 26.0,
         "operating_cash_flow": 18.0, "capex_tangible": 2.0, "rd_expense": 5.0, "sga_expense": 8.0,
         "intangible_capex_share": 0.68, "net_fund_flow_billions": 1.4, "impairment_loss": 0.0, "volatility_annual": 0.25},
        # QUALITY_CORE: 부채 안정(10 -> 11), GP/Rev 개선(0.50 -> 0.65), 강한 FCF/ROIC
        {"date": "2023-12-31", "year": 2023, "ticker": "QUALITY_CORE", "market": "US", "sector_type": "intangible_ai_ip",
         "market_cap": 200.0, "book_equity": 50.0, "total_debt": 10.0, "revenue": 100.0, "gross_profit": 50.0,
         "operating_cash_flow": 45.0, "capex_tangible": 3.0, "rd_expense": 8.0, "sga_expense": 10.0,
         "intangible_capex_share": 0.72, "net_fund_flow_billions": 5.0, "impairment_loss": 0.0, "volatility_annual": 0.20},
        {"date": "2024-12-31", "year": 2024, "ticker": "QUALITY_CORE", "market": "US", "sector_type": "intangible_ai_ip",
         "market_cap": 240.0, "book_equity": 58.0, "total_debt": 10.2, "revenue": 120.0, "gross_profit": 66.0,
         "operating_cash_flow": 55.0, "capex_tangible": 3.0, "rd_expense": 8.0, "sga_expense": 10.0,
         "intangible_capex_share": 0.74, "net_fund_flow_billions": 6.0, "impairment_loss": 0.0, "volatility_annual": 0.20},
        {"date": "2025-12-31", "year": 2025, "ticker": "QUALITY_CORE", "market": "US", "sector_type": "intangible_ai_ip",
         "market_cap": 290.0, "book_equity": 68.0, "total_debt": 10.5, "revenue": 145.0, "gross_profit": 87.0,
         "operating_cash_flow": 68.0, "capex_tangible": 3.0, "rd_expense": 8.0, "sga_expense": 10.0,
         "intangible_capex_share": 0.76, "net_fund_flow_billions": 7.0, "impairment_loss": 0.0, "volatility_annual": 0.20},
        {"date": "2026-12-31", "year": 2026, "ticker": "QUALITY_CORE", "market": "US", "sector_type": "intangible_ai_ip",
         "market_cap": 350.0, "book_equity": 80.0, "total_debt": 11.0, "revenue": 170.0, "gross_profit": 110.5,
         "operating_cash_flow": 82.0, "capex_tangible": 3.0, "rd_expense": 8.0, "sga_expense": 10.0,
         "intangible_capex_share": 0.78, "net_fund_flow_billions": 8.0, "impairment_loss": 0.0, "volatility_annual": 0.20},
    ])

    panel = load_or_generate_panel(raw_override_df=fixture)
    metrics = compute_all_metrics(panel)
    port = run_quant_screening_and_portfolio(metrics["panel"])
    scored = port["scored_universe"].set_index("ticker")

    assert scored.loc["DISTRESSED_LIM", "excluded_debt_stagnant_gpm"] is True
    assert scored.loc["DISTRESSED_LIM", "core_portfolio_eligible"] is False
    assert scored.loc["QUALITY_CORE", "excluded_debt_stagnant_gpm"] is False
    assert scored.loc["QUALITY_CORE", "core_portfolio_eligible"] is True
