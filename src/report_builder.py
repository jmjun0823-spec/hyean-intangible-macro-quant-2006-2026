import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd


def _read_text_safe(path: Path) -> str:
    if not path.exists():
        return "(해당 R 요약 파일 없음)"
    return path.read_text(encoding="utf-8", errors="replace").strip()


def build_blog_summary_markdown(
    report_data: Dict[str, Any],
    scored_universe: pd.DataFrame,
    core_portfolio: pd.DataFrame,
    risk_by_sector: pd.DataFrame,
    r_models_dir: Path = Path("outputs/r_models"),
    output_path: Path = Path("outputs/blog_verification_summary.md"),
) -> Path:
    """1~5장 검증 결과를 블로그 리포트 대조용 Markdown 문서로 출력합니다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = report_data.get("metrics_summary", {})
    portfolio = report_data.get("portfolio_summary", {})

    ch1 = metrics.get("chapter_1_macro_gap", {})
    ch2 = metrics.get("chapter_2_accounting_distortion_10y", {})
    ch3 = metrics.get("chapter_3_flows_and_capex_share", {})
    ch4_list = metrics.get("chapter_4_tail_risk_and_obsolescence", [])

    lines = [
        "# 혜안 정량 검증 요약 리포트",
        "",
        f"- **분석 기간**: `{metrics.get('analysis_period', 'N/A')}`",
        "- **검증 범위**: 1장 거시 GDP 괴리, 2장 회계 비용화 왜곡 및 Adjusted PBR/ROIC 검정, 3장 55% 무형투자 비중 및 수급 연동성, 4장 진부화·손상차손 테일 리스크, 5장 3년 전환율(1.5x+)·FCF Margin·Sharpe 및 한계기업 배제 포트폴리오",
        "",
        "---",
        "",
        "## 1. 거시 통계 괴리 (ΔY_Gap)",
        f"- 누적 ΔY_Gap: `{ch1.get('total_cumulative_delta_y_gap', 'N/A')}`",
        f"- 최근 연도 ΔY_Gap: `{ch1.get('latest_year_delta_y_gap', 'N/A')}` (ΔK_u: `{ch1.get('latest_dk_u', 'N/A')}`, ΔTFP_unmeasured: `{ch1.get('latest_dtfp_unmeasured', 'N/A')}`)",
        "",
        "## 2. 회계 비용 처리 왜곡 & 가설 검정 (최근 10년)",
        f"- 상위 20% vs 하위 20% 명목 PBR 평균: `{ch2.get('top20_mean_nominal_pbr', 'N/A')}` vs `{ch2.get('bot20_mean_nominal_pbr', 'N/A')}` (p-value: `{ch2.get('nominal_pbr_ttest_pval', 'N/A')}`)",
        f"- 상위 20% vs 하위 20% Adjusted PBR 평균: `{ch2.get('top20_mean_adjusted_pbr', 'N/A')}` vs `{ch2.get('bot20_mean_adjusted_pbr', 'N/A')}` (p-value: `{ch2.get('adjusted_pbr_ttest_pval', 'N/A')}`)",
        f"- 상위 20% vs 하위 20% Adjusted ROIC 평균: `{ch2.get('top20_mean_adjusted_roic', 'N/A')}` vs `{ch2.get('bot20_mean_adjusted_roic', 'N/A')}` (p-value: `{ch2.get('adjusted_roic_ttest_pval', 'N/A')}`)",
        f"- H0 기각 여부: `{ch2.get('h0_rejected', 'N/A')}`",
        "",
        "## 3. 수급 및 무형투자 비중 (55% 임계치)",
        f"- 무형투자 비중 55% 이상 연도: `{ch3.get('years_intangible_share_gte_55pct', [])}`",
        f"- 최근 무형투자 비중: `{ch3.get('latest_intangible_share', 'N/A')}`",
        f"- 무형투자 비중 vs 순자금유입 상관계수: `{ch3.get('flow_vs_intangible_share_corr', 'N/A')}` (p-value: `{ch3.get('flow_vs_intangible_share_pval', 'N/A')}`)",
        "",
        "## 4. 진부화 및 테일 리스크 (섹터별)",
    ]

    for row in ch4_list:
        lines.append(
            f"- **{row.get('sector_type')}**: 평균 변동성 `{row.get('mean_volatility')}`, P95 변동성 `{row.get('p95_volatility')}`, 평균 손상차손/자본 `{row.get('mean_impairment_ratio')}`, P95 손상차손/자본 `{row.get('tail_impairment_p95')}`"
        )

    lines.extend([
        "",
        "## 5. 퀀트 스크리닝 및 코어 포트폴리오",
        f"- 코어 편입 종목 수: `{portfolio.get('core_holdings_count', 0)}` / 차입+GPM 정체 배제 종목 수: `{portfolio.get('excluded_liminal_firms_count', 0)}`",
        f"- 코어 가중 Sharpe: `{portfolio.get('core_weighted_sharpe', 'N/A')}` | 가중 FCF Margin: `{portfolio.get('core_weighted_fcf_margin', 'N/A')}` | 가중 Adjusted ROIC: `{portfolio.get('core_weighted_adj_roic', 'N/A')}`",
        "",
    ])

    if core_portfolio is not None and not core_portfolio.empty:
        lines.append("### 코어 포트폴리오 편입 비중")
        for _, r in core_portfolio.iterrows():
            lines.append(
                f"- `{r['ticker']}` ({r['market']}): 비중 `{r['portfolio_weight']:.2%}`, 3Y 전환율 `{r['rd_to_gp_conversion_3y']}x`, FCF Margin `{r['fcf_margin_3y_avg']:.2%}`, Sharpe `{r['sharpe_ratio']}`"
            )
        lines.append("")

    if scored_universe is not None and not scored_universe.empty:
        excluded_df = scored_universe[scored_universe.get("excluded_debt_stagnant_gpm", False) == True]
        lines.append(f"### 차입 증가 + GPM 정체 한계기업 배제 목록 ({len(excluded_df)}개)")
        if excluded_df.empty:
            lines.append("- 해당 조건으로 배제된 종목 없음")
        else:
            for _, r in excluded_df.iterrows():
                lines.append(
                    f"- `{r['ticker']}`: 3Y 부채증가율 `{r['debt_growth_3y']:.2%}`, 3Y GPM 변화 `{r['gpm_delta_3y']:.4f}`"
                )
        lines.append("")

    lines.extend([
        "---",
        "## 6. R 회귀분석 요약 참조",
        "### `flow_intangible_ols_summary.txt`",
        "```text",
        _read_text_safe(r_models_dir / "flow_intangible_ols_summary.txt"),
        "```",
        "### `robustness_flow_regime.txt`",
        "```text",
        _read_text_safe(r_models_dir / "robustness_flow_regime.txt"),
        "```",
        "### `sharpe_factor_regression.txt`",
        "```text",
        _read_text_safe(r_models_dir / "sharpe_factor_regression.txt"),
        "```",
    ])

    _ = risk_by_sector
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    out = Path("outputs")
    rep_file = out / "verification_report.json"
    if rep_file.exists():
        data = json.loads(rep_file.read_text(encoding="utf-8"))
        scored = pd.read_csv(out / "quant_scored_universe.csv") if (out / "quant_scored_universe.csv").exists() else pd.DataFrame()
        core = pd.read_csv(out / "core_portfolio_weights.csv") if (out / "core_portfolio_weights.csv").exists() else pd.DataFrame()
        risk = pd.read_csv(out / "sector_tail_risk.csv") if (out / "sector_tail_risk.csv").exists() else pd.DataFrame()
        build_blog_summary_markdown(data, scored, core, risk)
