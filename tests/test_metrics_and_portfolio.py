import json
from pathlib import Path
from src.data_loader import load_or_generate_panel
from src.metrics_engine import compute_all_metrics
from src.portfolio_backtest import run_quant_screening_and_portfolio
from src.report_builder import build_blog_summary_markdown
from src.visualizer import generate_verification_charts


def test_end_to_end_pipeline(tmp_path: Path):
    df = load_or_generate_panel(cache_path=tmp_path / "panel.parquet", refresh_cache=True)
    assert not df.empty
    assert df["year"].nunique() >= 15

    metrics = compute_all_metrics(df)
    summary = metrics["summary"]

    assert "chapter_1_macro_gap" in summary
    assert "chapter_2_accounting_distortion_10y" in summary
    assert "chapter_3_flows_and_capex_share" in summary
    assert len(summary["chapter_4_tail_risk_and_obsolescence"]) == 2

    port = run_quant_screening_and_portfolio(metrics["panel"])
    assert not port["scored_universe"].empty
    assert "excluded_debt_stagnant_gpm" in port["scored_universe"].columns
    assert abs(port["core_portfolio"]["portfolio_weight"].sum() - 1.0) < 1e-3

    charts_dir = tmp_path / "charts"
    generate_verification_charts(
        metrics["yearly_macro"],
        metrics["panel"],
        port["scored_universe"],
        output_dir=charts_dir,
    )
    assert (charts_dir / "ch1_ch3_macro_and_intangible_share.png").exists()
    assert (charts_dir / "ch2_nominal_vs_adjusted_pbr.png").exists()
    assert (charts_dir / "ch5_quant_screening_map.png").exists()

    full_report = {
        "metrics_summary": summary,
        "portfolio_summary": port["portfolio_summary"],
    }
    md_path = build_blog_summary_markdown(
        report_data=full_report,
        scored_universe=port["scored_universe"],
        core_portfolio=port["core_portfolio"],
        risk_by_sector=metrics["risk_by_sector"],
        r_models_dir=tmp_path / "r_models",
        output_path=tmp_path / "blog_verification_summary.md",
    )
    assert md_path.exists()
    content = md_path.read_text(encoding="utf-8")
    assert "혜안 정량 검증 요약 리포트" in content
    assert json.loads(json.dumps(full_report))["metrics_summary"]["analysis_period"] == summary["analysis_period"]
