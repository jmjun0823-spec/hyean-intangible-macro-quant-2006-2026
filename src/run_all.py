import argparse
import json
from pathlib import Path

from src.data_loader import load_or_generate_panel
from src.metrics_engine import compute_all_metrics
from src.portfolio_backtest import run_quant_screening_and_portfolio
from src.visualizer import generate_verification_charts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 혜안 Macro & Quant Verification Pipeline")
    parser.add_argument("--refresh-cache", action="store_true", help="Rebuild panel dataset cache")
    parser.add_argument("--live-prices", action="store_true", help="Fetch benchmark live prices via yfinance")
    parser.add_argument("--no-charts", action="store_true", help="Skip generating PNG charts")
    args = parser.parse_args()

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_df = load_or_generate_panel(
        use_live_prices=args.live_prices,
        refresh_cache=args.refresh_cache,
    )
    metrics_res = compute_all_metrics(panel_df)
    port_res = run_quant_screening_and_portfolio(metrics_res["panel"])

    metrics_res["yearly_macro"].to_csv(out_dir / "yearly_macro_gap_and_flows.csv", index=False)
    metrics_res["risk_by_sector"].to_csv(out_dir / "sector_tail_risk.csv", index=False)
    port_res["scored_universe"].to_csv(out_dir / "quant_scored_universe.csv", index=False)
    port_res["core_portfolio"].to_csv(out_dir / "core_portfolio_weights.csv", index=False)

    if not args.no_charts:
        generate_verification_charts(
            metrics_res["yearly_macro"],
            metrics_res["panel"],
            port_res["scored_universe"],
            output_dir=out_dir / "charts",
        )

    full_report = {
        "metrics_summary": metrics_res["summary"],
        "portfolio_summary": port_res["portfolio_summary"],
    }
    (out_dir / "verification_report.json").write_text(
        json.dumps(full_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(full_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
