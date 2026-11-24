from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def generate_verification_charts(
    yearly_macro: pd.DataFrame,
    panel_df: pd.DataFrame,
    scored_universe: pd.DataFrame,
    output_dir: Path = Path("outputs/charts"),
) -> None:
    """1~5장 검증 차트를 PNG로 생성합니다."""
    if yearly_macro is None or yearly_macro.empty or panel_df is None or panel_df.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. 거시 GDP 괴리 및 무형자산 투자 비중(55% 임계치) 추이
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.bar(yearly_macro["year"], yearly_macro["delta_y_gap"], alpha=0.65, label="Delta Y_Gap")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Delta Y_Gap")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(
        yearly_macro["year"],
        yearly_macro["reported_intangible_capex_share"] * 100,
        linewidth=2.0,
        linestyle="--",
        label="Intangible Investment Share (%)",
    )
    ax2.axhline(55.0, linestyle=":", linewidth=1.2, label="55% Threshold")
    ax2.set_ylabel("Intangible Share (%)")
    ax2.legend(loc="upper right")

    plt.title("Macro Gap & Intangible Investment Share (2006-2026)")
    plt.tight_layout()
    plt.savefig(output_dir / "ch1_ch3_macro_and_intangible_share.png", dpi=150)
    plt.close(fig)

    # 2. 명목 PBR vs Adjusted PBR 비교
    recent_10y = panel_df[panel_df["year"] >= (panel_df["year"].max() - 9)]
    pbr_comp = (
        recent_10y.groupby("sector_type", as_index=False)[["nominal_pbr", "adjusted_pbr"]]
        .mean()
        .set_index("sector_type")
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pbr_comp.plot(kind="bar", ax=ax)
    ax.set_title("Nominal PBR vs. Adjusted PBR by Sector (Last 10Y Mean)")
    ax.set_ylabel("PBR Multiple (x)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "ch2_nominal_vs_adjusted_pbr.png", dpi=150)
    plt.close(fig)

    # 3. 5장 퀀트 스크리닝 맵
    if scored_universe is not None and not scored_universe.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.scatter(
            scored_universe["rd_to_gp_conversion_3y"],
            scored_universe["sharpe_ratio"],
            s=scored_universe["fcf_margin_3y_avg"].clip(lower=0.02) * 800,
            alpha=0.8,
        )
        ax.axvline(1.5, linestyle="--", linewidth=1.3, label="1.5x GP Conversion Cutoff")
        for _, row in scored_universe.iterrows():
            ax.annotate(
                row["ticker"],
                (row["rd_to_gp_conversion_3y"], row["sharpe_ratio"]),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
            )
        ax.set_xlabel("3Y Intangible Investment to Gross Profit Delta (x)")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title("Chapter 5 Quant Screening: GP Conversion vs. Risk-Adjusted Return")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "ch5_quant_screening_map.png", dpi=150)
        plt.close(fig)
