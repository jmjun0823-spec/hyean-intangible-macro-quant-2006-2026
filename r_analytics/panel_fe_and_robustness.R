suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
macro_path <- if (length(args) >= 1) args[1] else "outputs/yearly_macro_gap_and_flows.csv"
scored_path <- if (length(args) >= 2) args[2] else "outputs/quant_scored_universe.csv"
out_dir <- if (length(args) >= 3) args[3] else "outputs/r_models"

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

if (file.exists(macro_path)) {
  m_df <- read_csv(macro_path, show_col_types = FALSE) %>%
    mutate(post_55_regime = ifelse(reported_intangible_capex_share >= 0.55, 1, 0))

  robust_lm <- lm(
    net_fund_flow_b ~ reported_intangible_capex_share + post_55_regime + delta_y_gap,
    data = m_df
  )
  writeLines(capture.output(summary(robust_lm)), con = file.path(out_dir, "robustness_flow_regime.txt"))
}

if (file.exists(scored_path)) {
  s_df <- read_csv(scored_path, show_col_types = FALSE)
  sharpe_lm <- lm(
    sharpe_ratio ~ rd_to_gp_conversion_3y + fcf_margin_3y_avg + adjusted_roic_3y_avg + factor(sector_type),
    data = s_df
  )
  writeLines(capture.output(summary(sharpe_lm)), con = file.path(out_dir, "sharpe_factor_regression.txt"))
}

cat("R robustness models completed.\n")
