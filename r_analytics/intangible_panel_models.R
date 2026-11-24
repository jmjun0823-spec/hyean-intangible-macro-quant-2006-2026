suppressPackageStartupMessages({
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
input_path <- if (length(args) >= 1) args[1] else "outputs/yearly_macro_gap_and_flows.csv"
output_dir <- if (length(args) >= 2) args[2] else "outputs/r_models"

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (file.exists(input_path)) {
  macro_df <- read_csv(input_path, show_col_types = FALSE)
  ols_flow <- lm(net_fund_flow_b ~ reported_intangible_capex_share + delta_y_gap, data = macro_df)
  writeLines(capture.output(summary(ols_flow)), con = file.path(output_dir, "flow_intangible_ols_summary.txt"))
  cat("R OLS model completed successfully.\n")
} else {
  cat("Input CSV not found; skipping R OLS estimation.\n")
}
