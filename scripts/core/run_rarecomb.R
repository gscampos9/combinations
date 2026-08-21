#!/usr/bin/env Rscript
# Run RareComb's compare_enrichment() on an Input_*/Output_1 boolean matrix
# Usage:
#   Rscript run_rarecomb.R <input_matrix.txt> <output.txt> \
#       <combo_length> <min_indv_threshold> <max_freq_threshold> \
#       [pval_filter_threshold] [min_power_threshold]
#
#   Rscript run_rarecomb.R rarecomb_input.txt rarecomb_output.txt 2 3 0.25 0.05

library(arules)
library(RareComb)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 5) {
  stop(paste("Usage: Rscript run_rarecomb.R <input_matrix.txt> <output.txt>",
             "<combo_length> <min_indv_threshold> <max_freq_threshold>",
             "[pval_filter_threshold] [min_power_threshold]"))
}

input_path  <- args[1]
output_path <- args[2]
combo_length        <- as.integer(args[3])
min_indv_threshold  <- as.integer(args[4])
max_freq_threshold  <- as.numeric(args[5])
pval_filter_threshold <- if (length(args) >= 6) as.numeric(args[6]) else 0.05
min_power_threshold   <- if (length(args) >= 7) as.numeric(args[7]) else NULL

cat(sprintf("Loading input: %s\n", input_path))
if (requireNamespace("data.table", quietly = TRUE)) {
  input <- data.table::fread(input_path, header = TRUE, sep = "\t", data.table = FALSE)
} else {
  cat("[note] data.table not installed, falling back to slower read.table()\n")
  input <- read.table(input_path, header = TRUE, sep = "\t")
}

cat(sprintf("Running RareComb compare_enrichment() (combo_length=%d, min_indv_threshold=%d, max_freq_threshold=%s, pval_filter_threshold=%s, min_power_threshold=%s)...\n",
            combo_length, min_indv_threshold, max_freq_threshold, pval_filter_threshold,
            if (is.null(min_power_threshold)) "default" else min_power_threshold))

call_args <- list(
  input, combo_length, min_indv_threshold, max_freq_threshold,
  input_format = "Input_", output_format = "Output_",
  pval_filter_threshold = pval_filter_threshold, adj_pval_type = "BH",
  sample_names_ind = "Y"
)
if (!is.null(min_power_threshold)) {
  call_args$min_power_threshold <- min_power_threshold
}
output <- do.call(compare_enrichment, call_args)

write.csv(output, output_path, row.names = FALSE)
cat(sprintf("Saved: %s\n", output_path))
