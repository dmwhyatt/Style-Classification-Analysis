library(tidyverse)
library(psych)
library(jsonlite)

set.seed(42)

# Uses the same CSV and numeric column policy as logistic.py (includes corpus + LTM).
# Path mirrors output_paths.FEATURES_CSV (repo-root cache).
FEATURES_CSV <- "essen_china_europe_features.csv"
TRAIN_FRAC <- 0.8

# Canonical output locations, mirroring helpers/output_paths.py. Figure/table numbers
# match the "List of figures" block and the three result tables in paper.tex.
OUTPUTS_DIR <- "outputs"
FIGURES_DIR <- file.path(OUTPUTS_DIR, "figures")
TABLES_DIR <- file.path(OUTPUTS_DIR, "tables")
DATA_DIR <- file.path(OUTPUTS_DIR, "data")
dir.create(FIGURES_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(DATA_DIR, showWarnings = FALSE, recursive = TRUE)

FIG03_SCREE <- file.path(FIGURES_DIR, "fig03_factor_eigenvalues_elbow.pdf")
TABLE2_EFA_VARIANCE_CSV <- file.path(TABLES_DIR, "table2_efa_variance.csv")
TABLE2_EFA_VARIANCE_TEX <- file.path(TABLES_DIR, "table2_efa_variance.tex")
TABLE_S1_LOADINGS_CSV <- file.path(TABLES_DIR, "table_s1_factor_loadings_top10.csv")
TABLE_S1_LOADINGS_TEX <- file.path(TABLES_DIR, "table_s1_factor_loadings_top10.tex")

if (!file.exists(FEATURES_CSV)) {
  stop(
    "Missing features file: ", FEATURES_CSV, "\n",
    "Run logistic.py first to generate it."
  )
}

basename_no_ext <- function(x) {
  tools::file_path_sans_ext(basename(as.character(x)))
}

features <- read_csv(FEATURES_CSV, show_col_types = FALSE)

pearce_txt <- "pearce_default_idyom_basenames.txt"
if (!file.exists(pearce_txt)) {
  py <- if (file.exists("venv/bin/python3")) "venv/bin/python3" else Sys.which("python3")
  if (nzchar(py)) {
    code <- "from helpers.pearce_exclusion import write_pearce_basename_sidecar; write_pearce_basename_sidecar()"
    suppressWarnings(
      system2(py, args = c("-c", code), stdout = TRUE, stderr = TRUE, wait = TRUE)
    )
  }
}
if (file.exists(pearce_txt)) {
  pearce_bases <- readLines(pearce_txt, warn = FALSE)
  pearce_bases <- pearce_bases[nzchar(pearce_bases)]
  n_before <- nrow(features)
  features <- features %>%
    filter(!tolower(basename_no_ext(melody_id)) %in% pearce_bases)
  n_drop <- n_before - nrow(features)
  if (n_drop > 0) {
    message("Excluded ", n_drop, " melody row(s) overlapping pearce_default_idyom.")
  }
} else {
  warning(
    "pearce_default_idyom_basenames.txt not found; run `python logistic.py` once from ",
    "the project root to generate it. Proceeding without Pearce-IDyOM exclusion in R."
  )
}

features_numeric <- features %>%
  select(where(is.numeric)) %>%
  select(-melody_num)

# Match feature_selection.prepare_numeric_feature_matrix (Python): inf -> NA -> 0; keep all rows
features_numeric <- features_numeric %>%
  mutate(across(everything(), ~ replace(.x, is.infinite(.x), NA_real_))) %>%
  mutate(across(everything(), ~ replace_na(.x, 0)))
x_raw <- features_numeric

if (nrow(x_raw) == 0) {
  stop("No rows remain in feature matrix.")
}

features_model <- features %>%
  mutate(melody_key = tolower(basename_no_ext(melody_id)))

variances <- sapply(x_raw, var)
zero_var_cols <- names(variances[variances == 0 | is.na(variances)])
x_raw <- x_raw %>% select(-any_of(zero_var_cols))

x_scaled <- as.data.frame(scale(x_raw))

pa_mat <- as.matrix(x_scaled)
pa_result <- fa.parallel(
  pa_mat,
  fa = "fa",
  plot = FALSE,
  n.iter = 100
)

scree_df <- tibble(
  Factor = seq_along(pa_result$fa.values),
  Observed = pa_result$fa.values,
  Simulated = pa_result$fa.sim
)

p_scree <- ggplot(scree_df, aes(x = Factor)) +
  geom_line(aes(y = Observed, color = "Observed eigenvalues"), linewidth = 1) +
  geom_point(aes(y = Observed, color = "Observed eigenvalues"), size = 1.8) +
  geom_line(aes(y = Simulated, color = "Simulated eigenvalues"), linewidth = 1, linetype = "dashed") +
  labs(
    x = "Factor Number",
    y = "Eigenvalue",
    color = NULL
  ) +
  scale_color_manual(
    values = c(
      "Observed eigenvalues" = "#1f77b4",
      "Simulated eigenvalues" = "#d62728"
    )
  ) +
  theme_minimal(base_size = 12)

ggsave(FIG03_SCREE, plot = p_scree, width = 8, height = 6)
ggsave(
  sub("\\.pdf$", ".png", FIG03_SCREE),
  plot = p_scree, width = 8, height = 6, dpi = 150
)
cat("Saved", FIG03_SCREE, "(Figure 3)\n")

# inspect the scree/elbow outputs above to choose factor count
N_FACTORS <- 8

cat("Rows used:", nrow(x_scaled), "\n")
cat("Numeric features used:", ncol(x_scaled), "\n")
cat("Factors extracted:", N_FACTORS, "\n\n")

# EFA
fit <- fa(x_scaled, nfactors = N_FACTORS, rotate = "promax", fm = "pa")

loadings_mat <- as.matrix(fit$loadings[, ])
ss_loadings <- colSums(loadings_mat^2, na.rm = TRUE)
prop_var <- ss_loadings / nrow(loadings_mat)
cum_var <- cumsum(prop_var)

cat(
  "Cumulative variance explained by", N_FACTORS, "factors:",
  sprintf("%.2f%%", 100 * cum_var[length(cum_var)]), "\n\n"
)


# Interpreted using docs/index.html (3D network) and factor_top_loadings.csv
factor_names <- c(
  "1. Long Rhythms",
  "2. Irregular Rhythms",
  "3. Pitch-Class Variety",
  "4. Overall Complexity",
  "5. Wide Intervals",
  "6. Dense Rhythms",
  "7. Stepwise Complexity",
  "8. Corpus Familiarity"
)
if (length(factor_names) < N_FACTORS) {
  factor_names <- c(factor_names, paste0("Factor ", seq(length(factor_names) + 1, N_FACTORS)))
} else {
  factor_names <- factor_names[seq_len(N_FACTORS)]
}

# Supplementary materials: top loadings per factor (|loading|) for interpretation
TOP_LOADINGS_N <- 10
rownames(loadings_mat) <- colnames(x_scaled)
loading_top_list <- list()
for (j in seq_len(N_FACTORS)) {
  lj <- loadings_mat[, j, drop = TRUE]
  ord <- order(abs(lj), decreasing = TRUE)
  top_idx <- head(ord, TOP_LOADINGS_N)
  for (r in seq_along(top_idx)) {
    ii <- top_idx[r]
    loading_top_list[[length(loading_top_list) + 1]] <- tibble(
      factor_index = j,
      factor_code = paste0("F", j),
      factor_name = factor_names[j],
      rank = r,
      variable = rownames(loadings_mat)[ii],
      loading = unname(lj[ii]),
      abs_loading = abs(unname(lj[ii]))
    )
  }
}
loadings_top10_supplementary <- bind_rows(loading_top_list)
write_csv(loadings_top10_supplementary, TABLE_S1_LOADINGS_CSV)
cat(
  "Wrote ", TABLE_S1_LOADINGS_CSV, " (Table S1: top ",
  TOP_LOADINGS_N, " |loadings| per factor for supplementary interpretation)\n",
  sep = ""
)

# Table S1 as a LaTeX longtable fragment matching paper.tex's supplementary format
s1_tex_rows <- sprintf(
  "    %s & %d & %s & %.3f \\\\",
  loadings_top10_supplementary$factor_name,
  loadings_top10_supplementary$rank,
  loadings_top10_supplementary$variable,
  loadings_top10_supplementary$loading
)
s1_tex_lines <- c(
  "\\begingroup",
  "\\footnotesize",
  "\\setlength{\\tabcolsep}{3pt}",
  "\\begin{longtable}{@{}lr p{0.55\\textwidth} r@{}}",
  "\\toprule",
  "\\textbf{Factor Label} & \\textbf{Rank} & \\textbf{Feature} & \\textbf{Loading} \\\\",
  "\\midrule",
  "\\endfirsthead",
  "\\midrule",
  "\\textbf{Factor Label} & \\textbf{Rank} & \\textbf{Feature} & \\textbf{Loading} \\\\",
  "\\midrule",
  "\\endhead",
  "\\endfoot",
  "\\bottomrule",
  "\\caption{The ten highest loadings for each exploratory factor.}\\label{tab:supp-factor-loadings-top10}\\\\",
  "\\endlastfoot",
  s1_tex_rows,
  "\\end{longtable}",
  "\\endgroup"
)
writeLines(s1_tex_lines, TABLE_S1_LOADINGS_TEX)
cat("Wrote ", TABLE_S1_LOADINGS_TEX, "\n", sep = "")

# Top loadings per factor (CSV for tables / supplementary)
TOP_LOADINGS_PER_FACTOR <- 10L
top_loading_parts <- vector("list", N_FACTORS)
for (i in seq_len(N_FACTORS)) {
  v <- loadings_mat[, i]
  v[is.na(v)] <- 0
  ord <- order(abs(v), decreasing = TRUE)
  k <- min(TOP_LOADINGS_PER_FACTOR, length(ord))
  top_idx <- ord[seq_len(k)]
  top_loading_parts[[i]] <- tibble(
    factor = paste0("F", i),
    factor_name = factor_names[i],
    rank = seq_len(k),
    feature = rownames(loadings_mat)[top_idx],
    loading = v[top_idx],
    abs_loading = abs(v[top_idx])
  )
}
top_loadings_df <- bind_rows(top_loading_parts)
TOP_LOADINGS_CSV <- file.path(DATA_DIR, "factor_top_loadings.csv")
write_csv(top_loadings_df, TOP_LOADINGS_CSV)
cat(
  "Wrote ", TOP_LOADINGS_CSV, " (top ", TOP_LOADINGS_PER_FACTOR,
  " |loading| per factor)\n",
  sep = ""
)

# 3D network data for docs/index.html webapp (mirrors essen_new/create_3d_network_data.R)
NETWORK_CUTOFF <- 0.3
dir.create("docs", showWarnings = FALSE, recursive = TRUE)

build_3d_network_data <- function(loadings, cutoff = 0.3, factor_labels = NULL) {
  n_factors <- ncol(loadings)
  if (is.null(factor_labels)) {
    factor_labels <- paste0("Factor ", seq_len(n_factors))
  }

  nodes <- vector("list", 0)
  for (i in seq_len(n_factors)) {
    nodes[[length(nodes) + 1L]] <- list(
      id = paste0("F", i),
      name = factor_labels[i],
      type = "factor",
      val = 25L
    )
  }

  seen_vars <- character(0)
  links <- vector("list", 0)
  for (i in seq_len(n_factors)) {
    fl <- loadings[, i]
    fl[is.na(fl)] <- 0
    sig_idx <- which(abs(fl) > cutoff)
    for (j in sig_idx) {
      var_name <- rownames(loadings)[j]
      loading_value <- unname(fl[j])

      if (!var_name %in% seen_vars) {
        nodes[[length(nodes) + 1L]] <- list(
          id = var_name,
          name = var_name,
          type = "variable",
          val = 8L
        )
        seen_vars <- c(seen_vars, var_name)
      }

      links[[length(links) + 1L]] <- list(
        source = paste0("F", i),
        target = var_name,
        value = round(abs(loading_value), 4),
        sign = if (loading_value > 0) "positive" else "negative"
      )
    }
  }

  list(nodes = nodes, links = links)
}

network_data <- build_3d_network_data(
  loadings_mat,
  cutoff = NETWORK_CUTOFF,
  factor_labels = factor_names
)
# Top |loading| features per factor for docs/index.html "Loadings" tab
network_data$factor_loadings <- lapply(top_loading_parts, function(df) {
  lapply(seq_len(nrow(df)), function(r) {
    list(
      rank = as.integer(df$rank[r]),
      feature = df$feature[r],
      loading = round(unname(df$loading[r]), 4),
      abs_loading = round(unname(df$abs_loading[r]), 4)
    )
  })
})
names(network_data$factor_loadings) <- paste0("F", seq_len(N_FACTORS))
json_data <- toJSON(network_data, auto_unbox = TRUE, pretty = TRUE)
writeLines(json_data, "docs/network_data.json")
# JS shim so docs/index.html can be opened directly from disk (no HTTP server needed)
writeLines(c("const networkData =", json_data, ";"), "docs/network_data.js")
loadings_json <- toJSON(network_data$factor_loadings, auto_unbox = TRUE, pretty = TRUE)
writeLines(c("window.factorLoadingsData =", loadings_json, ";"), "docs/factor_loadings.js")
cat(sprintf(
  "Wrote docs/network_data.json and docs/network_data.js (%d nodes, %d links)\n",
  length(network_data$nodes), length(network_data$links)
))
cat("Wrote docs/factor_loadings.js\n")

cat("\nUsing factor names:\n")
for (i in seq_along(factor_names)) {
  cat(sprintf("  F%d: %s\n", i, factor_names[i]))
}
cat("\n")

# per-factor variance table
var_table <- data.frame(
  Factor     = paste0("F", seq_len(N_FACTORS)),
  Name       = factor_names,
  SS_Loading = round(ss_loadings, 4),
  Prop_Var   = round(100 * prop_var, 2),
  Cum_Var    = round(100 * cum_var,  2)
)
cat("Per-factor variance explained:\n")
print(var_table, row.names = FALSE)
cat("\n")

# Table 2: EFA proportion/cumulative variance explained (paper.tex tab:efa-variance)
write_csv(var_table, TABLE2_EFA_VARIANCE_CSV)
t2_tex_rows <- sprintf(
  "    %d & %s & %.2f & %.2f \\\\",
  seq_len(N_FACTORS), factor_names, var_table$Prop_Var, var_table$Cum_Var
)
t2_tex_lines <- c(
  "\\begin{table}[h]",
  "  \\centering",
  "  \\label{tab:efa-variance}",
  "  \\begin{tabular}{@{}clcc@{}}",
  "    Factor & Interpretation & Prop.\\ Var.\\ (\\%) & Cum.\\ Var.\\ (\\%) \\\\",
  "    \\midrule",
  t2_tex_rows,
  "  \\end{tabular}",
  "  \\caption{EFA: proportion and cumulative variance explained by each factor.}",
  "\\end{table}"
)
writeLines(t2_tex_lines, TABLE2_EFA_VARIANCE_TEX)
cat("Wrote ", TABLE2_EFA_VARIANCE_CSV, " and ", TABLE2_EFA_VARIANCE_TEX, " (Table 2)\n", sep = "")

# create factor scores for logistic regression
score_result <- factor.scores(x_scaled, fit, method = "regression")
factor_scores <- as.data.frame(score_result$scores)
colnames(factor_scores) <- paste0("F", seq_len(ncol(factor_scores)))

scores_df <- bind_cols(
  features_model %>% select(melody_id, melody_key),
  factor_scores
)
FACTOR_SCORES_CSV <- file.path(DATA_DIR, "factor_scores_for_logreg.csv")
write_csv(scores_df, FACTOR_SCORES_CSV)
cat("Wrote", FACTOR_SCORES_CSV, "\n")

model_df <- scores_df %>%
  left_join(
    features_model %>% select(melody_key, target = continent),
    by = "melody_key"
  ) %>%
  filter(!is.na(target))

if (nrow(model_df) == 0) {
  stop("No rows matched usable labels. Check melody_id formatting.")
}

class_counts <- table(model_df$target)
if (length(class_counts) != 2) {
  stop("Expected exactly two classes, got: ", paste(names(class_counts), collapse = ", "))
}

train_rows <- unlist(lapply(split(seq_len(nrow(model_df)), model_df$target), function(idx) {
  sample(idx, size = floor(TRAIN_FRAC * length(idx)))
}))
train_df <- model_df[train_rows, ]
test_df <- model_df[-train_rows, ]

feature_factor_cols <- paste0("F", seq_len(N_FACTORS))
logit_formula <- as.formula(paste("target ~", paste(feature_factor_cols, collapse = " + ")))

# 5-fold stratified CV on train
fold_ids <- integer(nrow(train_df))
for (cls in unique(train_df$target)) {
  idx <- which(train_df$target == cls)
  idx <- sample(idx)
  fold_ids[idx] <- rep(1:5, length.out = length(idx))
}

negative_class <- names(class_counts)[1]
positive_class <- names(class_counts)[2]
train_df$target_bin <- ifelse(train_df$target == positive_class, 1, 0)
test_df$target_bin <- ifelse(test_df$target == positive_class, 1, 0)

cv_acc      <- numeric(5)
cv_true_all <- character(0)
cv_pred_all <- character(0)

for (k in 1:5) {
  tr <- train_df[fold_ids != k, ]
  va <- train_df[fold_ids == k, ]
  model_k <- glm(update(logit_formula, target_bin ~ .), data = tr, family = binomial())
  prob_k <- predict(model_k, newdata = va, type = "response")
  pred_k <- ifelse(prob_k >= 0.5, positive_class, negative_class)
  cv_acc[k]   <- mean(pred_k == va$target)
  cv_true_all <- c(cv_true_all, va$target)
  cv_pred_all <- c(cv_pred_all, pred_k)
}

# Final model and held-out test
final_model <- glm(update(logit_formula, target_bin ~ .), data = train_df, family = binomial())
test_prob <- predict(final_model, newdata = test_df, type = "response")
test_pred <- ifelse(test_prob >= 0.5, positive_class, negative_class)
test_acc <- mean(test_pred == test_df$target)

coef_df <- as.data.frame(summary(final_model)$coefficients)
coef_df$term <- rownames(coef_df)
FACTOR_COEF_CSV <- file.path(DATA_DIR, "logistic_factor_coefficients.csv")
write_csv(coef_df, FACTOR_COEF_CSV)

metrics_df <- tibble(
  metric = c("cv_accuracy_mean", "cv_accuracy_sd", "test_accuracy"),
  value = c(mean(cv_acc), sd(cv_acc), test_acc)
)
FACTOR_METRICS_CSV <- file.path(DATA_DIR, "logistic_factor_metrics.csv")
write_csv(metrics_df, FACTOR_METRICS_CSV)

cat(sprintf("CV accuracy: %.4f +/- %.4f\n", mean(cv_acc), sd(cv_acc)))
cat(sprintf("Test accuracy: %.4f\n", test_acc))
cat("Wrote", FACTOR_METRICS_CSV, "and", FACTOR_COEF_CSV, "\n")

classes <- c(negative_class, positive_class)

# predictions for Python plots (Figures 4 and 5)
TEST_PRED_CSV <- file.path(DATA_DIR, "factor_logistic_predictions_test.csv")
CV_PRED_CSV <- file.path(DATA_DIR, "factor_logistic_predictions_cv.csv")
CLASS_ORDER_CSV <- file.path(DATA_DIR, "factor_logistic_class_order.csv")
write_csv(
  tibble(true_label = test_df$target, predicted = test_pred),
  TEST_PRED_CSV
)
write_csv(
  tibble(true_label = cv_true_all, predicted = cv_pred_all),
  CV_PRED_CSV
)
write_csv(tibble(class_label = classes), CLASS_ORDER_CSV)
cat(
  "Wrote", TEST_PRED_CSV, ",", CV_PRED_CSV, "and", CLASS_ORDER_CSV, "\n",
  "Run: python factor_logistic_plots.py\n"
)

# factor importance table: variance explained + logistic coefficient + significance
factor_coefs <- coef_df[coef_df$term != "(Intercept)", ]
factor_idx   <- as.integer(sub("^F", "", factor_coefs$term))
importance_df <- data.frame(
  Factor   = factor_coefs$term,
  Name     = factor_names[factor_idx],
  Prop_Var = round(100 * prop_var[factor_idx], 2),
  Coef     = round(factor_coefs$Estimate, 4),
  Std_Err  = round(factor_coefs$`Std. Error`, 4),
  Z        = round(factor_coefs$`z value`, 3),
  P_value  = signif(factor_coefs$`Pr(>|z|)`, 3),
  Sig      = ifelse(factor_coefs$`Pr(>|z|)` < 0.001, "***",
             ifelse(factor_coefs$`Pr(>|z|)` < 0.01,  "**",
             ifelse(factor_coefs$`Pr(>|z|)` < 0.05,  "*",
             ifelse(factor_coefs$`Pr(>|z|)` < 0.1,   ".", ""))))
)
importance_df <- importance_df[order(-abs(importance_df$Coef)), ]
cat("\nFactor importance (sorted by |coefficient|):\n")
print(importance_df, row.names = FALSE)
