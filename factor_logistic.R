library(tidyverse)
library(psych)
library(plotly)
library(htmlwidgets)
library(visNetwork)

set.seed(42)

# Uses the same CSV as logistic.py; corpus features are excluded.
FEATURES_CSV <- "essen_china_europe_features.csv"
TRAIN_FRAC <- 0.8

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

features_numeric <- features %>%
  select(where(is.numeric)) %>%
  select(-melody_num, -starts_with("corpus."), -contains("_ltm_"))
features_numeric_clean <- features_numeric %>% drop_na()
x_raw <- features_numeric_clean

if (nrow(x_raw) == 0) {
  stop("No complete rows remain after drop_na().")
}

row_idx <- as.integer(rownames(x_raw))
features_model <- features[row_idx, ] %>%
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

ggsave("factor_eigenvalues_elbow.pdf", plot = p_scree, width = 8, height = 6)

# interactive version makes it easier to count factors and identify the elbow
p_scree_interactive <- ggplotly(p_scree, tooltip = c("x", "y", "colour"))
saveWidget(
  p_scree_interactive,
  file = "factor_eigenvalues_elbow_interactive.html",
  selfcontained = FALSE
)
cat("Saved factor_eigenvalues_elbow.pdf and factor_eigenvalues_elbow_interactive.html\n")

# inspect the scree/elbow outputs above to choose factor count
N_FACTORS <- 9

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


# intepreted using the network diagram below
factor_names <- c(
  "1. Long Pulses",
  "2. Low Rhythmic Density",
  "3. Pitch Complexity",
  "4. Timing Variability",
  "5. Intervallic Complexity",
  "6. Higher Rhythmic Range",
  "7. Scale Conformity",
  "8. General Complexity",
  "9. Higher Absolute Pitch"
)
if (length(factor_names) < N_FACTORS) {
  factor_names <- c(factor_names, paste0("Factor ", seq(length(factor_names) + 1, N_FACTORS)))
} else {
  factor_names <- factor_names[seq_len(N_FACTORS)]
}

# Interactive loading graph to help interpret the factors
cat("\nCreating interactive factor network (visNetwork)...\n")
rownames(loadings_mat) <- colnames(x_scaled)

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
write_csv(top_loadings_df, "factor_top_loadings.csv")
cat(
  "Wrote factor_top_loadings.csv (top ", TOP_LOADINGS_PER_FACTOR,
  " |loading| per factor)\n",
  sep = ""
)

create_network_diagram <- function(loadings, cutoff = 0.3, max_factors = NULL,
                                   custom_names = NULL) {
  if (is.null(max_factors)) {
    max_factors <- ncol(loadings)
  }
  if (is.null(custom_names)) {
    factor_labels <- paste0("Factor ", seq_len(max_factors))
  } else {
    factor_labels <- custom_names[seq_len(max_factors)]
  }

  factor_nodes <- data.frame(
    id = paste0("F", seq_len(max_factors)),
    label = factor_labels,
    group = "factor",
    shape = "ellipse",
    color = "#ff7f0e",
    size = 30,
    font.size = 20,
    title = factor_labels,
    stringsAsFactors = FALSE
  )

  var_nodes_list <- list()
  edges_list <- list()

  for (i in seq_len(max_factors)) {
    factor_loadings <- loadings[, i]
    factor_loadings[is.na(factor_loadings)] <- 0
    sig_idx <- abs(factor_loadings) > cutoff

    if (sum(sig_idx) > 0) {
      sig_vars <- names(factor_loadings)[sig_idx]
      sig_values <- factor_loadings[sig_idx]

      for (j in seq_along(sig_vars)) {
        var_name <- sig_vars[j]
        loading_value <- sig_values[j]

        if (!var_name %in% names(var_nodes_list)) {
          var_nodes_list[[var_name]] <- data.frame(
            id = var_name,
            label = var_name,
            group = "variable",
            shape = "box",
            color = "#1f77b4",
            size = 20,
            font.size = 12,
            title = var_name,
            stringsAsFactors = FALSE
          )
        }

        edge_color <- ifelse(loading_value > 0, "green", "red")
        edges_list[[length(edges_list) + 1]] <- data.frame(
          from = paste0("F", i),
          to = var_name,
          value = abs(loading_value) * 10,
          title = paste0("Loading: ", round(loading_value, 3)),
          color = edge_color,
          arrows = "to",
          stringsAsFactors = FALSE
        )
      }
    }
  }

  if (length(edges_list) == 0) {
    warning("No edges above loading cutoff ", cutoff, "; network HTML not written.")
    return(NULL)
  }

  var_nodes <- do.call(rbind, var_nodes_list)
  all_nodes <- rbind(factor_nodes, var_nodes)
  all_edges <- do.call(rbind, edges_list)

  visNetwork(all_nodes, all_edges, width = "100%", height = "800px") %>%
    visGroups(groupname = "factor", color = "#ff7f0e", shape = "ellipse") %>%
    visGroups(groupname = "variable", color = "#1f77b4", shape = "box") %>%
    visOptions(
      highlightNearest = list(enabled = TRUE, degree = 1, hover = TRUE),
      nodesIdSelection = TRUE,
      selectedBy = "group"
    ) %>%
    visLayout(randomSeed = 123) %>%
    visPhysics(
      solver = "forceAtlas2Based",
      forceAtlas2Based = list(gravitationalConstant = -50)
    ) %>%
    visInteraction(
      navigationButtons = TRUE,
      dragNodes = TRUE,
      dragView = TRUE,
      zoomView = TRUE
    ) %>%
    visLegend(width = 0.1, position = "right", main = "Node Type")
}

full_diagram <- create_network_diagram(
  loadings_mat,
  cutoff = 0.3,
  max_factors = N_FACTORS,
  custom_names = factor_names
)
if (!is.null(full_diagram)) {
  visSave(full_diagram, "factor_network_full.html", selfcontained = FALSE)
  cat("Saved factor_network_full.html (open in browser; companion folder may be created)\n")
}

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

# create factor scores for logistic regression
score_result <- factor.scores(x_scaled, fit, method = "regression")
factor_scores <- as.data.frame(score_result$scores)
colnames(factor_scores) <- paste0("F", seq_len(ncol(factor_scores)))

scores_df <- bind_cols(
  features_model %>% select(melody_id, melody_key),
  factor_scores
)
write_csv(scores_df, "factor_scores_for_logreg.csv")
cat("Wrote factor_scores_for_logreg.csv\n")

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
write_csv(coef_df, "logistic_factor_coefficients.csv")

metrics_df <- tibble(
  metric = c("cv_accuracy_mean", "cv_accuracy_sd", "test_accuracy"),
  value = c(mean(cv_acc), sd(cv_acc), test_acc)
)
write_csv(metrics_df, "logistic_factor_metrics.csv")

cat(sprintf("CV accuracy: %.4f +/- %.4f\n", mean(cv_acc), sd(cv_acc)))
cat(sprintf("Test accuracy: %.4f\n", test_acc))
cat("Wrote logistic_factor_metrics.csv and logistic_factor_coefficients.csv\n")

classes <- c(negative_class, positive_class)

# predictions for Python plots
write_csv(
  tibble(true_label = test_df$target, predicted = test_pred),
  "factor_logistic_predictions_test.csv"
)
write_csv(
  tibble(true_label = cv_true_all, predicted = cv_pred_all),
  "factor_logistic_predictions_cv.csv"
)
write_csv(tibble(class_label = classes), "factor_logistic_class_order.csv")
cat(
  "Wrote factor_logistic_predictions_*.csv and factor_logistic_class_order.csv\n",
  "Run: python factor_logistic_plot_confusion.py\n"
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
