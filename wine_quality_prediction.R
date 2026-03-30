################################################################################
# Wine Quality Prediction using Machine Learning in R
#
# Dataset: UCI White Wine Quality
# Source: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
# Paper: Cortez et al., "Modeling wine preferences by data mining from
#        physicochemical properties", Decision Support Systems 47(4):547-553, 2009
#        DOI: 10.1016/j.dss.2009.05.016
#
# Models: Random Forest, GBM (Gradient Boosting Machine), Elastic Net
# Split: Train (70%) / Validation (10%) / Test (20%)
################################################################################

# ==============================================================================
# Section 0: Setup & Package Installation
# ==============================================================================

required_packages <- c("caret", "ranger", "gbm", "glmnet",
                       "corrplot", "ggplot2", "reshape2")

missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop(
    paste0(
      "The following required R packages are not installed: ",
      paste(missing_packages, collapse = ", "),
      ".\nPlease install them before running this script, for example:\n",
      "install.packages(c(\"",
      paste(missing_packages, collapse = "\", \""), "\"))"
    ),
    call. = FALSE
  )
}

library(caret)
library(ranger)
library(gbm)
library(glmnet)
library(corrplot)
library(ggplot2)
library(reshape2)

set.seed(42)

# ==============================================================================
# Section 1: Data & Paper Download
# ==============================================================================

dir.create("data", showWarnings = FALSE)
dir.create("plots", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

# Download white wine quality dataset
data_urls <- c(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",
  "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv?download=1",
  "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/winequality-white.csv"
)
data_file <- "data/winequality-white.csv"

if (!file.exists(data_file)) {
  cat("White wine quality dataset not found locally. Attempting to download...\n")
  data_downloaded <- FALSE
  for (url in data_urls) {
    tryCatch({
      download.file(url, data_file, method = "auto", quiet = FALSE)
      if (file.exists(data_file) && file.size(data_file) > 1000) {
        cat("Dataset downloaded successfully from:", url, "\n")
        data_downloaded <- TRUE
        break
      } else {
        file.remove(data_file)
      }
    }, error = function(e) {
      cat("  Could not download dataset from:", url, "\n")
    })
  }
  if (!data_downloaded) {
    stop(
      "Failed to download the white wine quality dataset.\n",
      "Please download it manually from one of the following URLs:\n  ",
      paste(data_urls, collapse = "\n  "),
      "\n and save it as '", data_file, "'."
    )
  }
}

# Research paper information (no automatic PDF download to avoid licensing issues)
paper_file <- "data/cortez2009_wine_quality.pdf"
if (!file.exists(paper_file)) {
  cat("\nThe research paper associated with this dataset is not downloaded automatically.\n")
  cat("Please access it manually using the citation and DOI below:\n\n")
  cat("  Title: Modeling wine preferences by data mining from physicochemical properties\n")
  cat("  Authors: P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis\n")
  cat("  Journal: Decision Support Systems, 47(4), 547-553, 2009\n")
  cat("  DOI: https://doi.org/10.1016/j.dss.2009.05.016\n\n")
  cat("You may also try the following URL in your browser (if accessible and permitted):\n")
  cat("  https://repositorium.sdum.uminho.pt/bitstream/1822/10029/1/wine5.pdf\n\n")
}

# ==============================================================================
# Section 2: Data Loading & Inspection
# ==============================================================================

cat("\n=== DATA LOADING ===\n")
df <- read.csv(data_file, sep = ";")

cat("Dimensions:", nrow(df), "rows x", ncol(df), "columns\n")
cat("Column names:", paste(names(df), collapse = ", "), "\n\n")

cat("Summary statistics:\n")
print(summary(df))

cat("\nMissing values per column:\n")
print(colSums(is.na(df)))

cat("\nData structure:\n")
str(df)

# ==============================================================================
# Section 3: Exploratory Data Analysis
# ==============================================================================

cat("\n=== EXPLORATORY DATA ANALYSIS ===\n")

# Quality distribution
png("plots/quality_distribution.png", width = 800, height = 500)
ggplot(df, aes(x = factor(quality))) +
  geom_bar(fill = "steelblue", color = "black") +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.5) +
  labs(title = "Distribution of White Wine Quality Scores",
       x = "Quality Score", y = "Count") +
  theme_minimal(base_size = 14)
dev.off()
cat("Saved: plots/quality_distribution.png\n")

# Correlation matrix
png("plots/correlation_matrix.png", width = 900, height = 800)
cor_matrix <- cor(df)
corrplot(cor_matrix, method = "color", type = "upper",
         addCoef.col = "black", number.cex = 0.7,
         tl.col = "black", tl.srt = 45,
         title = "Feature Correlation Matrix",
         mar = c(0, 0, 2, 0))
dev.off()
cat("Saved: plots/correlation_matrix.png\n")

# Boxplots of key features by quality
key_features <- c("alcohol", "volatile.acidity", "density", "residual.sugar")
png("plots/feature_boxplots.png", width = 1000, height = 800)
df_long <- melt(df[, c(key_features, "quality")], id.vars = "quality")
ggplot(df_long, aes(x = factor(quality), y = value, fill = factor(quality))) +
  geom_boxplot(show.legend = FALSE) +
  facet_wrap(~ variable, scales = "free_y", ncol = 2) +
  labs(title = "Key Features by Wine Quality",
       x = "Quality Score", y = "Value") +
  theme_minimal(base_size = 12)
dev.off()
cat("Saved: plots/feature_boxplots.png\n")

# ==============================================================================
# Section 4: Data Splitting (7:1:2 = Train:Validation:Test)
# ==============================================================================

cat("\n=== DATA SPLITTING (70% Train / 10% Validation / 20% Test) ===\n")

set.seed(42)

# First split: 80% (train+val) and 20% (test)
test_index <- createDataPartition(df$quality, p = 0.2, list = FALSE)
test_set <- df[test_index, ]
train_val_set <- df[-test_index, ]

# Second split: from the 80%, take 12.5% as validation (= 10% of total)
# 0.125 of 80% = 10% of total
val_index <- createDataPartition(train_val_set$quality, p = 0.125, list = FALSE)
val_set <- train_val_set[val_index, ]
train_set <- train_val_set[-val_index, ]

cat(sprintf("Training set:   %d samples (%.1f%%)\n", nrow(train_set), 100 * nrow(train_set) / nrow(df)))
cat(sprintf("Validation set: %d samples (%.1f%%)\n", nrow(val_set), 100 * nrow(val_set) / nrow(df)))
cat(sprintf("Test set:       %d samples (%.1f%%)\n", nrow(test_set), 100 * nrow(test_set) / nrow(df)))

# ==============================================================================
# Section 5: Feature Engineering & Preprocessing
# ==============================================================================

cat("\n=== FEATURE ENGINEERING & PREPROCESSING ===\n")

# Separate features and target
feature_cols <- setdiff(names(df), "quality")

# --- 5a: Outlier Detection & Capping (IQR-based Winsorizing) ---
cat("Detecting and capping outliers using IQR method (winsorizing)...\n")

# Compute bounds from training data only (3 * IQR rule)
IQR_MULTIPLIER <- 3.0
outlier_bounds <- lapply(feature_cols, function(col) {
  x <- train_set[[col]]
  q1 <- quantile(x, 0.25)
  q3 <- quantile(x, 0.75)
  iqr <- q3 - q1
  list(lower = q1 - IQR_MULTIPLIER * iqr, upper = q3 + IQR_MULTIPLIER * iqr)
})
names(outlier_bounds) <- feature_cols

cap_outliers <- function(df_subset) {
  for (col in feature_cols) {
    df_subset[[col]] <- pmax(pmin(df_subset[[col]],
                                  outlier_bounds[[col]]$upper),
                             outlier_bounds[[col]]$lower)
  }
  df_subset
}

train_set_capped <- cap_outliers(train_set)
val_set_capped   <- cap_outliers(val_set)
test_set_capped  <- cap_outliers(test_set)

total_capped <- sum(sapply(feature_cols, function(col) {
  sum(train_set[[col]] != train_set_capped[[col]])
}))
cat(sprintf("  Capped %d outlier values in training set (3 * IQR threshold).\n", total_capped))

# --- 5b: Polynomial & Interaction Feature Engineering ---
cat("Adding polynomial and interaction features...\n")

# Squared terms for physicochemical properties most correlated with quality
poly_features <- c("alcohol", "volatile.acidity", "density",
                   "residual.sugar", "free.sulfur.dioxide")

add_engineered_features <- function(df_subset) {
  # Squared (polynomial degree-2) terms
  for (f in poly_features) {
    df_subset[[paste0(f, "_sq")]] <- df_subset[[f]]^2
  }
  # Interaction terms between key feature pairs
  df_subset$alcohol_x_volatile.acidity <- df_subset$alcohol * df_subset$volatile.acidity
  df_subset$alcohol_x_density          <- df_subset$alcohol * df_subset$density
  df_subset$density_x_residual.sugar   <- df_subset$density * df_subset$residual.sugar
  df_subset
}

train_set_eng <- add_engineered_features(train_set_capped)
val_set_eng   <- add_engineered_features(val_set_capped)
test_set_eng  <- add_engineered_features(test_set_capped)

all_feature_cols <- setdiff(names(train_set_eng), "quality")
n_interaction_features <- length(all_feature_cols) - length(feature_cols) - length(poly_features)
cat(sprintf("  Features: %d original -> %d after engineering (%d poly + %d interaction)\n",
            length(feature_cols), length(all_feature_cols),
            length(poly_features), n_interaction_features))

# --- 5c: Center + Scale ---
cat("Fitting center/scale preprocessor on training data...\n")

# Fit preprocessing on training data only
preproc <- preProcess(train_set_eng[, all_feature_cols], method = c("center", "scale"))

train_x <- predict(preproc, train_set_eng[, all_feature_cols])
train_y <- train_set_eng$quality

val_x <- predict(preproc, val_set_eng[, all_feature_cols])
val_y <- val_set_eng$quality

test_x <- predict(preproc, test_set_eng[, all_feature_cols])
test_y <- test_set_eng$quality

cat("Preprocessing complete.\n")

# Recombine for caret::train
train_data <- cbind(train_x, quality = train_y)
val_data   <- cbind(val_x, quality = val_y)

# ==============================================================================
# Section 6: Model Training & Hyperparameter Tuning
# ==============================================================================

cat("\n=== MODEL TRAINING WITH HYPERPARAMETER TUNING ===\n")
cat("Using 5-fold cross-validation on training set for tuning.\n\n")

# Common train control: 5-fold CV
train_ctrl <- trainControl(method = "cv", number = 5, verboseIter = FALSE)

# Helper to compute metrics
calc_metrics <- function(actual, predicted) {
  data.frame(
    RMSE = sqrt(mean((actual - predicted)^2)),
    MAE = mean(abs(actual - predicted)),
    R2 = 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  )
}

# ---------- 6a: Random Forest (ranger) ----------
cat("--- Training Random Forest ---\n")

rf_grid <- expand.grid(
  mtry = c(2, 4, 6, 8, 11, 14, 19),  # 19 = total feature count after engineering
  splitrule = c("variance", "extratrees"),
  min.node.size = c(3, 5, 10)
)

set.seed(42)
rf_model <- train(
  quality ~ .,
  data = train_data,
  method = "ranger",
  trControl = train_ctrl,
  tuneGrid = rf_grid,
  importance = "impurity"
)

cat("Best RF parameters:\n")
print(rf_model$bestTune)
cat(sprintf("Best CV RMSE: %.4f\n\n", min(rf_model$results$RMSE)))

# ---------- 6b: GBM (Gradient Boosting Machine) ----------
cat("--- Training GBM ---\n")

gbm_grid <- expand.grid(
  n.trees = c(100, 200, 500, 1000),
  interaction.depth = c(3, 5, 7),
  shrinkage = c(0.01, 0.05, 0.1),
  n.minobsinnode = c(5, 10)
)

set.seed(42)
gbm_model <- train(
  quality ~ .,
  data = train_data,
  method = "gbm",
  trControl = train_ctrl,
  tuneGrid = gbm_grid,
  verbose = FALSE
)

cat("Best GBM parameters:\n")
print(gbm_model$bestTune)
cat(sprintf("Best CV RMSE: %.4f\n\n", min(gbm_model$results$RMSE)))

# ---------- 6c: Elastic Net (glmnet) ----------
cat("--- Training Elastic Net ---\n")

enet_grid <- expand.grid(
  alpha = seq(0, 1, by = 0.2),
  lambda = 10^seq(-4, 0, length.out = 20)
)

set.seed(42)
enet_model <- train(
  quality ~ .,
  data = train_data,
  method = "glmnet",
  trControl = train_ctrl,
  tuneGrid = enet_grid
)

cat("Best Elastic Net parameters:\n")
print(enet_model$bestTune)
cat(sprintf("Best CV RMSE: %.4f\n\n", min(enet_model$results$RMSE)))

# ==============================================================================
# Section 7: Validation Set Comparison
# ==============================================================================

cat("\n=== VALIDATION SET RESULTS ===\n")

rf_val_pred <- predict(rf_model, val_x)
gbm_val_pred <- predict(gbm_model, val_x)
enet_val_pred <- predict(enet_model, val_x)

val_results <- rbind(
  cbind(Model = "Random Forest", calc_metrics(val_y, rf_val_pred)),
  cbind(Model = "GBM", calc_metrics(val_y, gbm_val_pred)),
  cbind(Model = "Elastic Net", calc_metrics(val_y, enet_val_pred))
)

print(val_results, row.names = FALSE)

best_model_name <- val_results$Model[which.min(val_results$RMSE)]
cat(sprintf("\nBest model on validation set: %s\n", best_model_name))

# ==============================================================================
# Section 8: Final Test Set Evaluation
# ==============================================================================

cat("\n=== FINAL TEST SET RESULTS ===\n")

rf_test_pred <- predict(rf_model, test_x)
gbm_test_pred <- predict(gbm_model, test_x)
enet_test_pred <- predict(enet_model, test_x)

test_results <- rbind(
  cbind(Model = "Random Forest", calc_metrics(test_y, rf_test_pred)),
  cbind(Model = "GBM", calc_metrics(test_y, gbm_test_pred)),
  cbind(Model = "Elastic Net", calc_metrics(test_y, enet_test_pred))
)

print(test_results, row.names = FALSE)

best_test_model <- test_results$Model[which.min(test_results$RMSE)]
cat(sprintf("\nBest model on test set: %s\n", best_test_model))

# Actual vs Predicted plots
png("plots/actual_vs_predicted.png", width = 1200, height = 400)
par(mfrow = c(1, 3))

plot(test_y, rf_test_pred, main = "Random Forest",
     xlab = "Actual Quality", ylab = "Predicted Quality",
     pch = 16, col = rgb(0.2, 0.4, 0.8, 0.3))
abline(0, 1, col = "red", lwd = 2)

plot(test_y, gbm_test_pred, main = "GBM",
     xlab = "Actual Quality", ylab = "Predicted Quality",
     pch = 16, col = rgb(0.2, 0.4, 0.8, 0.3))
abline(0, 1, col = "red", lwd = 2)

plot(test_y, enet_test_pred, main = "Elastic Net",
     xlab = "Actual Quality", ylab = "Predicted Quality",
     pch = 16, col = rgb(0.2, 0.4, 0.8, 0.3))
abline(0, 1, col = "red", lwd = 2)

dev.off()
cat("Saved: plots/actual_vs_predicted.png\n")

# Residual distributions
png("plots/residuals.png", width = 1200, height = 400)
par(mfrow = c(1, 3))

hist(test_y - rf_test_pred, breaks = 30, main = "Random Forest Residuals",
     xlab = "Residual", col = "steelblue", border = "white")
hist(test_y - gbm_test_pred, breaks = 30, main = "GBM Residuals",
     xlab = "Residual", col = "steelblue", border = "white")
hist(test_y - enet_test_pred, breaks = 30, main = "Elastic Net Residuals",
     xlab = "Residual", col = "steelblue", border = "white")

dev.off()
cat("Saved: plots/residuals.png\n")

# ==============================================================================
# Section 9: Save Results
# ==============================================================================

# Combine validation and test results
all_results <- rbind(
  cbind(Set = "Validation", val_results),
  cbind(Set = "Test", test_results)
)

write.csv(all_results, "results/model_comparison.csv", row.names = FALSE)
cat("\nSaved: results/model_comparison.csv\n")

# Print final summary
cat("\n")
cat("================================================================================\n")
cat("                        FINAL SUMMARY\n")
cat("================================================================================\n")
cat(sprintf("Dataset: White Wine Quality (%d samples, %d features)\n", nrow(df), length(feature_cols)))
cat(sprintf("Split: Train=%d / Validation=%d / Test=%d\n", nrow(train_set), nrow(val_set), nrow(test_set)))
cat("\nTest Set Performance:\n")
print(test_results, row.names = FALSE)
cat(sprintf("\nBest overall model: %s\n", best_test_model))
cat("================================================================================\n")
