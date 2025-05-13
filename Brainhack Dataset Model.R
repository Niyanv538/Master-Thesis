# Brainhack Dataset model prediction

# Loading libraries
set.seed(123)
library(tidyr)
library(dplyr)
library(tidymodels)
library(themis)
library(ranger)
library(vip)
library(ggplot2)
library(corrplot)

# 1. Read in only the pre-processed EEG data
Analysis_Brainhack <- read.delim("data/df_analysis.txt", sep = ",")
View(Analysis_Brainhack)

# Class Balance
ggplot(EEG_Brainhack, aes(x = as.factor(adhdtype), fill = as.factor(adhdtype))) +
  geom_bar() +
  labs(
    title = "Overall Class Distribution (All ADHD Subtypes)",
    x = "ADHD Subtype (0 = Hyper, 1 = Inattentive, 2 = Combined)",
    y = "Count",
    fill = "Subtype"
  ) +
  theme_minimal()


# 2. Selecting only needed columns
EEG_brainhack <- Analysis_Brainhack %>%
  select(id, subtype, Gender, adhdtype = adhdtype, brain_oscillation, electrode, fft_abs_power)

# 3.Creating a unique EEG feature name per band/electrode combo
EEG_brainhack <- EEG_brainhack %>%
  unite("feature", brain_oscillation, electrode, sep = "_")

# 4. Converting to wide format for modeling
EEG_brainhack_wide <- EEG_brainhack %>%
  pivot_wider(names_from = feature, values_from = fft_abs_power, values_fill = 0)

# 5. Removing missing rows
EEG_brainhack_wide <- na.omit(EEG_brainhack_wide)
EEG_Brainhack <- EEG_brainhack_wide 

# 6. Converting target to factor
EEG_Brainhack$adhdtype <- as.factor(EEG_Brainhack$adhdtype)

View(EEG_Brainhack)

write.csv(EEG_Brainhack, "EEG_Brainhack.csv", row.names = FALSE)

# Check feature correlation
numeric_features <- augmented_data[, sapply(augmented_data, is.numeric)]
numeric_features <- numeric_features[, !names(numeric_features) %in% "id"]

# Compute correlation matrix
cor_matrix <- cor(numeric_features)
View(cor_matrix)

# Visualize as heatmap
upper_tri <- cor_matrix[upper.tri(cor_matrix)]

# Plot histogram
hist(upper_tri,
     breaks = 50,
     main = "Distribution of Feature Correlations",
     xlab = "Correlation Coefficient",
     col = "steelblue",
     border = "white")


# M. Logistic Regression with Hyperparameter tuning on all features

# Load libraries
library(tidymodels)
library(glmnet)
library(pROC)
library(ggplot2)
library(dplyr)
library(tidyr)
library(purrr)

set.seed(123)

# STEP 1 — Use only EEG numeric features (exclude ID, subtype, Gender) 
eeg_data <- EEG_Brainhack %>%
  select(where(is.numeric), id, Gender, adhdtype) %>% # keep id and gender for augmentation
  mutate(adhdtype = factor(adhdtype))

# STEP 2 — Augment underrepresented class
augment_by_interpolation <- function(data, class_col = "adhdtype", class_to_augment = "0", n_aug = 20) {
  minority_data <- data %>% filter(!!sym(class_col) == class_to_augment)
  
  synths <- map_dfr(1:n_aug, ~ {
    pair <- sample_n(minority_data, 2)
    synth <- summarise(pair, across(where(is.numeric), mean))
    synth[[class_col]] <- class_to_augment
    synth$subtype <- "hyper"  # optional
    synth$id <- max(data$id, na.rm = TRUE) + .x
    synth$Gender <- sample(minority_data$Gender, 1)
    synth
  })
  
  bind_rows(data, synths)
}

augmented_data <- augment_by_interpolation(eeg_data, class_to_augment = "0", n_aug = 20)

write.csv(augmented_data, "augmented_EEG_Brainhack.csv", row.names = FALSE)

# STEP 3 — Dropping non-predictive columns before modeling
eeg_data_final <- augmented_data %>%
  select(-id, -Gender, -subtype)  # Keep only EEG features + adhdtype

# STEP 3 — Cross-validation setup
cv_folds <- vfold_cv(augmented_data, v = 5, repeats = 10, strata = adhdtype)

# STEP 4 — Recipe (normalizing predictors)
log_recipe <- recipe(adhdtype ~ ., data = eeg_data_final) %>%
  step_normalize(all_predictors())

# STEP 5 — Multinomial logistic regression model (with tuning)
log_model <- multinom_reg(
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# STEP 6 — Workflow
log_wf <- workflow() %>%
  add_model(log_model) %>%
  add_recipe(log_recipe)

# STEP 7 — Tuning grid
grid_vals <- grid_regular(
  penalty(range = c(-4, 0)),   # 0.0001 to 1 (log10 scale)
  mixture(range = c(0, 1)),    # 0 = ridge, 1 = lasso
  levels = 5
)

# STEP 8 — Tune model using CV
val_metrics <- metric_set(accuracy, kap, yardstick::recall, yardstick::precision, f_meas)

tuned_results <- tune_grid(
  log_wf,
  resamples = cv_folds,
  grid = grid_vals,
  metrics = val_metrics,
  control = control_grid(save_pred = TRUE)
)

# STEP 9 — Select best model and finalize
best_params <- select_best(tuned_results, metric = "f_meas")

final_wf <- finalize_workflow(log_wf, best_params)

# STEP 10 — Fit final model on full data
log_final_fit <- fit(final_wf, data = augmented_data)

# STEP 11 — Predict class probabilities
log_probs <- predict(log_final_fit, augmented_data, type = "prob") %>%
  bind_cols(augmented_data %>% select(adhdtype)) %>%
  mutate(adhdtype = factor(adhdtype))

# STEP 12 — Multiclass ROC AUC
roc_auc_multiclass <- roc_auc(
  log_probs,
  truth = adhdtype,
  .pred_0, .pred_1, .pred_2,
  estimator = "macro"
)

# STEP 13 — Plot ROC curves
log_probs_long <- log_probs %>%
  pivot_longer(cols = starts_with(".pred_"),
               names_to = "class",
               names_prefix = ".pred_",
               values_to = "prob") %>%
  mutate(actual = as.character(adhdtype))

roc_plot_data <- log_probs_long %>%
  group_by(class) %>%
  summarise(
    roc_obj = list(roc(response = (actual == class), predictor = prob)),
    .groups = "drop"
  ) %>%
  mutate(
    auc = map_dbl(roc_obj, ~auc(.x)),
    coords = map(roc_obj, ~coords(.x, "all", ret = c("fpr", "tpr")))
  ) %>%
  unnest(coords) %>%
  rename(FPR = fpr, TPR = tpr)

roc_plot_data %>%
  ggplot(aes(x = FPR, y = TPR, color = class)) +
  geom_line(size = 1) +
  geom_abline(linetype = "dashed", color = "gray") +
  labs(title = "Multiclass ROC Curves",
       x = "False Positive Rate",
       y = "True Positive Rate") +
  theme_minimal()


# STEP 14 — Original CV metrics
collect_metrics(tuned_results)

# STEP 15 — Class predictions and confidence
log_preds <- predict(log_final_fit, augmented_data)
log_results <- bind_cols(
  augmented_data %>% select(id, adhdtype),  # keep adhdtype here
  log_preds %>% rename(.pred_class = .pred_class),
  log_probs %>% select(starts_with(".pred_"))  # only keep probabilities, not adhdtype again
) %>%
  mutate(
    adhdtype = factor(adhdtype),
    .pred_class = factor(.pred_class),
    max_prob = pmax(.pred_0, .pred_1, .pred_2),
    correct = if_else(.pred_class == adhdtype, "Correct", "Incorrect")
  )


# STEP 16 — Box plot of prediction confidence
log_results %>%
  ggplot(aes(x = .pred_class, y = max_prob, fill = correct)) +
  geom_boxplot(alpha = 0.8, outlier.shape = NA) +
  labs(
    title = "Prediction Confidence (Softmax Max Probability)",
    x = "Predicted Class",
    y = "Max Softmax Probability"
  ) +
  scale_fill_manual(values = c("Correct" = "seagreen", "Incorrect" = "firebrick")) +
  theme_minimal()

# Check how often each class was predicted - ADHD-H less often
log_results %>%
  count(.pred_class)

# Final model: confusion matrix
log_results %>%
  conf_mat(truth = adhdtype, estimate = .pred_class)



# Data Distribution comparison

# Combine both datasets for side-by-side comparison
original_counts <- eeg_data %>%
  count(adhdtype) %>%
  mutate(source = "Original")

augmented_counts <- augmented_data %>%
  count(adhdtype) %>%
  mutate(source = "Augmented")

# Merge both into one dataframe
class_dist <- bind_rows(original_counts, augmented_counts)

# Plotting
ggplot(class_dist, aes(x = as.factor(adhdtype), y = n, fill = source)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(
    title = "Class Distribution Before and After Augmentation",
    x = "ADHD Subtype (0 = Hyper, 1 = Inattentive, 2 = Combined)",
    y = "Count",
    fill = "Dataset"
  ) +
  theme_minimal() +
  scale_fill_manual(values = c("Original" = "#1f77b4", "Augmented" = "#ff7f0e"))


# Class distribution
eeg_data %>%
  count(adhdtype) %>%
  mutate(percentage = round(n / sum(n), 3))

augmented_data %>%
  count(adhdtype) %>%
  mutate(percentage = round(n / sum(n), 3))

# Within-Cluster analysis of the most prominent features per subtype
library(purrr)
library(tibble)
library(dplyr)

coef_matrix <- coef(manual_fit, s = best_params$penalty)

# Turn sparse matrices into a tidy dataframe
tidy_coef <- map_dfr(names(coef_matrix), function(class_name) {
  mat <- as.matrix(coef_matrix[[class_name]])
  tibble(
    term = rownames(mat),
    estimate = as.numeric(mat),
    class = class_name
  )
})


# Remove intercept and filter non-zero effects
non_zero_coef <- tidy_coef %>%
  filter(term != "(Intercept)", estimate != 0)

# View top 10 features per subtype
non_zero_coef %>%
  group_by(class) %>%
  slice_max(abs(estimate), n = 10) %>%
  arrange(class, desc(abs(estimate)))

print(non_zero_coef, n = Inf)

# Visualize 
non_zero_coef %>%
  group_by(class) %>%
  slice_max(abs(estimate), n = 8) %>%
  ggplot(aes(x = reorder(term, estimate), y = estimate, fill = estimate > 0)) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  facet_wrap(~class, scales = "free_y") +
  labs(
    title = "Top EEG Features Influencing Each ADHD Subtype",
    x = "EEG Feature",
    y = "Log-Odds Estimate"
  ) +
  theme_minimal()


# ----------------- The Bajestani Dataset --------------------------------------

library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)

EEG_Bajestani <- read.csv("/Users/niyaneykova/Desktop/MasterThesis/data/Matlab analysis/eeg_features.csv")
View(EEG_Bajestani)

# Step 1: Renaming columns to make it easier to reshape
EEG_Bajestani_long <- EEG_Bajestani %>%
  pivot_longer(
    cols = Delta:TBR,          
    names_to = "Band",
    values_to = "Power"
  )

# Step 2: Create a new feature name for band + channel
EEG_Bajestani_long <- EEG_Bajestani_long %>%
  mutate(Feature = paste0(Band, "_", Channel))

# Step 3: Pivot to wide format
EEG_Bajestani_wide <- EEG_Bajestani_long %>%
  select(Subject, Group, SegmentID, Feature, Power) %>%
  pivot_wider(
    names_from = Feature,
    values_from = Power
  )

View(EEG_Bajestani_wide)

# Save as a csv file
write.csv(
  EEG_Bajestani_wide,
  file = "/Users/niyaneykova/Desktop/MasterThesis/data/EEG_Bajestani_segmented.csv",
  row.names = FALSE
)

#  ------ EDA -------------
# EO EC distribution 

# Filter for both Eyes Open and Eyes Closed
EO_EC_rows <- EEG_Bajestani_wide %>%
  filter(grepl("^(EO|EC)[1-2]", SegmentID)) %>%
  mutate(
    Condition = case_when(
      grepl("^EO[1-2]", SegmentID) ~ "Eyes Open",
      grepl("^EC[1-2]", SegmentID) ~ "Eyes Closed",
      TRUE ~ "Other"
    )
  )

# Count rows per Group and Condition
condition_counts <- EO_EC_rows %>%
  group_by(Group, Condition) %>%
  summarise(Count = n(), .groups = "drop")

# Plot
ggplot(condition_counts, aes(x = Condition, y = Count, fill = Group)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(
    title = "Distribution per Condition and Group",
    x = "Condition",
    y = "Number of Segments"
  ) +
  theme_minimal()

 
# Create a binary variable for ADHD vs Control for easier analysis
EEG_Bajestani_binary <- EEG_Bajestani_wide %>%
  mutate(ADHD_status = ifelse(grepl("ADHD", Group), "ADHD", "Control"))

EEG_Bajestani_binary %>%
  count(ADHD_status)

# Check balance
ggplot(EEG_Bajestani_binary, aes(x = ADHD_status, fill = ADHD_status)) +
  geom_bar() +
  labs(title = "Distribution of ADHD vs Control", x = "Group", y = "Count") +
  theme_minimal()

View(EEG_Bajestani_binary)

# How many rows are in the eyes open condition -> nrow(4582)
EO_rows <- EEG_Bajestani_wide %>%
  filter(grepl("^EO[1-2]", SegmentID))  # matches SegmentIDs starting with EO1 or EO2

nrow(EO_rows) #  nrow(4582)

EO_rows %>%
  group_by(Group) %>%
  summarise(Count = n())

# Count number of rows per group EO
group_counts <- EO_rows %>%
  group_by(Group) %>%
  summarise(Count = n())

# Bar plot EO
ggplot(group_counts, aes(x = Group, y = Count, fill = Group)) +
  geom_bar(stat = "identity") +
  labs(
    title = "EO Group Distribution",
    x = "Group",
    y = "Number of Segments"
  ) +
  theme_minimal()


# EO filter EDA

# EO-only dataset

EEG_EO <- EEG_Bajestani_wide %>%
  filter(grepl("^EO[1-2]", SegmentID))

# Combine ADHD subtypes (regardless of sex) into one group
EEG_EO_binary <- EEG_EO %>%
  mutate(Group_binary = ifelse(grepl("ADHD", Group), "ADHD", "Control"))

View(EEG_EO_binary)

EEG_EO_binary %>%
  count(Group_binary)

ggplot(EEG_EO_binary, aes(x = Group_binary, fill = Group_binary)) +
  geom_bar() +
  labs(
    title = "EO Segment Counts: ADHD vs Control",
    x = "Group",
    y = "Number of Segments"
  ) +
  theme_minimal()

# summary stats
summary(EEG_EO_binary)
colSums(is.na(EEG_EO_binary))

# Select numeric EEG features
ADHD_EO_numeric <- EEG_EO_binary %>%
  select(where(is.numeric))

# Compute correlation matrix
ADHD_EO_corr_matrix <- cor(ADHD_EO_numeric, use = "complete.obs")

# Extract lower triangle
ADHD_EO_corr_values <- ADHD_EO_corr_matrix[lower.tri(ADHD_EO_corr_matrix)]

# Plot histogram
ggplot(data.frame(Correlation = ADHD_EO_corr_values), aes(x = Correlation)) +
  geom_histogram(binwidth = 0.05, fill = "steelblue", color = "white") +
  geom_vline(xintercept = c(-0.9, 0.9), linetype = "dashed", color = "red") +
  labs(
    title = "Correlation Distribution of EEG Features (ADHD, Eyes Open)",
    x = "Correlation Coefficient",
    y = "Frequency"
  ) +
  theme_minimal()

View(EEG_EO_binary)

# Save the subsetted EO + ADHD participant data only
write.csv(EEG_EO_binary, "EEG_EO_binary.csv", row.names = FALSE)


