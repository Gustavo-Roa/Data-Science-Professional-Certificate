#' ---
#' title: "Sulfur Fertilization in Wheat - A Machine Learning Approach"
#' author: "Gustavo Roa"
#' date: "2024-09-06"
#'
#' 
## ----setup, include=FALSE---------------------------------------------------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)

#' 
#' # Introduction
#' 
## ----library, warning=FALSE, include=TRUE, echo=FALSE, include=FALSE--------------------------------------------------------------------------------
# Define the packages you need
packages <- c("tidyverse", "caret", "randomForest", "xgboost", "grf", 
              "pdp", "corrplot", "glmnet", "rpart", 
              "rpart.plot", "e1071", "mice", "readxl")

# Function to check if a package is installed and install it if not
install_if_needed <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
  library(pkg, character.only = TRUE)
}

# Apply the function to each package
lapply(packages, install_if_needed)


#' 
#' ## Data Preparation
#' ## Dataset
#' 
## ----data_download, warning=FALSE-------------------------------------------------------------------------------------------------------------------
# Load your data
url <- "https://dataverse.harvard.edu/api/access/datafile/10360340"
download.file(url, destfile = "Sulfur_datasets_GRoa.xlsx", mode = "wb")

# Read the Excel file
dataset <- read_excel("Sulfur_datasets_GRoa.xlsx", sheet = "database")

dataset <- dataset %>% 
  select(pH, om_percent, texture_group, soil_sulfate_mg_kg, control_yield_kg_ha, treated_yield_kg_ha )


#' 
#' ## Data Imputation

## ----data_preparation-------------------------------------------------------------------------------------------------------------------------------
# Display the first few rows and summary of the dataset
print(head(dataset)) 
summary(dataset)

# Check for missing values
missing_values <- colSums(is.na(dataset))
print(missing_values)

# Handle missing values
# For numeric columns, we'll impute with median
# For categorical columns, we'll impute with mode
dataset_imputed <- dataset %>%
  mutate(
    om_percent = ifelse(is.na(om_percent), median(om_percent, na.rm = TRUE), om_percent),
    soil_sulfate_mg_kg = ifelse(is.na(soil_sulfate_mg_kg), median(soil_sulfate_mg_kg, na.rm = TRUE), soil_sulfate_mg_kg),
    texture_group = ifelse(is.na(texture_group), names(which.max(table(texture_group))), texture_group)
  )


#' 
#' ## Data Exploration

## ----data_visualization-----------------------------------------------------------------------------------------------------------------------------

# Calculate yield difference
dataset_imputed$yield_difference <- dataset_imputed$treated_yield_kg_ha - dataset_imputed$control_yield_kg_ha

# Correlation matrix for numeric variables
cor_matrix <- cor(dataset_imputed[, sapply(dataset_imputed, is.numeric)], use = "complete.obs")
corrplot(cor_matrix, method = "color")

# Boxplot of yield difference by texture group
ggplot(dataset_imputed, aes(x = texture_group, y = yield_difference)) +
  geom_boxplot() +
  labs(x = "Soil Texture Group", y = "Yield Difference (Treated - Control)", 
       title = "Yield Difference by Soil Texture")

# Scatter plot of yield difference vs pH
ggplot(dataset_imputed, aes(x = pH, y = yield_difference)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(x = "pH", y = "Yield Difference (Treated - Control)", 
       title = "Yield Difference vs pH")

# Scatter plot of yield difference vs soil sulfate
ggplot(dataset_imputed, aes(x = soil_sulfate_mg_kg, y = yield_difference)) +
  geom_point() +
  geom_smooth(method = "lm") +
  labs(x = "soil sulfate_ (mg/kg)", y = "Yield Difference (Treated - Control)", 
       title = "Yield Difference vs Soil Sulfate")


#' 
#' ## Feature Engineering

## ----feature_engineering----------------------------------------------------------------------------------------------------------------------------
# Convert texture_group to factor
dataset_imputed$texture_group <- as.factor(dataset_imputed$texture_group)

# Create dummy variables for texture_group
dataset_engineered <- model.matrix(~ . - 1, data = dataset_imputed) %>% as.data.frame()

# Split data into features (X) and target variable (y)
X <- dataset_engineered %>% select(-yield_difference, -control_yield_kg_ha, -treated_yield_kg_ha)
y <- dataset_engineered$yield_difference

# Split data into training and testing sets
set.seed(2024)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]


#' 
#' ## Causal Inference Modeling

## ----causal-----------------------------------------------------------------------------------------------------------------------------------------
# Prepare data for causal forest
W <- as.numeric(dataset_engineered$treated_yield_kg_ha > dataset_engineered$control_yield_kg_ha)

# Train causal forest
cf <- causal_forest(X, y, W)

# Estimate treatment effects
te_estimate <- predict(cf)$predictions

# Variable importance
variable_importance <- variable_importance(cf)

# Combine variable names with their importance scores and order them from highest to lowest
variable_importance_df <- data.frame(
  Variable = colnames(X),
  Importance = variable_importance
) %>% arrange(desc(Importance))

# Print ordered variable importance with names
print(variable_importance_df)


#' 
#' ## Predictive Modeling
#' ### Random Forest
#' 
## ----random_forest----------------------------------------------------------------------------------------------------------------------------------
rf_model <- randomForest(x = X_train, y = y_train, ntree = 500)
rf_pred <- predict(rf_model, newdata = X_test)
rf_rmse <- sqrt(mean((rf_pred - y_test)^2))
print(paste("Random Forest RMSE:", rf_rmse))


## ----xgboost----------------------------------------------------------------------------------------------------------------------------------------
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

xgb_params <- list(
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 6,
  nrounds = 100
)

xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 100)
xgb_pred <- predict(xgb_model, dtest)
xgb_rmse <- sqrt(mean((xgb_pred - y_test)^2))
print(paste("XGBoost RMSE:", xgb_rmse))

#' ### Decision Tree
#' 
## ----decision_tree----------------------------------------------------------------------------------------------------------------------------------
dt_model <- rpart(y_train ~ ., data = as.data.frame(X_train), method = "anova")
rpart.plot(dt_model)
dt_pred <- predict(dt_model, newdata = as.data.frame(X_test))
dt_rmse <- sqrt(mean((dt_pred - y_test)^2))
print(paste("Decision Tree RMSE:", dt_rmse))


## ----svm--------------------------------------------------------------------------------------------------------------------------------------------
svm_model <- svm(y_train ~ ., data = as.data.frame(X_train))
svm_pred <- predict(svm_model, newdata = as.data.frame(X_test))
svm_rmse <- sqrt(mean((svm_pred - y_test)^2))
print(paste("SVM RMSE:", svm_rmse))

#' ### Linear Regression
#' 
## ----linear_regression------------------------------------------------------------------------------------------------------------------------------
lm_model <- lm(y_train ~ ., data = as.data.frame(X_train))
lm_pred <- predict(lm_model, newdata = as.data.frame(X_test))
lm_rmse <- sqrt(mean((lm_pred - y_test)^2))
print(paste("Linear Regression RMSE:", lm_rmse))

#' # Results

#' ## Model Performance

## ----model_performance------------------------------------------------------------------------------------------------------------------------------
# Create a data frame with model names and RMSE values
model_comparison <- data.frame(
  Model = c("Random Forest", "XGBoost", "Decision Tree", "SVM", "Linear Regression"),
  RMSE = c(rf_rmse, xgb_rmse, lm_rmse, dt_rmse, svm_rmse)
)

# Sort the data frame by RMSE in ascending order
model_comparison <- model_comparison[order(model_comparison$RMSE), ]

# Print the comparison table
print(knitr::kable(model_comparison, caption = "Model Comparison by RMSE"))


#' ## Select Best Model
#' 
## ----model_selection--------------------------------------------------------------------------------------------------------------------------------
# Specify the best model here (e.g., xgb_model or rf_model)

best_model <- xgb_model  # Example: If XGBoost is the best model

#best_model <- rf_model   # Example: If Ramdom Forest is the best model


#' ## Variable Importance
#' 
## ----model_importance-------------------------------------------------------------------------------------------------------------------------------
# Model type detection and corresponding variable importance function
if (inherits(best_model, "xgb.Booster")) {
  importance <- xgb.importance(feature_names = colnames(X_train), model = best_model)
  plot_importance <- function(importance) {
    xgb.plot.importance(importance, main = "Variable Importance Plot")
  }
} else if (inherits(best_model, "randomForest")) {
  importance <- randomForest::importance(best_model)
  plot_importance <- function(importance) {
    randomForest::varImpPlot(best_model, 
                             main = "Variable Importance Plot",
                             n.var = min(20, ncol(X_train)))
  }
} else {
  stop("Model type not supported. Please add corresponding variable importance handling.")
}

# Plot the variable importance
plot_importance(importance)


#' 
#' ## Partial Dependence and ICE Plots

## ----model_dependence-------------------------------------------------------------------------------------------------------------------------------
# Partial Dependence Plots
# Function to safely create and print partial dependence plots
safe_partial_plot <- function(model, var, data) {
  tryCatch({
    if (inherits(model, "xgb.Booster")) {
      partial_plot <- pdp::partial(object = model, 
                                   pred.var = var, 
                                   train = data,
                                   plot = TRUE,
                                   plot.engine = "ggplot2",
                                   which.class = 1)  # For classification models, specify class
    } else {
      partial_plot <- pdp::partial(object = model, 
                                   pred.var = var, 
                                   train = data,
                                   plot = TRUE,
                                   plot.engine = "ggplot2")
    }
    print(partial_plot)
  }, error = function(e) {
    message(paste("Could not create partial plot for", var, ":", e$message))
  })
}

# Create partial plots for top 5 important variables
if (inherits(best_model, "xgb.Booster")) {
  important_vars <- importance %>%
    .[1:5, "Feature"]
} else if (inherits(best_model, "randomForest")) {
  important_vars <- names(sort(importance, decreasing = TRUE))[1:5]
} else {
  stop("Model type not supported for extracting important variables.")
}

for (var in important_vars) {
  safe_partial_plot(best_model, var, X_train)
}

# ICE plots for top 2 important variables
safe_ice_plot <- function(model, var, data) {
  tryCatch({
    if (is.factor(data[[var]])) {
      message(paste(var, "is categorical. Skipping ICE plot."))
    } else {
      if (inherits(model, "xgb.Booster")) {
        ice_plot <- pdp::partial(object = model, 
                                 pred.var = var, 
                                 train = data,
                                 plot = TRUE,
                                 plot.engine = "ggplot2",
                                 ice = TRUE,
                                 center = TRUE,
                                 which.class = 1)  # For classification models, specify class
      } else {
        ice_plot <- pdp::partial(object = model, 
                                 pred.var = var, 
                                 train = data,
                                 plot = TRUE,
                                 plot.engine = "ggplot2",
                                 ice = TRUE,
                                 center = TRUE)
      }
      print(ice_plot)
    }
  }, error = function(e) {
    message(paste("Could not create ICE plot for", var, ":", e$message))
  })
}

for (var in important_vars[1:2]) {
  safe_ice_plot(best_model, var, X_train)
}


#' 
#' ## Model Validation
## ----validation-------------------------------------------------------------------------------------------------------------------------------------

# Define cross-validation control
ctrl <- trainControl(method = "cv", number = 5)

# Cross-validation for the best model
if (inherits(best_model, "xgb.Booster")) {
  best_model_cv <- train(
    x = X_train,
    y = y_train,
    method = "xgbTree",
    trControl = ctrl,
    tuneGrid = expand.grid(
      nrounds = 100,
      max_depth = 6,
      eta = 0.1,
      gamma = 0,
      colsample_bytree = 1,
      min_child_weight = 1,
      subsample = 1
    )
  )
} else if (inherits(best_model, "randomForest")) {
  best_model_cv <- train(
    x = X_train,
    y = y_train,
    method = "rf",
    trControl = ctrl
  )
} else {
  stop("Model type not supported for cross-validation.")
}

# Print cross-validation results
print(best_model_cv)


#' 
#' ## Actual vs Predicted
#' 
## ----actual_vs_predicted----------------------------------------------------------------------------------------------------------------------------
# Generate predictions on the test set based on the best model
if (inherits(best_model, "xgb.Booster")) {
  pred_test <- predict(best_model, newdata = as.matrix(X_test))
} else if (inherits(best_model, "randomForest")) {
  pred_test <- predict(best_model, newdata = X_test)
} else {
  stop("Model type not supported for predictions.")
}

# Create a data frame for plotting
actual_vs_predicted_df <- data.frame(
  Actual = y_test,
  Predicted = pred_test
)

# Calculate R-squared and RMSE
r_squared <- cor(actual_vs_predicted_df$Actual, actual_vs_predicted_df$Predicted) ^ 2
rmse <- sqrt(mean((actual_vs_predicted_df$Predicted - actual_vs_predicted_df$Actual) ^ 2))

# Plot Actual vs Predicted with R-squared and RMSE
ggplot(actual_vs_predicted_df, aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue", alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Actual vs Predicted Yield Difference",
       x = "Actual Yield Difference",
       y = "Predicted Yield Difference") +
  theme_minimal() +
  annotate("text", x = min(actual_vs_predicted_df$Actual), 
           y = max(actual_vs_predicted_df$Predicted), 
           label = paste("R-squared =", round(r_squared, 2)), 
           hjust = 0, vjust = 1, size = 5, color = "darkred") +
  annotate("text", x = min(actual_vs_predicted_df$Actual), 
           y = max(actual_vs_predicted_df$Predicted) - 0.1 * (max(actual_vs_predicted_df$Predicted) - min(actual_vs_predicted_df$Predicted)), 
           label = paste("RMSE =", round(rmse, 2)), 
           hjust = 0, vjust = 1, size = 5, color = "darkred")



#' 
#' ### Decision Support Tool

## ----decision_tool----------------------------------------------------------------------------------------------------------------------------------
# Function to make predictions based on user inputs
predict_sulfur_fertilizer <- function(pH, om_percent, soil_sulfate_mg_kg, texture_group) {
  
  # Ensure the input values are within the range of the training data
  pH <- min(max(pH, min(dataset_imputed$pH, na.rm = TRUE)), max(dataset_imputed$pH, na.rm = TRUE))
  om_percent <- min(max(om_percent, min(dataset_imputed$om_percent, na.rm = TRUE)), max(dataset_imputed$om_percent, na.rm = TRUE))
  soil_sulfate_mg_kg <- min(max(soil_sulfate_mg_kg, min(dataset_imputed$soil_sulfate_mg_kg, na.rm = TRUE)), max(dataset_imputed$soil_sulfate_mg_kg, na.rm = TRUE))
  
  # Convert texture_group to the appropriate factor levels
  texture_group <- factor(texture_group, levels = levels(dataset_imputed$texture_group))
  
  # Create a data frame with the inputs
  input_data <- data.frame(
    pH = pH,
    om_percent = om_percent,
    soil_sulfate_mg_kg = soil_sulfate_mg_kg,
    texture_group = texture_group
  )
  
  # Create dummy variables for texture_group and ensure consistent structure with training data
  input_data_engineered <- model.matrix(~ pH + om_percent + soil_sulfate_mg_kg + texture_group, data = input_data)
  input_data_engineered <- as.data.frame(input_data_engineered)
  
  # Ensure that all necessary columns are present
  missing_cols <- setdiff(names(X_train), names(input_data_engineered))
  input_data_engineered[missing_cols] <- 0
  
  # Reorder columns to match the training data
  input_data_engineered <- input_data_engineered[, names(X_train)]
  
  # Make prediction using the best model
  if (inherits(best_model, "xgb.Booster")) {
    predicted_yield_diff <- predict(best_model, newdata = as.matrix(input_data_engineered))
  } else if (inherits(best_model, "randomForest")) {
    predicted_yield_diff <- predict(best_model, newdata = input_data_engineered)
  } else {
    stop("Model type not supported for predictions.")
  }
  
  # Decision rule based on predicted yield difference
  if (predicted_yield_diff > 0) {
    decision <- "Apply Sulfur Fertilizer"
    yield_increase <- predicted_yield_diff
    result <- paste(decision, "- Expected yield increase:", round(yield_increase, 2), "kg/ha")
  } else {
    decision <- "Don't Apply Sulfur Fertilizer"
    result <- decision
  }
  
  return(result)
}


#' 
#' #### Example
 
## ----example----------------------------------------------------------------------------------------------------------------------------------------
# Example usage:
# Replace these values with the actual soil characteristics the farmer inputs
example_pH <- 6.5
example_om_percent <- 5
example_soil_sulfate_mg_kg <- 15
example_texture_group <- "Loam"

# Predict and print the decision
decision <- predict_sulfur_fertilizer(example_pH, example_om_percent, example_soil_sulfate_mg_kg, example_texture_group)
print(decision)


#' # Conclusion
