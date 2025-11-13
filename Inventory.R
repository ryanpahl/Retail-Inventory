df <- read.csv("Retail inventory.csv")
library(tidymodels)
library(lubridate)
library(slider)
library(tidyverse) # For data manipulation
library(ranger)
library(vip)
library(broom)

df$Store <- as.factor(df$Store)
# Create time-series and date features
df
df_features <- df %>%
  
  # 1. Convert text date to a real Date object
  mutate(Date = dmy(Date)) %>% 
  
  # 2. Sort by group and date (now works)
  arrange(Store, Product, Date) %>%
  
  # 3. Group by each unique time series
  group_by(Store, Product) %>%
  
  # 4. Create time-series features
  mutate(
    Sales_Lag_1 = lag(`Weekly_Sales`, n = 1),
    Sales_Lag_4 = lag(`Weekly_Sales`, n = 4),
    
    Sales_Roll_Avg_4 = slide_dbl(
      .x = `Weekly_Sales`,
      .f = ~mean(.x, na.rm = TRUE),
      .before = 4, 
      .after = -1   
    )
  ) %>%
  
  # 5. Ungroup to create date features
  ungroup() %>%
  
  # 6. Create calendar features
  mutate(
    Month = month(Date),
    WeekOfYear = week(Date),
    Year = year(Date),
    DayOfYear = yday(Date)
  ) %>%
  
  na.omit()

glimpse(df_features)
set.seed(123)

data_split <- initial_time_split(df_features, prop = 0.8)

train_data <- training(data_split)
test_data <- testing(data_split)
sales_recipe <- recipe(Weekly_Sales ~ ., data = train_data) %>%

  step_rm(Date, demand.forecast) %>%
  
  step_string2factor(Store, Product) %>%
  step_dummy(Store, Product) %>%
 
  step_novel(all_nominal_predictors()) %>%
 
  step_zv(all_predictors()) %>%
 
  step_normalize(all_numeric_predictors(), -all_outcomes())

metrics_set <- metric_set(rmse, mae)

# Calculate the baseline error from the provided forecast
baseline_results <- test_data %>%
  summarize(
    rmse = rmse_vec(truth = `Weekly_Sales`, estimate = `demand.forecast`),
    mae = mae_vec(truth = `Weekly_Sales`, estimate = `demand.forecast`)
  )

print("--- Baseline Model Error (The 'Score to Beat') ---")
print(baseline_results)
# Define the model specification
rf_spec <- rand_forest(mode = "regression") %>%
  set_engine("ranger", importance = "permutation") # 'importance' lets us see feature importance later

# Create the complete workflow
rf_workflow <- workflow() %>%
  add_recipe(sales_recipe) %>%
  add_model(rf_spec)

# Train the model
print("Training model...")
rf_fit <- fit(rf_workflow, data = train_data)
print("Model training complete.")
# Get predictions on the test set
predictions <- predict(rf_fit, new_data = test_data)

# Combine predictions with the truth
results <- bind_cols(predictions, test_data)

# Calculate our model's metrics
model_results <- results %>%
  metrics_set(truth = `Weekly_Sales`, estimate = .pred)

# Compare!
print("--- Baseline Model Error ---")
print(baseline_results)

print("--- Our Random Forest Model Error ---")
print(model_results)
# Extract and plot feature importance
dev.new()
rf_fit %>%
  extract_fit_parsnip() %>%
  vip(num_features = 15) # vip = Variable Importance Plot

# Calculate the correlation between sales and promotions for each product
promo_impact_table <- df_features %>%
  group_by(Product) %>%
  summarize(
    # Calculate correlation
    promo_correlation = cor(Weekly_Sales, `Past.promotion.of.product.in.lac`, use = "complete.obs"),
    avg_promo_spend = mean(`Past.promotion.of.product.in.lac`, na.rm = TRUE),
    total_sales = sum(Weekly_Sales)
  ) %>%
  arrange(desc(promo_correlation))

# Print the table
print("--- Promotion Impact by Product ---")
print(promo_impact_table)
# Plot Sales vs. Promotion, with a separate plot for each product
dev.new() # Open a new plot window

ggplot(df_features, aes(x = `Past.promotion.of.product.in.lac`, y = Weekly_Sales)) +
  geom_point(alpha = 0.3) + # Makes points semi-transparent
  geom_smooth(method = "lm", se = FALSE, color = "blue") + # Adds a linear trend line
  facet_wrap(~ Product, scales = "free") + # Creates a grid of plots, one per product
  labs(
    title = "Promotion Impact on Sales, by Product",
    x = "Weekly Promotion Spend (in lac)",
    y = "Weekly Sales"
  ) +
  theme_minimal()


promo_impact_table <- df_features %>%
  group_by(Product) %>%
  
  # Filter out groups with < 3 points, as cor.test will fail
  filter(n() >= 3) %>% 
  
  summarize(
    # Run the full correlation test and tidy the results
    promo_test = list(tidy(cor.test(Weekly_Sales, `Past.promotion.of.product.in.lac`))),
    n_points = n(),
    avg_promo_spend = mean(`Past.promotion.of.product.in.lac`, na.rm = TRUE),
    total_sales = sum(Weekly_Sales)
  ) %>%
  unnest(promo_test) %>%
  
  # Select and rename the most important columns
  select(
    Product, 
    promo_correlation = estimate, # 'estimate' is the correlation
    p_value = p.value,            # This is the p-value
    n_points,                     # Number of data points
    avg_promo_spend,
    total_sales
  ) %>%
  
  # Sort by the correlation
  arrange(desc(promo_correlation))


# Print the new, more reliable table
print("--- Promotion Impact by Product (with p-values) ---")
print(promo_impact_table)
# 1. Get the list of products that meet p-value criteria
significant_products <- promo_impact_table %>%
  filter(p_value < 0.15) %>%
  pull(Product)

dev.new()

# Filter the data before plotting
df_features %>%
  filter(Product %in% significant_products) %>% 
  ggplot(aes(x = `Past.promotion.of.product.in.lac`, y = Weekly_Sales)) +
  geom_point(alpha = 0.3) + 
  geom_smooth(method = "lm", se = FALSE, color = "blue") + 
  facet_wrap(~ Product, scales = "free") + 
  labs(
    title = "Promotion Impact on Sales (Statistically Significant Products, p < 0.15)",
    x = "Weekly Promotion Spend (in lac)",
    y = "Weekly Sales"
  ) +
  theme_minimal()

# --------- Forecasting ---------

df <- read.csv("Retail inventory.csv")
df$Store <- as.factor(df$Store)
df_source_clean <- df %>%
  mutate(Date = dmy(Date))


df_features_regression <- df_source_clean %>%
  mutate(
    Month = month(Date),
    WeekOfYear = week(Date),
    Year = year(Date),
    DayOfYear = yday(Date)
  )

sales_recipe_regression <- recipe(
  Weekly_Sales ~ Store + Product + inventory.level + Temperature +
    Past.promotion.of.product.in.lac + Month +
    WeekOfYear + Year + DayOfYear,
  data = df_features_regression
) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors(), -all_outcomes())

rf_spec <- rand_forest(mode = "regression") %>%
  set_engine("ranger")

rf_workflow_regression <- workflow() %>%
  add_recipe(sales_recipe_regression) %>%
  add_model(rf_spec)

final_model_fit <- fit(rf_workflow_regression, data = df_features_regression)

unique_combos <- df_features_regression %>%
  distinct(Store, Product)

all_predictions <- list()

forecast_date <- as_date("2012-11-02")

for (i in 1:nrow(unique_combos)) {
  
  current_store <- unique_combos$Store[i]
  current_product <- unique_combos$Product[i]
  
  future_data_to_predict <- tibble(
    Store = current_store,
    Product = current_product,
    
    inventory.level = 1,
    Temperature = 45.0,
    Past.promotion.of.product.in.lac = 3.0,
    
    Date = forecast_date
  ) %>%
    mutate(
      Month = month(Date),
      WeekOfYear = week(Date),
      Year = year(Date),
      DayOfYear = yday(Date)
    )
  
  prediction <- predict(final_model_fit, new_data = future_data_to_predict)
  
  all_predictions[[i]] <- tibble(
    Store = current_store,
    Product = current_product,
    Date = forecast_date,
    Predicted_Sales = prediction$.pred
  )
}

final_forecast_table <- bind_rows(all_predictions)

print("--- Final Forecast Table ---")
print(final_forecast_table, n = 20)

