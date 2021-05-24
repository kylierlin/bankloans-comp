# Final RF Attempt

library(tidyverse)
library(tidymodels)
library(tictoc)
library(janitor)
library(ranger)

set.seed(3013)

bl_test <- read_csv("data/test.csv") %>% 
  clean_names()
bl_train <- read_csv("data/train.csv") %>% 
  clean_names()

bl_folds <- bl_train %>%
  vfold_cv(v = 10, repeats = 3, strata = money_made_inv)
bl_folds

bl_recipe <- recipe(money_made_inv ~ out_prncp_inv + int_rate + 
                      loan_amnt + term + grade,
                    data = bl_train) %>% 
  step_dummy(term, grade) %>% 
  step_normalize(all_predictors())

# Define model
rf_model <- rand_forest(mode = "regression",
                        min_n = tune(),
                        mtry = tune()) %>% 
  set_engine("ranger")

rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(1, 8)))

# Grid
rf_grid <- grid_regular(rf_params, levels = 3)

# Workflow
rf_wf <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(bl_recipe)

# tuning and fitting
tic("Random Forest Model")
rf_tuned <- rf_wf %>% 
  tune_grid(bl_folds, grid = rf_grid)

toc(log = TRUE)

# save runtime info
rf_runtime <- tic.log(format = TRUE)

# fit to train data
rf_wf_tune <- rf_wf %>% 
  finalize_workflow(select_best(rf_tuned, metric = "rmse"))

rf_results <- fit(rf_wf_tune, bl_train)

final_results <- rf_results %>%
  predict(new_data = bl_test) %>%
  bind_cols(bl_test %>% select(id)) %>% 
  rename("Id" = id,
         "Predicted" = .pred)

final_results <- final_results[c(2,1)]

write_csv(final_results, path="Lin_Kylie_RF_NEW_RegComp.csv")


# Write out results & workflow
save(rf_tuned, rf_wf, rf_runtime, final_results, file = "rf_NEW_tuned.rda")

