# Boosted Tree

library(tidyverse)
library(tictoc)
library(janitor)
library(rsample)
library(tidymodels)
library(xgboost)


# set seed
set.seed(3013)

bl_test <- read_csv("data/test.csv") %>% 
  clean_names()
bl_train <- read_csv("data/train.csv") %>% 
  clean_names()

bl_folds <- bl_train %>%
  vfold_cv(v = 5, repeats = 2)
bl_folds

bl_recipe <- recipe(money_made_inv ~ out_prncp_inv + int_rate + 
                      loan_amnt + term + grade,
                    data = bl_train) %>% 
  step_dummy(term, grade) %>% 
  step_normalize(all_predictors())

# Define model
boost_model <- boost_tree(mode = "regression",
                          min_n = tune(),
                          mtry = tune(),
                          learn_rate = tune()) %>% 
  set_engine("xgboost")

boost_params <- parameters(boost_model) %>% 
  update(mtry = mtry(range = c(2, 8)))

# Grid
boost_grid <- grid_regular(boost_params, levels = 3)

# Workflow
boost_wf <- workflow() %>% 
  add_model(boost_model) %>% 
  add_recipe(bl_recipe)

# tuning and fitting - DON"T RUN HERE
tic("Boosted Trees Model")
boost_tuned <- boost_wf %>% 
  tune_grid(bl_folds, grid = boost_grid)

toc(log = TRUE)

# save runtime info
boost_runtime <- tic.log(format = TRUE)


# fit to train data
boost_wf_tune <- boost_wf %>% 
  finalize_workflow(select_best(boost_tuned, metric = "rmse"))

boost_results <- fit(boost_wf_tune, bl_train)

boost_results <- boost_results %>%
  predict(new_data = bl_test) %>%
  bind_cols(bl_test %>% select(id)) %>% 
  rename("Id" = id,
         "Predicted" = .pred)

boost_results <- boost_results[c(2,1)]

write_csv(boost_results, path="Lin_Kylie_Boost_RegComp.csv")


# Write out results & workflow
save(boost_tuned, boost_wf, boost_runtime, boost_results, file = "boost_tuned.rda")

