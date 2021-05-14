# Mars tuning ----

# Load package(s) ----
library(tidyverse)
library(tictoc)
library(rsample)
library(tidymodels)
library(earth)

# set seed
set.seed(3013)

# load required objects ----
load("bl_setup.rda")

# Define Model
mars_model <- mars(num_terms = tune(),
                   prod_degree = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("earth")

mars_params <- parameters(mars_model)

# update parameters to go for a wider range of num_terms
mars_params <- mars_params %>%
  update(num_terms = num_terms(range = c(2, 9)),
         prod_degree = prod_degree(range = c(1, 4)))

# Grid
# play around with levels
mars_grid <- grid_regular(mars_params, levels = 5)

mars_grid

# Workflow
mars_wf <- workflow() %>% 
  add_model(mars_model) %>% 
  add_recipe(basic_recipe)

# tuning and fitting
tic("Mars Model")
mars_tuned <- mars_wf %>% 
  tune_grid(bl_folds, grid = mars_grid,)

toc(log = TRUE)

# save runtime info
mars_runtime <- tic.log(format = TRUE)

mars_wf_tune <- mars_wf %>% 
  finalize_workflow(select_best(mars_tuned, metric = "rmse"))

mars_results <- fit(mars_wf_tune, bl_train)

final_mars_results <- mars_results %>%
  predict(new_data = bl_test) %>%
  bind_cols(bl_test %>% select(id)) %>% 
  rename("Id" = id,
         "Predicted" = .pred)

final_mars_results <- final_mars_results[c(2,1)]

write_csv(final_mars_results, path="Lin_Kylie_MARS_RegComp.csv")


# Write out results & workflow
save(mars_tuned, mars_wf, mars_runtime, final_mars_results, file = "mars_tuned.rda")

