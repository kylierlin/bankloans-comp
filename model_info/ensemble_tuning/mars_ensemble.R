# Mars Ensemble

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
  add_recipe(barebones_recipe)

# tuning
mars_res <- mars_wf %>% 
  tune_grid(resamples = bl_folds, 
            grid = mars_grid,
            control = stacks::control_stack_grid())

save(mars_res, file = "mars_res.rda")




