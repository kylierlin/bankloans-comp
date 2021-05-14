# Linear model ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(stacks)

# Handle common conflicts
tidymodels_prefer()

# load required objects ----
load("bl_setup.rda")

# Define model ----
lin_reg_model <- linear_reg(
  mode = "regression",
) %>%
  set_engine("lm")

# workflow ----
lin_reg_workflow <- workflow() %>%
  add_model(lin_reg_model) %>%
  add_recipe(barebones_recipe)

# Tuning/fitting ----
lin_reg_res <- lin_reg_workflow %>%
  fit_resamples(
    resamples = bl_folds,
    # NEED TO ADD AN ARGUMENT HERE
    control = stacks::control_stack_resamples()
  )

# Write out results & workflow
save(lin_reg_res, file = "lin_reg_res.rda")