# Ensemble Tuning

# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(stacks)

# Handle common conflicts
tidymodels_prefer()

# Load candidate model info ----
load("knn_res.rda")
load("mars_res.rda")
load("lin_reg_res.rda")

# Load split data object & get testing data
load("bl_setup.rda")

# Create data stack ----
bl_data_stack <- stacks() %>% 
  add_candidates(knn_res) %>% 
  add_candidates(mars_res) %>% 
  add_candidates(lin_reg_res)

# Fit the stack ----
# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

# Blend predictions using penalty defined above (tuning step, set seed)
set.seed(3013)
bl_blend <- bl_data_stack %>% 
  blend_predictions(penalty = blend_penalty)

# Save blended model stack for reproducibility & easy reference (Rmd report)
# save(bl_blend, file = "model_info/wildfires_blend.rda")

# Explore the blended model stack
bl_blend

plot1 <- autoplot(bl_blend)
plot2 <- autoplot(bl_blend, type = "weights")

# fit to ensemble to entire training set ----
bl_model_st <- bl_blend %>%
  fit_members()

# Save trained ensemble model for reproducibility & easy reference (Rmd report)
# save(bl_model_st, file = "bl_model_st.rda")

# Explore and assess trained ensemble model
# assessment: model retained 7 candidates. The top candidate was a linear_reg model with
# a weight of 0.871.
# wildfires_model_st

# bl_test <- bl_test %>%
#   bind_cols(predict(bl_model_st, .)) %>%
#   select(id, .pred) %>%
#   rename("Id" = id,
#          "Predicted" = .pred)
# 
# write_csv(bl_test, "Ensemble_Reg.csv")

# pred_plot <- ggplot(bl_test) +
#   aes(x = money_made_inv, 
#       y = .pred) +
#   geom_point() + 
#   coord_obs_pred()

# member_preds <-
#   wildfires_test %>%
#   select(burned) %>%
#   bind_cols(predict(wildfires_model_st, wildfires_test, members = TRUE))

# evaluating the root mean squared error from each model:
# rmse_all <- map_dfr(member_preds, rmse, truth = burned, data = member_preds) %>%
#   mutate(member = colnames(member_preds)) %>% 
#   arrange(.estimate)

save(bl_model_st, bl_test, plot1, plot2, file = "blend_results.rda")
