library('ggmap')
library("rjson")
library(tidyverse)
setwd("~/Desktop/ML1030")

register_google(key = "",  # your Static Maps API key
                                account_type = "standard")
pred <- read.csv("final_pred_toronto.csv", stringsAsFactors = FALSE)
location <- fromJSON("toronto_metadata.json")


pred %>% left_join(location, by = c('X_file' = '_file')) %>% select(X_file, X0, X1, pred_safety, location) -> pred

pred %>% mutate (lat = pred$location$lat, lng = pred$location$lng) %>% select(-location) -> pred

pred$pred_safety =  factor(pred$pred_safety)

sbbox <- make_bbox(lon = pred$lng, lat = pred$lat, f = .1)

sq_map <- get_map(location = sbbox, maptype = "roadmap", source = "google")

ggmap(sq_map) + 
  geom_point(data = pred, 
             aes(x = lng, y = lat, colour = pred_safety), alpha = .8)

