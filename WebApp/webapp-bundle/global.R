# set up -----
rm(list = ls())
invisible(gc(verbose = F))

options(
  scipen = 999, 
  max.print = 100
)

# pakages -----
suppressMessages({
  library(shiny)
  library(shinyWidgets)
  library(shinyjs)
  library(dplyr)
  library(googlesheets)
  library(googledrive)
  library(xlsx)
})

# source -----
source("functions.R")

# google authorization -----
gs_auth()
drive_auth()

# static data -----
scale_anchors <- list(
  evaluative = list(c("Bad", "Awful"), c("Good", "Nice")),
  potency = list(c("Powerless", "Little"), c("Powerful", "Big")),
  activity = list(c("Slow", "Quiet", "Inactive"), c("Fast", "Noisy", "Active"))
)
