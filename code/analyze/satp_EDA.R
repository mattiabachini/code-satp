#------------------------------------------------------------------------------#
### Irrigating Peace:
# SATP EDA
# Author: Mattia Bachini
#------------------------------------------------------------------------------#

rm(list = ls())
options(scipen = 20, width = 250)

lop <- c("dplyr", "tidyr", "stringr", "sf", "terra", "exactextractr")

loaded <- sapply(lop, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org", dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
  TRUE
})

setwd("/Users/mattiabachini/Library/CloudStorage/Dropbox/irrigating_peace/mattia")
wris    <- "../data/input/public/india_wris"
out_dir <- "sandbox"

sink(file.path(out_dir, "TA_wris_landuse_explore.txt"), split = TRUE)
t0 <- Sys.time()
# clock time and minutes since the start on every progress line
msg <- function(...) message(format(Sys.time(), "[%H:%M:%S] "),
                             sprintf("(%.1f min) ", as.numeric(difftime(Sys.time(), t0, units = "mins"))), ...)
msg("start")


setwd("/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp")
out_dir <- "data/intermediate/validation"

#------------------------------------------------------------------------------#
####                              Import Data                               ####
#------------------------------------------------------------------------------#
df1 <- read.csv('/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/data/input/satp_scrape/scraped_data-3.csv')
df2 <- read.csv('/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/data/input/satp_scrape/scraped_data-2.csv')
df3 <- read.csv('/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/data/input/satp_scrape/scraped_data-1.csv')


df <- rbind(df1, df2, df3)
rm(df1, df2, df3)
colnames(df)

# Check for duplicate rows
sum(duplicated(df))
sum(nrow(distinct(df)))
nrow(df)

# row-level copies: only the two re-scraped months
sum(duplicated(df))                      # 860

# the duplication that matters: same incident across series pages
nrow(distinct(df, Date, Incident_Summary))   # ~127,469

# safest: reuse the merge output, which also normalises whitespace
incidents <- readRDS("data/intermediate/scraped/scraped_incidents.rds")   # 127,466

df  <- df  %>% 
  mutate(Incident_Summary = str_squish(Incident_Summary))  %>% 
  group_by(Date, Incident_Summary)  %>% 
  summarise(series = paste(sort(unique(Series)), collapse = "|"),
            n_series = n_distinct(Series),
            .groups = "drop")
