#------------------------------------------------------------------------------#
### Irrigating Peace:
# Input: SATP scraped data
# Author: Mattia Bachini
#------------------------------------------------------------------------------#

rm(list=ls())
options(scipen=20)

lop <- c("sf", "dplyr", "stringr", "tidyr", "tidyverse", "tmap", "countrycode", 
         "readxl", "data.table", "tmap", "haven", "xtable", "doParallel", "foreach",
         "lubridate", "tidygeocoder", "RColorBrewer", "purrr", "rnaturalearth",
         "rnaturalearthdata")

loaded <- sapply(lop, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org", dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
  TRUE
})

sf_use_s2(FALSE)

setwd("/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/code")

#------------------------------------------------------------------------------#
#scraped data path:
path <- "/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/scraped_data"

list.files(path, pattern = "scraped_data-.*\\.csv", full.names = TRUE) %>%
  map(read.csv) %>%
  bind_rows() %>%
  arrange(lubridate::parse_date_time(as.character(Date), orders = c("ymd", "dmy", "mdy"))) %>% #sort by date when rbinding since two files are separate from the main one 
  saveRDS(file = "/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/scraped_data/scraped_data.rds")

# read the newly merged file
df_merged <- readRDS("/Users/mattiabachini/Library/CloudStorage/Dropbox/code-satp/scraped_data/scraped_data.rds")

colnames(df_merged)


#------------------------------------------------------------------------------#
# check for duplicate incidents: same date and summary text, within or across series
# (Series/Incident_ID differ across series, so whole-row duplicates would miss these)
dup_key <- df_merged[c("Date", "Incident_Summary")]
duplicate_rows <- df_merged[duplicated(dup_key) | duplicated(dup_key, fromLast = TRUE), , drop = FALSE]
if (nrow(duplicate_rows) > 0) {
  dup_groups <- duplicate_rows %>%
    group_by(Date, Incident_Summary) %>%
    summarise(n_rows = n(), n_series = n_distinct(Series), .groups = "drop")
  message("Found ", nrow(duplicate_rows), " rows in ", nrow(dup_groups), " duplicated incidents.")
  message("  repeated within a series: ", sum(dup_groups$n_rows > dup_groups$n_series))
  message("  appearing in more than one series: ", sum(dup_groups$n_series > 1))
  print(dup_groups %>% count(n_series, name = "n_incidents"))
} else {
  message("No duplicate incidents found.")
}

# Incident_ID is series + date + daily counter; the same ID with different text means
# files overlap on a day but disagree on its contents
id_clash <- df_merged %>%
  group_by(Incident_ID) %>%
  filter(n() > 1, n_distinct(Incident_Summary) > 1) %>%
  ungroup()
message("Incident_IDs shared by rows with different summaries: ", n_distinct(id_clash$Incident_ID))


#------------------------------------------------------------------------------#
# rows per series and month, to find gaps (0 = no incidents scraped that month)
first_month <- as.Date("2000-01-01")
last_month  <- floor_date(max(as.Date(df_merged$Date), na.rm = TRUE), "month")

monthly_counts <- df_merged %>%
  mutate(month = floor_date(as.Date(Date), "month")) %>%
  count(Series, month) %>%
  complete(Series, month = seq(first_month, last_month, by = "month"), fill = list(n = 0))

# zero months: failed/skipped scrapes, or genuinely empty months (e.g. Punjab)
gaps <- monthly_counts %>% filter(n == 0) %>% arrange(Series, month)
message("Series-months with zero incidents: ", nrow(gaps))
print(gaps %>% count(Series, name = "n_zero_months"))
print(gaps, n = Inf)

# year x month grid per series, for the manual check
monthly_counts %>%
  mutate(year = year(month), mon = month(month, label = TRUE)) %>%
  select(Series, year, mon, n) %>%
  pivot_wider(names_from = mon, values_from = n) %>%
  split(.$Series) %>%
  walk(~ { message("\n", .x$Series[1]); print(select(.x, -Series), n = Inf) })




