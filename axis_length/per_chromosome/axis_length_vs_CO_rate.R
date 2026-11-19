
# run from the folder that contains this script and its input files (see README)

axis_lengths <- read.table("axis_lengths.txt", header = 1, sep="\t")

species_list <- c("Rhynchospora_breviuscula", "Rhynchospora_cephalotes", "Rhynchospora_ciliata",
                  "Rhynchospora_colorata", "Rhynchospora_nervosa", "Rhynchospora_pubera")
co_rates <- data.frame(species=species_list,
                       co_rate=c(0.669840395, 1.455593945, 0.562917532,
                                 0.541144111, 0.55812295, 0.2582767),
                       mean_co_num=c(2.466179159, 4.419271791, 2.601340034,
                                     2.34032634, 2.543904518, 3.803571),
                       mean_total_length=c(87.39637, 100.1398, 59.75133,
                                           57.49325, 78.6395, 130.4728))

# merge axis length and CO rates
library(dplyr)

df_merged <- left_join(axis_lengths, co_rates, by = "species")


# axis length vs. mean CO num
plot(co_rates$mean_total_length, co_rates$mean_co_num)

plot(df_merged$mean_axis_length, df_merged$co_rate)

plot(co_rates$mean_total_length, co_rates$mean_co_num*100000000/co_rates$mean_total_length)

plot(co_rates$mean_total_length, co_rates$co_rate)
