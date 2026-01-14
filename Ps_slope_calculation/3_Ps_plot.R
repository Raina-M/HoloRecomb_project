library(dplyr)
library(ggplot2)
library(scales)

setwd("/netscratch/dep_mercier/grp_marques/mzhang/Manuscripts/HoloRecom/Figure_4/4b_Ps/")

############# Custom Functions ##############
calculate_Ps <- function(df, species){
  colnames(df) <- c("bin1", "bin2", "contact")
  
  # Calculate genomic distance
  df <- df %>%
    mutate(distance = abs(bin1 - bin2)+1)
    
  # Define bins for genomic distance
  # Create bins between the minimum and maximum distance
  num_bins <- 100000
  bins <- seq(min(df$distance), max(df$distance), length.out = num_bins+1)

  # Assign each distance to a bin
  df$binned_distance <- cut(df$distance, breaks = bins, include.lowest = TRUE)
  
  # Calculate average contact probability within each bin
  p_s <- df %>%
    group_by(binned_distance) %>%
    summarize(mean_contact = mean(contact, na.rm = TRUE)) %>%
    mutate(bin_lft = as.numeric(sub("^[[(]([^,]*),.*", "\\1", binned_distance)),
           bin_rgt = as.numeric(sub(".*,([^]]*)\\]", "\\1", binned_distance)) )
  
  sumPs <- sum(p_s$mean_contact)
  p_s$Ps <- p_s$mean_contact/sumPs
  return(p_s)
}

########### End of Functions ##############


# ----- Main ----- #
species_list <- c("Rhynchospora_breviuscula",
                  "Rhynchospora_cephalotes",
                  "Rhynchospora_ciliata",
                  "Rhynchospora_corymbosa",
                  "Rhynchospora_filiformis",
                  "Rhynchospora_gaudichaudii",
                  "Rhynchospora_holoschoenoides",
                  "Rhynchospora_nervosa",
                  "Rhynchospora_radicans",
                  "Rhynchospora_ridleyi",
                  "Rhynchospora_rugosa",
                  "Rhynchospora_tenuis_PECP2",
                  "Rhynchospora_tenuis_REF",
                  "Rhynchospora_watsonii")
#species_list <- c("Rhynchospora_pubera")
genomeDir="/netscratch/dep_mercier/grp_marques/mzhang/HiC_maps/Rhynchospora_deep_HiC/"
inputDir="/netscratch/dep_mercier/grp_marques/mzhang/HiC_maps/Rhynchospora_deep_HiC/Hi-C_results_analysis/HiC_contacts/"

df <- c()
for (sp in species_list) {
  # list all chromosomes of this species
  genomeFile <- list.files(paste(genomeDir, sp, "/references/", sep=""), pattern="*.genome", full.names=T)
  chrsizes <- read.table(genomeFile)
  
  this_sp <- c()
  for (chr in chrsizes$V1) {
    # read Hi-C counts
    hic_file <- paste(inputDir, sp, '/', chr,'_',chr,'_hic_counts_5k.txt', sep="")
    this_chr <- read.table(hic_file, header = FALSE, sep = "\t")
    
    # Compute Ps
    this_Ps <- calculate_Ps(this_chr, sp)
    
    # attach species name
    this_Ps$sp <- rep(sp, nrow(this_Ps))
    
    this_sp <- rbind(this_sp, this_Ps)
  }
  
  df <- rbind(df, this_sp)
}
# write.table(df, "Ps.txt", quote = F, col.names = F, row.names = F, sep = "\t")


# Plot Ps
pdf(file="Ps_by_species_100kBin_5kRes_0.3.pdf", width=6, height=6)

pl <- ggplot(df, aes(x=bin_lft, y=Ps)) +
  #geom_line(aes(color=sp), linewidth=0.5) +
  geom_smooth(method = "loess", span=0.3,
              aes(color=sp), linewidth=1, se=F) +
  scale_x_log10(
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  scale_y_log10(
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  coord_cartesian(xlim = c(1e3, max(df$bin_lft, na.rm = TRUE))) +
  labs(x = "Genomic distance (log)", y = "Average contact probability P(s) (log)") +
  theme_minimal()

print(pl)
dev.off()


# -------- calculate Ps slope ------------ #
df_slope <- c()
for (sp in species_list) {
  this_df <- df[df$sp==sp,]
  # 1. Fit loess model (same as geom_smooth(method = "loess"))
  loess_fit <- loess(log10(Ps)~log10(bin_lft), data = this_df, span = 0.1)
  
  # 2. Predict smoothed values
  y_smooth <- predict(loess_fit, newdata = log10(this_df$bin_lft))
  
  # 3. Calculate slope (dy/dx)
  slope <- diff(y_smooth) / diff(log10(this_df$bin_lft))
  
  # 4. Record for plot
  df_slope <- rbind(df_slope,
                    cbind(log10(this_df$bin_lft[-1]),
                          slope,
                          rep(sp, length(slope))))
}

df2 <- data.frame(df_slope)
colnames(df2) <- c("bin_lft", "slope", "sp")
df2$bin_lft <- as.numeric(df2$bin_lft)
df2$slope <- as.numeric(df2$slope)

ordercolors <- c("#1B9E77", "#D95F02","#3399FF", "#66A61E", "#7570B3", "#E6AB02")

pdf("slope_100kBin_5kRes_01_ZoomIn.pdf", width = 8, height = 5)
ggplot(df2, aes(x=10^bin_lft, y=slope)) +
  #geom_line(aes(color=sp), linewidth=1.2) +
  geom_smooth(method = "loess", span=0.1,
              aes(color=sp), linewidth=1.2, se=F) +
  #scale_color_manual(values=ordercolors) + 
  scale_x_log10(
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  coord_cartesian(xlim = c(1e5, 8e6), ylim=c(-0.1, 0.1)) +
  labs(x = "Genomic distance (log)", y = "P(s) slope") +
  theme_classic() + annotation_logticks()
dev.off()

