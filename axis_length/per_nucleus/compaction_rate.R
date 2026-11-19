library(ggplot2)
library(dplyr)
library(gghalves)

# working directory
# run from the folder that contains this script and its input files (see README)


df_axis <- read.table("axis_lengths.txt", header=T, sep="\t")
df_chrs <- read.table("chrs_lengths.txt", header=T, sep="\t")
df_chrs$mean_hap <- apply(df_chrs[, c("h1_total","h2_total")], 1, max)

# Join genome size into dataset1
df_out <- df_axis %>%
  left_join(df_chrs, by = "species") %>%
  mutate(axis_norm_chr = mean_axis_length * 1000000 / mean_chr_length,
         axis_norm_cell = total_axis_length * 1000000 / mean_hap)

# order by chromosome length in bp
df_out$species <- reorder(df_out$species, df_out$mean_hap, mean)
ordercolors <- c("#D95F02", "#7570B3", "#1B9E77", "#66A61E", "#3399FF", "#E6AB02")

scale_factor=max(df_out$axis_norm_cell)/max(df_out$mean_hap)

pdf("normalized_axis_length_and_genome_size.pdf", width=8, height = 4.5)
ggplot(df_out, aes(x=species)) +
  geom_boxplot(
    aes(y=axis_norm_cell, fill=species),
    width = 0.3,
    position= position_nudge(x=0.2),
    linewidth = 0.5,
    alpha = 0.25,
    outlier.shape = NA
  ) +
  geom_point(
    position = position_jitter(width = 0.15, height = 0),
    aes(x = as.numeric(species) - 0.2, y=axis_norm_chr, fill=species),
    shape = 21,
    size = 3,
    color = "white"
  ) +
  # line for chromosome size
  geom_line(aes(y=mean_hap * scale_factor, group = 1),
            color='#585858', linewidth=1, linetype="dashed") +
  scale_fill_manual(values = ordercolors) +
  labs(x="") + 
  scale_y_continuous(name="Normalized axis length (mm/Mb)",
                     sec.axis = sec_axis(~ . / scale_factor,
                                         name = "Genome size (bp)")
                     ) +
  theme_classic() +
  theme(
    legend.position = "none",
    axis.title = element_text(size = 15),
    axis.text = element_text(size = 15),
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.margin = margin(20, 20, 10, 20)
  )
dev.off()



