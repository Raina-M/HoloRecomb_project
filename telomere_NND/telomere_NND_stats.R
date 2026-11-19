library(ggstatsplot)
library(ggplot2)

# run from the folder that contains this script and its input files (see README)
                     
# ---------- custom settings ----------
mytexttheme <- theme(panel.background = element_rect(fill = "white", colour = "white", linewidth = 0.5, linetype = "solid"),
                     #panel.grid.major.y = element_line(size = 0.5, linetype = 'dashed', colour = "darkgrey"),
                     panel.grid.major.x = element_line(linewidth = 0.3, linetype = 'dashed', colour = "grey"),
                     #panel.grid.minor.x = element_line(size = 0.25, linetype = 'dashed', colour = "lightgrey"),
                     plot.title = element_text(family = "Helvetica", face = "bold", size = (15), hjust = 0.5),
                     legend.title = element_text(family = "Helvetica", size = (15)), 
                     legend.text = element_text(size = (15)),
                     axis.title.x = element_text(vjust = 5, hjust = 1.075, size = (15)),
                     axis.title.y = element_text(family = "Helvetica", vjust=3, size = (15)),
                     axis.text.x = element_text(size = (15), colour = "black", hjust = 0.5, vjust=0.5, angle=45),
                     axis.text.y = element_text(size = (15), colour = "black", hjust = 0.05)
                     #axis.line = element_line(colour = "black", size = 0.75, linetype = "solid")
                     )
                     
# read file
df <- read.table("telomere_NND.tsv")
colnames(df) <- c("species", "NND")

# order species by their mean values
df$species <- reorder(df$species, df$NND, mean)

# prepare output plot
pdf("telomere_NND.pdf",
    family="Helvetica", height=6, width=9)
par(mai = c(1, 1, 1, 1))

set.seed(123)
# CO from scRNA and MLH1 foci
ggbetweenstats(data=df, x=species, y=NND,
               plot.type = "violin", # title = "Distribution of CO number",
               xlab="", ylab = 'NND (micrometer)',
               centrality.label.args = list(size = 5, nudge_x = 0.6, nudge_y=0.5),
               results.subtitle=FALSE,
               ggtheme = mytexttheme) +
  ggplot2::scale_y_continuous(breaks=seq(0, 8, 2)) +
  theme(plot.margin = unit(c(0.5,0.5,0.5,1), "cm"),
        panel.border = element_rect(colour = "black", fill=NA, size=1))
dev.off()

# -------------------------------------------------------------------------------------------------------------#

library(ggplot2)
library(gghalves)

pdf("telomere_NND_boxplot.pdf", width=10, height = 6)
ordercolors <- c("#D95F02", "#1B9E77", "#66A61E", "#7570B3", "#3399FF", "#E729BA", "#A6761D", "#E6AB02")

ggplot(df, aes(x = species, y = NND, fill = species)) +
  geom_boxplot(
    width = 0.3,
    position= position_nudge(x=0.2),
    linewidth = 0.5,
    alpha = 0.25,
    outlier.shape = NA
  ) +
  geom_point(
    position = position_jitter(width = 0.15, height = 0),
    aes(x = as.numeric(species) - 0.2),
    shape = 21,
    size = 3,
    color = "white"
  ) +
  scale_fill_manual(values = ordercolors) +
  labs(y = "NND (micrometer)", x = NULL) +
  theme_classic() +
  theme(
    legend.position = "none",
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 15),
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.margin = margin(20, 20, 10, 20)
  )
dev.off()

########## Normalize NND by cell area ###################
library(dplyr)
library(tibble)

df_area <- read.table("cell_areas.tsv")
colnames(df_area) <- c("species", "area")

# Mean cell area of each species
mean_area <- tapply(df_area$area, df_area$species, mean)

df_mean_area <- data.frame(mean_area) %>%
  rownames_to_column("species")

nnd_norm <- df %>%
  left_join(df_mean_area, by = "species") %>%
  mutate(normalized_NND = NND / (mean_area)^0.5)

nnd_norm$species <- reorder(nnd_norm$species, nnd_norm$normalized_NND, mean)


pdf("telomere_NND_norm_boxplot.pdf", width=10, height = 6)
ordercolors <- c("#1B9E77", "#E729BA", "#D95F02", "#E6AB02", "#66A61E", "#A6761D", "#7570B3", "#3399FF")


ggplot(nnd_norm, aes(x = species, y = normalized_NND, fill = species)) +
  geom_boxplot(
    width = 0.3,
    position= position_nudge(x=0.2),
    linewidth = 0.5,
    alpha = 0.25,
    outlier.shape = NA
  ) +
  geom_point(
    position = position_jitter(width = 0.15, height = 0),
    aes(x = as.numeric(species) - 0.2),
    shape = 21,
    size = 3,
    color = "white"
  ) +
  scale_fill_manual(values = ordercolors) +
  labs(y = "Normalized NND", x = NULL) +
  theme_classic() +
  theme(
    legend.position = "none",
    axis.title = element_text(size = 18),
    axis.text = element_text(size = 15),
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.margin = margin(20, 20, 10, 20)
  )
dev.off()

set.seed(123)
# CO from scRNA and MLH1 foci
pdf("telomere_NND_norm_with_sigTest.pdf", width=10, height = 6)
ggbetweenstats(data=nnd_norm, x=species, y=normalized_NND,
               plot.type = "violin", # title = "Distribution of CO number",
               xlab="", ylab = 'Normalized NND',
               centrality.label.args = list(size = 5, nudge_x = 0.6, nudge_y=0.05),
               results.subtitle=FALSE,
               ggtheme = mytexttheme) +
  ggplot2::scale_y_continuous(breaks=seq(0, 8, 2)) +
  theme(plot.margin = unit(c(0.5,0.5,0.5,1), "cm"),
        panel.border = element_rect(colour = "black", fill=NA, size=1))
dev.off()

#########

# stats
shapiro.test(nnd_norm[nnd_norm$species=="R_breviuscula", 4]) # not normal
shapiro.test(nnd_norm[nnd_norm$species=="R_pubera", 4]) # normal
shapiro.test(nnd_norm[nnd_norm$species=="R_cephalotes", 4]) # normal
shapiro.test(nnd_norm[nnd_norm$species=="R_nervosa", 4]) # normal

# variance
var.test(nnd_norm[nnd_norm$species=="R_pubera", 4], nnd_norm[nnd_norm$species=="R_nervosa", 4]) # equal
var.test(nnd_norm[nnd_norm$species=="R_cephalotes", 4], nnd_norm[nnd_norm$species=="R_nervosa", 4]) # equal

# t test
t.test(nnd_norm[nnd_norm$species=="R_pubera", 4], nnd_norm[nnd_norm$species=="R_nervosa", 4],
       alternative = "less", var.equal = T)

t.test(nnd_norm[nnd_norm$species=="R_cephalotes", 4], nnd_norm[nnd_norm$species=="R_nervosa", 4],
       alternative = "less", var.equal = T)

# mann-whitney test
wilcox.test(nnd_norm[nnd_norm$species=="R_breviuscula", 4], nnd_norm[nnd_norm$species=="R_nervosa", 4],
            alternative = "less")

