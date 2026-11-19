library(ggplot2)
library(tidyverse)
library(dplyr)

# run from the folder that contains this script and its input files (see README)


co_num_barb <- read.table("R_barbata_CO_num_per_cell.list", header = FALSE)
colnames(co_num_barb) <- c('co_num')

co_num_brev <- read.table("R_breviuscula_CO_num_per_cell.list", header = FALSE)
colnames(co_num_brev) <- c('co_num')

co_num_ceph <- read.table("R_cephalotes_CO_num_per_cell.list", header = FALSE)
colnames(co_num_ceph) <- c('co_num')

co_num_cili <- read.table("R_ciliata_CO_num_per_cell.list", header = FALSE)
colnames(co_num_cili) <- c('co_num')

co_num_colo <- read.table("R_colorata_CO_num_per_cell.list", header = FALSE)
colnames(co_num_colo) <- c('co_num')

co_num_nerv <- read.table("R_nervosa_CO_num_per_cell.list", header = FALSE)
colnames(co_num_nerv) <- c('co_num')

co_num_pube <- read.table("R_pubera_CO_num_per_meiocyte.list", header = FALSE)
colnames(co_num_pube) <- c('co_num')

co_num_barb <- co_num_barb %>%
  add_column(source="R. barbata", .before="co_num")
co_num_brev <- co_num_brev %>%
  add_column(source="R. breviuscula", .before="co_num")
co_num_ceph <- co_num_ceph %>%
  add_column(source="R. cephalotes", .before="co_num")
co_num_cili <- co_num_cili %>%
  add_column(source="R. ciliata", .before="co_num")
co_num_colo <- co_num_colo %>%
  add_column(source="R. colorata", .before="co_num")
co_num_nerv <- co_num_nerv %>%
  add_column(source="R. nervosa", .before="co_num")
co_num_pube <- mutate(co_num_pube, co_num = 0.5 * co_num) %>%
  add_column(source="R. pubera", .before="co_num")

df_co_count <- rbind(co_num_brev, co_num_ceph, co_num_cili, co_num_colo, co_num_nerv, co_num_barb, co_num_pube)
df_co_count$source <- reorder(df_co_count$source, df_co_count$co_num, mean)
#df_co_count$source <- factor(df_co_count$source, levels=c("R. colorata", "R. breviuscula", "R. nervosa", "R. ciliata", "R. barbata", "R. pubera", "R. cephalotes"))

ordercolors <- c("#66A61E", "#1B9E77", "#E6AB02", "#3399FF", "#7570B3", "#E7298A", "#D95F02")


#########################################################
# violin plot of CO number distribution for each species
#########################################################

# CO from scRNA and MLH1 foci
p <- ggplot(df_co_count, aes(x=source, y=co_num, fill=source)) +
       geom_violin(trim=F, adjust=2, linewidth=0.5, alpha=0.7) +
       geom_boxplot(width=0.2, fill="white", notch = F, outlier.size = 0.75, linewidth=0.5) +
       labs(title="") + xlab("") + ylab("Count of COs") +
       theme_classic() + 
       scale_fill_manual(values=ordercolors) +
       scale_y_continuous(breaks=seq(0, 10, 5)) +
       theme(axis.title.y = element_text(family = "Helvetica", vjust=3, size = (15)),
             axis.text.x = element_text(size = (15), colour = "black", vjust = 0.6, angle=45),
             axis.text.y = element_text(size = (15), colour = "black", hjust = 0.05),
             legend.position = "none",
             plot.margin = margin(t=10, r=10, b=10 ,l=10),
             panel.border = element_rect(colour = "black", fill=NA, linewidth = 0.75))
print(p)
ggsave("violin_boxplot_of_CO_num_per_species.pdf", p, width = 7, height = 4)

