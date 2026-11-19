library(LinkageMapView)
library(RColorBrewer)
library(ggplot2)

# run from the folder that contains this script and its input files (see README)

map_markers <- read.table("merged_linkage_brev_ceph_cili.txt", header = FALSE, sep = "\t")
colnames(map_markers) <- c('chrname', 'cm_dist', 'marker')

# Marey map
pdf("Marey_map_brev_ceph_cili.pdf",
    family="Helvetica", height=8, width=10)
par(mai = c(2, 2, 1, 0.5)); # margin: bottom, left, top, right
#
ggplot(map_markers, aes(x = marker, y = cm_dist, col = chrname)) +
  geom_line(size=1.2) +
  xlab("Physical position (Mb)") +
  ylab("Genetic distance (cM)") +
  scale_color_manual(values = c("#00441B","#238B45","#2eb62c","#A1D99B","#E5F5E0",
                                "#99000D","#CB181D","#EF3B2C","#FB6A4A","#FC9272","#5b2912","#b14612","#ed8d40","#d0a765",
                                "#084594","#2171B5","#4292C6","#6BAED6","#9ECAE1"),
                     name = "Species.Chr",
                     labels = c("Breviscula.1","Breviscula.2","Breviscula.3", "Breviscula.4", "Breviscula.5",
                                "Cephalotes.1","Cephalotes.2","Cephalotes.3","Cephalotes.4","Cephalotes.5","Cephalotes.6","Cephalotes.7","Cephalotes.8","Cephalotes.9",
                                "Ciliata.1","Ciliata.2","Ciliata.3","Ciliata.4","Ciliata.5")) +
  scale_x_continuous(breaks=seq(0, max(map_markers$marker), 1e+07),
                     labels=round(seq(0, max(map_markers$marker), 1e+07)/1e+06),
                     expand = c(0, 0)) +
  scale_y_continuous(breaks = seq(0, 70, by = 10)) +
  theme(axis.title.x = element_text(family = "Helvetica", vjust=-1, size = (15)),
        axis.title.y = element_text(family = "Helvetica", vjust=3, size = (15)),
        axis.text.x = element_text(size = (15), colour = "black", vjust = 0.05),
        axis.text.y = element_text(size = (15), colour = "black", hjust = 0.05),
        panel.background = element_rect(fill = "white", colour = "white", linewidth = 0.5, linetype = "solid"),
        panel.grid.major.x = element_line(linewidth = 0.3, linetype = 'dashed', colour = "grey"),
        panel.grid.major.y = element_line(linewidth = 0.5, linetype = 'dashed', colour = "darkgrey"),
        panel.border = element_rect(colour = "black", fill=NA, linewidth=1),
        plot.margin = unit(c(0.5,0.5,0.5,1), "cm"))
dev.off()



# merged linkage map

# testd <- mstmap(markers, dist.fun = "kosambi", trace = FALSE)
# pull.map(testd)
# 
# lmv.linkage.plot(testd, "test.pdf", denmap = TRUE)
#sectcoldf <- lmvdencolor(map_markers, colorin = rev(brewer.pal(8,"Spectral")))
sectcoldf <- lmvdencolor(map_markers, colorin = rev( c("#D53E4F","#FEE08B","#66C2A5", "#3288BD","#5E4FA2")))

lmv.linkage.plot(map_markers, "linkage_map.pdf", denmap=TRUE, sectcoldf=sectcoldf,
                 pdf.height = 10, pdf.height = 25)

# ----- R. breviuscula -----
map_markers <- read.table("linkage_info_for_plot_Rb.txt", header = FALSE, sep = "\t")
colnames(map_markers) <- c('chrname', 'cm_dist', 'marker')

sectcoldf <- lmvdencolor(map_markers, colorin = rev(c("#D53E4F","#FEE08B","#66C2A5", "#3288BD","#5E4FA2")))

lmv.linkage.plot(map_markers, "linkage_map_Rbrev.pdf", denmap=TRUE, sectcoldf=sectcoldf,
                 pdf.height = 8, pdf.width = 6)

# ----- R. colorata -----
map_markers <- read.table("linkage_info_for_plot_Rcol.txt", header = FALSE, sep = "\t")
colnames(map_markers) <- c('chrname', 'cm_dist', 'marker')

sectcoldf <- lmvdencolor(map_markers, colorin = rev(c("#D53E4F","#FEE08B","#66C2A5", "#3288BD","#5E4FA2")))

lmv.linkage.plot(map_markers, "linkage_map_Rcol.pdf", denmap=TRUE, sectcoldf=sectcoldf,
                 pdf.height = 7, pdf.width = 6)

# ----- R. cephalotes -----
map_markers <- read.table("linkage_info_for_plot_Rcep.txt", header = FALSE, sep = "\t")
colnames(map_markers) <- c('chrname', 'cm_dist', 'marker')

sectcoldf <- lmvdencolor(map_markers, colorin = rev(c("#D53E4F","#FEE08B","#66C2A5", "#3288BD","#5E4FA2")))

lmv.linkage.plot(map_markers, "linkage_map_Rcep.pdf", denmap=TRUE, sectcoldf=sectcoldf,
                 pdf.height = 10, pdf.width = 10)
