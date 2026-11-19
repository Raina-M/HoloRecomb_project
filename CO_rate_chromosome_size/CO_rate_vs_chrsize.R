library("RColorBrewer")
library(ggplot2)
library(scales)

# run from the folder that contains this script and its input files (see README)

co_rate_barb <- read.table("R_barbata_CO_num_per_chr.txt", header = FALSE)[,2:3]
colnames(co_rate_barb) <- c('chrsize', 'co_num')
co_rate_barb$co_rate <- co_rate_barb$co_num * 1e08 /(co_rate_barb$chrsize * 2495)
co_rate_barb$color <- "#7570B3"
co_rate_barb$shape <- 16

co_rate_brev <- read.table("R_breviuscula_CO_num_per_chr.txt", header = FALSE)[,2:3]
colnames(co_rate_brev) <- c('chrsize', 'co_num')
co_rate_brev$co_rate <- co_rate_brev$co_num * 1e08 /(co_rate_barb$chrsize * 1641)
co_rate_brev$color <- "#1B9E77"
co_rate_brev$shape <- 16

co_rate_ceph <- read.table("R_cephalotes_CO_num_per_chr.txt", header = FALSE)[,2:3]
colnames(co_rate_ceph) <- c('chrsize', 'co_num')
co_rate_ceph$co_rate <- co_rate_ceph$co_num * 1e08 /(co_rate_ceph$chrsize * 2719)
co_rate_ceph$color <- "#D95F02"
co_rate_ceph$shape <- 16

co_rate_cili <- read.table("R_ciliata_CO_num_per_chr.txt", header = FALSE)[,2:3]
colnames(co_rate_cili) <- c('chrsize', 'co_num')
co_rate_cili$co_rate <- co_rate_cili$co_num * 1e08 /(co_rate_cili$chrsize * 597)
co_rate_cili$color <- "#3399FF"
co_rate_cili$shape <- 16

co_rate_colo <- read.table("R_colorata_CO_num_per_chr.txt", header = FALSE)[,2:3]
colnames(co_rate_colo) <- c('chrsize', 'co_num')
co_rate_colo$co_rate <- co_rate_colo$co_num * 1e08 /(co_rate_colo$chrsize * 429)
co_rate_colo$color <- "#66A61E"
co_rate_colo$shape <- 16

co_rate_nerv <- read.table("R_nervosa_CO_num_per_chr.txt", header = FALSE)[,2:3]
colnames(co_rate_nerv) <- c('chrsize', 'co_num')
co_rate_nerv$co_rate <- co_rate_nerv$co_num * 1e08 /(co_rate_nerv$chrsize * 1173)
co_rate_nerv$color <- "#E6AB02"
co_rate_nerv$shape <- 16

co_rate_pube <- read.table("R_pubera_CO_num_per_chr.txt", header = FALSE)[,2:3]
colnames(co_rate_pube) <- c('chrsize', 'co_num')
co_rate_pube$co_rate <- co_rate_pube$co_num * 1e08 /(co_rate_pube$chrsize * 489 *2)
co_rate_pube$color <- "#E7298A"
co_rate_pube$shape <- 16

co_rate_pube_immuno <- data.frame(
  chrsize=1472673118,
  co_num=639,
  co_rate=639 * 1e08 / (1472673118 * 84 * 2),
  color="#E7298A",
  shape=17
)

co_rate_brev_immuno <- data.frame(
  chrsize=368174147,
  co_num=921,
  co_rate=921 * 1e08 / (368174147 * 152 * 2),
  color="#1B9E77",
  shape=17
)

df <- data.frame(rbind(co_rate_barb, co_rate_brev, co_rate_ceph, co_rate_cili,
                       co_rate_colo, co_rate_nerv, co_rate_pube))

df_immu <- data.frame(rbind(co_rate_barb, co_rate_brev, co_rate_ceph, co_rate_cili,
                            co_rate_colo, co_rate_nerv, co_rate_pube_immuno))

df_all <- data.frame(rbind(co_rate_barb, co_rate_brev, co_rate_ceph, co_rate_cili,
                           co_rate_colo, co_rate_nerv, co_rate_pube,
                           co_rate_pube_immuno, co_rate_brev_immuno))
  
# Fitting the Exponential Regression Model
#t <- data.frame(cbind(p$size, p$cM*1000000/p$size))
#colnames(t) <- c('size','co_rate')
model <- lm(log(df$co_rate) ~ df$chrsize)
summary(model)

model2 <- lm(log(df_immu$co_rate) ~ log(df_immu$chrsize))
summary(model2)

model_all <- lm(log(df_all$co_rate) ~ log(df_all$chrsize))
summary(model_all)


pdf("CO_rate_vs_chrsize.pdf",
    family="Helvetica", height=4, width=5)
par(mai = c(1.2, 1, 1, 0.5)); # margin: bottom, left, top, right

ggplot(df_all, aes(chrsize, co_rate,
               color = color,
               shape = factor(shape))) +
  geom_point(size = 2) +
  geom_smooth(method = "lm", se = FALSE, color = "#585858") +
  scale_fill_manual(values=unique(df_all$color)) +
  scale_x_log10(breaks = 10^(7:10),
                labels = trans_format("log10", math_format(10^.x)),
                minor_breaks = c(2:9 * 10^7, 2:9 * 10^8)) +
  annotation_logticks(sides = "b") +
  scale_y_log10(breaks = seq(0,2,0.5),
                labels = seq(0,2,0.5)) +
  coord_cartesian(xlim = c(min(df_all$chrsize), max(df_all$chrsize, na.rm = TRUE))) +
  xlab("Chromosome size (bp)") +
  ylab("Crossover rate (cM/Mb)") +
  scale_color_identity() +
  theme_classic() +
  theme(axis.title.x = element_text(family = "Helvetica", vjust=3, size = (15)),
        axis.title.y = element_text(family = "Helvetica", vjust=3, size = (15)),
        axis.text.x = element_text(size = (15)),
        axis.text.y = element_text(size = (15)),
        plot.margin = margin(t=10, r=10, b=10 ,l=10),
        panel.border = element_rect(colour = "black", fill=NA, linewidth = 0.75))

dev.off()
