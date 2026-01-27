# Author: JT Lovell
# Date: 18-Dec 2025

# Notes:
# Code below is a combination of internal tools to geeViz (cite DEEPSPACE), and
# ad hoc functions outlined below.
# DEEPSPACE cite: github.com/jtlovell/DEEPSPACE

# 3rd party libraries (see session info below)
library(geeviz)
library(GenomicRanges)
library(Biostrings)
library(data.table)
library(ggplot2)

################################################################################
# 1. Read in files
# -- paths to all the files
wd <- "/PATH/recombination_landscapes_and_features_corrected_chrID"
assems <- pull_namedPaths(
  path = wd,
  pattern = "chrs.fasta$", recursive = TRUE, full.names = TRUE)
tybas <- pull_namedPaths(
  path = wd,
  pattern = "Tyba_arrays.gff3", recursive = TRUE, full.names = TRUE)
cms <- pull_namedPaths(
  path = wd,
  pattern = "rates.txt", recursive = TRUE, full.names = TRUE)
genes <- pull_namedPaths(
  path = wd,
  pattern = "genes.bed", recursive = TRUE, full.names = TRUE)
reps <- pull_namedPaths(
  path = wd,
  pattern = "repeat_clustering.gff3", recursive = TRUE, full.names = TRUE)
tes <- pull_namedPaths(
  path = wd,
  pattern = "TE.gff3", recursive = TRUE, full.names = TRUE)
gids <- rev(c(
  "R_cephalotes", "R_breviuscula", "R_nervosa", "R_ciliata", "R_colorata", "R_barbata"))

# -- get dictionary of chrNames
ord <- c(3,9,4,6,7,8,5,2,1)
nn <- paste0(letters[1:9],ord)
names(nn) <- as.character(ord)

# -- assemblies
chrNamesStrip <- "Chr|_h1|_h2|cr|chr|HiC_scaffold_"
minChrLen <- 1e6
ssList <- sapply(assems, simplify = FALSE, USE.NAMES = TRUE, function(x){
  ss <- Biostrings::readDNAStringSet(x)
  ss <- ss[width(ss) >= minChrLen]
  names(ss) <- gsub(chrNamesStrip, "", names(ss))
  return(ss)
})
names(ssList[["R_cephalotes"]]) <- nn[as.character(names(ssList[["R_cephalotes"]]))]
ssList[["R_cephalotes"]] <- ssList[["R_cephalotes"]][order(names(ssList[["R_cephalotes"]]))]

# -- centromeres
tybaList <- sapply(tybas, simplify = FALSE, USE.NAMES = TRUE, function(x){
  repGr <- read_gffAsGr(
    gff3File = x, chrNamesStrip = chrNamesStrip,
    reduceIt = TRUE)
  return(repGr)
})
x <- as.data.frame(tybaList$R_cephalotes)
x$seqnames <- nn[as.character(x$seqnames)]
tybaList$R_cephalotes <- makeGRangesFromDataFrame(x)

# -- cm
cm <- lapply(cms, function(x){
  y <- gsub("\t", " ", readLines(x))
  y <- as.data.table(tstrsplit(y[-1], " "))[,c(1,2,4)]
  setnames(y, c("chr", "start", "xo"))
  y[,`:=`(start = as.numeric(start),
          end = as.numeric(start),
          xo = as.numeric(xo))]
  return(y)
})
x <- data.table(cm$R_cephalotes)
x$chr <- nn[as.character(x$chr)]
cm$R_cephalotes <- x

# -- genes
geneList <- sapply(names(genes), simplify = FALSE, USE.NAMES = TRUE, function(i){
  x <- genes[i]
  if(i == "R_colorata"){
    tmp <- fread(x,
                 col.names = c("chr", "start", "end"), select = c(1,4,5))
  }else{
    tmp <- fread(x,
                 col.names = c("chr", "start", "end"), select = 1:3)
  }

  tmp[,chr := gsub(chrNamesStrip, "", chr)]

  if(i == "R_cephalotes"){
    tmp$chr <- as.character(tmp$chr)
    tmp$chr <- nn[tmp$chr]
  }

  gr <- makeGRangesFromDataFrame(
    tmp,
    keep.extra.columns = TRUE,
    ignore.strand = TRUE)
  gr <- reduce(gr)
  return(gr)
})

# -- tandem repeats
repList <- sapply(reps, simplify = FALSE, USE.NAMES = TRUE, function(x){
  repGr <- read_gffAsGr(
    gff3File = x,
    chrNamesStrip = chrNamesStrip,
    reduceIt = TRUE)
  return(repGr)
})

x <- as.data.frame(repList$R_cephalotes)
x$seqnames <- nn[as.character(x$seqnames)]
repList$R_cephalotes <- makeGRangesFromDataFrame(x)

# -- tes
teList <- sapply(tes, simplify = FALSE, USE.NAMES = TRUE, function(x){
  teGr <- read_gffAsGr(
    gff3File = x,
    chrNamesStrip = chrNamesStrip,
    reduceIt = TRUE)
  return(teGr)
})

x <- as.data.frame(teList$R_cephalotes)
x$seqnames <- nn[as.character(x$seqnames)]
teList$R_cephalotes <- makeGRangesFromDataFrame(x)


################################################################################
# 2. Get synteny

i = 1
syns <- lapply(1:(length(gids)-1), function(i){
  gi <- gids[i:(i+1)]
  si <- sapply(ssList[gi], function(x) sum(width(x)))
  if(si[1] > si[2]){
    qgen <- gi[1]
    tgen <- gi[2]
  }else{
    qgen <- gi[2]
    tgen <- gi[1]
  }

  qss <- ssList[[qgen]]
  tss <- ssList[[tgen]]

  rhits <- DEEPSPACE::mm2_windows(
    query = qss,
    target = tss,
    speedPreset = "fast",
    xPreset = "10",
    nCores = 8,
    minimap2call = "/opt/miniconda3/envs/minimap2/bin/minimap2")
  shits2 <- DEEPSPACE::synteny_windows(
    paf = rhits, topHits2keep = 1,
    maxGapInBlk = 1e6, minBlkSize = 5, minMapq = 30, minSplitSize = 5,
    MCScanX_hCall = "/PATH/programs/MCScanX/MCScanX_h")
  print(ggplot(shits2$hits, aes(x = qstart/1e6, y = tstart/1e6,
                                colour = factor(blkID, levels = sample(unique(blkID)))))+
          geom_point(pch = ".", alpha = .2)+
          facet_grid(tname ~ qname, scale = "free", space = "free")+
          scale_color_discrete(guide = "none"))
  blkMd <- data.table(query = qgen, target = tgen, top = gi[1], bottom = gi[2], y = i)
  return(list(md = blkMd, blks = shits2$blkCoords))
})

################################################################################
# 3. Do sliding windows for simplified features (crossovers)
swList <- lapply(gids, function(i){
  cmi <- cm[[i]]
  cmi$class <- "nXOs"
  cmi[,`:=`(start = start, end = start, region = 1:.N,  prop = xo/2)]
  cmi$prop[cmi$prop > 1] <- 1
  cmi[,`:=`(nbp = (end - start) * prop,
            uniqueRegionID = paste0("region", region),
            genome = i, color = "dodgerblue")]
  cmu <- data.table(cmi)
  cmu[,`:=`(prop = 1-prop, color = "white",  class = "missing")]

  sw <- makeGRangesFromDataFrame(rbind(cmi, cmu), keep.extra.columns = TRUE)
  return(sw)
})
swGr <- do.call(c, swList)
swGr$grp <- factor(swGr$class, levels = c("nXOs", "missing"))


featList <- sapply(gids, simplify = FALSE, USE.NAMES = TRUE, function(i){
  geneList[[i]]$type = "gene"
  repList[[i]]$type = "tandemRepeat"
  teList[[i]]$type = "te"

  feats <- c(geneList[[i]],  repList[[i]],  teList[[i]])


  return(feats)
})

################################################################################
# 4. Do sliding windows for simplified features (genes + repeats)
swList2 <- lapply(gids, function(i){
  grdi <- window_gr(x = ssList[[i]], width = 1e6, step = 100e3)
  feati <- featList[[i]]
  feati$class <- feati$type
  feati$class <- factor(feati$class, levels = c("gene", "tandemRepeat", "te"))
  feati$type <- NULL
  sw <- count_overlapsByGroup(regions = grdi, features = feati)
  sw$genome <- i
  return(sw)
})
swGr2 <- do.call(c, swList2)
swGr2$color <- ifelse(swGr2$grp == "gene", "darkorange",
                     ifelse(swGr2$grp == "tandemRepeat", "dodgerblue4",
                            ifelse(swGr2$grp == "te", "lightblue", "white")))
swGr2$grp <- factor(swGr2$grp, levels = c("gene", "missing","tandemRepeat", "te"))

################################################################################
# 5. Parse the DEEPSPACE synteny blocks
md <- rbindlist(lapply(syns, function(x) x$md))
dsBlks <- rbindlist(lapply(syns, function(x) data.table(
  qgenome = x$md$query,
  tgenome = x$md$target,
  gindex = x$md$y,
  x$blks)))

dsPrep <- prep_paf4rip(paf = dsBlks, genomeIDs = gids)
dsPrep[,`:=`(braidColor = "white", braidAlpha = .5,
             braidOutlineColor = NA, braidOutlineWidth = NA)]


################################################################################
# 6. Simplify linearize function
linearize_chrCoords <- function(x, gapSize = 0, stripChrName = "", xoffset = 0){
  si <- x

  # -- convert to data table
  dt <- data.table(chr = names(si), width = si)

  # -- get the starts and ends
  dt[,gp := gapSize]
  dt[,xStart := c(1, cumsum(width[-.N]) + cumsum(gp[-.N]) + 1)]
  dt[,`:=`(xEnd = xStart + width, gp = NULL, mid = (xStart + width + xStart)/2)]

  # -- rename chrs if necessary
  if(stripChrName != ""){
    dt[,chr := gsub(stripChrName, "", chr)]
  }

  if(xoffset != 0){
    dt[,`:=`(xStart = xStart + xoffset,
             xEnd = xEnd + xoffset,
             mid = mid + xoffset)]
  }
  return(dt)
}

################################################################################
# 7. Get tyba into format
cengr <- do.call(c, lapply(gids, function(i){
  x <- tybaList[[i]]
  x$genome <- i
  x$grp <- "tyba"
  x$color <- "gold"
  return(x)
}))

################################################################################
# 8. Get crossovers into a heatmap
classGrList <- lapply(gids, function(i){
  x <- data.table(cm[[i]])
  x[,grp := ifelse(xo == 0, "none",
                   ifelse(xo < .5, "low",
                          ifelse(xo < 1, "mid", "high")))]
  x[,genome := i]
  x[,`:=`(end = start + 50e3, start = start -50e3)]
  x[,xo := NULL]
  d1 <- makeGRangesFromDataFrame(
    x, keep.extra.columns = TRUE)
  return(d1)
})
classGr <- do.call(c, classGrList)
classGr$color <- ifelse(classGr$grp == "none", "black",
                        ifelse(classGr$grp == "low", "darkblue",
                               ifelse(classGr$grp == "mid", "purple", "magenta")))

################################################################################
# 9 Make plots
pdf(file.path(wd, "outRips_reord.pdf"), height = 6, width = 8.5)
dsPrep2 <- data.table(dsPrep)
dsPrep2[,braidColor := ifelse(strand == "-", "green", "white")]

annotated_riparian(
  paf = dsPrep,
  genomeIDs = gids,
  gapWidth = .01,
  yBuffer = .02,
  chrHeight = .05,
  chrColorsHeight = .1,
  chrDensity = swGr,
  chrColors = cengr,
  chrDensityHeight = .3)

annotated_riparian(
  paf = dsPrep,
  genomeIDs = gids,
  gapWidth = .01,
  yBuffer = .02,
  chrHeight = .05,
  chrColorsHeight = .1,
  chrDensity = swGr2,
  chrColors = classGr,
  chrDensityHeight = .3)

annotated_riparian(
  paf = dsPrep2,
  genomeIDs = gids,
  gapWidth = .01,
  yBuffer = .02,
  chrHeight = .05,
  chrColorsHeight = .1,
  chrDensity = swGr,
  chrColors = cengr,
  chrDensityHeight = .3)

annotated_riparian(
  paf = dsPrep2,
  genomeIDs = gids,
  gapWidth = .01,
  yBuffer = .02,
  chrHeight = .05,
  chrColorsHeight = .1,
  chrDensity = swGr2,
  chrColors = classGr,
  chrDensityHeight = .3)
dev.off()

# -- session info
# R version 4.4.1 (2024-06-14)
# Platform: aarch64-apple-darwin20
# Running under: macOS 15.6
#
# Matrix products: default
# BLAS:   /System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/libBLAS.dylib
# LAPACK: /Library/Frameworks/R.framework/Versions/4.4-arm64/Resources/lib/libRlapack.dylib;  LAPACK version 3.12.0
#
# locale:
#   [1] en_US.UTF-8/en_US.UTF-8/en_US.UTF-8/C/en_US.UTF-8/en_US.UTF-8
#
# time zone: America/Denver
# tzcode source: internal
#
# attached base packages:
#   [1] stats4    stats     graphics  grDevices utils     datasets  methods   base
#
# other attached packages:
#   [1] ggplot2_4.0.0        data.table_1.16.4    Biostrings_2.72.1    XVector_0.44.0       GenomicRanges_1.56.2
# [6] GenomeInfoDb_1.40.1  IRanges_2.38.1       S4Vectors_0.42.1     BiocGenerics_0.50.0  geeviz_0.1.0
#
# loaded via a namespace (and not attached):
#   [1] SummarizedExperiment_1.34.0 DEEPSPACE_0.2.1             gtable_0.3.6                rjson_0.2.23
# [5] Biobase_2.64.0              lattice_0.22-6              vctrs_0.6.5                 tools_4.4.1
# [9] bitops_1.0-9                generics_0.1.3              curl_6.0.1                  parallel_4.4.1
# [13] tibble_3.2.1                pkgconfig_2.0.3             R.oo_1.27.0                 Matrix_1.7-1
# [17] RColorBrewer_1.1-3          S7_0.2.0                    lifecycle_1.0.4             GenomeInfoDbData_1.2.12
# [21] compiler_4.4.1              farver_2.1.2                Rsamtools_2.20.0            codetools_0.2-20
# [25] RCurl_1.98-1.16             yaml_2.3.10                 pillar_1.10.0               crayon_1.5.3
# [29] R.utils_2.12.3              BiocParallel_1.38.0         DelayedArray_0.30.1         dbscan_1.2-0
# [33] neuralnet_1.44.2            abind_1.4-8                 tidyselect_1.2.1            dplyr_1.1.4
# [37] restfulr_0.0.15             labeling_0.4.3              grid_4.4.1                  cli_3.6.5
# [41] SparseArray_1.4.8           magrittr_2.0.3              S4Arrays_1.4.1              XML_3.99-0.17
# [45] withr_3.0.2                 scales_1.4.0                UCSC.utils_1.0.0            GENESPACE_1.3.1
# [49] httr_1.4.7                  matrixStats_1.4.1           igraph_2.1.2                R.methodsS3_1.8.2
# [53] BiocIO_1.14.0               rtracklayer_1.64.0          rlang_1.1.6                 Rcpp_1.0.13-1
# [57] glue_1.8.0                  rstudioapi_0.17.1           jsonlite_1.8.9              R6_2.5.1
# [61] MatrixGenerics_1.16.0       GenomicAlignments_1.40.0    zlibbioc_1.50.0

