# read input arguments
args <- commandArgs(trailingOnly = TRUE)
bed <- args[1]
compos <- args[2]
output <- args[3]

df <- read.table(bed)
comp <- read.table(compos)

#chrlist <- unique(comp$V1)
# count repeat time of target chr
t <- table(comp$V1)
chrlist <- names(t)[t > 1]

cleaned_df <- c()
for (chr in chrlist) {
  cleaned_df <- rbind(cleaned_df, df[df$V1==chr & (df$V4 %in% comp[comp$V1==chr, 2]),])
}

df_out <- data.frame(cleaned_df)

write.table(df_out, output, sep="\t", quote = F, row.names = F, col.names = F)
