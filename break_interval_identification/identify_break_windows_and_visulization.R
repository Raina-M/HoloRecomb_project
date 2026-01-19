# read input arguments
args <- commandArgs(trailingOnly = TRUE)

win_bed <- args[1]        # bed file that has window smoothed chr ID numbers
org_bed <- args[2]        # bed file that has original chr ID numbers (only main compositions)
genome <- args[3]         # genome file with chr sizes
output_refine <- args[4]  # output file that has chromosome and break intervals
plot1 <- args[5]          # identification of breaks on chr
plot2 <- args[6]          # identification of breaks in fine regions
median_size <- args[7]    # size of median filter
# median size 101 works well for most species except for R. pubera
# due to its large genome size

#####
library(stats)

df <- read.table(win_bed)
colnames(df) <- c('chr', 'start', 'end', 'chrnum')

chrlist <- unique(df$chr)

chrsizes <- read.table(genome)

# visualization output
pdf(plot1, width = 5*length(chrlist), height = 6)
layout(matrix(1:(3*length(chrlist)), 3, length(chrlist), byrow = F),
       widths=10*c(chrsizes$V2)/sum(chrsizes$V2),
       heights=c(2,2,2))

break_intervals <- c()
#df_all <- c()
for (chr in chrlist) {
  df_chr <- df[df$chr==chr & df$chrnum!=0,]
  
  # median filter to remove noisy signals
  df_chr$medianSig <- runmed(df_chr$chrnum, median_size, endrule="constant")
  
  # first derivative
  df_chr$change_to_prev <- c(0, diff(df_chr$medianSig))
  
  #df_all <- rbind(df_all, df_chr)
  
  # non-zero derivatives (potential changing points)
  potential_cp <- df_chr[df_chr$change_to_prev!=0,]
  
  # find changing intervals in the non-zero derivatives
  # as long as this windows are in the distance of 1Mbp to the next widow
  # they are regarded as the same changing points
  prev_record <- df_chr[which(rownames(df_chr)==rownames(potential_cp[1,]))-1,]
  
  interval_s <- c(prev_record$end+1)
  interval_e <- c()
  old_chrnum <- c(prev_record$medianSig)
  new_chrnum <- c()
  
  if (nrow(potential_cp) < 2){
    print(paste(chr, "has only one row at the changing point."))
  }
  else{
    for (i in 2:nrow(potential_cp)) {
      current_s <- potential_cp[i, 'start']
      
      if((current_s - potential_cp[i-1, 'end'])>=1e+06){
        # when current position is not close to previous,
        # update
        interval_e <- c(interval_e, potential_cp[i-1, 'end'])
        new_chrnum <- c(new_chrnum, potential_cp[i-1, 'medianSig'])
        
        prev_record <- df_chr[which(rownames(df_chr)==rownames(potential_cp[i,]))-1,]
        interval_s <- c(interval_s, prev_record$end+1)
        old_chrnum <- c(old_chrnum, potential_cp[i-1, 'medianSig'])
      }
    }
  }
  interval_e <- c(interval_e, potential_cp[nrow(potential_cp), 'end'])
  new_chrnum <- c(new_chrnum, potential_cp[nrow(potential_cp), 'medianSig'])
  
  # define break interval in windows
  break_this_chr <- cbind(rep(chr, length(interval_s)), interval_s, interval_e, old_chrnum, new_chrnum)
  break_intervals <- rbind(break_intervals, break_this_chr)
  
  # visualization
  x <- apply(df_chr[, c('start', 'end')], 1, mean)
  plot(x, df_chr$chrnum, type="l",
       main=chr,
       xlab="Chromosome position",
       ylab="Chromosome ID",
       cex=1.2, cex.lab=1.2)
  plot(x, df_chr$medianSig, type="l",
       xlab="Chromosome position",
       ylab="Chr ID after median filter",
       cex=1.2, cex.lab=1.2)
  plot(x, df_chr$change_to_prev, type="l",
       xlab="Chromosome position",
       ylab="1st derivative after median filter",
       cex=1.2, cex.lab=1.2)
}
dev.off()

break_intervals <- data.frame(break_intervals)
colnames(break_intervals) <- c('target_chr', 'interval_s', 'interval_e', 'old_chr', 'new_chr')
break_intervals$interval_s <- as.numeric(break_intervals$interval_s)
break_intervals$interval_e <- as.numeric(break_intervals$interval_e)
break_intervals$old_chr <- as.numeric(break_intervals$old_chr)
break_intervals$new_chr <- as.numeric(break_intervals$new_chr)

# output the window intervals
# write.table(break_intervals, output_win, sep="\t", quote = F, row.names = F, col.names = F)



# refine break intervals
df_org <- read.table(org_bed)

pdf(plot2, width = 6, height = 4)

df_refine <- c()
for (i in 1:nrow(break_intervals)) {
  target_chr <- break_intervals[i, 'target_chr']
  old_chr <- break_intervals[i, 'old_chr']
  new_chr <- break_intervals[i, 'new_chr']
  
  this_cp <- df_org[df_org$V1==target_chr &
                    df_org$V3>=(break_intervals[i, 'interval_s']-1e+06) &
                    df_org$V2<=(break_intervals[i, 'interval_e']+1e+06) &
                    (df_org$V4==old_chr | df_org$V4==new_chr),]
  
  # compute cumulative counts of old and new chrs
  count_old <- cumsum(this_cp$V4==old_chr)
  count_new <- cumsum(this_cp$V4==new_chr)
  
  # find partition position with minimum cost
  min_cost <- Inf
  best_k <- 0
  
  # Iterate through possible partition points (k = 1 to n-1)
  n <- nrow(this_cp)
  for (k in 1:(n-1)) {
    # Misclassifications:
    # - Left region (indices 1 to k): count of new chr (should be mostly old chr)
    # - Right region (indices k+1 to n): count of old chr (should be mostly new chr)
    left_new <- count_new[k]
    right_old <- count_old[n] - count_old[k]
    cost <- left_new + right_old
    
    if (cost < min_cost) {
      min_cost <- cost
      best_k <- k
    }
  }
  
  # best partition location
  df_refine <- rbind(df_refine, c(target_chr, this_cp[best_k, 3], this_cp[best_k+1, 2], old_chr, new_chr))
  
  # visualization
  plot(this_cp$V2, this_cp$V4,
       main=target_chr,
       xlab="Chromosome position", ylab="Chromosome ID number")
  abline(v=this_cp[best_k, 3], col='red')
}
dev.off()

# output the refined intervals
write.table(df_refine, output_refine, sep="\t", quote = F, row.names = F, col.names = F)
