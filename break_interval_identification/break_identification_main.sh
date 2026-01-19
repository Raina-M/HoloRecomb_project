#!bin/bash


paf=$1
outDir=$2
queryChrNum=$3
compos=$4
medianSize=$5
winSize=50000
stepSize=10000


# if the chromosome from the smaller genome has more than 10% can map to a chromosome from the bigger genome, then this chromsome from the smaller genome is considered as a "core" composition

# chromosome ids and lengths of the smaller genome (query)
cut -f1-2 $paf | sort | uniq > $outDir/genome_query.genome

# chromosome ids and lengths of the bigger genome (target)
cut -f6-7 $paf | sort | uniq > $outDir/genome_target.genome

# TODO:
# find the main components of this chromosomses
# i.e. the consecutive alignment length of the query chr > 0.05 * query chromosome size
#cut -f1 $outDir/genome_target.genome | while read chr
#do
#  awk -v c=$chr '$6==c {print $1"\t"$2"\t"$3"\t"$4}' $paf > $outDir/this_chr.tmp
  
#  cut -f1 $outDir/this_chr.tmp | sort -u | while read qchr
#  do
#    aln_len=`awk -v qc=$qchr '$1==qc {print $4-$3}' $outDir/this_chr.tmp | paste -sd+ | bc`
#    chrsize=`awk -v qc=$qchr '$1==qc {print $2}' $outDir/genome_query.genome`
#    perc=`echo -e "scale=4;$aln_len/$chrsize" | bc`
#    if (( $(echo "$perc > 0.05" | bc -l) ))
#    then
#      echo -e "$chr\t$qchr" >> $outDir/target_composition.txt
#    else
#      continue
#    fi
#  done
  
  # purge
#  rm $outDir/this_chr.tmp 
#done


# ---------- TARTGET BREAK POINTS ---------- #

# extract alignment positions
cut -f1,3,4,6,8,9 $paf | sort -k4,4 -k5,5n > $outDir/aln_pairs.txt

# replace the query chromosome IDs with number
#!/bin/bash

awk '
    NR==FNR { map[$1] = $2; next }   # Read file that assign chr ID numeric value
    { $1 = map[$1]; print $4"\t"$5"\t"$6"\t"$1}   # Replace $1 in aln_pairs.txt with numeric value
' $queryChrNum $outDir/aln_pairs.txt > $outDir/target_genome_composition.bed

# only include query sequences that are main compositions of a certain target chromosomes
# information of compositions of target chromosomse are in $compos
Rscript /netscratch/dep_mercier/grp_marques/mzhang/GENESPACE/Compare_20_Rhynchospora/scripts_identify_breaks/extract_main_compos.R $outDir/target_genome_composition.bed $compos $outDir/target_main_composition.bed


# make windows

bedtools makewindows -g $outDir/genome_target.genome -w $winSize -s $stepSize > $outDir/windows.bed
# calculate mean in sliding windows
bedtools map -a $outDir/windows.bed -b $outDir/target_main_composition.bed -c 4 -o mean -null 0 > $outDir/target_genome_composition_smoothed.tmp
# only use the target chromosomes having more than 2 main queries
cut -f1 $compos | sort | uniq -c | awk '$1>1 {print $2}' > $outDir/viable_chrs.list
grep -Ff $outDir/viable_chrs.list $outDir/target_genome_composition_smoothed.tmp | awk '$NF>0' > $outDir/target_genome_composition_smoothed.bed
rm $outDir/target_genome_composition_smoothed.tmp

# find the intervals that the breaking happened
# also check the R plots
Rscript /netscratch/dep_mercier/grp_marques/mzhang/GENESPACE/Compare_20_Rhynchospora/scripts_identify_breaks/identify_break_windows_and_visulization.R $outDir/target_genome_composition_smoothed.bed $outDir/target_main_composition.bed $outDir/genome_target.genome $outDir/break_intervals.txt $outDir/identify_breaks_chr.pdf $outDir/identify_breaks_refined.pdf $medianSize

# refine break intervals based on the breaking windows



# TODO:
# ---------- QUERY BREAK POINTS ---------- #
