#!bin/bash

JSD="/netscratch/dep_mercier/grp_marques/mzhang/HiC_maps/Rhynchospora_deep_HiC/Rhynchospora_barbata"

# merge 2 haplotypes
cd $JSD/references/
REF=$JSD/references/rhyBarHap1_chrs.fasta

# index genome
bwa index $REF
#samtools faidx $REF

cut -f1,2 ${REF}.fai > ${REF}.genome

# Run juicer
cd $JSD
$JSD/scripts/juicer.sh -g ${REF%.fasta} \
 -s none -t 40 -D $JSD \
 -d $JSD/HiC_work -z $REF \
 -p $REF.genome
