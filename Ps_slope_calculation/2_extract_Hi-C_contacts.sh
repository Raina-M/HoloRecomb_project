#!bin/bash
WD="/netscratch/dep_mercier/grp_marques/mzhang/HiC_maps/Rhynchospora_deep_HiC/Hi-C_results_analysis/HiC_contacts"
INDIR="/netscratch/dep_mercier/grp_marques/mzhang/HiC_maps/Rhynchospora_deep_HiC"

JUICER_TOOLS=/home/mzhang/software/juicer/CPU/common/juicer_tools.jar

hic_dir_pattern="HiC_work/aligned"
hicfile="inter_30.hic"
chrfile_suffix=".genome"

# juicer_tool parameters
RES=5000    # bin size
resH="5k"
NR="VC_SQRT"  # normalization method



for species in Rhynchospora_gaudichaudii
do
  cd ${INDIR}/${species}
  
  # inspect existence of hic file
  echo "Check Hi-C file of $species ..."
  if [ -s ./$hic_dir_pattern/$hicfile ]
  then
    echo ":) Hi-C file: $hicfile exists."
  else
    echo ":( Hi-C file does not exist. Please check directory ${INDIR}/${species}."
    continue;
  fi
  
  # inspect existence of genome file
  echo "Check the genome file of $species ..."
  if [ -s ./references/*${chrfile_suffix} ]
  then
    echo ":) Genome file exists."
    # read chromsome names
    chrs=`awk '{print $1}' ./references/*${chrfile_suffix}`
    chr_array=($chrs)
  else
    echo ":( Genome file does not exist. Please check directory ${INDIR}/${species}."
    continue;
  fi
  
  
  OUTDIR="${WD}/${species}"
  [ ! -d "$OUTDIR" ] && mkdir -p "$OUTDIR"
  cd "$OUTDIR"
  
  for i in "${!chr_array[@]}"
  do
	echo "Extract the Hi-C counts for ${chr_array[i]}:"
    #s=$( expr $i + 1 )
    
    java -Xmx16g -jar $JUICER_TOOLS dump observed $NR $INDIR/$species/$hic_dir_pattern/$hicfile ${chr_array[i]} ${chr_array[i]} BP $RES ${chr_array[i]}_${chr_array[i]}_hic_counts_${resH}.txt
  done
  
done





