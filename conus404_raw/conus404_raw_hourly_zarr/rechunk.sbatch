#!/bin/bash
#SBATCH --mail-user=pnorton@usgs.gov
#SBATCH --mail-type=begin,end,fail
#SBATCH -J rechunk
#SBATCH -t 00:20:00
#SBATCH -o %j-rechunk.out
#SBATCH -p workq
#SBATCH -A wbeep
# #SBATCH --hint=nomultithread
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
# #SBATCH --array=0-2495%30

export PATH="$PATH:$HOME/anaconda3/bin"

source activate pangeo

base_dir=<some_path>/wbeep/conus404_work

# Path to the source WRF output files
wrf_dir=<some_path>/wbeep/conus404_work/wrfout_post

# Path to the source WRF constant variables to include
const_file=<some_path>/wbeep/conus404_work/wrf_constants_conus404_final.nc

# Path to the attribute metadata for WRF variables
metadata_file=wrfout_metadata.csv

# File containing list of variables to process
proc_vars_file=2022-09-28_solrad.csv

# Destination path for rechunked WRF output
target_store=<some_path>/scratch/conus404/pnorton_tmp5

python <some_path>/wbeep/conus404_work/conus404_rechunk_ja.py -i $SLURM_ARRAY_TASK_ID -b $base_dir -w $wrf_dir -c $const_file -v $proc_vars_file -m $metadata_file -d $target_store
