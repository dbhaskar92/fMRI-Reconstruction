#!/bin/bash
#SBATCH --output log/%A.o
#SBATCH --job-name prep_meg
#SBATCH --mem-per-cpu 5G -t 24:00:00 -N 1 -n 16
#SBATCH --mail-type ALL
#SBATCH --partition=psych_day
#SBATCH --account=turk-browne

module load miniconda
source activate tphate_env
echo "ISC $1"
#echo "MEG SUBJECT $1"
python -u run_iscs_meg.py $1
#python -u prep_reliability_data.py $1