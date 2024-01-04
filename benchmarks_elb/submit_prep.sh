#!/bin/bash
#SBATCH --output log/%A.o
#SBATCH --job-name fc_meg
#SBATCH --mem-per-cpu 100G -t 24:00:00 -N 1 -n 1
#SBATCH --mail-type ALL
#SBATCH --partition=psych_day
#SBATCH --account=turk-browne

module load miniconda
source activate env_tda
# echo "ISC $1"
#echo "MEG SUBJECT $1"
python -u fc_graph.py $1
#python -u prep_reliability_data.py $1