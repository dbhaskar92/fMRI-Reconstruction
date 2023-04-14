#!/bin/bash
#SBATCH --partition psych_gpu
#SBATCH --gpus 1
#SBATCH --mem=64G
#SBATCH --cpus-per-gpu 8
#SBATCH --time 0-12:00:00
#SBATCH --job-name fmri-recon
#SBATCH --output fmri-recon-%J.log

# load modules
module load miniconda CUDA/CUDA-10.1.105 cuDNN/7.6.2.24-CUDA-10.1.105

# activate env
conda activate pytorch_env
python train_lstm_ae.py
