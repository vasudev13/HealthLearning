#!/bin/bash

#SBATCH --job-name=error_samples
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vasudev.w13@gmail.com

#SBATCH --accout=csci_ga_2565_0001
#SBATCH --requeue

bash /scratch/SP21ML/run-pytorch.bash python /scratch/va2134/HealthLearning/nli/RUN_ERROR_ANALYSIS.py --model-checkpoint=/scratch/va2134/HealthLearning/nli/checkpoints/