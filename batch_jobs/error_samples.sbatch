#!/bin/batch

#SBATCH --job-name=error_samples
#SBATCH --accout=csci_ga_2565_0001
#SBATCH --requeue

bash /scratch/SP21ML/run-pytorch.bash python /scratch/va2134/HealthLearning/nli/RUN_ERROR_ANALYSIS.py