#!/bin/bash
#SBATCH --job-name=albert_training
#SBATCH --account=csci_ga_2565_0001
#SBATCH --partition=n1s8-t4-1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --requeue

singularity exec --nv --bind /scratch --bind /share/apps --overlay /share/apps/pytorch/1.8.1/pytorch-1.8.1.sqf:ro /share/apps/images/cuda11.1.1-cudnn8-devel-ubuntu20.04.sif /bin/bash -c "

source /ext3/env.sh

cd HealthLearning/pre-training/

mkdir clinic_albert_v1

python transformers/examples/pytorch/language-modeling/run_mlm.py \
    --model_name_or_path albert-base-v2 \
    --train_file discharge_summaries/ds_sentences_train_v1.txt \
    --validation_file discharge_summaries/ds_sentences_val_v1.txt \
    --do_train \
    --do_eval \
    --output_dir clinic_albert_v1 \
    --fp16 \
    --dataloader_num_workers 8\
    --line_by_line 
"