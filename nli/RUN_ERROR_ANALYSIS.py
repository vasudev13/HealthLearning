"""This script is a runner script for executing code during batch jobs."""

import torch
import pytorch_lightning as pl
import transformers
import pandas as pd
import argparse
from path import Path

from config import CONFIG
from nli_finetune import NLIFineTuningModel
from mnli import mnli_df
from error_analysis import get_error_samples

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model-checkpoint', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to model checkpoint')

args = parser.parse_args()
# print(args.model_checkpoint), should be similar to something like Path('/checkpoints/biobert_v1-epoch=00-val_loss=0.55.ckpt')
trained_model = NLIFineTuningModel.load_from_checkpoint(checkpoint_path=args.model_checkpoint,  # model checkpoint path
                                                        num_labels=CONFIG['NUM_CLASSES'],
                                                        model_name_or_path=CONFIG['MODEL_NAME_OR_PATH'])
trained_model.freeze()

# Unfortunately cannot save tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    CONFIG['MODEL_NAME_OR_PATH'])

train_df = mnli_df('train')
error_train = get_error_samples(trained_model,
                                train_df,
                                tokenizer  # reusing tokenizer object
                                )
error_train.to_csv(
    f"CONFIG['MODEL_SAVE_NAME']_error_samples.csv", index=False)
