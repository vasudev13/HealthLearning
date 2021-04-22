import torch
import transformers
import pytorch_lightning as pl

from config import CONFIG
from nli_finetune import NLIFineTuningModel

import pandas as pd


def get_error_samples(trained_model: NLIFineTuningModel, df: pd.DataFrame, tokenizer: transformers.AutoTokenizer, max_length=256):
    """Get samples where model predicts incorrectly

    Args:
        trained_model (NLIFineTuningModel): saved model to make predictions
        df (pd.DataFrame): Dataframe with input text and labels
        tokenizer (transformers.AutoTokenizer): Tokenizer object to encode text input
        max_length (int, optional): Maximum permissible length of text to be considered. Defaults to 256

    Returns:
        [type]: [description]
    """
    error_samples = []
    for _, row in df.iterrows():
        sentence_1 = row[CONFIG['sentence1']]
        sentence_2 = row[CONFIG['sentence2']]
        gold_label = row[CONFIG['labels']]
        encoded_input= tokenizer.encode_plus(
            text=sentence_1,
            text_pair=sentence_2,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        output = trained_model(encoded_input)
        predicted_label = torch.argmax(output.logits)
        if predicted_label.item() != gold_label:
            error_samples.append(
                {
                    'Sentence 1': sentence_1,
                    'Sentence 2': sentence_2,
                    'Ground Label': gold_label,
                    'Predicted Label': predicted_label.item()
                }
            )
    reverse_map = {0: 'entailment', 1: 'contradiction', 2: 'neutral'}
    if error_samples:
        error_df = pd.DataFrame(error_samples)
        error_df['Ground Label Text'] = error_df['Ground Label'].map(reverse_map)
        error_df['Predicted Label Text'] = error_df['Predicted Label'].map(reverse_map)
        return error_df
    else:
        print('LOL, No Errors!')
