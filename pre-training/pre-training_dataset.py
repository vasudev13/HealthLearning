import torch
import pytorch_lightning as pl

import transformers

import pandas as pd
from config import CONFIG
from sklearn.model_selection import train_test_split

from typing import List


class ContrastiveLearningDataset(torch.utils.data.Dataset):
    """Prepares a positive sample for Contrastive Learning.
    """

    def __init__(self, max_len: int, tokenizer: transformers.AutoTokenizer, sentences: List[str], transforms, transforms_prime) -> torch.utils.data.Dataset:
        super().__init__()
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.transforms = transforms
        self.transforms_prime = transforms_prime

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        text = self.sentences[idx]
        z1 = self.transforms(text)
        z2 = self.transforms_prime(text)

        z1 = self.tokenizer.encode_plus(
            text=z1,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        z2 = self.tokenizer.encode_plus(
            text=z2,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'z1': z1,
            'z2': z2
        }


class ContrastiveLearningDataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()

    def prepare_data(self):
        self.dataset = pd.read_csv(CONFIG['MIMIC_PATH'])
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            CONFIG['MODEL_NAME_OR_PATH'])

    def setup(self, stage):
        if stage == 'fit':
            self.train_df, self.val_df = train_test_split(
                self.dataset, seed=CONFIG['SEED'])

    def get_dataset(self, df):
        dataset = ContrastiveLearningDataset(max_len=CONFIG['MAX_LEN'],
                                             tokenizer=self.tokenizer,
                                             sentences=df['TEXT'].values,
                                             transforms=1,
                                             transforms_prime=1
                                            )
        return dataset

    def train_dataloader(self):
        train_dataset = self.get_dataset(self.train_df)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=CONFIG['TRAIN_BS'],
                                                       shuffle=True,
                                                       num_workers=CONFIG['NUM_WORKERS'])

        return train_dataloader

    def val_dataloader(self):
        val_dataset = self.get_dataset(self.val_df)
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=CONFIG['VAL_BS'],
                                                     shuffle=False,
                                                     num_workers=CONFIG['NUM_WORKERS'])

        return val_dataloader
