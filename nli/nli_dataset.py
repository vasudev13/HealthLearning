import torch
import pytorch_lightning as pl


import transformers

import zipfile
from typing import List

from config import CONFIG


class NLIDataset(torch.utils.data.Dataset):
    """Natural Language Inferece: Given a pair of sentence, predict the relation between them among 3 possible outcomes: Entailment, Contradiction, Neutral
    """

    def __init__(self, max_len: int, tokenizer: transformers.AutoTokenizer, sentence1: List[str], sentence2: List[str], labels: List[str]) -> torch.utils.data.Dataset:
        """Dataset class for Natural Language Inference task

        Args:
            max_len (int): Maximum permissible length of text to be considered.
            tokenizer (transformers.AutoTokenizer): Tokenizer object to encode text input
            sentence1 (List[str]): List of `sentence 1`
            sentence2 (List[str]): List of `sentence 1`
            labels (List[str]): List of `labels` specifying relation between the two sentences: 0:'entailment',1:'contradiction',2:'neutral'

        Returns:
            torch.utils.data.Dataset: Instance of NLIDataset
        """
        super().__init__()
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.labels = labels

    def __len__(self):
        return len(self.sentence1)

    def __getitem__(self, idx: int):
        sentence_1 = self.sentence1[idx]
        sentence_2 = self.sentence2[idx]
        encoded_input = self.tokenizer.encode_plus(
            text=sentence_1,
            text_pair=sentence_2,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'labels': torch.Tensor(self.labels[idx]),
            'input_ids': encoded_input['input_ids'].view(-1),
            'attention_mask': encoded_input['attention_mask'].view(-1),
            'token_type_ids': encoded_input['token_type_ids'].view(-1),
        }


class NLIDataModule(pl.LightningDataModule):
    """Lightning Data Module for Natural Language Inference task
    """

    def __init__(self, get_split_def):
        super().__init__()
        self.get_split_def = get_split_def

    def prepare_data(self):
        if CONFIG['UNZIP']:
            zip = zipfile.ZipFile(CONFIG['ZIP_PATH'])
            zip.extractall()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            CONFIG['MODEL_NAME_OR_PATH'])

    def setup(self, stage):

        if stage == 'fit':
            self.train_df, self.val_df = self.get_split_def(
                'train'), self.get_split_def('dev')

        if stage == 'test':
            self.test_df = self.get_split_def('test')

    def get_dataset(self, df):
        dataset = NLIDataset(max_len=CONFIG['MAX_LEN'],
                             tokenizer=self.tokenizer,
                             sentence1=df[CONFIG['sentence1']].values,
                             sentence2=df[CONFIG['sentence2']].values,
                             labels=df[CONFIG['labels']].values)
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

    def test_dataloader(self):
        test_dataset = self.get_dataset(self.test_df)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                      batch_size=CONFIG['VAL_BS'],
                                                      shuffle=False,
                                                      num_workers=CONFIG['NUM_WORKERS'])

        return test_dataloader
