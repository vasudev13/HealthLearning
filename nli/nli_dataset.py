import torch
import pytorch_lightning as pl


import transformers

import zipfile

from config import CONFIG


class NLIDataset(torch.utils.data.Dataset):

  def __init__(self,max_len:int,tokenizer,sentence1,sentence2,labels,transforms=None):
    super().__init__()
    self.max_len=max_len
    self.tokenizer=tokenizer
    self.sentence1=sentence1
    self.sentence2=sentence2
    self.labels=labels
    self.transforms=transforms
  
  def __len__(self):
    return len(self.sentence1)

  def __getitem__(self,idx):
    if self.transforms:
        sentence_1=self.transforms(self.sentence1[idx])
        sentence_2=self.transforms(self.sentence2[idx])
    else:
        sentence_1=self.sentence1[idx]
        sentence_2=self.sentence2[idx]
    encoded_input=self.tokenizer.encode_plus(
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
        'labels':torch.tensor(self.labels[idx]),
        'input_ids':encoded_input['input_ids'].view(-1),
        'attention_mask':encoded_input['attention_mask'].view(-1),
        'token_type_ids':encoded_input['token_type_ids'].view(-1),
    }

class NLIDataModule(pl.LightningDataModule):
    """Lightning Data Module for Natural Language Inference task
    """

    def __init__(self, get_split_def,train_transforms=None,val_transforms=None):
        super().__init__()
        self.get_split_def = get_split_def
        self.train_transforms=train_transforms
        self.val_transforms=val_transforms

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

    def get_dataset(self, df,transforms):
        dataset = NLIDataset(max_len=CONFIG['MAX_LEN'],
                             tokenizer=self.tokenizer,
                             sentence1=df[CONFIG['sentence1']].values,
                             sentence2=df[CONFIG['sentence2']].values,
                             labels=df[CONFIG['labels']].values,
                             transforms=transforms
                            )
        return dataset

    def train_dataloader(self):
        train_dataset = self.get_dataset(self.train_df,self.train_transforms)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=CONFIG['TRAIN_BS'],
                                                       shuffle=True,
                                                       num_workers=CONFIG['NUM_WORKERS'])

        return train_dataloader

    def val_dataloader(self):
        val_dataset = self.get_dataset(self.val_df,self.val_transforms)
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
