

class NLIDataModel(pl.LightningDataModule):

    def __init__(self,get_split_def):
        super().__init__()
        self.get_split_def=get_split_def

    def prepare_data(self):
        zip = zipfile.ZipFile(CONFIG['ZIP_PATH'])
        zip.extractall()
        self.tokenizer=transformers.AutoTokenizer.from_pretrained(CONFIG['MODEL_NAME_OR_PATH'])

    def setup(self, stage):

      if stage=='fit':
        self.train_df,self.val_df=self.get_split_def('train'),self.get_split_def('dev')

      if stage=='test':
        self.test_df=self.get_split_def('test')

    def get_dataset(self,df):
      dataset = NLIDataset(max_len=CONFIG['MAX_LEN'],
                               tokenizer=self.tokenizer,
                               concept1=df[CONFIG['concept1']].values,
                               concept2=df[CONFIG['concept2']].values,
                               labels=df[CONFIG['labels']].values)
      return dataset

    def train_dataloader(self):
      train_dataset=self.get_dataset(self.train_df)
      train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                     batch_size=CONFIG['TRAIN_BS'], 
                                                     shuffle=True, 
                                                     num_workers=CONFIG['NUM_WORKERS'])
      
      return train_dataloader

    def val_dataloader(self):
      val_dataset=self.get_dataset(self.val_df)
      val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                     batch_size=CONFIG['VAL_BS'], 
                                                     shuffle=False, 
                                                     num_workers=CONFIG['NUM_WORKERS'])
      
      return val_dataloader

    def test_dataloader(self):
      test_dataset=self.get_dataset(self.test_df)
      test_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                     batch_size=CONFIG['VAL_BS'], 
                                                     shuffle=False, 
                                                     num_workers=CONFIG['NUM_WORKERS'])
      
      return test_dataloader

