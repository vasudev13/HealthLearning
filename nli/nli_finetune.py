import torch
import pytorch_lightning as pl

import transformers
import torchmetrics

from config import CONFIG


class NLIFineTuningModel(pl.LightningModule):
    """Natural Language Inference model to fine-tune BERTology models over given dataset

    """

    def __init__(self, model_name_or_path: str,
                 num_labels: int,
                 learning_rate: float,
                 adam_epsilon: float,
                 weight_decay: float,
                 max_len: int,
                 warmup_steps: int,
                 gpus: int, max_epochs: int, accumulate_grad_batches: int):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels

        self.save_hyperparameters('learning_rate', 'adam_epsilon', 'weight_decay',
                                  'max_len', 'gpus', 'accumulate_grad_batches', 'max_epochs', 'warmup_steps')

        self.config = transformers.AutoConfig.from_pretrained(
            model_name_or_path, num_labels=self.num_labels)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config)
        # self.model = nn.Sequential(
        #     OrderedDict(
        #         [
        #          ('base',transformers.AutoModel.from_pretrained(model_name_or_path)),
        #          ('classifier',nn.Linear(in_features=768,out_features=self.num_labels)),
        #          ('softmax',nn.Softmax())
        #         ]
        #     )
        # )
        metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(),
            torchmetrics.F1(num_classes=3, average='macro')
        ])
        self.train_metrics = metrics.clone()
        self.val_metrics = metrics.clone()

    def forward(self, inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        loss, logits = self(batch)[:2]
        predictions = torch.argmax(logits, dim=1)
        self.train_metrics(predictions, batch['labels'])
        self.log_dict({'train_accuracy': self.train_metrics['Accuracy'],
                       'train_f1': self.train_metrics['F1']}, on_step=False, on_epoch=True)
        return {
            'loss': loss,
            'predictions': predictions,
            'labels': batch['labels']
        }

    def validation_step(self, batch, batch_idx):
        loss, logits = self(batch)[:2]
        predictions = torch.argmax(logits, dim=1)
        self.val_metrics(predictions, batch['labels'])
        self.log_dict({'val_accuracy': self.val_metrics['Accuracy'],
                       'val_f1': self.val_metrics['F1']}, on_step=False, on_epoch=True)
        return {
            'loss': loss,
            'predictions': predictions,
            'labels': batch['labels']
        }

    def test_step(self, batch, batch_idx):
        loss, logits = self(batch)[:2]
        predictions = torch.argmax(logits, dim=1)
        self.val_metrics(predictions, batch['labels'])
        self.log_dict({'test_accuracy': self.val_metrics['Accuracy'],
                       'test_f1': self.val_metrics['F1']}, on_step=False, on_epoch=True)
        return {
            'loss': loss,
            'predictions': predictions,
            'labels': batch['labels']
        }

#     def test_epoch_end(self, outputs):
#         loss = torch.Tensor([x['loss'] for x in outputs])
#         loss = loss.mean()
#         self.log('test_loss', loss, prog_bar=True,
#                  on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs):
        loss = torch.Tensor([x['loss'] for x in outputs])
        loss = loss.mean()
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)

    def training_epoch_end(self, outputs):
        loss = torch.Tensor([x['loss'] for x in outputs])
        loss = loss.mean()
        self.log('train_loss', loss, prog_bar=True,
                 on_step=False, on_epoch=True)

    def setup(self, stage):
        if stage == 'fit':
            train_loader = self.train_dataloader()
            self.total_steps = (
                (len(train_loader.dataset) //
                 (train_loader.batch_size * max(1, self.hparams.gpus)))
                // self.hparams.accumulate_grad_batches * float(self.hparams.max_epochs)
            )

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = transformers.AdamW(
            optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]
