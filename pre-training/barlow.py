import torch
import pytorch_lightning as pl

import transformers
import torchmetrics


class BarlowBERT(pl.LightningModule):
    """Contrastive Learning model for training BERT on sentence-level.
    """

    def __init__(self, model_name_or_path: str,
                 learning_rate: float,
                 adam_epsilon: float,
                 weight_decay: float,
                 max_len: int,
                 warmup_steps: int,
                 gpus: int, max_epochs: int, accumulate_grad_batches: int):
        super().__init__()
        self.model_name_or_path = model_name_or_path

        self.save_hyperparameters('learning_rate', 'adam_epsilon', 'weight_decay',
                                  'max_len', 'gpus', 'accumulate_grad_batches', 'max_epochs', 'warmup_steps')

        self.encoder = transformers.AutoModel.from_pretrained(
            model_name_or_path)

    def forward(self, inputs):
        z1, z2 = inputs['z1'], inputs['z2']

    def training_step(self, batch, batch_idx):
        loss, logits = self(batch)[:2]
        return {
            'loss': loss,
        }

    def validation_step(self, batch, batch_idx):
        loss, logits = self(batch)[:2]
        return {
            'loss': loss,
        }

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
        pass

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
        return [optimizer]
