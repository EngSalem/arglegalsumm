import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report
import torch.multiprocessing
import argparse

my_parser = argparse.ArgumentParser()
my_parser.add_argument("-model_path", type=str,
                       default='bert-base-uncased')
my_parser.add_argument("-tokenizer", type=str, default='bert-base-uncased')
my_parser.add_argument("-test", type=str, default="full")
my_parser.add_argument("-predictions", type=str)

args = my_parser.parse_args()

torch.multiprocessing.set_sharing_strategy('file_system')

from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    BertModel,
    get_linear_schedule_with_warmup,
    RobertaModel,
    AutoModelForSequenceClassification,
)


class LegalBERTModel(LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            num_labels: int,
            learning_rate: float = 2e-5,
            adam_epsilon: float = 1e-8,
            weight_decay: float = 0.0,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            dataset_name: str = "full",
            n_warmup_steps: int = 1000,
            n_training_steps: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained('zlucia/legalbert',
                                                                        return_dict=True)
        self.classifier = nn.Linear(int(self.model.config.hidden_size), num_labels)
        self.n_warmup_steps = n_warmup_steps
        self.n_training_steps = n_training_steps
        weights = torch.tensor([1000., 1.])
        self.criterion = nn.CrossEntropyLoss(weight=weights)

    def forward(self, input_ids, attn_masks, labels=None):
        output = self.model(input_ids, attention_mask=attn_masks)
        #print(output.logits)
        #output = output['hidden_states']

        # raise EOFError
        #output = self.classifier(output.logits)
        output = output.logits
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        print(outputs)
        if self.hparams.num_labels >= 1:
            preds = torch.argmax(outputs, axis=1)
        elif self.hparams.num_labels == 1:
            preds = outputs.squeeze()

        return {"loss": loss, "preds": preds, "labels": labels}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(outputs, axis=1)
        elif self.hparams.num_labels == 1:
            preds = outputs.squeeze()


        return {"loss": loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"]
                            for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        print("validition")
        import pickle
        self.metric = classification_report
        print(self.metric(labels, preds))
        eval_result = self.metric(labels, preds)

        with open("lightning_logs/%s_%s_%s/dev_result.txt" % ('irc', 'binary', '1000.1'), "w") as fin:
            fin.write(self.metric(labels, preds))
        return loss

    def test_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"]
                            for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("test_loss", loss, prog_bar=True)
        print("TEST RESULT")
        import pickle
        self.metric = classification_report
        print(self.metric(labels, preds))

        with open("lightning_logs/%s_%s_%s/test_result.txt" % ('irc', 'binary', '1000.1'), "w") as fin:
            fin.write(self.metric(labels, preds))

        lbl2ix = {'IRC': 0, 'Non_IRC': 1}
        ix2lbl = {ix:lbl for lbl,ix in lbl2ix.items()}

        with open(args.predictions,"w") as fin:
            for pred_lbl in preds:
                fin.write(ix2lbl[pred_lbl]+'\n')

        return loss


class IRCDataset(Dataset):
    def __init__(self, ds_path, max_seq_length, tokenizer):
        self.tokenizer = tokenizer
        self.df = pd.read_csv(ds_path)
        print("Load data with length ", len(self.df))
        self.max_len = max_seq_length
        # self.text = self.df['sentence'].values
        # self.labels = self.df['IRC_type'].values
        self.label2id = {'Issue': 0, 'Reason': 0,
                         'Conclusion': 0, 'Non_IRC': 1}
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        data_row = self.df.iloc[idx]
        text = data_row.sentence
        label = data_row.IRC_type
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,  # Truncate all sentences.
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        labelid = torch.tensor(self.label2id[label])

        return dict(raw_text=text,
                    input_ids=encoding['input_ids'].flatten(),
                    attention_mask=encoding["attention_mask"].flatten(),
                    labels=labelid)



test_df = args.test
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

test_dataset = IRCDataset(
                test_df,
                128,
                tokenizer,
            )
i
test_dataloader = DataLoader(
            test_dataset,
            batch_size=8,
            num_workers=2
        )

print('Load from checkpoint ..')

model = LegalBERTModel.load_from_checkpoint(checkpoint_path=args.model_path)
trainer = Trainer()
trainer.test(model, test_dataloader)
