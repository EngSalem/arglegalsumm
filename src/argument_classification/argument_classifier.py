import torch
import torch.nn as nn
import input_formatter as formatter
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
import data_helper as dh
from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    BertModel,
    get_linear_schedule_with_warmup,
    RobertaModel,
    AutoModelForSequenceClassification,
)
from dataset import *
from sklearn.metrics import classification_report

import argparse


def loss_fn(out, target):
    loss = nn.CrossEntropyLoss()(out, target)
    return loss


class LegalIRCDataModule(LightningDataModule):
    def __init__(self, train_df, valid_df, test_df, tokenizer, train_batch_size, eval_batch_size, max_tok_len=128,
                 num_class=2):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.valid_df = valid_df
        self.tokenizer = tokenizer
        self.max_token_len = max_tok_len
        self.num_class = num_class

    def setup(self, stage=None):
        if self.num_class == 2:
            self.train_dataset = SentDataset_binary(
                self.train_df,
                self.max_token_len,
                self.tokenizer,
            )
            self.valid_dataset = SentDataset_binary(
                self.valid_df,
                self.max_token_len,
                self.tokenizer,
            )
            self.test_dataset = SentDataset_binary(
                self.test_df,
                self.max_token_len,
                self.tokenizer,
            )
        else:
            self.train_dataset = SentDataset(
                self.train_df,
                self.max_token_len,
                self.tokenizer,
            )
            self.valid_dataset = SentDataset(
                self.valid_df,
                self.max_token_len,
                self.tokenizer,
            )
            self.test_dataset = SentDataset(
                self.test_df,
                self.max_token_len,
                self.tokenizer,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=2
        )


class BERTmodel(LightningModule):
    def __init__(
            self,
            model_name_or_path: str,
            num_labels: int,
            learning_rate: float = 2e-5,
            adam_epsilon: float = 1e-8,
            weight_decay: float = 0.0,
            train_batch_size: int = 32,
            eval_batch_size: int = 32,
            eval_splits: Optional[list] = None,
            dataset_name: str = "full",
            n_warmup_steps: int = 1000,
            n_training_steps: int = 1000,
            **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.bert = BertModel.from_pretrained(
            model_name_or_path)

        self.classifier = nn.Linear(
            int(self.bert.config.hidden_size), num_labels)
        self.n_warmup_steps = n_warmup_steps
        self.n_training_steps = n_training_steps

        # picking loss.
        if self.hparams.args.loss_fn == "CE":
            if self.hparams.args.class_weight:
                if self.hparams.num_labels == 4:
                    weights = torch.tensor([1., 1., 1., 1.])
                else:
                    weights = torch.tensor([1., 1.])
                self.criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attn_masks, labels=None):
        output = self.bert(input_ids, attention_mask=attn_masks)
        output = output['pooler_output']

        # raise EOFError
        output = self.classifier(output)

        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    # def setup(self, stage=None) -> None:
    #     if stage != "fit":
    #         return
    #     # Get dataloader by calling it - train_dataloader() is called after setup() by default
    #     train_loader = self.train_dataloader()

    #     # Calculate total steps
    #     tb_size = self.hparams.train_batch_size * max(1, 1)
    #     ab_size = self.trainer.accumulate_grad_batches * \
    #         float(self.trainer.max_epochs)
    #     self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

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

        with open("lightning_logs/%s_%s_%s/dev_result.txt" % (
        self.hparams.args.data, self.hparams.args.num_class, self.hparams.args.class_weight), "w") as fin:
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

        with open("lightning_logs/%s_%s_%s/test_result.txt" % (
        self.hparams.args.data, self.hparams.args.num_class, self.hparams.args.class_weight), "w") as fin:
            fin.write(self.metric(labels, preds))

        return loss


if __name__ == "__main__":
    # parser set up.
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument("-model_path", type=str,
                           default='bert-base-uncased')
    my_parser.add_argument("-tokenizer", type=str, default='bert-base-uncased')
    my_parser.add_argument("-data", type=str, default="full")
    my_parser.add_argument("-num_class", type=int, default=4)
    my_parser.add_argument("-class_weight", type=bool, default=False)

    # model hyper-parameters
    my_parser.add_argument("-seed", type=int, default=42)
    my_parser.add_argument("-lr", type=str, default="2e-5")
    my_parser.add_argument("-epochs", type=int, default=3)
    my_parser.add_argument("-loss_fn", type=str, default="CE",
                           help="Choose between normal Cross Entropy Loss (CE)")
    my_parser.add_argument("-train_batch_size", type=int, default=8)
    my_parser.add_argument("-eval_batch_size", type=int, default=16)
    
    # data location
    my_parser.add_argument("-train", type=str)
    my_parser.add_argument("-valid", type=str)
    my_parser.add_argument("-test", type=str)
    my_parser.add_argument("-context", type=bool, default=False) 
    
    # model output
    my_parser.add_argument("-model_out", type=str, default='checkpoints')
    
    args = my_parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    AVAIL_GPUS = min(1, torch.cuda.device_count())
    seed_everything(args.seed)



    # get contextualized rep of data
    data_params = formatter.get_train_params()
    
    train_df = args.train
    valid_df = args.valid
    test_df = args.test

    train_data = pd.read_csv(train_df)
    steps_per_epoch = len(train_data)
    total_training_steps = steps_per_epoch * args.epochs
    warmup_steps = total_training_steps // 5

    data_module = LegalIRCDataModule(
        train_df,
        valid_df,
        test_df,
        tokenizer,
        args.train_batch_size,
        args.eval_batch_size,
        max_tok_len=128,
        num_class=args.num_class)

    model = BERTmodel(
        model_name_or_path=args.model_path,
        num_labels=args.num_class,
        learning_rate=args.lr,
        n_warmup_steps=warmup_steps,
        n_training_steps=total_training_steps,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        dataset_name=args.data,
        args=args,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.model_out,
        filename="%s_class-%s_%s_best-checkpoint" % (
            args.data, args.num_class, args.class_weight),
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )

    logger = TensorBoardLogger(
        "lightning_logs", name="%s_%s_%s" % (args.data, args.num_class, args.class_weight))
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=3)

    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=args.epochs,
        gpus=AVAIL_GPUS,
        progress_bar_refresh_rate=30
    )

    
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
