# -*- coding: utf-8 -*-
# author: Yang Zhong
## mail: YAZ118@pitt.edu

from datetime import datetime
from typing import Optional

import torch
import pandas as pd
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.utils.data import Dataset, DataLoader, TensorDataset

from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    BertModel,
    get_linear_schedule_with_warmup,
    RobertaModel
)


class SentDataset(Dataset):
    """Sentence dataset """

    def __init__(self, ds_path, max_seq_length, tokenizer):
        self.tokenizer = tokenizer
        self.df = pd.read_csv(ds_path)
        print("Load data with length ", len(self.df))
        self.max_len = max_seq_length
        # self.text = self.df['sentence'].values
        # self.labels = self.df['IRC_type'].values
        self.label2id = {'Issue': 0, 'Reason': 1,
                         'Conclusion': 2, 'Non_IRC': 3}

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


class SentDataset_binary(Dataset):
    """Sentence dataset """

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

# if __name__ == "__main__":
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#     a = SentDataset("./../data/full_articles_test.csv",
#                     max_seq_length=128, tokenizer=tokenizer)

    # class LegalDataModule(LightningDataModule):

    #     task_text_field_map = {
    #         'sentence': ['name', 'position', 'sentence']
    #     }

    #     task_num_labels = {
    #         'sentence': 4
    #     }

    #     loader_columns = [
    #         "datasets_idx",
    #         "input_ids",
    #         "token_type_ids",
    #         "attention_mask",
    #         "start_positions",
    #         "end_positions",
    #         "labels",
    #     ]

    #     def __init__(
    #         self,
    #         model_name_or_path: str,
    #         task_name: str = 'sentence',
    #         max_seq_length: int = 128,
    #         train_batch_size: int = 32,
    #         eval_batch_size: int = 32,
    #         **kwargs,
    #     ):
    #         '''
    #         :param model_name_or_path: BERT/RoBerta/LegalBERT model name
    #         :param task_name: level of classification
    #         :param max_seq_length: actual class label
    #         '''
    #         super().__init__()
    #         self.model_name_or_path = model_name_or_path
    #         self.task_name = task_name
    #         self.max_seq_length = max_seq_length
    #         self.train_batch_size = train_batch_size
    #         self.eval_batch_size = eval_batch_size

    #         self.text_fields = self.task_text_field_map[task_name]
    #         self.num_labels = self.glue_task_num_labels[task_name]
    #         self.tokenizer = AutoTokenizer.from_pretrained(
    #             self.model_name_or_path)

    #     def setup(self, stage: str):
    #         self.train_df = pd.read_csv("./../data/full_articles_train.csv")
    #         self.valid_df = pd.read_csv("./../data/full_articles_validation.csv")
    #         self.test_df = pd.read_csv("../data/full_articles_test.csv")