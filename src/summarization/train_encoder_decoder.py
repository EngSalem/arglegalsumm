## author: Mohamed Salem Elaraby
## email: mse30@pitt.edu
## USAGE: This script is used to train multiple encoder decoder architectures based on the huggingface library models.

import argparse

import yaml
# from IPython.display import display, HTML
from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback


parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store', dest='train',
                    help='train csv')
parser.add_argument('--valid', action='store', dest='valid',
                    help='valid csv')
parser.add_argument('--config', action='store', dest='config',
                    help='yaml configuration file')
parser.add_argument('--outdir', action='store', dest='outdir',
                    help='output directory')
parser.add_argument('--input_col', action='store', dest='input_col',
                    help='input column name for original article')
parser.add_argument('--summ_col', action='store', dest='summ_col',
                    help='Summary column name')
args = parser.parse_args()


def parse_config(cofig_file):
    with open(cofig_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


######### Loading data #############
print('Loading datasets ...')
train_dataset = load_dataset("csv", data_files=args.train)['train']
val_dataset = load_dataset("csv", data_files=args.valid)['train']

######## processing data to models ########


print('Processing data to model ...')
config_dict = parse_config(args.config)
max_input_length = config_dict['data']['max_input_length']
max_output_length = config_dict['data']['max_output_length']
batch_size = config_dict['model']['batchsize']

print('Loading tokenizer ...')
tokenizer = AutoTokenizer.from_pretrained(config_dict['model']['checkpoint'])


def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch[args.input_col],
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
    )
    outputs = tokenizer(
        batch[args.summ_col],
        padding="max_length",
        truncation=True,
        max_length=max_output_length,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch


def get_optimizer_scheduler():
    ##
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "anneal_w": args.recadam_anneal_w,
            "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                not any(nd in p_n for nd in no_decay)]
        },
        {
            "params": [p for n, p in new_model.named_parameters() if
                       not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "anneal_w": 0.0,
            "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                not any(nd in p_n for nd in no_decay)]
        },
        {
            "params": [p for n, p in new_model.named_parameters() if
                       any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "anneal_w": args.recadam_anneal_w,
            "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                any(nd in p_n for nd in no_decay)]
        },
        {
            "params": [p for n, p in new_model.named_parameters() if
                       any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "anneal_w": 0.0,
            "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                any(nd in p_n for nd in no_decay)]
        }
    ]

    ## define optimizer
    optimizer = RecAdam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                        anneal_fun=args.recadam_anneal_fun, anneal_k=args.recadam_anneal_k,
                        anneal_t0=args.recadam_anneal_t0, pretrain_cof=args.recadam_pretrain_cof)

    ## define scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )




train_dataset = train_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=[args.input_col, args.summ_col],
)

val_dataset = val_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=[args.input_col, args.summ_col],
)
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

####### Constucting Model ########

model = AutoModelForSeq2SeqLM.from_pretrained(
    config_dict['model']['checkpoint'], use_cache=False)
# set generate hyperparameters
model.config.num_beams = 4
model.config.max_length = 100
model.config.min_length = 32
model.config.length_penalty = 2.0
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
rouge = load_metric("rouge")


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_outputs = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge1", "rouge2", "rougeL"]
    )
    rouge2_output = rouge_outputs["rouge2"].mid
    rouge1_output = rouge_outputs['rouge1'].mid
    rougeL_output = rouge_outputs['rougeL'].mid

    return {
        "rouge1_precision": round(rouge1_output.precision, 4),
        "rouge2_precision": round(rouge2_output.precision, 4),
        "rougeL_precision": round(rougeL_output.precision, 4),
        "rouge1_recall": round(rouge1_output.recall, 4),
        "rouge1_fmeasure": round(rouge1_output.fmeasure, 4),
        "rouge2_recall": round(rouge2_output.recall, 4),
        "rouge2_fmeasure": round(rouge2_output.fmeasure, 4),
        "rougeL_recall": round(rougeL_output.recall, 4),
        "rougeL_fmeasure": round(rougeL_output.fmeasure, 4),
    }


training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=config_dict['model']['batchsize'],
    per_device_eval_batch_size=config_dict['model']['batchsize'],
    fp16=True,
    output_dir=args.outdir,
    logging_steps=50,
    eval_steps=config_dict['training']['eval_steps'],
    save_steps=config_dict['training']['save_steps'],
    save_total_limit=1,
    gradient_accumulation_steps=4,
    num_train_epochs=config_dict['training']['epochs'],
    metric_for_best_model='rouge1_fmeasure',
    load_best_model_at_end=True
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])