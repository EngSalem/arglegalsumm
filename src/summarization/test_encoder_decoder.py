import torch

from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import argparse
import yaml
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store', dest='test',
                    help='test csv')
parser.add_argument('--prediction', action='store', dest='predict',
                    help='output prediction')
parser.add_argument('--model', action='store', dest='model',
                    help='model')
parser.add_argument('--config', action='store', dest='config',
                    help='config yaml file')
parser.add_argument('--input_col', action='store', dest='input_col')
args = parser.parse_args()

## Load config file
print('Loading configuration file ...')
def parse_config(cofig_file):
    with open(cofig_file, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

config_dict = parse_config(args.config)

## Load test file
print('Load test file ...')
test_doc = load_dataset(
    "csv", data_files=args.test)['train']

## Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(config_dict['model']['checkpoint'])
## half is to convert the model to fp16
model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to("cuda").half()

def generate_answer(batch):
    inputs_dict = tokenizer(batch[args.input_col], padding="max_length",
                            max_length=config_dict['data']['max_input_length'], return_tensors="pt", truncation=True)
    input_ids = inputs_dict.input_ids.to("cuda")
    attention_mask = inputs_dict.attention_mask.to("cuda")
    global_attention_mask = torch.zeros_like(attention_mask)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1

    predicted_abstract_ids = model.generate(
        input_ids, attention_mask=attention_mask)
    batch["predicted_abstract"] = tokenizer.batch_decode(
        predicted_abstract_ids, skip_special_tokens=True)
    return batch

result = test_doc.map(generate_answer, batched=True, batch_size=4)

## constructing output dataframe
pd.DataFrame.from_dict({'oracle': result['summary'],
                             'generated_summary': result['predicted_abstract']}).to_csv(args.predict, index=False)

print('Done generating summaries and writing to disc')

#rouge = load_metric("rouge")

#all_results = rouge.compute(predictions=result["predicted_abstract"], references=result["summary"], rouge_types=["rouge2", 'rouge1', 'rougeL'])
#print("Results")
#print("ROUGE1: ", all_results['rouge1'].mid)
#print("ROUGE2: ", all_results['rouge2'].mid)
#print("ROGUEL: ", all_results['rougeL'].mid)
