import pandas as pd
import argparse
from summ_eval.rouge_metric import RougeMetric
import nltk

my_parser = argparse.ArgumentParser()
my_parser.add_argument("-summary_out", type=str)
args = my_parser.parse_args()


## get rouge

rouge = RougeMetric()

## read summaries

df_summaries = pd.read_csv(args.summary_out)
df_summaries['tokenized_summary'] = df_summaries.apply(lambda row: '.\n'.join(nltk.sent_tokenize(row['generated_summary'])), axis=1)
df_summaries['tokenized_oracle'] = df_summaries.apply(lambda  row: '.\n'.join(nltk.sent_tokenize(row['oracle'])), axis=1)
rouge_dict = rouge.evaluate_batch(df_summaries['tokenized_summary'].tolist(), df_summaries['oracle'].tolist())

rouge1 = rouge_dict['rouge']['rouge_1_f_score']
rouge2 = rouge_dict['rouge']['rouge_2_f_score']
rougeL = rouge_dict['rouge']['rouge_l_f_score']

print({'rouge 1': rouge1,
        'rouge 2': rouge2,
        'rouge L': rougeL})


