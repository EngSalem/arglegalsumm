## author: Mohamed Salem Elaraby
## mail: mse30@pitt.edu
## This script is used to train the summarization model automatically
set -e
export CUDA_VISIBLE_DEVICES=1,2,3,4 ## set number of GPUs to parallelize training

trainDIR='../data/train_summ.csv'
validDIR='../data/valid_summ.csv'
modelDIR='../models/finetune_led'

echo 'training started ...'
python train_encoder_decoder.py --train $trainDIR\
 --valid $validDIR --config ./config/train_config.yml  --outdir $modelDIR --input_col article --summ_col summary

echo 'Done training and saved best check point to '$modelDIR