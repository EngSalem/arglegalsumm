export CUDA_VISIBLE_DEVICES=0,1,2,3

set -e

trainDIR='../data/irc_train.csv'
validDIR='../data/irc_valid.csv'
testDIR='../data/irc_test.csv'
modelDIR='./models/legalBERTArgumentClassifier'
modelType='zlucia/legalbert'

## train with legal-bert
python argument_classifier.py -model_path  $modelType  -num_class 4 -tokenizer 'zlucia/legalbert'  -epochs 10 -train $trainDIR  -valid $validDIR -test $testDIR -model_out $modelDIR

echo 'Done training and saved best checkpoint to '$modelDIR