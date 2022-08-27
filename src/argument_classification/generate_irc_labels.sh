set -e

argumentClassifier = '' ## path to best checkpoint
test_set = '' ## path to test set
prediction_outs = '' ## select the path
python generate_ircs_predictions.py -model_path $argumentClassifier -tokenizer 'zlucia/legalbert' -test $test_set -predictions $prediction_outs

echo 'Done Generating labels per sentences ...'