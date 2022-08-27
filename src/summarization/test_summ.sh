## author: Mohamed Salem Elaraby
## mail: mse30@pitt.edu

set -e

testDIR='../data/test_summ.csv'
predictedSummaries = './summ_outputs/finetune_longformer_outputs.csv'
bestModel = '../models/finetune_led'

echo 'testing started ...'
python test_encoder_decoder.py --test $testDIR --prediction $predictedSummaries --model $bestModel'/checkpoint-500'

echo 'running evaluation with pyrouge and summEval'

python score_summaries.py summary_out $predictedSummaries

