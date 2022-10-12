# ArgLegalSumm

* This repository contains the source code for the paper "ArgLegalSumm: Improving Abstractive Summarization of Legal Documents with Argument Mining" to appeat at COLING 2022

## Data

*  To request the annotations of both summaries and articles with argument roles , please contact Dr. Kevin D. Ashley (ashley@pitt.edu).  However, you must first obtain the unannotated data through an agreement with the Canadian Legal Information Institute (CanLII) (https://www.canlii.org/en/)

## The code is split into two parts 
- Argument Classification [[link](https://github.com/EngSalem/arglegalsumm/tree/master/src/argument_classification)]
- Document Summaization [[link](https://github.com/EngSalem/arglegalsumm/tree/master/src/summarization)]

* The argument classification uses by default *Legalbert* while the Document summarization uses by default the *Logformer Encoder-Decoder*.



## To run the code

### Requirements 
- transformers
- pytorch
- pylightining for training argument classifier.
- SummEval [[link](https://github.com/Yale-LILY/SummEval)]

### training the summaries.
- training script [[link](https://github.com/EngSalem/arglegalsumm/blob/master/src/summarization/train_summ.sh)]

### testing the summaries.

- generation script [[link](https://github.com/EngSalem/arglegalsumm/blob/master/src/summarization/test_summ.sh)]

* notice that you can easily choose the model and modify input and summary length through the config file without the need to modify much in the training scripts. 

### The special tokens used to highlight the argument roles in our data , they are split into two groups
- Binary special tokens [[link](https://github.com/EngSalem/arglegalsumm/blob/master/src/summarization/binary_tokens.txt)]
- Finegrained special tokens [[link](https://github.com/EngSalem/arglegalsumm/blob/master/src/summarization/fine_grained_tokens.txt)]

### training and testing argument classifier

- training script [[link](https://github.com/EngSalem/arglegalsumm/blob/master/src/argument_classification/label_sentences.sh)]
- testing script [[link](https://github.com/EngSalem/arglegalsumm/blob/master/src/argument_classification/generate_irc_labels.sh)]

* Note that we made our best predictions on the test set obtained by the model available to use.
- predictions [[link](https://github.com/EngSalem/arglegalsumm/blob/master/src/argument_classification/artifacts/legal_bert_predicts.txt)]


If you are going to follow up on this project please cite this work using the following bibtext:*


```
@inproceedings{elaraby-litman-2022-arglegalsumm,
    title = "{A}rg{L}egal{S}umm: Improving Abstractive Summarization of Legal Documents with Argument Mining",
    author = "Elaraby, Mohamed  and
      Litman, Diane",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.540",
    pages = "6187--6194",
    abstract = "A challenging task when generating summaries of legal documents is the ability to address their argumentative nature. We introduce a simple technique to capture the argumentative structure of legal documents by integrating argument role labeling into the summarization process. Experiments with pretrained language models show that our proposed approach improves performance over strong baselines.",
}
```

