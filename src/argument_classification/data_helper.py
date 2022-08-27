## author: Mohamed Salem Elaraby
## mail: mse30@pitt.edu

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np


def read_data():
    try:
        df_train = pd.read_csv('./data/full_articles_train.csv')
        df_validation = pd.read_csv('./data/full_articles_validation.csv')
        df_test = pd.read_csv('./data/full_articles_test.csv')
    except:
        raise Exception('Error in loading training validation and test files from data directory')
    return df_train, df_test, df_validation


def random_down_sample(df_train):
    len_ircs = df_train[df_train['IRC_type'].isin(['Conclusion', 'Issue', 'Reason'])].shape[0]
    df_nonircs = df_train[df_train['IRC_type'] == 'Non_IRC'].sample(n=len_ircs)
    return pd.concat([df_train[df_train['IRC_type'].isin(['Conclusion', 'Issue', 'Reason'])], df_nonircs]).sample(
        frac=1)


def quantize_down_sample(df_train):
    ## encoder
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df_train[df_train['IRC_type'] == 'Non_IRC']['sentence'].tolist())
    len_ircs = df_train[df_train['IRC_type'].isin(['Conclusion', 'Issue', 'Reason'])].shape[0]
    clustering = AgglomerativeClustering()
    clustering.n_clusters_ = len_ircs
    clustering.fit(embeddings)
    df_train['cluster_label'] = clustering.labels_

    ##
    non_ircs = []
    for lbl, df_ in df_train.groupby(by=['cluster_label']):
        mean_vec = np.mean(model.encode(df_['sentence'].tolist()), axis=0)
        mean_vec_norm = mean_vec / np.linalg.norm(mean_vec)
        ##
        bank_norm = model.encode(df_['sentence'].tolist()) / np.linalg.norm(model.encode(df_['sentence'].tolist()),
                                                                            axis=0)
        ## dot product
        similarities = np.dot(mean_vec_norm, bank_norm)
        centroid_ix = np.argsort(similarities)[0]
        non_ircs.append(df_.iloc[centroid_ix]['sentence'].tolist()[0])

    return pd.concat([pd.DataFrame.from_dict({'sentence': non_ircs, 'IRC_type': ['Non_IRC'] * len(non_ircs)}),
                      df_train[df_train['IRC_type'].isin(['Conclusion', 'Issue', 'Reason'])]]).sample(frac=1.)


def get_contextualized_sentence_rep(full_article, position, label, context_size=5):
    '''
    :param full_article: article data frame for either train validation or split
    :param position: sentence position
    :param label: actual class label
    :param context_size: window size (fixed first at 5)
    :return: sentence surrounded by <left> left context of size context_size </left> sentence <right> right context</right>
    '''
    max_position = full_article.position.max()
    left_positions = [i for i in range(max(0, position - context_size), position) if i != position]
    right_positions = [i for i in range(position, min(position + context_size, max_position)) if i != position]

    df_left = full_article[full_article['position'].isin(left_positions)]
    df_right = full_article[full_article['position'].isin(right_positions)]

    left_context = '<left> ' + '\n'.join(df_left.sentence.tolist()) + ' </left>'
    right_context = '<right> ' + '\n'.join(df_right.sentence.tolist()) + ' </right>'
    candidate_sentence = full_article[full_article['position'] == position]['sentence'].tolist()[0]

    return ' '.join([left_context, candidate_sentence, right_context]), label


def get_contextualized_article_inputs(df_articles, window_size=5):
    '''
    :param df_articles:
    :return: a data frame of df['input','label'] where input is the contextualized input
    '''
    dfs = []
    for article in set(df_articles['name'].tolist()):
        input_text, labels = [], []
        df_article = df_articles[df_articles['name'] == article]
        for position, label in zip(df_article.position.tolist(), df_article.IRC_type.tolist()):
            input_rep, lbl = get_contextualized_sentence_rep(df_article, position, label, window_size)
            input_text.append(input_rep)
            labels.append(lbl)
        dfs.append(pd.DataFrame.from_dict({'input': input_text, 'label': labels}))
    return pd.concat(dfs)


def create_label_dict(label_list):
    '''
    :param label_list: training list
    :return: lbl2id dictionary
    '''
    return {lbl: id for id, lbl in enumerate(list(set(label_list)))}
