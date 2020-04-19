import os
import pandas as pd
import data_processing
import random
import collections
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, SpaceTokenizer
import math
from enum import Enum

# if the data size of a label is less than NOISE_NUM, then we consider it a Noise
NOISE_THRESHOLD = 10

def pandas_setting():
    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', 10)
    pd.set_option('max_colwidth', 100)


# read csv from the result of proprecessing.py
def read_csv(input_filepath):
    lines, label = [], []
    with open(input_filepath, 'r') as file:
        for raw_line in file:
            line = raw_line.strip().split(",")
            label.append(line[0])
            lines.append(line)
    df = pd.DataFrame(data={'label': label, 'data': lines})
    return df

# remove the label with less than NOISE_NUM data
def noise_remove(df):
    # find the minimal count value for undersampling
    min_count, avail_labels = data_processing.available_label(
        df, 'label', data_processing.Type.STRING)
    new_df = df[df['label'].isin(avail_labels)]
    return new_df


# undersampling to produce the label balanced training data
def undersampling(df):

    # get the min_count of labels
    min_counts, _ = data_processing.available_label(df, 'label', data_processing.Type.STRING)
    newdf = df.groupby('label')['data'].apply(list)

    sampled_lines = []
    train_data, dev_data, test_data = [], [], []
    for single_labelled in newdf:
        if len(single_labelled) < min_counts-1: continue
        cur_labelled_lines = random.sample(single_labelled[1:], min_counts-1)
        train_N, dev_N, test_N = int(0.8 * min_counts), int(0.1 * min_counts), int(0.1 * min_counts)
        output_lines = []
        for lines in cur_labelled_lines:
            purified_line = [line.replace("\'", "") for line in lines]
            output_line = ",".join(line for line in purified_line) + "\n"
            output_lines.append(output_line)
        train_data += output_lines[0: train_N]
        dev_data += output_lines[train_N: train_N + dev_N]
        test_data += output_lines[train_N + dev_N:]
    data_processing.output_file(train_data, output_filepath + "sampled_train.csv")
    data_processing.output_file(dev_data, output_filepath + "sampled_dev.csv")
    data_processing.output_file(test_data, output_filepath + "sampled_test.csv")


def random_splitting(df):
    df_num = len(df)
    data = []
    for line in df['data']:
        data.append(",".join(line) + "\n")
    train_N, dev_N, test_N = int(0.8 * df_num), int(0.1 * df_num), int(0.1 * df_num)
    train_data, dev_data, test_data = \
        data[0:train_N], data[train_N: train_N+dev_N], data[train_N+dev_N:]
    data_processing.output_file(train_data, output_filepath + "ranspt_train.csv")
    data_processing.output_file(dev_data, output_filepath + "ranspt_dev.csv")
    data_processing.output_file(test_data, output_filepath + "ranspt_test.csv")


def vocabulary_count(df):

    # df_cate = df.groupby('label')['data'].apply(sum)
    # print("hello world")
    #
    # for index, row in df_cate.iteritems():
    #     text = " ".join([str(sent).replace("\"", "") for sent in row])
    #     counter = collections.Counter(text.split(" "))
    #     distinct_words_N = len(counter)
    #     words_N = sum(counter.values())
    #     print("{} distinct words, {} words in category {}".format(distinct_words_N, words_N, index))

    hashmap_sent = collections.defaultdict(int)
    hashmap = collections.defaultdict(lambda: collections.defaultdict(int))
    for index, row in df.iterrows():
        occasion = row['label']
        text = " ".join([str(sent).replace("\"", "").replace("\'", "") for sent in row['data']])
        text = text.split(" ")
        for word in text:
            hashmap[occasion][word] += 1
        hashmap_sent[occasion] += 1
    print("hahaha")

    for key, submap in hashmap.items():
        distinct_words_N = len(submap)
        words_N = sum(submap.values())
        print("{} distinct words, {} words in category {}".format(distinct_words_N, words_N, key))
    for key, value in hashmap_sent.items():
        print("{} sentences in category {}".format(value, key))


def human_shuffle(input_filepath):
    lines = []
    random.seed(3)
    with open(input_filepath, 'r') as file:
        for line in file:
            lines.append(line)
    random.shuffle(lines)
    data_processing.output_file(lines, output_filepath + "human_test_shuffle.csv")



if __name__ == "__main__":
    # get the absolute path to the original dataset
    root_filepath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    rel_input_filepath = "data/interim/processed_data.csv"
    rel_output_filepath = "data/interim/"
    input_filepath = os.path.join(root_filepath, rel_input_filepath)
    output_filepath = os.path.join(root_filepath, rel_output_filepath)

    pandas_setting()
    dataframe = read_csv(input_filepath)
    dataframe = noise_remove(dataframe)

    # output the random_splitting
    # random_splitting(dataframe)

    # output the random_sampling
    # undersampling(dataframe)

    # print the vocabulary count of each catalog
    # vocabulary_count(dataframe)

    # output human_ramdon_shuffle file
    human_shuffle(os.path.join(root_filepath, "data/interim/human_test.csv"))


