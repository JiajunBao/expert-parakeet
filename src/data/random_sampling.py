import os
import pandas as pd
import data_processing
import random
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


if __name__ == "__main__":
    # get the absolute path to the original dataset
    root_filepath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    rel_input_filepath = "data/processed/processed_data.csv"
    rel_output_filepath = "data/processed/"
    input_filepath = os.path.join(root_filepath, rel_input_filepath)
    output_filepath = os.path.join(root_filepath, rel_output_filepath)

    pandas_setting()
    dataframe = read_csv(input_filepath)
    dataframe = noise_remove(dataframe)


    # output the random_splitting
    random_splitting(dataframe)

    # output the random_sampling
    undersampling(dataframe)
