import os
import pandas as pd
import collections
import math
from nltk.tokenize import sent_tokenize
from enum import Enum

# if the data size of a label is less than NOISE_NUM, then we consider it a Noise
NOISE_THRESHOLD = 10

class Type(Enum):
    STRING = 1
    FLOAT = 2


# read file and set pandas
def read_file(input_filepath, output_filepath):
    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', 10)
    pd.set_option('max_colwidth', 1000)
    data = pd.read_json(input_filepath, lines=True)
    return data


def is_float(val):
    try:
        num = float(val)
    except ValueError:
        return False
    return True


# get the same item counts for each attribute
def attribute_counts(data, attribute, type):
    hashmap = collections.defaultdict(int)
    for line in data[attribute]:
        # if math.isnan(float(line)): continue
        hashmap[line] += 1
    sorted_counts = sorted(hashmap.items(), key=lambda item: item[0])
    purified_counts = []
    if type == Type.FLOAT:
        purified_counts = [count for count in sorted_counts if is_float(str(count[0]).replace("\"", ""))]
    elif type == Type.STRING:
        purified_counts = [count for count in sorted_counts]
    return purified_counts


# remove the label with less than NOISE_NUM data, return the available label and min count
# for the available label
def available_label(df, feature, type):
    counts = attribute_counts(df, feature, type)
    sorted_counts = sorted(counts, key=lambda x: x[1])
    noise_remove_counts = [count for count in sorted_counts if count[1] >= NOISE_THRESHOLD]
    labels = [count[0] for count in noise_remove_counts]
    # print("labels is " + " ".join(labels))
    min_counts = noise_remove_counts[0]
    # print("min_count is {}".format(min_counts[1]))
    return min_counts[1], labels


# feature selection for embedding layer
def feature_selection(data, data_attribute, label_attribute, type):
    output_data = []
    min_count, avail_labels = available_label(data, feature, type)
    for idx, (content, label) in enumerate(zip(data[data_attribute], data[label_attribute])):
        if label not in avail_labels: continue
        lines = sent_tokenize(content)
        lines = [line.replace("\"", "") for line in lines]
        lines = [line.replace("\n", "") for line in lines]
        output_line = "\"" + str(label) + "\",\"" + "\",\"".join(lines) + "\"\n"
        output_data.append(output_line)
        if idx % 100 == 0:
            print("processing: {}%".format(idx/len(data[data_attribute])*100))
    print("processing: {}%".format(100.0))
    return output_data


def output_file(data, output_filepath):
    with open(output_filepath, "w") as file:
        for line in data:
            file.write(line)


if __name__ == "__main__":

    # get the absolute path to the original dataset
    root_filepath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    rel_input_filepath = "../raw/renttherunway_final_data.json"
    rel_output_filepath = "data/processed/processed_data.csv"
    input_filepath = os.path.join(root_filepath, rel_input_filepath)
    output_filepath = os.path.join(root_filepath, rel_output_filepath)

    dataset = read_file(input_filepath, "")
    feature = "rented for"

    # get unique value of a column
    # print(dataset['category'].unique())

    # get rating
    # dataset = dataset[dataset["rating"].notna()]
    # print(attribute_counts(dataset, "rating", Type.FLOAT))
    # output_data = feature_selection(dataset, "review_text", "rating")
    # output_file(output_data, output_filepath)

    # get rented for
    dataset = dataset[dataset[feature].notna()]
    available_label(dataset, feature, Type.STRING)
    print(attribute_counts(dataset, feature, Type.STRING))
    output_data = feature_selection(dataset, "review_text", feature, Type.STRING)
    output_file(output_data, output_filepath)
