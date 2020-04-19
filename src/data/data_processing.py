import os
import pandas as pd
import collections
import csv
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
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.max_colwidth', 1000)
    pd.set_option('display.max_rows', None)
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


def groupby_attribute(df, key, value):
    itemCounts = df.user_id.value_counts()
    reduced_df = df[df.user_id.isin(itemCounts.index[itemCounts.gt(1)])]
    item_user_map = reduced_df.groupby(key)[value].apply(set)
    for key, values in item_user_map.items():
        if len(values) < 2: del item_user_map[key]
    return item_user_map


def output_groupby(map, outputpath):
    lines = [",".join(list(value)) + "\n" for key, value in map.items()]
    output_file(lines, outputpath)


def output_file(data, output_filepath):
    with open(output_filepath, "w") as file:
        for line in data:
            file.write(line)


if __name__ == "__main__":

    # get the absolute path to the original dataset
    root_filepath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    rel_input_filepath = "../raw/renttherunway_final_data.json"
    rel_output_filepath = "data/interim/processed_data.csv"
    input_filepath = os.path.join(root_filepath, rel_input_filepath)
    output_filepath = os.path.join(root_filepath, rel_output_filepath)

    dataset = read_file(input_filepath, "")
    feature = ["rented for", "rating", "category"]

    # get rating
    # dataset = dataset[dataset["rating"].notna()]
    # print(attribute_counts(dataset, "rating", Type.FLOAT))
    # output_data = feature_selection(dataset, "review_text", "rating")
    # output_file(output_data, output_filepath)

    # get rented for
    # dataset = dataset[dataset["rented for"].notna()]
    # available_label(dataset, "rented for", Type.STRING)
    # print(attribute_counts(dataset, "rented for", Type.STRING))
    # output_data = feature_selection(dataset, "review_text", feature, Type.STRING)
    # output_file(output_data, output_filepath)

    # get category
    # dataset = dataset[dataset["category"].notna()]
    # available_label(dataset, "category", Type.STRING)
    # counts = attribute_counts(dataset, "category", Type.STRING)
    # print(sorted(counts, key=lambda x: x[1]))

    # get hashmap key = user_id, value = count of items user bought
    # item_user_map = dataset.groupby('user_id')['item_id'].count().sort_values(ascending=False)
    # item_user_map.to_csv(os.path.join(root_filepath, "data/interim/user_by_item.csv"))

    # get hashmap key = user_id, value = list of item_id
    item_user_map = groupby_attribute(dataset, 'user_id', 'category')
    output_groupby(item_user_map, os.path.join(root_filepath, "data/interim/user_by_item.csv"))
    # item_user_map.to_csv(os.path.join(root_filepath, "data/interim/user_by_item.csv"), quoting=csv.QUOTE_NONNUMERIC)

