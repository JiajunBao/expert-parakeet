import os
import pandas as pd
import collections
import math
from nltk.tokenize import sent_tokenize

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
def attribute_counts(data, attribute):
    hashmap = collections.defaultdict(int)
    for line in data[attribute]:
        # if math.isnan(float(line)): continue
        hashmap[line] += 1
    sorted_counts = sorted(hashmap.items(), key=lambda item: item[0])
    purified_counts = [count for count in sorted_counts if is_float(str(count[0]).replace("\"", ""))]
    return purified_counts


# feature selection for embedding layer
def feature_selection(data, data_attribute, label_attribute):
    output_data = []
    for idx, (content, label) in enumerate(zip(data[data_attribute], data[label_attribute])):
        lines = sent_tokenize(content)
        lines = [line.replace("\"", "") for line in lines]
        lines = [line.replace("\n", "") for line in lines]
        output_line = "\"" + str(label) + "\",\"" + "\",\"".join(lines) + "\"\n"
        output_data.append(output_line)
        # print(idx+1, output_line)
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
    rel_input_filepath = "dataset/raw/renttherunway_final_data.json"
    rel_output_filepath = "dataset/processed/processed_data.csv"
    input_filepath = os.path.join(root_filepath, rel_input_filepath)
    output_filepath = os.path.join(root_filepath, rel_output_filepath)

    dataset = read_file(input_filepath, "")
    dataset = dataset[dataset["rating"].notna()]
    # print(dataset)

    print(attribute_counts(dataset, "rating"))
    # output_data = feature_selection(dataset, "review_text", "rating")
    # output_file(output_data, output_filepath)
