from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np


# read csv from the result of proprecessing.py
def read_csv(input_filepath):
    lines, label = [], []
    with open(input_filepath, 'r') as file:
        for raw_line in file:
            line = raw_line.strip().split(",")
            label.append(line[0])
            lines.append(" ".join(line[1:]))
    return label, lines


if __name__ == "__main__":
    train_y, train_X = read_csv("../data/interim/sampled_train.csv")
    test_y, test_X = read_csv("../data/interim/sampled_test.csv")

    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', SVC())])
    text_clf.fit(train_X, train_y)

    predicted = text_clf.predict(test_X)
    print(np.mean(predicted == test_y))

