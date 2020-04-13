import torch
from torch import nn
import numpy as np
from collections import Counter
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from tqdm import tqdm
from torchtext import vocab
import os
import json
import csv
import logging

classes = ['date', 'everyday', 'formal affair', 'other', 'party', 'vacation', 'wedding', 'work']
label_map = {k: v for v, k in enumerate(classes)}
rev_label_map = {v: k for k, v in label_map.items()}

# Tokenizers
sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()


def preprocess(text):
    """
    Pre-process text for use in the model. This includes lower-casing, standardizing newlines, removing junk.

    :param text: a string
    :return: cleaner string
    """
    if isinstance(text, float):
        return ''

    return text.lower().replace('<br />', '\n').replace('<br>', '\n').replace('\\n', '\n').replace('&#xd;', '\n')


def read_csv(csv_folder, split, sentence_limit, word_limit):
    """
    Read CSVs containing raw training data, clean documents and labels, and do a word-count.

    :param csv_folder: folder containing the CSV
    :param split: train or test CSV?
    :param sentence_limit: truncate long documents to these many sentences
    :param word_limit: truncate long sentences to these many words
    :return: documents, labels, a word-count
    """
    assert split in {'train', 'dev', 'test'}

    docs = []
    labels = []
    word_counter = Counter()
    # no header
    with open(os.path.join(csv_folder, split + '.csv'), 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in tqdm(reader, desc='Read the (%s) data' % split):

            sentences = list()

            for text in row[1:]:
                for paragraph in preprocess(text).splitlines():
                    sentences.extend([s for s in sent_tokenizer.tokenize(paragraph)])

            words = list()
            for s in sentences[:sentence_limit]:
                w = word_tokenizer.tokenize(s)[:word_limit]
                # If sentence is empty (due to removing punctuation, digits, etc.)
                if len(w) == 0:
                    continue
                words.append(w)
                word_counter.update(w)

            # If all sentences were empty
            if len(words) == 0:
                continue

            labels.append(label_map[row[0]])  # since labels are detailed category labels
            docs.append(words)
    return docs, labels, word_counter


def create_input_files(csv_folder, output_folder, sentence_limit, word_limit, min_word_count=5,
                       cache_raw_data=True):
    """
    Create data files to be used for training the model.
    :param csv_folder: folder where the CSVs with the raw data are located
    :param output_folder: folder where files must be created
    :param sentence_limit: truncate long documents to these many sentences
    :param word_limit: truncate long sentences to these many words
    :param min_word_count: discard rare words which occur fewer times than this number
    :param cache_raw_data: whether to save the data required for training word2vec embeddings
    """
    # Read training data
    print('\nReading and preprocessing training data...\n')
    train_docs, train_labels, word_counter = read_csv(csv_folder, 'train', sentence_limit, word_limit)

    # create the output folder
    os.makedirs(output_folder, exist_ok=True)

    # Save raw text data
    if cache_raw_data:
        torch.save(train_docs, os.path.join(output_folder, 'raw_data.pth.tar'))
        print('\nRaw text data saved to %s.\n' % os.path.abspath(output_folder))

    # Create word map
    word_map = dict()
    word_map['<pad>'] = 0
    for word, count in word_counter.items():
        if count >= min_word_count:
            word_map[word] = len(word_map)
    word_map['<unk>'] = len(word_map)
    print('\nDiscarding words with counts less than %d, the size of the vocabulary is %d.\n' % (
        min_word_count, len(word_map)))

    with open(os.path.join(output_folder, 'word_map.json'), 'w') as j:
        json.dump(word_map, j)
    print('Word map saved to %s.\n' % os.path.abspath(output_folder))

    # Encode and pad
    print('Encoding and padding training data...\n')
    encoded_train_docs = list(map(lambda doc: list(
        map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [0] * (word_limit - len(s)),
            doc)) + [[0] * word_limit] * (sentence_limit - len(doc)), train_docs))
    sentences_per_train_document = list(map(lambda doc: len(doc), train_docs))
    words_per_train_sentence = list(
        map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (sentence_limit - len(doc)), train_docs))

    # Save
    print('Saving...\n')
    assert len(encoded_train_docs) == len(train_labels) == len(sentences_per_train_document) == len(
        words_per_train_sentence)
    # Because of the large data, saving as a JSON can be very slow
    torch.save({'docs': encoded_train_docs,
                'labels': train_labels,
                'sentences_per_document': sentences_per_train_document,
                'words_per_sentence': words_per_train_sentence},
               os.path.join(output_folder, 'TRAIN_data.pth.tar'))
    print('Encoded, padded training data saved to (%s).\n' % os.path.abspath(output_folder))

    # Free some memory
    del train_docs, encoded_train_docs, train_labels, sentences_per_train_document, words_per_train_sentence

    # Read dev data
    print('Reading and preprocessing dev data...\n')
    dev_docs, dev_labels, _ = read_csv(csv_folder, 'dev', sentence_limit, word_limit)

    # Encode and pad
    print('\nEncoding and padding dev data...\n')
    encoded_dev_docs = list(map(lambda doc: list(
        map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [0] * (word_limit - len(s)),
            doc)) + [[0] * word_limit] * (sentence_limit - len(doc)), dev_docs))
    sentences_per_dev_document = list(map(lambda doc: len(doc), dev_docs))
    words_per_dev_sentence = list(
        map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (sentence_limit - len(doc)), dev_docs))

    # Save
    print('Saving...\n')
    assert len(encoded_dev_docs) == len(dev_labels) == len(sentences_per_dev_document) == len(
        words_per_dev_sentence)
    torch.save({'docs': encoded_dev_docs,
                'labels': dev_labels,
                'sentences_per_document': sentences_per_dev_document,
                'words_per_sentence': words_per_dev_sentence},
               os.path.join(output_folder, 'DEV_data.pth.tar'))
    print('Encoded, padded dev data saved to (%s).\n' % os.path.abspath(output_folder))

    # Read test data
    print('Reading and preprocessing test data...\n')
    test_docs, test_labels, _ = read_csv(csv_folder, 'test', sentence_limit, word_limit)

    # Encode and pad
    print('\nEncoding and padding test data...\n')
    encoded_test_docs = list(map(lambda doc: list(
        map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [0] * (word_limit - len(s)),
            doc)) + [[0] * word_limit] * (sentence_limit - len(doc)), test_docs))
    sentences_per_test_document = list(map(lambda doc: len(doc), test_docs))
    words_per_test_sentence = list(
        map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (sentence_limit - len(doc)), test_docs))

    # Save
    print('Saving...\n')
    assert len(encoded_test_docs) == len(test_labels) == len(sentences_per_test_document) == len(
        words_per_test_sentence)
    torch.save({'docs': encoded_test_docs,
                'labels': test_labels,
                'sentences_per_document': sentences_per_test_document,
                'words_per_sentence': words_per_test_sentence},
               os.path.join(output_folder, 'TEST_data.pth.tar'))
    print('Encoded, padded test data saved to (%s).\n' % os.path.abspath(output_folder))

    print('All done!\n')


def init_embedding(input_embedding):
    """
    Initialize embedding tensor with values from the uniform distribution.

    :param input_embedding: embedding tensor
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def load_embeddings(word_map):
    """
    Load pre-trained embeddings for words in the word map.

    :param word_map: word map
    :return: embeddings for words in the word map, embedding size
    """
    # Load glove model into memory
    glove = vocab.GloVe(name='6B', dim=50)
    print("\nEmbedding length is %d.\n" % glove.dim)

    # Create tensor to hold embeddings for words that are in-corpus
    embeddings = torch.FloatTensor(len(word_map), glove.dim)
    init_embedding(embeddings)

    # Read embedding file
    print("Loading embeddings...")
    for word in word_map:
        vec = glove[word]
        if vec.sum() != 0.:  # this condition may vary
            embeddings[word_map[word]] = torch.FloatTensor(vec)

    print("Done.\n Embedding vocabulary: %d.\n" % len(word_map))

    return embeddings, glove.dim


def clip_gradient(optimizer, grad_clip):
    """
    Clip gradients computed during backpropagation to prevent gradient explosion.

    :param optimizer: optimized with the gradients to be clipped
    :param grad_clip: gradient clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(state):
    """
    Save model checkpoint.

    :param state: the state dictionary to be saved
    """
    filename = 'checkpoint_han.pth.tar'
    torch.save(state, filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, scale_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rates must be decayed
    :param scale_factor: factor to scale by
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
