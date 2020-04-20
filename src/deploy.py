import torch
from torch import nn
from src.utils import preprocess, rev_label_map
import json
import os
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer

from models.model import HierarchialAttentionNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

asset_folder = './src'
# Load model
checkpoint = os.path.join(asset_folder, 'checkpoint_han.pth.tar')
checkpoint = torch.load(checkpoint, map_location='cpu')
model = HierarchialAttentionNetwork(**checkpoint['opt'])
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()

# Pad limits, can use any high-enough value since our model does not compute over the pads
sentence_limit = 15
word_limit = 20

# Word map to encode with
with open(os.path.join('./src/checkpoints/word_map.json'), 'r') as j:
    word_map = json.load(j)

# Tokenizers
sent_tokenizer = PunktSentenceTokenizer()
word_tokenizer = TreebankWordTokenizer()


def classify(document):
    """
    Classify a document with the Hierarchial Attention Network (HAN).

    :param document: a document in text form
    :return: pre-processed tokenized document, class scores, attention weights for words, attention weights for sentences, sentence lengths
    """
    # A list to store the document tokenized into words
    doc = list()

    # Tokenize document into sentences
    sentences = list()
    for paragraph in preprocess(document).splitlines():
        sentences.extend([s for s in sent_tokenizer.tokenize(paragraph)])

    # Tokenize sentences into words
    for s in sentences[:sentence_limit]:
        w = word_tokenizer.tokenize(s)[:word_limit]
        if len(w) == 0:
            continue
        doc.append(w)

    # Number of sentences in the document
    sentences_in_doc = len(doc)
    sentences_in_doc = torch.LongTensor([sentences_in_doc]).to(device)  # (1)

    # Number of words in each sentence
    words_in_each_sentence = list(map(lambda s: len(s), doc))
    words_in_each_sentence = torch.LongTensor(words_in_each_sentence).unsqueeze(0).to(device)  # (1, n_sentences)

    # Encode document with indices from the word map
    encoded_doc = list(
        map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [0] * (word_limit - len(s)),
            doc)) + [[0] * word_limit] * (sentence_limit - len(doc))
    encoded_doc = torch.LongTensor(encoded_doc).unsqueeze(0).to(device)

    # Apply the HAN model
    scores, word_alphas, sentence_alphas = model(encoded_doc, sentences_in_doc,
                                                 words_in_each_sentence)
    # (1, n_classes), (1, n_sentences, max_sent_len_in_document), (1, n_sentences)
    scores = scores.squeeze(0)  # (n_classes)
    scores = nn.functional.softmax(scores, dim=0)  # (n_classes)
    word_alphas = word_alphas.squeeze(0)  # (n_sentences, max_sent_len_in_document)
    sentence_alphas = sentence_alphas.squeeze(0)  # (n_sentences)
    words_in_each_sentence = words_in_each_sentence.squeeze(0)  # (n_sentences)

    return doc, scores, word_alphas, sentence_alphas, words_in_each_sentence


def get_activation(document):
    """
    Visualize important sentences and words, as seen by the HAN model.

    :param document: a string to be classified
    return: a dictionary with {activation_maps, doc, scores}
    """

    doc, scores, word_alphas, sentence_alphas, words_in_each_sentence = classify(document)
    # :param doc: pre-processed tokenized document (number of sentences, number of words in each sentence)
    # :param scores: class scores, a tensor of size (n_classes)
    # :param word_alphas: attention weights of words, a tensor of size (n_sentences, max_sent_len_in_document)
    # :param sentence_alphas: attention weights of sentences, a tensor of size (n_sentences)
    # :param words_in_each_sentence: sentence lengths, a tensor of size (n_sentences)

    activation_maps = list()
    for i, length in enumerate(words_in_each_sentence):
        wattns = word_alphas[i][:length]
        activation_maps.append([wattns.tolist(), sentence_alphas[i].tolist()])
    _, prediction = scores.max(dim=0)
    return {
        'activations': activation_maps,
        'doc': doc,
        'scores': scores.tolist(),
        'categories': rev_label_map[prediction.item()]
    }
