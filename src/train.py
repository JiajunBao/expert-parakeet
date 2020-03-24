import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from model import HierarchialAttentionNetwork
from utils import *
from datasets import HANDataset
import copy

# Data parameters
data_folder = './checkpoints'
with open(os.path.join(data_folder, 'word_map.json'), 'r') as j:
    word_map = json.load(j)

# Model parameters
n_classes = len(label_map)
word_rnn_size = 50  # word RNN size
sentence_rnn_size = 50  # character RNN size
word_rnn_layers = 1  # number of layers in character RNN
sentence_rnn_layers = 1  # number of layers in word RNN
word_att_size = 100  # size of the word-level attention layer (also the size of the word context vector)
sentence_att_size = 100  # size of the sentence-level attention layer (also the size of the sentence context vector)
dropout = 0.3  # dropout
fine_tune_word_embeddings = True  # fine-tune word embeddings?

# Training parameters
start_epoch = 0  # start at this epoch
batch_size = 64  # batch size
lr = 1e-3  # learning rate
momentum = 0.9  # momentum
workers = 4  # number of workers for loading data in the DataLoader
epochs = 2  # number of epochs to run
grad_clip = None  # clip gradients at this value
print_freq = 2000  # print training or validation status every __ batches
checkpoint = None  # path to model checkpoint, None if none

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


def main():
    """
    Training and validation.
    """
    global checkpoint, start_epoch, word_map

    # Initialize model or load checkpoint
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model = HierarchialAttentionNetwork(**checkpoint['opt'])
        model.load_state_dict(checkpoint['model'])
        optimizer = checkpoint['optimizer']
        word_map = checkpoint['word_map']
        start_epoch = checkpoint['epoch'] + 1
        print(
            '\nLoaded checkpoint from epoch %d.\n' % (start_epoch - 1))
    else:
        embeddings, emb_size = load_embeddings(word_map)  # load pre-trained embeddings

        model = HierarchialAttentionNetwork(n_classes=n_classes,
                                            vocab_size=len(word_map),
                                            emb_size=emb_size,
                                            word_rnn_size=word_rnn_size,
                                            sentence_rnn_size=sentence_rnn_size,
                                            word_rnn_layers=word_rnn_layers,
                                            sentence_rnn_layers=sentence_rnn_layers,
                                            word_att_size=word_att_size,
                                            sentence_att_size=sentence_att_size,
                                            dropout=dropout)
        model.sentence_attention.word_attention.init_embeddings(
            embeddings)  # initialize embedding layer with pre-trained embeddings
        model.sentence_attention.word_attention.fine_tune_embeddings(fine_tune_word_embeddings)  # fine-tune
        optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Loss functions
    criterion = nn.CrossEntropyLoss()

    # Move to device
    model = model.to(device)
    criterion = criterion.to(device)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(HANDataset(data_folder, 'train'), batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True)
    dev_loader = torch.utils.data.DataLoader(HANDataset(data_folder, 'dev'), batch_size=batch_size, shuffle=False,
                                             num_workers=workers, pin_memory=True)
    best_avg_loss = float('inf')
    best_state = {'epoch': -1,
                  'model': None,
                  'optimizer': None,
                  'loss': float('inf'),
                  'acc': -1.}
    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Decay learning rate every epoch
        adjust_learning_rate(optimizer, 0.1)

        avg_acc, avg_loss = evaluate(dev_loader=dev_loader,
                                     model=model,
                                     criterion=criterion,
                                     epoch=epoch)
        if avg_loss < best_state['loss']:
            best_state['loss'] = avg_loss
            best_state['acc'] = avg_acc
            best_state['model'] = model.state_dict()
            best_state['optimizer'] = optimizer.state_dict()
            best_state['epoch'] = epoch

    best_state['word_map'] = word_map
    best_state['opt'] = model.opt
    # Save checkpoint
    save_checkpoint(best_state)


def evaluate(dev_loader, model, criterion, epoch):
    """
    Performs one epoch's training.

    :param dev_loader: DataLoader for evaluating data
    :param model: model
    :param criterion: cross entropy loss layer
    :param epoch: epoch number
    """

    model.eval()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    losses = AverageMeter()  # cross entropy loss
    accs = AverageMeter()  # accuracies

    start = time.time()

    with torch.no_grad():
        # Batches
        for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(dev_loader):

            data_time.update(time.time() - start)

            documents = documents.to(device)  # (batch_size, sentence_limit, word_limit)
            sentences_per_document = sentences_per_document.squeeze(1).to(device)  # (batch_size)
            words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
            labels = labels.squeeze(1).to(device)  # (batch_size)

            # Forward prop.
            scores, word_alphas, sentence_alphas = model(documents, sentences_per_document,
                                                         words_per_sentence)
            # (n_documents, n_classes),
            # (n_documents, max_doc_len_in_batch, max_sent_len_in_batch),
            # (n_documents, max_doc_len_in_batch)

            # Loss
            loss = criterion(scores, labels)  # scalar

            # Find accuracy
            _, predictions = scores.max(dim=1)  # (n_documents)
            correct_predictions = torch.eq(predictions, labels).sum().item()
            accuracy = correct_predictions / labels.size(0)

            # Keep track of metrics
            losses.update(loss.item(), labels.size(0))
            batch_time.update(time.time() - start)
            accs.update(accuracy, labels.size(0))

            start = time.time()

            # Print training status
            if i % print_freq == 0:
                print('Evaluating: '
                      'Epoch: [{0}][{1}/{2}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, i, len(dev_loader),
                                                                      batch_time=batch_time,
                                                                      data_time=data_time, loss=losses,
                                                                      acc=accs))
    return accs.avg, losses.avg


def train(train_loader, model, criterion, optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param dev_loader: DataLoader for evaluating data
    :param model: model
    :param criterion: cross entropy loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    """

    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    losses = AverageMeter()  # cross entropy loss
    accs = AverageMeter()  # accuracies

    start = time.time()

    # Batches
    for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(train_loader):

        data_time.update(time.time() - start)

        documents = documents.to(device)  # (batch_size, sentence_limit, word_limit)
        sentences_per_document = sentences_per_document.squeeze(1).to(device)  # (batch_size)
        words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
        labels = labels.squeeze(1).to(device)  # (batch_size)

        # Forward prop.
        scores, word_alphas, sentence_alphas = model(documents, sentences_per_document,
                                                     words_per_sentence)
        # (n_documents, n_classes),
        # (n_documents, max_doc_len_in_batch, max_sent_len_in_batch),
        # (n_documents, max_doc_len_in_batch)

        # Loss
        loss = criterion(scores, labels)  # scalar

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update
        optimizer.step()

        # Find accuracy
        _, predictions = scores.max(dim=1)  # (n_documents)
        correct_predictions = torch.eq(predictions, labels).sum().item()
        accuracy = correct_predictions / labels.size(0)

        # Keep track of metrics
        losses.update(loss.item(), labels.size(0))
        batch_time.update(time.time() - start)
        accs.update(accuracy, labels.size(0))

        start = time.time()

        # Print training status
        if i % print_freq == 0:
            print('Training: '
                  'Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses,
                                                                  acc=accs))


if __name__ == '__main__':
    main()