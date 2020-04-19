import time
from utils import *
from datasets import HANDataset

from model import HierarchialAttentionNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data parameters
data_folder = './checkpoints'

# Evaluation parameters
batch_size = 64  # batch size
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 2000  # print training or validation status every __ batches
checkpoint = 'checkpoint_han.pth.tar'

# Load model
checkpoint = torch.load(checkpoint)
model = HierarchialAttentionNetwork(**checkpoint['opt'])
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()

# Load test data
test_loader = torch.utils.data.DataLoader(HANDataset(data_folder, 'test'), batch_size=batch_size, shuffle=False,
                                          num_workers=workers, pin_memory=True)

# Track metrics
accs = AverageMeter()  # accuracies

# Evaluate in batches
preds = list()
golds = list()

for i, (documents, sentences_per_document, words_per_sentence, labels) in enumerate(
        tqdm(test_loader, desc='Evaluating')):

    documents = documents.to(device)  # (batch_size, sentence_limit, word_limit)
    sentences_per_document = sentences_per_document.squeeze(1).to(device)  # (batch_size)
    words_per_sentence = words_per_sentence.to(device)  # (batch_size, sentence_limit)
    labels = labels.squeeze(1).to(device)  # (batch_size)

    # Forward prop.
    scores, word_alphas, sentence_alphas = model(documents, sentences_per_document,
                                                 words_per_sentence)  # (n_documents, n_classes), (n_documents, max_doc_len_in_batch, max_sent_len_in_batch), (n_documents, max_doc_len_in_batch)

    # Find accuracy
    _, predictions = scores.max(dim=1)  # (n_documents)
    correct_predictions = torch.eq(predictions, labels).sum().item()
    accuracy = correct_predictions / labels.size(0)
    preds.append(correct_predictions)
    golds.append(labels)
    # Keep track of metrics
    accs.update(accuracy, labels.size(0))

    start = time.time()

# Print final result
preds = torch.tensor(preds, dtype=torch.long)
golds = torch.tensor(golds, dtype=torch.long)
print('\n * TEST ACCURACY - %.1f per cent\n' % (accs.avg * 100))
lb = ['date', 'everyday', 'formal affair', 'other', 'party', 'vacation', 'wedding', 'work']
print(preds.eq(golds).sum().item() / preds.shape[0])

for i in range(8):
    print(lb[i], (preds[golds == i] == i).sum().item() / (golds == i).sum().item())

