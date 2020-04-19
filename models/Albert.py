import logging
from torch import nn
import torch

from tqdm.auto import tqdm

from transformers.optimization import AdamW
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AlbertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)


class AlbertForReviewClassification(AlbertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.label_names = ['date', 'everyday', 'formal affair', 'other', 'party', 'vacation', 'wedding', 'work']
        # self.num_labels = len(self.label_names)

    @staticmethod
    def training_step(self, batch):
        """
        a single forwarding step for training
        used by solver
        e.g.
        meta_features, input_ids, input_mask, segment_ids, labels = batch
        batch_input = {'meta_features': meta_features.to(self.device),
                       'input_ids': input_ids.to(self.device),
                       'attention_mask': input_mask.to(self.device),
                       'token_type_ids': segment_ids.to(self.device),
                       'labels': labels}
        logits = self.model(**batch_input)
        loss = self.criterion(logits.view(-1), labels.to(self.device).view(-1))
        return loss
        :param self: a solver
        :param batch: a batch of input for model
        :return: training loss for this batch
        """
        input_ids, attention_mask, token_type_ids, labels = batch
        batch_input = {
            'input_ids': input_ids.to(self.device),
            'attention_mask': attention_mask.to(self.device),
            'token_type_ids': token_type_ids.to(self.device),
            'labels': labels.to(self.device)
        }
        outputs = self.model(**batch_input)
        loss = outputs[0]
        return loss

    @staticmethod
    def validating_step(self, batch):
        """
        a single forwarding pass for the validation
        e.g.
        meta_features, input_ids, input_mask, segment_ids, labels = batch
        batch_input = {'meta_features': meta_features.to(self.device),
                       'input_ids': input_ids.to(self.device),
                       'attention_mask': input_mask.to(self.device),
                       'token_type_ids': segment_ids.to(self.device),
                       'labels': labels}
        logits = self.model(**batch_input)
        return logits, labels
        :param self: a solver
        :param batch: a batch of input for model
        :return: logits and ground true label for this batch
        """
        input_ids, attention_mask, token_type_ids, labels = batch
        batch_input = {
            'input_ids': input_ids.to(self.device),
            'attention_mask': attention_mask.to(self.device),
            'token_type_ids': token_type_ids.to(self.device),
            'labels': labels.to(self.device)
        }
        outputs = self.model(**batch_input)
        logits = outputs[1].view(-1, self.model.num_labels)
        return logits

    @staticmethod
    def get_dataset(data_path):
        """
        read dataset
        :param data_path: root path to the data
        :param label_header: the label name for the dataset
        :return: dataset (TensorDataset)
        e.g.
        logger.info(f'Loading data from {data_path}')
        iids = torch.load(data_path / 'iid.pt')
        imask = torch.load(data_path / 'imask.pt')
        sids = torch.load(data_path / 'tid.pt')
        label_file = f'{label}.pt'
        labels = torch.load(data_path / label_file).to(torch.float32)
        meta_features = torch.load(data_path / 'meta-features.pt').to(torch.float32)
        assert iids.shape[0] == imask.shape[0] == sids.shape[0] == labels.shape[0] == meta_features.shape[0], \
            'inconsistent num_train in pt-s'
        dataset = TensorDataset(meta_features, iids, imask, sids, labels)
        return dataset
        """
        if (data_path / 'cache.pth.tar').exists():
            dataset = torch.load(data_path / 'cache.pth.tar')
            logger.info(f'Loaded the dataset at {data_path}.')
        else:
            tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            sequence_length = 128
            # encode of the Albert encoding
            texts = list()
            labels = list()
            label_names = ['date', 'everyday', 'formal affair', 'other', 'party', 'vacation', 'wedding', 'work']
            label2id = dict(zip(label_names, range(len(label_names))))
            with open(data_path / 'raw.csv', 'r') as istream:
                for line in tqdm(istream, 'Process data'):
                    parts = line.split(',', 1)
                    review = parts[1].strip('"').strip().replace('","', '')
                    texts.append(review)
                    labels.append(label2id[parts[0].strip('"')])
            instances = tokenizer.batch_encode_plus(texts,
                                                    max_length=sequence_length,
                                                    pad_to_max_length=True, return_attention_masks=True,
                                                    return_token_type_ids=True)
            input_ids = torch.tensor(instances['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(instances['attention_mask'], dtype=torch.long)
            token_type_ids = torch.tensor(instances['token_type_ids'], dtype=torch.long)
            # label
            labels = torch.tensor(labels, dtype=torch.long)
            dataset = TensorDataset(input_ids, attention_mask, token_type_ids, labels)
            # cache the dataset
            torch.save(dataset, data_path / 'cache.pth.tar')
            logger.info(f'Cached the dataset at {data_path / "cache.pth.tar"}.')
        return dataset

    @staticmethod
    def get_optimizer(self):
        """
        return the optimizer and learning rate scheduler
            the parameters for updates with the optimizer are set in this function
        :param self: the solver
        :return: the optimizer, and the learning rate scheduler
        e.g.
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = None  # not implemented yet
        # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.args.warmup_steps,
                                           t_total=self.args.num_train_optimization_steps)
        return optimizer, scheduler
        """
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        # get a linear scheduler
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
        #                                             num_training_steps=self.args.num_train_optimization_steps)
        scheduler = None
        return optimizer, scheduler

    def get_criterion(self):
        if self.num_labels == 1:
            #  We are doing regression
            loss_fct = nn.MSELoss()
            logger.info('The solver validates with MSELoss.')
        else:
            loss_fct = nn.CrossEntropyLoss()
            logger.info('The solver validates with CrossEntropyLoss.')
        return loss_fct

    @staticmethod
    def get_scores(pred, gold):
        smax = nn.Softmax(dim=1)
        pred_idx = torch.argmax(smax(pred))
        correct = pred_idx.eq(gold).sum()
        return {"Accuracy": correct / len(gold), "CrossEntropyLoss": nn.CrossEntropyLoss()(gold, pred)}

