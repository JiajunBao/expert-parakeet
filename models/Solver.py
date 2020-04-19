import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import os

import copy
import logging
import numpy
import random
from pathlib import Path


class Solver(object):
    def __init__(self, args, model):
        # check output dir status
        # function registering
        self.__training_step = model.training_step
        self.__validating_step = model.validating_step
        self.__get_scores = model.get_scores
        self.get_dataset = model.get_dataset
        logging.info("Functions registered.")

        # arguments
        self.args = args
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.n_gpu = torch.cuda.device_count()
        logging.info(f'Number of GPU: {self.n_gpu}.')

        # data utilities
        self.train_dataloader = None
        self.dev_dataloader = None
        self.test_dataloader = None

        # logging utilities
        self.logger = None
        self.global_step = 0

        # training utilities
        self.model = model
        self.criterion = model.get_criterion()
        self.optimizer = None
        self.scheduler = None  # not implemented yet # ReduceLROnPlateau(self.optimizer, 'min')
        self.batch_size = args.per_gpu_batch_size * max(1, self.n_gpu)

        # optimizer and scheduler
        self.optimizer, self.scheduler = model.get_optimizer(self)

        # checkpoints
        self.best_optimizer_dict = None
        self.best_model_dict = None
        self.held_loss = float("Inf")

    def fit(self, num_eval_per_epoch=5):
        if os.path.exists(self.args.output_dir) and os.listdir(self.args.output_dir):
            raise ValueError(f"Output directory ({self.args.output_dir}) already exists "
                             "and is not empty")
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        self.setup()
        self.set_train_dataloader()
        self.set_dev_dataloader()
        steps_per_eval = len(self.train_dataloader) // num_eval_per_epoch
        self.train(steps_per_eval)
        self.save_best_checkpoints()
        self.save_final_checkpoints()

    def setup(self):
        # logger
        log_format = '%(asctime)-10s: %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)
        self.logger = logging.getLogger(__name__)
        # put onto cuda
        self.model.to(self.device)
        # self.model.data_parallel(self.n_gpu)
        # fix random seed
        self.fix_random_seed()

    def fix_random_seed(self):
        # Set seed
        random.seed(self.args.seed)
        numpy.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.args.seed)

    def train(self, steps_per_eval):
        # TensorBoard
        tb_writer = SummaryWriter(self.args.output_dir, self.args.comment)
        for epoch_idx in range(self.args.epochs):
            self.model.train()
            self.__train_per_epoch(epoch_idx, steps_per_eval, tb_writer)
        tb_writer.close()

    def validate(self, callback):
        preds, golds = self.__infer(self, self.dev_dataloader, self.model)
        preds = preds.detach().cpu()
        golds = golds.detach().cpu()
        mean_loss = self.criterion(preds, golds)
        if self.n_gpu > 1:
            mean_loss = mean_loss.mean()  # mean() to average on multi-gpu.
        callback.add_scalar('mean_evaluation_loss', mean_loss.item(), self.global_step)
        self.logger.info(f'Eval_loss: {mean_loss: .5f}')
        for k, v in self.__get_scores(preds, golds).items():
            self.logger.info(f'{k}: {v: .5f}')
            callback.add_scalar(k, v, self.global_step)
        return mean_loss

    def __train_per_epoch(self, epoch_idx, steps_per_eval, callback):
        self.model.zero_grad()  # reset gradients
        self.model.train()
        with tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch_idx}") as pbar:
            for batch_idx, batch in enumerate(self.train_dataloader):
                if batch_idx % steps_per_eval == 0:
                    mean_loss = self.validate(callback)
                    if mean_loss < self.held_loss:
                        self.best_model_dict = self.model.state_dict() if self.n_gpu <= 1 \
                            else self.model.module.state_dict()
                        self.best_optimizer_dict = self.optimizer.state_dict()
                        self.held_loss = mean_loss

                        # self.scheduler.step()  # Update learning rate schedule
                    self.model.train()
                self.model.zero_grad()
                loss = self.__training_step(batch)
                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                loss.backward()
                callback.add_scalar('training_loss', loss.item(), self.global_step)
                pbar.set_postfix_str(f"tr_loss: {loss.item():.5f}")
                self.optimizer.step()
                self.model.zero_grad()  # reset gradient
                self.global_step += 1
                pbar.update(1)

    def __training_step(self, batch):
        """
        It is going to be registered.
        :param batch: a batch of input
        :return:
        """
        pass

    def __validating_step(self, batch):
        """
        It is going to be registered.
        :param batch: a batch of input
        :return:
        """
        pass

    @staticmethod
    def __get_scores(preds, golds):
        """
        It is going to be registered.
        :param preds:
        :param golds:
        :return:
        """
        pass

    @staticmethod
    def __infer(self, dataloader, model):
        model.eval()
        preds_list = list()
        golds_list = list()
        with tqdm(total=len(dataloader), desc=f"Evaluating: ") as pbar:
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    logits, labels = model.validating_step(self, batch)
                    preds_list.append(logits)
                    golds_list.append(labels)
                    pbar.update(1)
        # collect the whole chunk
        preds = torch.cat(preds_list, dim=0).cpu()
        print(preds.shape)
        golds = torch.cat(golds_list, dim=0).cpu()
        return preds, golds

    def set_train_dataloader(self):
        data_path = self.args.input_dir / 'train'
        dataset = self.__set_dataset(data_path)
        self.train_dataloader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)
        return self.train_dataloader

    def set_dev_dataloader(self):
        data_path = self.args.input_dir / 'dev'
        dataset = self.__set_dataset(data_path)
        self.dev_dataloader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size)

    def set_test_dataloader(self):
        data_path = self.args.input_dir / 'test'
        dataset = self.__set_dataset(data_path)
        self.test_dataloader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size)

    def __set_dataset(self, data_path):
        return self.get_dataset(data_path)

    @staticmethod
    def get_dataset(data_path, label):
        pass

    def save_best_checkpoints(self):
        self.__save_checkpoints(self.args, self.n_gpu, self.best_optimizer_dict, self.best_model_dict,
                                self.args.output_dir, "best", self.logger)

    def save_final_checkpoints(self):
        self.__save_checkpoints(self.args, self.n_gpu, self.optimizer, self.model,
                                self.args.output_dir, "final", self.logger)

    @staticmethod
    def __save_checkpoints(args, n_gpu, optimizer_dict, model_dict, output_dir, prefix, logger):
        save_directory = output_dir
        os.makedirs(save_directory, exist_ok=True)
        output_path = output_dir / f'{prefix}_trainer.pth.tar'
        # save the trainer
        torch.save(
            {'args': args,
             'optimizer': optimizer_dict},
            output_path
        )
        logger.info(f'Saved the trainer at {output_path}')
        # save the model
        output_path = output_dir / f'{prefix}_model.pth.tar'
        if n_gpu > 1:
            torch.save({"opt": model_dict.opt, "model": model_dict}, output_path)
        else:
            torch.save({"opt": model_dict.opt, "model": model_dict}, output_path)
        logger.info(f'Saved model checkpoints to {output_path}')

    def infer(self, data_path):
        data_path = Path(data_path)
        self.setup()
        dataset = self.get_dataset(data_path, self.args.label)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=self.batch_size)
        preds, golds = self.__infer(self, dataloader, self.model)
        return preds, golds, dataset
