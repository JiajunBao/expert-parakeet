import argparse
from pathlib import Path
from models.Albert import AlbertForReviewClassification
from models.Solver import Solver
import logging

logger = logging.getLogger(__name__)


def get_arguments():
    parser = argparse.ArgumentParser()
    # parameters to trainer
    parser.add_argument('--input_dir', type=Path, required=True,
                        help='directory to the input data, e.g. ./data')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='directory to the output data, e.g. ./output')
    parser.add_argument('--label', type=str, required=True,
                        help='name of the label, e.g. PC0')
    parser.add_argument('--learning_rate', type=float, required=True,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, required=True,
                        help='number of training epochs')
    parser.add_argument('--per_gpu_batch_size', type=int, required=True,
                        help='batch size')
    parser.add_argument('--comment', type=str, required=True,
                        help='comment for the tensorboard record')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                        help='epsilon for adam optimizer')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed to fix')
    # parameters for the model
    parser.add_argument('--resume_path', type=Path, required=False, default=None,
                        help='model to resume the training from')
    parser.add_argument('--freeze', action='store_true',
                        help='where to freeze weights of the bert model')
    parameters = parser.parse_args()
    # does not split
    return parameters, parameters


if __name__ == '__main__':
    args, opt = get_arguments()
    if opt.resuming_path:
        model = AlbertForReviewClassification.from_pretrained(opt.resuming_path)
    else:
        model = AlbertForReviewClassification.from_pretrained('albert-base-v2')
    logger.info("model load completed")
    trainer = Solver(args, model)
    logger.info("trainer load completed")
    trainer.fit(num_eval_per_epoch=5)
