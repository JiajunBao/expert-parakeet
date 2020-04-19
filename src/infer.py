import argparse
from pathlib import Path
import torch
from models.Albert import AlbertForReviewClassification
from models.Solver import Solver
import logging

logger = logging.getLogger(__name__)



checkpoints = torch.load('checkpoints/albert/best_model.pth.tar')
model = AlbertForReviewClassification.from_pretrained('albert-base-v2', num_labels=8)
model.load_state_dict(checkpoints["model"])
args = torch.load('checkpoints/albert/best_trainer.pth.tar')['args']
logger.info("model load completed")
trainer = Solver(args, model)
preds, golds, _, _ = trainer.infer(Path('data/processed/full/test'))
torch.save(preds, 'checkpoints/albert/preds.pth')
torch.save(golds, 'checkpoints/albert/golds.pth')

