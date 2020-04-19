import argparse
from pathlib import Path
import torch
from models.Albert import AlbertForReviewClassification
from models.Solver import Solver
import logging

logger = logging.getLogger(__name__)


# item()
checkpoints = torch.load('checkpoints/albert_balanced/best_model.pth.tar')
model = AlbertForReviewClassification.from_pretrained('albert-base-v2', num_labels=8)
model.load_state_dict(checkpoints["model"])
args = torch.load('checkpoints/albert_balanced/best_trainer.pth.tar')['args']
logger.info("model load completed")
trainer = Solver(args, model)
preds, golds, _, _ = trainer.infer(Path('data/processed/balanced/test'))
torch.save(preds, 'checkpoints/albert_balanced/preds.pth')
torch.save(golds, 'checkpoints/albert_balanced/golds.pth')
print(preds.eq(golds).sum().item() / preds.shape[0])
