import argparse
from pathlib import Path
import torch
from models.Albert import AlbertForReviewClassification
from models.Solver import Solver
import logging

logger = logging.getLogger(__name__)



# dirty code
checkpoints_path = Path('checkpoints/albert')

checkpoints = torch.load(checkpoints_path / 'best_model.pth.tar')
model = AlbertForReviewClassification.from_pretrained('albert-base-v2', num_labels=8)
model.load_state_dict(checkpoints["model"])
args = torch.load(checkpoints_path / 'best_trainer.pth.tar')['args']
logger.info("model load completed")
trainer = Solver(args, model)
preds, golds, _, _ = trainer.infer(Path('data/processed/full/test'))
torch.save(preds, checkpoints_path / 'preds.pth')
torch.save(golds, checkpoints_path / 'golds.pth')
print(preds.eq(golds).sum().item() / preds.shape[0])
print((preds[golds == 0] == 0).sum().item() / (golds == 0).sum().item())
print((preds[golds == 1] == 1).sum().item() / (golds == 1).sum().item())
print((preds[golds == 2] == 2).sum().item() / (golds == 2).sum().item())
print((preds[golds == 3] == 3).sum().item() / (golds == 3).sum().item())
print((preds[golds == 4] == 4).sum().item() / (golds == 4).sum().item())
print((preds[golds == 5] == 5).sum().item() / (golds == 5).sum().item())
print((preds[golds == 6] == 6).sum().item() / (golds == 6).sum().item())
print((preds[golds == 7] == 7).sum().item() / (golds == 7).sum().item())

