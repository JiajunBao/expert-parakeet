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

lb = ['date', 'everyday', 'formal affair', 'other', 'party', 'vacation', 'wedding', 'work']
for i in range(8):
    print(lb[i], (preds[golds == i] == i).sum().item() / (golds == i).sum().item())
