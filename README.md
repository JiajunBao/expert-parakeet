# expert-parakeet

Code Release

Reproduce results:
```
# set up environment
python3 -m venv env 
source env/bin/activate
pip install -r requirements.txt

# train ALBERT
python3 src/albert_driver.py \
--input_dir data/processed/full \
--output_dir checkpoints/albert/ \
--epochs 5 \
--learning_rate 1e-5 \
--per_gpu_batch_size 32 \
--weight_decay 1e-5 \
--warmup_steps 10 \
--comment albert

# test
python3 src/infer.py

# train hattn
set -x
python3 src/create_input_files.py
python3 src/train.py 
# test
python3 src/eval.py

# deploy the online webapp for inference (with our hattn model)
python3 src/run_hattn_viewer.py
```
