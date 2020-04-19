/opt/anaconda/bin/python3 src/albert_driver.py \
--input_dir data/processed/full \
--output_dir checkpoints/albert/ \
--epochs 5 \
--learning_rate 1e-5 \
--per_gpu_batch_size 32 \
--weight_decay 1e-5 \
--warmup_steps 10 \
--comment albert