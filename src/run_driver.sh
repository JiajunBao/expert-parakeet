python albert_driver.py \
--input_dir data/processed/balanced \
--output_dir checkpoints/albert/ \
--epochs 5 \
--learning_rate 1e-5 \
--per_gpu_batch_size 64 \
--comment albert