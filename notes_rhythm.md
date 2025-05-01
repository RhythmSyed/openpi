
The following commands need to be run to setup the environment:

pip install pydantic==2.10.6


python scripts/compute_norm_stats.py --config-name pi0_libero_all_no_pretrain

Run the following to train:
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_fast_libero_low_mem_finetune --exp-name=pi0_example --overwrite

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py pi0_libero_all_no_pretrain --exp-name=pi0_libero_all_no_pretrain_experiment --overwrite

