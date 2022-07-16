import os

root_dir = os.path.expanduser("~")

train_data_path = os.path.join(root_dir, "finished_files_m1/chunked/train_*")
eval_data_path = os.path.join(root_dir, "finished_files_m1/val.bin")
decode_data_path = os.path.join(root_dir, "finished_files_m1/test.bin")
vocab_path = os.path.join(root_dir, "finished_files_m1/vocab")

log_root = os.path.join(root_dir, "log_models")#s2s
#log_root = os.path.join(root_dir, "log_models_pg")#pg
#log_root = os.path.join(root_dir, "log_models_pgcv")#pg



# Hyperparameters
use_gpu=True

hidden_dim= 256
emb_dim= 300 
batch_size= 16
max_enc_steps=400
max_dec_steps=100
min_dec_steps=35
beam_size=4

vocab_size=50000
max_iterations = 100000 


lr=0.15
eps = 1e-12
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4



pointer_gen = True #False #
is_coverage = False 

cov_loss_wt = 1.0
lr_coverage	= 0.15

adagrad_init_acc=0.1
max_grad_norm=2.0







