import os

root_dir = os.path.expanduser("~")

train_data_path = os.path.join(root_dir, "pointer_generator/pg_data/chunked/train_*")
eval_data_path = os.path.join(root_dir, "pointer_generator/pg_data/dev.bin")
# decode_data_path = os.path.join(root_dir, "pointer_generator/finished_files_m1/test.bin")
decode_data_path = os.path.join(root_dir, "pointer_generator/pg_data/test.bin")
vocab_path = os.path.join(root_dir, "pointer_generator/pg_data/vocab")

# log_root = os.path.join(root_dir, "pointer_generator/log_models")#s2s
#log_root = os.path.join(root_dir, "log_models_pg")#pg
log_root = os.path.join(root_dir, "pointer_generator/log_models_pg_cv_ex_mup")#pg_cv_w2v

modelname = "pg_cv_ex_mup"

# Hyperparameters
use_gpu=True

hidden_dim= 256
emb_dim= 300 #300(used with without pretrained embeddings and fasttext embeddings) AND 60 used for word2vec 
batch_size= 16
max_enc_steps=512 #250
max_dec_steps=120
min_dec_steps=50
beam_size=4

vocab_size=100000
max_iterations = 50000# Used 50k iterations at the time of training and 3k iterations added for further training with pg+cv model 


lr=0.15
eps = 1e-12
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4



pointer_gen = True  #
is_coverage = True #True # False 

cov_loss_wt = 1.0
lr_coverage = 0.15

adagrad_init_acc=0.1
max_grad_norm=2.0


