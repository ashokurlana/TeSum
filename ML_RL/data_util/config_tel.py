
train_data_path	 =       "telugu_data/chunked/train/train_*"
valid_data_path	 =       "telugu_data/chunked/main_valid/valid_*"
test_data_path	 =       "telugu_data/chunked/main_test/test_*"
vocab_path		   =       "telugu_data/vocab"



## Hyperparameters
hidden_dim = 512
emb_dim = 60 #300

batch_size = 1 ## At the time of testing make sure the batch_size is 1 to maintain the order of outputs. For training use batch_size=8
max_enc_steps = 400 
max_dec_steps = 100 
min_dec_steps= 35

vocab_size = 50000
beam_size = 4
max_iterations = 100000 

lr = 0.001
eps = 1e-12
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4


intra_encoder = True
intra_decoder = True ### with intra attention;


#save_model_path = "ml_with_intra/"
save_model_path = "saved_models/"
















