from __future__ import unicode_literals, print_function, division
import os
import time
import argparse
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import torch
from model import Model
#from model_file import ModelName as Model
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adagrad
import os,sys
import config
from batcher import Batcher
from data import Vocab
from utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch
from eval_387 import *



logfile = open("pg_cv_ex_mup.txt","a+")
# logfile = open("m1_log.txt","a+")

use_cuda = config.use_gpu and torch.cuda.is_available()

class Train(object):
    def __init__(self):
        #config("print.vocab_path ",config.vocab_path)
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train', batch_size=config.batch_size, single_pass=False)
        time.sleep(15)

        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        # self.summary_writer = tf.compat.v1.summary.FileWriter(train_dir)
        # self.summary_writer = tf.summary.FileWriter(train_dir)
        # self.summary_writer = tf.train.SummaryWriter(train_dir)
        self.summary_writer = tf.summary.create_file_writer(train_dir)


    def save_model(self, running_avg_loss, iter):
        state = {
            'iter': iter,
            'encoder_state_dict': self.model.encoder.state_dict(),
            'decoder_state_dict': self.model.decoder.state_dict(),
            'reduce_state_dict': self.model.reduce_state.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_loss
        }
        model_save_path = os.path.join(self.model_dir, 'model_%d_%d' % (iter, int(time.time())))
        torch.save(state, model_save_path)
        return model_save_path

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())

        initial_lr = config.lr_coverage if config.is_coverage else config.lr
        self.optimizer = Adagrad(params, lr=initial_lr, initial_accumulator_value=config.adagrad_init_acc)

        start_iter, start_loss = 0, 0

        #### Loading state where the training stopped earlier use that to train for future epoches ####
        if model_file_path is not None:
            state = torch.load(model_file_path, map_location= lambda storage, location: storage)
            start_iter = state['iter']
            start_loss = state['current_loss']

            if not config.is_coverage:
                self.optimizer.load_state_dict(state['optimizer'])
                ###### Making into GPU/server accessable Variables #####
                if use_cuda:
                    for state in self.optimizer.state.values():
                        for k, v in state.items():
                            if torch.is_tensor(v):
                                state[k] = v.cuda()
        return start_iter, start_loss

    def train_one_batch(self, batch):

        ########### Below Two lines of code is for just initialization of Encoder and Decoder sizes,vocab, lenghts etc : ######
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        self.optimizer.zero_grad()
        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden) ### Here initially encoder final hiddenstate==decoder first/prev word at timestamp=0
        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            ############ Traing [ Teacher Forcing ] ###########
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab,
                                                                           coverage, di)
            target = target_batch[:, di]

            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)   #################################################### Eqn_6
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)  ###############################Eqn_13a
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss    ###############################Eqn_13b
                coverage = next_coverage
                
            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses/dec_lens_var
        loss = torch.mean(batch_avg_loss)

        loss.backward()

        self.norm = clip_grad_norm_(self.model.encoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.decoder.parameters(), config.max_grad_norm)
        clip_grad_norm_(self.model.reduce_state.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()



    def trainIters(self, n_iters, model_file_path=None):
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        print("Max iteration : n_iters = ",n_iters)
        print("\n")

        best_model = ""
        best_loss  = 90000000.00 

        while iter < n_iters:
            batch = self.batcher.next_batch()
            loss = self.train_one_batch(batch)
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            iter += 1
            if iter % 100 == 0: ##100
                self.summary_writer.flush()

            print_interval =500#1000 #500 #1000
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval, time.time() - start, loss))
                start = time.time()

            if iter % 1000== 0: #10000#15000#3000 #20000
                model_save_path = self.save_model(running_avg_loss, iter)
                eval_processor = Evaluate(model_save_path)
                running_avg_loss_of_val = eval_processor.run_eval()

                print('Iteration_No: %d, train_avg_loss: %f , val_avg_loss: %f' % (iter,running_avg_loss,running_avg_loss_of_val))
                logfile.write(('Iteration_No: %d, train_avg_loss: %f , val_avg_loss: %f' % (iter,running_avg_loss,running_avg_loss_of_val)))
                logfile.write("\n")

                ###### Early Stopping Criteria: #####                
                if(running_avg_loss_of_val<best_loss):
                    best_loss = running_avg_loss_of_val
                    best_model = iter


        print("\n")
        print("best_model , low loss : ",best_model,best_loss)

        logfile.write(("best_model = %s "%str(best_model)))
        logfile.write("\n")
        logfile.write(("best_model_low loss = %s "%str(best_loss)))
        logfile.write("\n")





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path", 
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()
    print("\n******************************** Starting ***********************************\n")
    print("config.hidden_dim = %f, config.emb_dim = %f, config.batch_size = %f, config.max_enc_steps = %f, config.max_dec_steps = %f, config.min_dec_steps = %f , config.beam_size  = %f, config.vocab_size =  = %f, config.max_iterations = %f, config.lr  = %f, config.eps  = %f, config.pointer_gen  = %s, config.is_coverage  = %s "%(config.hidden_dim, config.emb_dim, config.batch_size, config.max_enc_steps, config.max_dec_steps, config.min_dec_steps, config.beam_size, config.vocab_size, config.max_iterations, config.lr, config.eps, config.pointer_gen, config.is_coverage))
    print("\n-------------------------------------------------------\n")
    logfile.write(("config.hidden_dim = %f, config.emb_dim = %f, config.batch_size = %f, config.max_enc_steps = %f, config.max_dec_steps = %f, config.min_dec_steps = %f , config.beam_size  = %f, config.vocab_size =  = %f, config.max_iterations = %f, config.lr  = %f, config.eps  = %f, config.pointer_gen  = %s, config.is_coverage  = %s "%(config.hidden_dim, config.emb_dim, config.batch_size, config.max_enc_steps, config.max_dec_steps, config.min_dec_steps, config.beam_size, config.vocab_size, config.max_iterations, config.lr, config.eps, config.pointer_gen, config.is_coverage)))
    logfile.write("\n")

    train_processor = Train()
    print("Starting to Train")
    print("Max iteration is",config.max_iterations)
    train_processor.trainIters(config.max_iterations, args.model_file_path)
    print("Completed...")
    print("\n############################# ENDING #######################################\n\n")






logfile.close()
