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
#sys.path.insert(1, "/home/ravva_priyanka/Summarization/get_to_point_model_1/data_util/")

#import data_util
#import data_util
#from data_util import config
#from data_util.batcher import Batcher
#from data_util.data import Vocab
#from data_util.utils import calc_running_avg_loss

#import data_util
#import data_util

import config
from batcher import Batcher
from data import Vocab
from utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch




use_cuda = config.use_gpu and torch.cuda.is_available()

class Train(object):
    def __init__(self):
        #config("print.vocab_path ",config.vocab_path)
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train',
                               batch_size=config.batch_size, single_pass=False)
        time.sleep(15)

        train_dir = os.path.join(config.log_root, 'train_%d' % (int(time.time())))
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        self.model_dir = os.path.join(train_dir, 'model')
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        #self.summary_writer = tf.compat.v1.summary.FileWriter(train_dir)
        self.summary_writer = tf.summary.FileWriter(train_dir)
        #self.summary_writer = tf.contrib.summary.create_file_writer(train_dir)

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

    def setup_train(self, model_file_path=None):
        self.model = Model(model_file_path)

        params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters()) + \
                 list(self.model.reduce_state.parameters())
        #print("params : ",params)
        #print("params collection is completed....")
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
        #print("train_one_batch function ......")
        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden) ### Here initially encoder final hiddenstate==decoder first/prev word at timestamp=0
        #print("s_t_1 : ",len(s_t_1),s_t_1[0].shape,s_t_1[1].shape)

        #print("steps.....")
        #print("max_dec_len = ",max_dec_len)
        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            ############ Traing [ Teacher Forcing ] ###########
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            #print("y_t_1 : ",len(y_t_1))
            final_dist, s_t_1,  c_t_1, attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab,
                                                                           coverage, di)
            #print("attn_dist : ",len(attn_dist),attn_dist[0].shape)
            #print("final_dist : ",len(final_dist),final_dist[0].shape) ############## vocab_Size
            target = target_batch[:, di]
            #print("target = ",len(target))

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
        print("trainIters__Started___model_file_path is : ", model_file_path)
        iter, running_avg_loss = self.setup_train(model_file_path)
        start = time.time()
        print("Max iteration : n_iters = ",n_iters)
        print("going to start running iter NO : ",iter)
        print("\n******************************\n")
        while iter < n_iters:
            #print("\n###################################\n")
            #print("iter : ",iter)
            batch = self.batcher.next_batch()
            #print("batch data loading : ",len(batch))
            loss = self.train_one_batch(batch)
            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            #print("running_avg_loss : ",running_avg_loss)
            iter += 1
            if iter % 100 == 0: ##100
                self.summary_writer.flush()

            print_interval = 100 #1000
            if iter % print_interval == 0:
                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval,
                                                                           time.time() - start, loss))
                start = time.time()
            if iter % 500 == 0: ##5000
                self.save_model(running_avg_loss, iter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train script")
    parser.add_argument("-m",
                        dest="model_file_path", 
                        required=False,
                        default=None,
                        help="Model file for retraining (default: None).")
    args = parser.parse_args()
    print(args)
    
    train_processor = Train()
    #print("train_processor object created....")
    train_processor.trainIters(config.max_iterations, args.model_file_path)
    print("Train iteration process completed...")
