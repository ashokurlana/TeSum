from __future__ import unicode_literals, print_function, division

import os
import time
import sys

import tensorflow as tf
import torch

import config
from batcher import Batcher
from data import Vocab

from utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch
from model import Model

use_cuda = config.use_gpu and torch.cuda.is_available()

class Evaluate(object):
    def __init__(self, model_file_path):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.eval_data_path, self.vocab, mode='eval', batch_size=config.batch_size, single_pass=True)
        time.sleep(15)
        model_name = os.path.basename(model_file_path)

        eval_dir = os.path.join(config.log_root, 'eval_%s' % (model_name))
        if not os.path.exists(eval_dir):
            os.mkdir(eval_dir)
        # self.summary_writer = tf.compat.v1.summary.FileWriter(eval_dir)
        self.summary_writer = tf.summary.create_file_writer(eval_dir)
        self.model = Model(model_file_path, is_eval=True)

    def eval_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)
        dec_batch, dec_padding_mask, max_dec_len, dec_lens_var, target_batch = \
            get_output_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_1 = self.model.reduce_state(encoder_hidden)

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t_1 = dec_batch[:, di]  # Teacher forcing
            final_dist, s_t_1, c_t_1,attn_dist, p_gen, next_coverage = self.model.decoder(y_t_1, s_t_1,
                                                        encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
                                                        extra_zeros, enc_batch_extend_vocab, coverage, di)
            target = target_batch[:, di]
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()
            step_loss = -torch.log(gold_probs + config.eps)
            if config.is_coverage:
                step_coverage_loss = torch.sum(torch.min(attn_dist, coverage), 1)
                step_loss = step_loss + config.cov_loss_wt * step_coverage_loss
                coverage = next_coverage

            step_mask = dec_padding_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_step_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_step_losses / dec_lens_var
        loss = torch.mean(batch_avg_loss)

        return loss.item()

    def run_eval(self):
        running_avg_loss, iter = 0, 0
        start = time.time()
        batch = self.batcher.next_batch()
        while batch is not None:
            loss = self.eval_one_batch(batch)

            running_avg_loss = calc_running_avg_loss(loss, running_avg_loss, self.summary_writer, iter)
            #print("iter, running_avg_loss: ",iter, running_avg_loss)
            iter += 1

            if iter % 100 == 0:
                self.summary_writer.flush()

            epoch_equivalent_iteration_num = int(3336/config.batch_size) ##(validation dataset size/batchsize; may be some flies can be left)

#            print_interval = 100 ###1000
            print_interval = epoch_equivalent_iteration_num

            if iter % print_interval == 0:
#                print('steps %d, seconds for %d batch: %.2f , loss: %f' % (iter, print_interval, time.time() - start, running_avg_loss))
                #print('Iteration_No: %d,  avg_loss: %f' % (model_iteration_num,running_avg_loss))
                start = time.time()

            batch = self.batcher.next_batch()
        return running_avg_loss


if __name__ == '__main__':
    path = config.log_root ## basically log file
    list_of_directories = sorted(os.listdir(path), reverse=True)
    list_of_directories = [idx for idx in list_of_directories if idx.lower().startswith("train_")]
    #print("list_of_directories of trained model: ",list_of_directories)
    if(len(list_of_directories)!=0):
        directory_for_validation = list_of_directories[0] ## considering the recent files for validation
    
    model_filename = os.path.join(path,directory_for_validation,"model")
    list_of_decoder_files = sorted(os.listdir(model_filename))
    for model_name in list_of_decoder_files:
        full_path_model = os.path.join(model_filename,model_name)
        model_iteration_num = model_name.split("_")[1]
        
        eval_processor = Evaluate(full_path_model)
        val_loss = eval_processor.run_eval()
        print("validation_loss: ", val_loss)


