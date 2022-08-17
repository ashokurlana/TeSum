import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from model import Model

from data_util import config_tel as config
from data_util import data
from data_util.batcher import Batcher
from data_util.data import Vocab
from train_util import *
from beam_search import *
from rouge import Rouge
import argparse


logfile = open("backup_log/m2_val_log.txt",'a+')

def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


class Evaluate(object):
    def __init__(self, data_path, opt, batch_size = config.batch_size):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(data_path, self.vocab, mode='eval',
                               batch_size=batch_size, single_pass=True)
        self.test_set_name = data_path.split("/")[-2] ## test_set1/test_set2  sftp://rakesh.vemula@ada/home2/rakesh.vemula/ml_rl/telugu_data/chunked/main_test/test_0000.bin

        self.opt = opt
        time.sleep(5)

    def setup_valid(self):
        self.model = Model()
        self.model = get_cuda(self.model)
        checkpoint = T.load(os.path.join(config.save_model_path, self.opt.load_model))
        self.model.load_state_dict(checkpoint["model_dict"])


    def print_original_predicted(self, decoded_sents, ref_sents, article_sents, loadfile):
        print("The loaded file is: ", loadfile)
        setname = opt.model_type
#        filename = "test_"+loadfile.split(".")[0]+".txt"
#        filename = setname+"__"+"test_"+loadfile.split(".")[0]+".txt"
        set_num = self.test_set_name
#        filename = save_model_path+setname+"__"+set_num+"_test_"+loadfile.split(".")[0]+".txt"
        filename = config.save_model_path+setname+"__"+set_num+"_"+loadfile.split(".")[0]+".txt"

        #with open(os.path.join("telugu_data_m2",filename), "w") as f:
        with open(os.path.join("",filename), "w") as f:
            for i in range(len(decoded_sents)):
                f.write("article: "+article_sents[i] + "\n")
                f.write("ref: " + ref_sents[i] + "\n")
                f.write("dec: " + decoded_sents[i] + "\n\n")

    def evaluate_batch(self, print_sents = True):
    #def evaluate_batch(self, print_sents = False):

        self.setup_valid()
        batch = self.batcher.next_batch()
        #print("batch = ",(batch[0].shape))
        start_id = self.vocab.word2id(data.START_DECODING)
        end_id = self.vocab.word2id(data.STOP_DECODING)
        unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        #print("start_id =%d , end_id =%d , unk_id =%d "%(start_id , end_id , unk_id))
        decoded_sents = []
        ref_sents = []
        article_sents = []
        rouge = Rouge()
        count = 0
        while batch is not None:
            count+=1
            #print("\n########################################\n")
            #print("Current Batche No = ",count)
            enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e = get_enc_data(batch)

            with T.autograd.no_grad():
                enc_batch = self.model.embeds(enc_batch)
                enc_out, enc_hidden = self.model.encoder(enc_batch, enc_lens)
                #print("enc_out, enc_hidden  = ",enc_out, enc_hidden )

            ###-----------------------Summarization----------------------------------------------------
            with T.autograd.no_grad():
                pred_ids = beam_search(enc_hidden, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, self.model, start_id, end_id, unk_id)


            #######
            for i in range(len(pred_ids)):
                #print("i = ",i)
                decoded_words = data.outputids2words(pred_ids[i], self.vocab, batch.art_oovs[i])
                if len(decoded_words) < 2:
                    decoded_words = "xxx"
                else:
                    decoded_words = " ".join(decoded_words)

                #print("decoded_words = ",decoded_words)
                decoded_sents.append(decoded_words)
                abstract = batch.original_abstracts[i]
                article = batch.original_articles[i]
                ref_sents.append(abstract)
                article_sents.append(article)


            batch = self.batcher.next_batch()
            #print("batch = ",batch)
           
        #print("all batches completed...") 
        #print("Last count = ",count)
        #print("\n**********************************\n")
        #print("self.opt = ",self.opt)
        #print("self.opt.load_model = ",self.opt.load_model)

        load_file = self.opt.load_model
        #print("load_file = ",load_file)

        if print_sents:
            self.print_original_predicted(decoded_sents, ref_sents, article_sents, load_file)

        scores = rouge.get_scores(decoded_sents, ref_sents, avg = True)
        if self.opt.task == "test":
            print(load_file, "scores:", scores)
            rouge_l = scores ### giving all scores(optional):
        else:
            rouge_l = scores["rouge-l"]["f"]
            print(load_file, "rouge_l:", "%.4f" % rouge_l)
            #print(load_file, "rouge_l = %.4f , so_far_best_score = %.4f , best_iter = %d" % (rouge_l))

            #logfile.write((str(load_file)+" rouge_l: "+" %.4f" % rouge_l))
            #logfile.write('\n')
        return load_file, rouge_l




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="validate", choices=["validate","test"])
    parser.add_argument("--start_from", type=str, default="0001000.tar")#0020000.tar
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="ml")

    opt = parser.parse_args()
    print("opt = ",opt)

    logfile.write("\n#########################[ Start Point ] ###########################\n")
    best_score_record = 0.000
    best_score_record_filename =""

    if opt.task == "validate":
        print("config.save_model_path = ",config.save_model_path)
        saved_models = os.listdir(config.save_model_path)
        saved_models.sort()
        file_idx = saved_models.index(opt.start_from)
        saved_models = saved_models[file_idx:]

        for f in saved_models:
            opt.load_model = f
            eval_processor = Evaluate(config.valid_data_path, opt)
            load_file, rouge_l = eval_processor.evaluate_batch()
            
            if(best_score_record < rouge_l):
                best_score_record = rouge_l
                best_score_record_filename = load_file

            #print(load_file, "rouge_l = %.4f , so_far_best_score = %.4f and it's best_iter No = %s \n" % (rouge_l, best_score_record, str(best_score_record_filename)))#showing present & best iter

            logfile.write("load_file = %s , rouge_l = %.4f , so_far_best_score = %.4f and it's (best)iter No = %s \n" % (str(load_file), rouge_l, best_score_record, str(best_score_record_filename)))
            #logfile.write('\n')


        print("\n--------------------------------------------\n")
        print("best_score_record : ",best_score_record)
        print("best_score_record_filename : ",best_score_record_filename)
        print("\n--------------------------------------------\n")

        logfile.write("\n************************************\n best_score_record_filename : "+str(best_score_record_filename))
        logfile.write("\n")
        logfile.write("best_score_record : "+str(best_score_record))
        logfile.write('\n')

    else:   #test
        print("config.test_data_path = ",config.test_data_path)
        print("\n")
        eval_processor = Evaluate(config.test_data_path, opt)
        load_file_name, rouge_l = eval_processor.evaluate_batch()
    logfile.write("\n#########################[ End Point ] ###########################\n")



logfile.close()





