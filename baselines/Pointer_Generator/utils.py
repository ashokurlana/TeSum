#Content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/
import os
import csv
import pyrouge
import logging
import tensorflow as tf
import config
from config import *
from TeluguTokenizer.tokenizer import *
from rouge_score import rouge_scorer


def print_results(article, abstract, decoded_output):
  print ("")
  print('ARTICLE:  %s', article)
  print('REFERENCE SUMMARY: %s', abstract)
  print('GENERATED SUMMARY: %s', decoded_output)
  print( "")


def make_html_safe(s):
  s.replace("<", "&lt;")
  s.replace(">", "&gt;")
  return s

def sent_tokenize(text):
    processed_data = preprocess_data(text)        ### Preprocessing data
    sentences = sentence_tokenize(processed_data) ### Sentencification
    data = ""
    for sent in sentences:
        data += sent +"\n"
    return data.strip()


def rouge_eval(ref_dir, dec_dir):
  rouge_scores = []
  count = 0
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], lang="telugu")
  for dfile in sorted(os.listdir(dec_dir)):
    found = False
    if dfile.endswith('.txt'):
      print(dfile)
      for rfile in sorted(os.listdir(ref_dir)):
        if rfile.endswith('.txt') and dfile.split('_')[0] == rfile.split('_')[0]:
          print(rfile)
          found = True
          break
      if found:
        count += 1
        hypo = open(dec_dir+'/'+ dfile, 'r', encoding='utf-8').read()
        hypo_sents = sent_tokenize(hypo)
        ref = open(ref_dir +'/'+rfile, 'r', encoding='utf-8').read()
        ref_sents = sent_tokenize(ref)
        # hypo_wx = con.convert(hypo)
        # ref_wx = con.convert(ref)
        scores = scorer.score(ref_sents, hypo_sents)
        rouge_scores.append({'file': dfile, 'rouge-1_f': scores['rouge1'][2], 'rouge-2_f': scores['rouge2'][2], 'rouge-l_f': scores['rougeL'][2], 'rouge-l-sum_f': scores['rougeLsum'][2]})
        # print(rouge_scores)
        # exit()
        print(count, '  ', config.modelname)

  print("\n------------------------------------------------------\n")
  # print(rouge_scores)        
  score_tags = list(rouge_scores[0].keys())
  print("score_tags are: ", score_tags)
  print("Writing final ROUGE results into a csv file")
  filename = str(config.modelname)+"_rouge.csv"
  with open(filename, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=score_tags)
    writer.writeheader()
    writer.writerows(rouge_scores)

  # r = pyrouge.Rouge155()
  # r.model_filename_pattern = '#ID#_reference.txt'
  # r.system_filename_pattern = '(\d+)_decoded.txt'
  # r.model_dir = ref_dir
  # r.system_dir = dec_dir
  # logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
  # rouge_results = r.convert_and_evaluate()
  # return r.output_to_dict(rouge_results)


# def rouge_log(results_dict, dir_to_write):
#   log_str = ""
#   for x in ["1","2","l"]:
#     log_str += "\nROUGE-%s:\n" % x
#     for y in ["f_score", "recall", "precision"]:
#       key = "rouge_%s_%s" % (x,y)
#       key_cb = key + "_cb"
#       key_ce = key + "_ce"
#       val = results_dict[key]
#       val_cb = results_dict[key_cb]
#       val_ce = results_dict[key_ce]
#       log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
#   print(log_str)
#   results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
#   print("Writing final ROUGE results to %s..."%(results_file))
#   with open(results_file, "w") as f:
#     f.write(log_str)


def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
  writer = summary_writer
  if running_avg_loss == 0:  # on the first iteration just take the loss
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)  # clip
  loss_sum = tf.compat.v1.Summary()
  # loss_sum = tf.Summary()
  tag_name = 'running_avg_loss/decay=%f' % (decay)
  loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
  #summary_writer.add_summary(loss_sum, step)
  with writer.as_default():
    tf.summary.scalar(tag_name, running_avg_loss, step=step)
    writer.flush()
  return running_avg_loss


out = open(log_root+"/"+"Output_pg_wo_emb.txt","w")

def write_for_rouge(input_article, reference_sents, decoded_words, ex_index, _rouge_ref_dir, _rouge_dec_dir, _rouge_article_dir):
  decoded_sents = []
  while len(decoded_words) > 0:
    try:
      fst_period_idx = decoded_words.index(".")
    except ValueError:
      fst_period_idx = len(decoded_words)
    sent = decoded_words[:fst_period_idx + 1]
    decoded_words = decoded_words[fst_period_idx + 1:]
    decoded_sents.append(' '.join(sent))

  # pyrouge calls a perl script that puts the data into HTML files.
  # Therefore we need to make our output HTML safe.
  decoded_sents = [make_html_safe(w) for w in decoded_sents]
  reference_sents = [make_html_safe(w) for w in reference_sents]

  ref_file = os.path.join(_rouge_ref_dir, "%06d_reference.txt" % ex_index) ##list of string formate
  decoded_file = os.path.join(_rouge_dec_dir, "%06d_decoded.txt" % ex_index) ##list of string
  article_file = os.path.join(_rouge_article_dir, "%06d_article.txt" % ex_index) ### whole context as one string

  with open(ref_file, "w") as f:
    for idx, sent in enumerate(reference_sents):
      f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")

  with open(decoded_file, "w") as f:
    for idx, sent in enumerate(decoded_sents):
      f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")

  with open(article_file, "w") as f:
    f.write(input_article)



  #print("Wrote example %i to file" % ex_index)
  out.write("Article: %s \n"%str(input_article))
  out.write("Reference_Summary: %s \n"%str(("".join(reference_sents))))
  out.write("System_Generated_Summary: %s \n"%str(("".join(decoded_sents))))
  out.write("@------------------------------------------------------------------------------------@\n")






