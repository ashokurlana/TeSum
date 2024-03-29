#Most of this file is copied form https://github.com/abisee/pointer-generator/blob/master/data.py

import glob
import random
import struct
import csv
from tensorflow.core.example import example_pb2
from torch.autograd import Variable
from gensim.models import KeyedVectors
from data_util import config_tel
from gensim.models import Word2Vec, KeyedVectors

#import config_tel as config

import torch
import numpy as np

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences

# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.


class Vocab(object):

  def __init__(self, vocab_file, max_size):
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0 # keeps track of total number of words in the Vocab

    # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
    for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
      self._word_to_id[w] = self._count
      self._id_to_word[self._count] = w
      self._count += 1

    # Read the vocab file and add words up to max_size
    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        pieces = line.split()
        if len(pieces) != 2:
          #print("Warning: incorrectly formatted line in vocabulary file: %s\n" % line)
          continue
        w = pieces[0]
        if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
          raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
        if w in self._word_to_id:
          raise Exception('Duplicated word in vocabulary file: %s' % w)
        self._word_to_id[w] = self._count
        self._id_to_word[self._count] = w
        self._count += 1
        if max_size != 0 and self._count >= max_size:
          #print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
          break
    #print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))


  def load_pretrained_wordembeddings(self, embedding_dim,vocabs):
    #fasttext_path = "/home2/manshri/Priyanka/Embeddings/Cbow_fasttext/cc.te.300.vec"
    #fasttext_path = "/home2/manshri/Priyanka/Embeddings/word2vec/irishad_tel_w2v_emb.txt"
    fasttext_path = "word2vec/irishad_tel_w2v_emb.txt"
    #fasttext_path = "/home/ravva_priyanka/Telugu_word_embeddings/word2vec/tel_word_embeddings.txt"

    word_embeddings = KeyedVectors.load_word2vec_format(fasttext_path)
    word_vectors = word_embeddings.wv
    #word_vectors = word_embeddings

    print("word_vectors vocab :: ",len(word_vectors.vocab))
    print(len(self._word_to_id),type(self._word_to_id))

    embeddings = 1 * np.random.randn(len(self._word_to_id) + 1, embedding_dim)

    embeddings[0] = 0
    for word, index in self._word_to_id.items():
        if word in word_vectors.vocab:
            embeddings[index] = word_vectors[word]
    return torch.tensor(embeddings, dtype=torch.float)



  def word2id(self, word):
    if word not in self._word_to_id:
      return self._word_to_id[UNKNOWN_TOKEN]
    return self._word_to_id[word]

  def id2word(self, word_id):
    if word_id not in self._id_to_word:
      raise ValueError('Id not found in vocab: %d' % word_id)
    return self._id_to_word[word_id]

  def size(self):
    return self._count

  def write_metadata(self, fpath):
    print("Writing word embedding metadata file to %s..." % (fpath))
    with open(fpath, "w") as f:
      fieldnames = ['word']
      writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
      for i in xrange(self.size()):
        writer.writerow({"word": self._id_to_word[i]})



class Create_Vocab(object):

  def __init__(self, vocab_file, max_size):
    self._word_to_id = {}
    self._id_to_word = {}
    self._count = 0 # keeps track of total number of words in the Vocab

    # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
    for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
      self._word_to_id[w] = self._count
      self._id_to_word[self._count] = w
      self._count += 1

    # Read the vocab file and add words up to max_size
    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        pieces = line.split()
        if len(pieces) != 2:
          print("Warning: incorrectly formatted line in vocabulary file: %s\n" % line)
          continue
        w = pieces[0]
        if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
          raise Exception('<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
        if w in self._word_to_id:
          raise Exception('Duplicated word in vocabulary file: %s' % w)
        self._word_to_id[w] = self._count
        self._id_to_word[self._count] = w
        self._count += 1
        if max_size != 0 and self._count >= max_size:
          print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
          return self._word_to_id
          #break
    print("Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1]))
    return self._word_to_id



def example_generator(data_path, single_pass):
  while True:
    #print("data_path = ",data_path)
    filelist = glob.glob(data_path) # get the list of datafiles
    assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
    if single_pass:
      filelist = sorted(filelist)
      # print("filelist: ", filelist)
    else:
      # random.shuffle(filelist)
      filelist = sorted(filelist)
  
    files_names_list = []
    files_names_list2 = []
    #print("total filelist = ",len(filelist))
    total_samples_count = 0
    for f in filelist:
      # print("file is: ", f)
      reader = open(f, 'rb')
      tkn = f.split("/")[-1]
      files_names_list.append(tkn)
      sample_count = 0
      while True:
        len_bytes = reader.read(8)
        if not len_bytes: break # finished reading this file
        sample_count+=1
        total_samples_count+=1
        files_names_list2.append(tkn)
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        #print("f:",f,sample_count,total_samples_count,tkn)

        yield example_pb2.Example.FromString(example_str)

    #print("One-Go__files_names_list : ",files_names_list, len(files_names_list), len(set(files_names_list)))
    #print("\n")
    #print("One-Go__files_names_list2 : ",set(files_names_list2), len(files_names_list2), len(set(files_names_list2)))
    #print("\n")

    if single_pass:
      #print("example_generator completed reading all datafiles. No more data.",total_samples_count,sample_count)
      #print("files_names_list : ",files_names_list, len(files_names_list), len(set(files_names_list)))
      break



def article2ids(article_words, vocab):
  ids = []
  oovs = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in article_words:
    i = vocab.word2id(w)
    if i == unk_id: # If w is OOV
      if w not in oovs: # Add to list of OOVs
        oovs.append(w)
      oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
      ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
    else:
      ids.append(i)
  return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
  ids = []
  unk_id = vocab.word2id(UNKNOWN_TOKEN)
  for w in abstract_words:
    i = vocab.word2id(w)
    if i == unk_id: # If w is an OOV word
      if w in article_oovs: # If w is an in-article OOV
        vocab_idx = vocab.size() + article_oovs.index(w) # Map to its temporary article OOV number
        ids.append(vocab_idx)
      else: # If w is an out-of-article OOV
        ids.append(unk_id) # Map to the UNK token id
    else:
      ids.append(i)
  return ids


def outputids2words(id_list, vocab, article_oovs):
  words = []
  for i in id_list:
    try:
      w = vocab.id2word(i) # might be [UNK]
    except ValueError as e: # w is OOV
      assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
      article_oov_idx = i - vocab.size()
      try:
        w = article_oovs[article_oov_idx]
      except ValueError as e: # i doesn't correspond to an article oov
        raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
    words.append(w)
  return words


def abstract2sents(abstract):
  cur = 0
  sents = []
  while True:
    try:
      start_p = abstract.index(SENTENCE_START, cur)
      end_p = abstract.index(SENTENCE_END, start_p + 1)
      cur = end_p + len(SENTENCE_END)
      sents.append(abstract[start_p+len(SENTENCE_START):end_p])
    except ValueError as e: # no more sentences
      return sents


def show_art_oovs(article, vocab):
  unk_token = vocab.word2id(UNKNOWN_TOKEN)
  words = article.split(' ')
  words = [("__%s__" % w) if vocab.word2id(w)==unk_token else w for w in words]
  out_str = ' '.join(words)
  return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
  unk_token = vocab.word2id(UNKNOWN_TOKEN)
  words = abstract.split(' ')
  new_words = []
  for w in words:
    if vocab.word2id(w) == unk_token: # w is oov
      if article_oovs is None: # baseline mode
        new_words.append("__%s__" % w)
      else: # pointer-generator mode
        if w in article_oovs:
          new_words.append("__%s__" % w)
        else:
          new_words.append("!!__%s__!!" % w)
    else: # w is in-vocab word
      new_words.append(w)
  out_str = ' '.join(new_words)
  return out_str





