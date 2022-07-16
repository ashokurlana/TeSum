### (if the standford corenlp didn't work) run this command -->  export CLASSPATH=/home/priyanka/Desktop/bin_file_creation/stanford-corenlp-latest/stanford-corenlp-4.1.0/stanford-corenlp-4.1.0.jar 

import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'


finished_files_dir = "finished_files_m1"
chunks_dir = os.path.join(finished_files_dir, "chunked")
VOCAB_SIZE = 200000
CHUNK_SIZE = 250 # num examples per chunk, for the chunked data


### For Get to the point formate:
def chunk_file(set_name):
  in_file = finished_files_dir+'/%s.bin' % set_name
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1

def chunk_all():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['train', 'val', 'test']:
    print("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name)
  print("Saved chunked data in %s" % chunks_dir)

def tokenize_stories(stories_dir, tokenized_stories_dir):
  """
  Maps a whole directory of .txt files to a tokenized version using Stanford CoreNLP Tokenizer
  """
  print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
  stories = os.listdir(stories_dir)
  print("stories len = %d"%len(stories))

  # make IO list file
  print("Making list of files to tokenize...")
  with open("mapping.txt", "w") as f:
    for s in stories:
      f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, s)))

  command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
  print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
  subprocess.call(command)
  print("Stanford CoreNLP Tokenizer has finished.")
  #os.remove("mapping.txt")

  # Check that the tokenized stories directory contains the same number of files as the original directory
  num_orig = len(os.listdir(stories_dir))
  num_tokenized = len(os.listdir(tokenized_stories_dir))
  print("num_orig = %d, num_tokenized = %d"%(num_orig, num_tokenized))
  if num_orig != num_tokenized:
    raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
  print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def get_art_abs(story_file, story_out_file):
  ilines = read_text_file(story_file)
  olines = read_text_file(story_out_file)


  # Lowercase everything
  ilines = [line.lower() for line in ilines]
  olines = [line.lower() for line in olines]

  # Put periods on the ends of lines that are missing them:
  ilines = [fix_missing_period(line) for line in ilines]
  olines = [fix_missing_period(line) for line in olines]

  # Separate out article and abstract sentences
  article_lines = []
  abs_lines = []
  next_is_highlight = False

  for idx,line in enumerate(ilines):
    if line == "":
      continue # empty line
    else:
      article_lines.append(line)

  for idx,line in enumerate(olines):
    if line == "":
      continue # empty line
    else:
      abs_lines.append(line)

  # Make article into a single string
  article = ' '.join(article_lines)

  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in abs_lines])

  return article, abstract


def write_to_bin(url_file, url_output_file, out_file, makevocab=False):
  print("Making bin file for files listed in %s..." % url_file)

  story_fname = os.listdir(url_file)
  story_fname.sort()
  ostory_fname = os.listdir(url_output_file)
  ostory_fname.sort()
  num_stories = len(story_fname)

  if makevocab:
    vocab_counter = collections.Counter()

  with open(out_file, 'wb') as writer:
    for idx,s in enumerate(story_fname):
      if idx % 1000 == 0:
        print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))

      # Look in the tokenized story dirs to find the .story file corresponding to this url
      if os.path.isfile(os.path.join(url_file, s)):
        story_file = os.path.join(url_file, s)

        story_out_file = os.path.join(url_output_file, s[:-8]+'summ.sent.txt') ### Cross check THIS, if you get any error in imbalance of samples:
        story_out_file = os.path.join(url_output_file,ostory_fname[idx]) ## story_fname == ostory_fname both are in sorted same aligned order.

        print("story_file =%s, story_out_file=%s "%(story_file,story_out_file))
      else:
        print("Error: Couldn't find tokenized story file %s in tokenized story directories %s. Was there an error during tokenization?" % (s, url_file))
        # Check again if tokenized stories directories contain correct number of files
        print("Checking that the tokenized stories directory %s contain correct number of files..." % (url_file))
        raise Exception("Tokenized stories directory %s contain correct number of files but story file %s found in neither." % (url_file, s))
		
      # Get the strings to write to .bin file
      article, abstract = get_art_abs(story_file, story_out_file)

      # Write to tf.Example
      tf_example = example_pb2.Example()

      #tf_example.features.feature['article'].bytes_list.value.extend([article])######## for python2
      #tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])

      tf_example.features.feature['article'].bytes_list.value.extend([article.encode()])##### for python3
      tf_example.features.feature['abstract'].bytes_list.value.extend([abstract.encode()])

      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

      # Write the vocab to file, if applicable
      if makevocab:
        art_tokens = article.split(' ')
        abs_tokens = abstract.split(' ')
        abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
        tokens = art_tokens + abs_tokens
        tokens = [t.strip() for t in tokens] # strip
        tokens = [t for t in tokens if t!=""] # remove empty
        vocab_counter.update(tokens)
  print("Finished writing file %s\n" % out_file)

  # write vocab to file
  if makevocab:
    print("Writing vocab file...")
    with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print("Finished writing vocab file")

def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


############################################################################## Input directories:
itrain_dir = "../data/In_files/zi_train_dir"
ival_dir = "../data/In_files/zi_val_dir"
itest_dir = "../data/In_files/zi_test_dir"

otrain_dir = "../data/Out_files/zo_train_dir"
oval_dir = "../data/Out_files/zo_val_dir"
otest_dir = "../data/Out_files/zo_test_dir"


############################################################################# Tokenized Directories:
itrain_tkd_dir = "Tokenized_dir/zi_train_tkd_dir"
ival_tkd_dir = "Tokenized_dir/zi_val_tkd_dir"
itest_tkd_dir = "Tokenized_dir/zi_test_tkd_dir"

otrain_tkd_dir = "Tokenized_dir/zo_train_tkd_dir"
oval_tkd_dir = "Tokenized_dir/zo_val_tkd_dir"
otest_tkd_dir = "Tokenized_dir/zo_test_tkd_dir"



#########################################################################################################


if __name__ == '__main__':

  # Create some new directories
  if not os.path.exists(itrain_tkd_dir): os.makedirs(itrain_tkd_dir)
  if not os.path.exists(itest_tkd_dir): os.makedirs(itest_tkd_dir)
  if not os.path.exists(ival_tkd_dir): os.makedirs(ival_tkd_dir)

  if not os.path.exists(otrain_tkd_dir): os.makedirs(otrain_tkd_dir)
  if not os.path.exists(otest_tkd_dir): os.makedirs(otest_tkd_dir)
  if not os.path.exists(oval_tkd_dir): os.makedirs(oval_tkd_dir)

  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  #Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
  tokenize_stories(ival_dir, ival_tkd_dir)
  tokenize_stories(itrain_dir, itrain_tkd_dir)
  tokenize_stories(itest_dir, itest_tkd_dir)


  tokenize_stories(oval_dir, oval_tkd_dir)
  tokenize_stories(otrain_dir, otrain_tkd_dir)
  tokenize_stories(otest_dir, otest_tkd_dir)
 
  #Read the tokenized stories, do a little postprocessing then write to bin files ##[ with tokenized files]
  write_to_bin(itest_tkd_dir, otest_tkd_dir, os.path.join(finished_files_dir, "test.bin"))
  write_to_bin(ival_tkd_dir, oval_tkd_dir, os.path.join(finished_files_dir, "val.bin"))
  write_to_bin(itrain_tkd_dir, otrain_tkd_dir, os.path.join(finished_files_dir, "train.bin"), makevocab=True)


  # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks

  chunk_all()

  ################################## Completed..!! ################################################








