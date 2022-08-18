import json
import os
import argparse

# Read a jsonl file into list-of-dictionaries
def read_jsonl(filename):
    pairs = []
    for line in open(filename, 'r', encoding='utf-8'):
            pairs.append(json.loads(line))
    if len(pairs) == 1:
        pairs = pairs[0]
    print("\n# Samples: ", len(pairs))
    print("# Keys: ", pairs[0].keys())
    return pairs

def make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# unfinished_path = "telugu_data/unfinished/"
# make_folder(unfinished_path)
def make_data_format(train_path, dev_path, test_path, save_path):
	set_names = ['train', 'dev', 'test']
	for name in set_names:
		print("Started preparing %s data"%name)
		if name == 'train':
			# path = "tesum_data/"+str(name)+'.jsonl'
			data_samples = read_jsonl(train_path)
			with open(save_path+'train.article.txt', 'w', encoding='utf-8') as train_text:
				for i in range(len(data_samples)):
					text = data_samples[i]['cleaned_text']
					train_text.write(text+"\n")
			with open(save_path+'train.title.txt' , 'w', encoding='utf-8') as train_summ:
				for i in range(len(data_samples)):
					summary = data_samples[i]['summary']
					train_summ.write(summary+"\n")
		elif name == 'dev':
			# path = 'tesum_data/'+str(name)+'.jsonl'
			data_samples = read_jsonl(dev_path)
			with open(save_path+'valid.article.filter.txt', 'w', encoding='utf-8') as valid_text:
				for i in range(len(data_samples)):
					text = data_samples[i]['cleaned_text']
					valid_text.write(text+"\n")
			with open(save_path+'valid.title.filter.txt', 'w', encoding='utf-8') as valid_summ:
				for i in range(len(data_samples)):	
					summary = data_samples[i]['summary']
					valid_summ.write(summary+"\n")
		else:
			# path = 'tesum_data/'+str(name)+'.jsonl'
			data_samples = read_jsonl(test_path)
			with open(save_path+'test.article.filter.txt', 'w', encoding='utf-8') as test_text:
				for i in range(len(data_samples)):
					text = data_samples[i]['cleaned_text']
					test_text.write(text+"\n")
			with open(save_path+'test.title.filter.txt', 'w', encoding='utf-8') as test_summ:
				for i in range(len(data_samples)):
					summary = data_samples[i]['summary']
					test_summ.write(summary+"\n")

def main():
	parser = argparse.ArgumentParser(description='ML_RL data formation')
	parser.add_argument('-train_path', type=str, default="tesum_data/train.jsonl", help='training data path')
	parser.add_argument('-dev_path', type=str, default="tesum_data/dev.jsonl", help='validation data path') 
	parser.add_argument('-test_path', type=str, default="tesum_data/test.jsonl", help='test data path')
	parser.add_argument('-save_path', type=str, default="telugu_data/unfinished/", help='data save path')
	args = parser.parse_args()

	make_folder(args.save_path)
	make_data_format(args.train_path, args.dev_path, args.test_path, args.save_path)

if __name__ == '__main__':
    main()
