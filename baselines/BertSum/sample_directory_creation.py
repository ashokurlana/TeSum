import os
import json
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

def sample_data_creation(setnames, paths, save_path):
	for n, name in enumerate(setnames):
		c=0
		data_samples = read_jsonl(paths[n])
		print("%s set samples count  = %d "%(name,len(data_samples)))
		print("\n")
		data_path = save_path+str(name)+"/"
		for i in range(len(data_samples)):
			if not os.path.exists(data_path+str(i)): os.makedirs(data_path+str(i))		
			fname_i = open(data_path+str(i)+"/"+str(i)+".sent.txt","w")
			fname_o = open(data_path+str(i)+"/"+str(i)+".summ.sent.txt","w")
			
			article = data_samples[i]['cleaned_text']
			summary = data_samples[i]['summary']

			fname_i.write(str(article))
			fname_o.write(str(summary))
			c+=1
		print("Total number of samples in %s are: %d"%(name, c))

def main():
	parser = argparse.ArgumentParser(description='Sample Directory Creation')
	parser.add_argument('-train_path', type=str, default="tesum_data/train.jsonl", help='training data path')
	parser.add_argument('-dev_path', type=str, default="tesum_data/dev.jsonl", help='validation data path') 
	parser.add_argument('-test_path', type=str, default="tesum_data/test.jsonl", help='test data path')
	parser.add_argument('-save_path', type=str, default="data_samples/", help='sample data save path')
	args = parser.parse_args()
	setnames = ["train", "dev", "test"]
	paths = [args.train_path, args.dev_path, args.test_path]
	sample_data_creation(setnames, paths, args.save_path)

if __name__ == '__main__':
    main()