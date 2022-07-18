import pandas as pd
import csv
from tqdm import tqdm
import json

def read_jsonl(filename):
    pairs = []
    for line in open(filename, 'r', encoding='utf-8'):
        pairs.append(json.loads(line))
    if len(pairs)==1:
        pairs = pairs[0]
    print("\n# Samples: ", len(pairs))
    print("# Keys: ", pairs[0].keys())
    return pairs

def jsonl_to_csv(set_name):
    pairs = read_jsonl(str(set_name)+".jsonl")
    samples_data = []
    for i in tqdm(range(len(pairs))):
        text = pairs[i]['cleaned_text']
        summary = pairs[i]['summary']
        sample = {"text":str(text), "summary":str(summary)}
        samples_data.append(sample)
    to_csv = samples_data
    keys = to_csv[0].keys()
    with open(str(set_name)+'.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(to_csv)
    return str(set_name)+'.csv'

def main():
	set_names = ['train','dev', 'test']
	for set_name in set_names:
		jsonl_to_csv(set_name)

if __name__ == '__main__':
    main()