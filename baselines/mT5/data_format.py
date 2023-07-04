import pandas as pd
import csv
from tqdm import tqdm
import json
import argparse

def read_jsonl(filename):
    pairs = []
    for line in open(filename, 'r', encoding='utf-8'):
        pairs.append(json.loads(line))
    if len(pairs)==1:
        pairs = pairs[0]
    print("\n# Samples: ", len(pairs))
    print("# Keys: ", pairs[0].keys())
    return pairs

def jsonl_to_csv(paths):
    for n in range(len(paths)):
        pairs = read_jsonl(paths[n])
        samples_data = []
        for i in tqdm(range(len(pairs))):
            text = pairs[i]['cleaned_text']
            summary = pairs[i]['summary']
            sample = {"text":str(text), "summary":str(summary)}
            samples_data.append(sample)
        to_csv = samples_data
        keys = to_csv[0].keys()
        set_name = paths[n].split('/')[-1].split('.')[0]
        print(set_name)
        with open(str(set_name)+'.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(to_csv)
    return str(set_name)+'.csv'

def main():
    parser = argparse.ArgumentParser(description="mBART data preparation")
    parser.add_argument('-train_data', type=str, default="tesum_data/train.jsonl", help="train data path")
    parser.add_argument('-dev_data', type=str, default="tesum_data/dev.jsonl", help="validation data path")
    parser.add_argument('-test_data', type=str, default="tesum_data/test.jsonl", help="test data path")
    args = parser.parse_args()
    paths =[args.train_data, args.dev_data, args.test_data]
    jsonl_to_csv(paths)

if __name__ == '__main__':
    main()