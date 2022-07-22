import os
import glob
import json
import shutil
import argparse
import re
from tqdm import tqdm

def get_contents(line):
    
    #print(line.keys())
    return line["summary"], line["cleaned_text"]

def extract_data(input_dir, output_dir):
    multilingual_dir = os.path.join(output_dir,"multilingual")
    os.makedirs(multilingual_dir, exist_ok=True)

    f_iterator = glob.glob(os.path.join(input_dir,"*.jsonl"))
    
    for input_file in tqdm(f_iterator):
        print(input_file)
        lang = "_".join(os.path.basename(input_file).rsplit("_")[:-1])
        lang_dir = os.path.join(output_dir, "individual", lang)
        os.makedirs(lang_dir, exist_ok=True)

        source_file = os.path.join(lang_dir,os.path.basename(input_file).replace(".jsonl", ".source").rsplit("_", 1)[1])

        target_file = os.path.join(lang_dir,os.path.basename(input_file).replace(".jsonl", ".target").rsplit("_", 1)[1])
        # print("reached_here")
        # print(input_file)
        with open(input_file, 'r', encoding='utf-8') as inpf:
            with open(source_file, 'w') as srcf, open(target_file, 'w') as tgtf:
                for line in inpf:
                    obj = json.loads(line)
                    # for key, val in obj.items():
                    summary, text = get_contents(obj)
                    summary = re.sub('[\r\n]+', ' ', summary)
                    text = re.sub('[\r\n]+', ' ', text)
                    if summary.strip()!="" and text.strip()!="":
                        print(text, file=srcf)
                        print(summary, file=tgtf)

        if source_file.endswith("train.source"):
            shutil.copy(source_file,os.path.join(multilingual_dir,lang + "_" + os.path.basename(source_file)))

            shutil.copy(target_file,os.path.join(multilingual_dir,lang + "_" + os.path.basename(target_file)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', type=str,required=True,metavar='PATH',help="Input directory")
    parser.add_argument('--output_dir', '-o', type=str,required=True,metavar='PATH',help="Output directory")
    args = parser.parse_args()
    extract_data(args.input_dir, args.output_dir)
