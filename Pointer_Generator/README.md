* We used a modified fork of [pointer_generator model implementation](https://github.com/atulkum/pointer_summarizer) for our experiments.

## Installations
```
pip install -r requirements.txt

```

### 1. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
```
export CLASSPATH=/path/to/stanford-corenlp-full-2016-10-31/stanford-corenlp-3.7.0.jar
```
replacing `/path/to/` with the path to where you saved the `stanford-corenlp-full-2016-10-31` directory. You can check if it's working by running
```
echo "Please tokenize this text." | java edu.stanford.nlp.process.PTBTokenizer
```
You should see something like:
```
Please
tokenize
this
text
.
PTBTokenizer tokenized 5 tokens at 68.97 tokens per second.
```

### 2. Data formate:
```
python3 make_PG_data.py
```

This will create a `pg_data` directory which process into .bin and vocab files.

#### How to run
##### Training & Validation
```
python3 train_Telugu.py >> log_txt_files/PG_train_val.txt
```
#### Testing:

* After finding the best model by checking which iteration is converged (lets say ```model_30000_1629407459```), evaluate it on test data:
```
python3 decode_m1_latest.py log_models/train_1617467291/model/model_30000_1629407459 
```
This will generate decoded test summaries in a directory (lets say ```decode_model_30000_1629407459```) and ```pg_rouge_scores.csv```.

