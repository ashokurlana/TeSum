We use a modified fork of [huggingface transformers](https://github.com/huggingface/transformers) for our experiments.

#### Install environment

```sh
pip install -r requirements.txt
```

### Data format:
```
python3 data_format.py
```

This script will create a `train, dev, test csv files'


### Run the script

To run the IndicBART_XLSUM you can use the `run.sh` script. 

Note: Change the model name from IndicBART_XLSUM to IndicBARTSS to finetune the IndicBART model on TeSum corpus.

```sh
sh run.sh
```


