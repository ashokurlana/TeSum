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

### Train the Adapter

To train the adapter with mBART-large-50 you can use the `train.sh` script

```sh
sh train.sh
```
You will see the `summarization` directory in the `tmp/outputs` path. 
### Testing

To perform the testing with trained adapter you can use the `test.sh` script. To load the trained adapter pass the generated summarization directory path.

```sh
sh test.sh
```


