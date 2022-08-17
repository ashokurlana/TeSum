* We used a modified fork of [ML_RL model implementation](https://github.com/rohithreddy024/Text-Summarizer-Pytorch) for our experiments.


## Installations
```
pip install -r requirements.txt
```

## Data preprocessing

```
python data_preparation.py
```
You will see `telugu_data` directory.

```
python make_data_files_Telugu_ML_RL.py
```
You will see `telugu_data/finished_data` directory.

## Training
* As suggested in [Paulus et al. (2018)](https://arxiv.org/pdf/1705.04304.pdf), first pretrain the seq-to-seq model using MLE (with Python 3):
```
python train_tel.py --train_mle=yes --train_rl=no --mle_weight=1.0
```
* Next, find the best saved model on validation data by running (with Python 3):
```
python eval_tel.py --task=validate --start_from=0005000.tar
```
* After finding the best model (lets say ```0100000.tar```) with high rouge-l f score, load it and run (with Python 3):
```
python train_tel.py --train_mle=yes --train_rl=yes --mle_weight=0.25 --load_model=0100000.tar --new_lr=0.0001
```
for MLE + RL training (or)
```
python train_tel.py --train_mle=no --train_rl=yes --mle_weight=0.0 --load_model=0100000.tar --new_lr=0.0001
```
for RL training

## Validation
* To perform validation of RL training, run (with Python 3):
```
python eval_tel.py --task=validate --start_from=0100000.tar 
```
## Testing
* After finding the best model of RL training (lets say ```0200000.tar```),evaluate it on test data & get all rouge metrics by running (Python 3):
```
python eval_tel.py --task=test --load_model=0200000.tar
```

### Decode summaries from the pretrained model
Download the pretrained model from(yet to update) and put it in `saved_models` directory. You will also need a preprocessed version of the TeSum dataset.  Please follow the `installations` and `Data preprocessing` instructions. 

After that, Run
```
python eval_tel.py --task=test --load_model=0014000.tar
```

