## Telugu Abstractive Summarization
This repository contain TeSum dataset and benchmark models implementation for Telugu language.


### Step 1 Download the data
Download the TeSum dataset from ```tesum directory```

### Step 2 preprocessing of the data
Run 
```
 python sample_directory_creation.py
```
```
 python models_data_formates.py
````
 
You will see `data_samples` and `data` directories will be created. These will be used further to run benchmark models on Tesum data.


.
## Benchmark Models for Telugu Abstractive Summarization

#### Pointer_Generator : [ Get To The Point: Summarization with Pointer-Generator Networks ](https://arxiv.org/pdf/1704.04368.pdf)
#### ML_RL: [A Deep Reinforced Model For Abstractive Summarization ](https://arxiv.org/pdf/1705.04304.pdf)
#### BERTSum : [Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf)
#### mT5 : [XL-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages](https://aclanthology.org/2021.findings-acl.413.pdf)




You can see the individual directories for above mentioned models. Go to the respective model directory and follow the corresponding instructions to setup the models.

##### Note:  All the experiments were performed on a single NVIDIA GeForce GTX 1080 GPU.

## Intrinsic Evaluation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/googlecolab/colabtools/bl)

## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />Contents of this repository are restricted to non-commerical research purpose under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. Copyright of dataset contents belongs to original copyright holder.
