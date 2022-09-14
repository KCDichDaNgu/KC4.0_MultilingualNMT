The open-source Multilingual-UET-KC4.0 is developed by UETNLPLab.

# Setup

**Note**:
Note:
Multilingual-UET-KC4.0 requires:
 - python >= 3.6 
 - torch 1.8.0+

```bash
git clone https://github.com/KCDichDaNgu/KC4.0_MultilingualNMT.git
cd KC4.0_MultilingualNMT
pip install -r requirements.txt

# Quickstart

```

## Step 1: Data Preparation

Iwslt's English-Vietnamese parallel corpus contains 133k sentence pairs:

```bash
cd data/iwslt_en_vi
```

The dataset contains the source and target sentences, which were all tokenized:

* `train.en`
* `train.vi`
* `tst2012.en`
* `tst2012.vi`

| Data set    | Sentences  |                    Download                   |
| :---------: | :--------: | :-------------------------------------------: |
| Training    | 133,317    | via GitHub or located in data/train-en-vi.tgz |
| Development | 1,553      | via GitHub or located in data/train-en-vi.tgz |
| Test        | 1,268      | via GitHub or located in data/train-en-vi.tgz |


**Note**:
Note:
- Before training the NMT model, the dataset should be tokenized. 
- $CONFIG is the path containing the config file.

The dev set (tst2012) contains 1553 sentence pairs utilized to compute the coverage of the model.

```text
$ head -n 5 data/iwslt_en_vi/train.en
Rachel Pike : The science behind a climate headline
In 4 minutes , atmospheric chemist Rachel Pike provides a glimpse of the massive scientific effort behind the bold headlines on climate change , with her team -- one of thousands who contributed -- taking a risky flight over the rainforest in pursuit of data on a key molecule .
I &apos;d like to talk to you today about the scale of the scientific effort that goes into making the headlines you see in the paper .
Headlines that look like this when they have to do with climate change , and headlines that look like this when they have to do with air quality or smog .
They are both two branches of the same field of atmospheric science .
```

## Step 2: Train a new model

Please change the file config YML to train a new model **YAML config**:
If you don't change the file config en_vi.yml, the model will be trained by default hyper-parameters:

```yaml
# data location and config section
data:
  train_data_location: data/iwslt_en_vi/train
  eval_data_location:  data/iwslt_en_vi/tst2013
  src_lang: .en 
  trg_lang: .vi 
log_file_models: 'model.log'
lowercase: false
build_vocab_kwargs: # additional arguments for build_vocab. See torchtext.vocab.Vocab for mode details
#  max_size: 50000
  min_freq: 5
# model parameters section
device: cuda
d_model: 512
n_layers: 6
heads: 8
# inference section
eval_batch_size: 8
decode_strategy: BeamSearch
decode_strategy_kwargs:
  beam_size: 5 # beam search size
  length_normalize: 0.6 # recalculate beam position by length. Currently only work in default BeamSearch
  replace_unk: # tuple of layer/head attention to replace unknown words
    - 0 # layer
    - 0 # head
input_max_length: 200 # input longer than this value will be trimmed in inference. Note that this values are to be used during cached PE, hence, validation set with more than this much tokens will call a warning for the trimming.
max_length: 160 # only perform up to this much timestep during inference
train_max_length: 50 # training samples with this much length in src/trg will be discarded
# optimizer and learning arguments section
lr: 0.2
optimizer: AdaBelief
optimizer_params:
  betas:
    - 0.9 # beta1
    - 0.98 # beta2
  eps: !!float 1e-9
n_warmup_steps: 4000
label_smoothing: 0.1
dropout: 0.1
# training config, evaluation, save & load section
batch_size: 64
epochs: 20
printevery: 200
save_checkpoint_epochs: 1
maximum_saved_model_eval: 5
maximum_saved_model_train: 5

```

Please run the below command to train the model:

```bash
python -m bin.main train --model Transformer --model_dir $MODEL/en-vi.model --config $CONFIG/en_vi.yml
```

**Note**:
Where:
- $MODEL is the path that saves the model. After the trained model, the directory contains models, file config, file log, and vocabulary.
- $CONFIG is the path containing the config file.

## Step 3: Inference 

The beam search algorithm is utilized during inference, and the translated file is saved at `$your_data_path/translate.en2vi.vi`.

```bash
python -m bin.main infer --model Transformer --model_dir $MODEL/en-vi.model --features_file $your_data_path/tst2012.en --predictions_file $your_data_path/translate.en2vi.vi
```

## Step 4: Evaluation BLEU score

Multi-bleu BLEU is utilized to evaluate quality.

```bash
perl thrid-party/multi-bleu.perl $your_data_path/translate.en2vi.vi < $your_data_path/tst2012.vi
```

|        MODEL       | BLEU (Beam Search) |
| :-----------------:| :----------------: |
| Transformer (Base) |        25.64       |


## The details refer to 
[nmtuet.ddns.net](http://nmtuet.ddns.net:1190/)

## If you have any feedback, please send to mail kcdichdangu@gmail.com

## Citations:

```bash
@inproceedings{ViNMT2022,
  title = {ViNMT: Neural Machine Translation Toolkit},
  author = {Nguyen Hoang Quan, Nguyen Thanh Dat, Nguyen Hoang Minh Cong, Nguyen Van Vinh, Ngo Thi Vinh, Nguyen Phuong Thai, Tran Hong Viet},
  booktitle = {https://arxiv.org/abs/2112.15272},
  year = {2022},
}
```