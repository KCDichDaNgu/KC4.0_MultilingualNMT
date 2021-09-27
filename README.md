Dự án MultilingualMT-UET-KC4.0 là dự án open-source được phát triển bởi nhóm UETNLPLab.

# Setup
## Cài đặt công cụ Multilingual-NMT

**Note**:
Lưu ý:
Phiên bản hiện tại chỉ tương thích với python>=3.6
```bash
git clone https://github.com/KCDichDaNgu/MultilingualMT-UET-KC4.0.git
cd MultilingualMT-UET-KC4.0
pip install -r requirements.txt

# Quickstart

```

## Bước 1: Chuẩn bị dữ liệu

Ví dụ thực nghiệm dựa trên cặp dữ liệu Anh-Việt nguồn từ iwlst với 133k cặp câu:

```bash
cd data/en-vi
```

Dữ liệu bao gồm câu nguồn (`src`) và câu đích (`tgt`) dữ liệu đã được tách từ:

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
Lưu ý:
- Dữ liệu trước khi đưa vào huấn luyện cần phải được tokenize. 
- $CONFIG là đường dẫn tới vị trí chứa file config

Tách dữ liệu dev để tính toán hội tụ trong quá trình huấn luyện, thường không lớn hơn 5k câu.

```text
$ head -n 3 data/iwlst_en_vi/train.en
It is not acceptable that , with the help of the national bureaucracies , Parliament &apos;s legislative prerogative should be made null and void by means of implementing provisions whose content , purpose and extent are not laid down in advance .
Federal Master Trainer and Senior Instructor of the Italian Federation of Aerobic Fitness , Group Fitness , Postural Gym , Stretching and Pilates; from 2004 , he has been collaborating with Antiche Terme as personal Trainer and Instructor of Stretching , Pilates and Postural Gym .
&quot; Two soldiers came up to me and told me that if I refuse to sleep with them , they will kill me . They beat me and ripped my clothes .
```

## Bước 2: Huấn luyện mô hình

Để huấn luyện một mô hình mới **hãy chỉnh sửa file YAML config**:
Cần phải sửa lại file config en_vi.yml chỉnh siêu tham số và đường dẫn tới dữ liệu huấn luyện:

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

Sau đó có thể chạy với câu lệnh:

```bash
python -m bin.main train --model Transformer --model_dir $MODEL/en-vi.model --config $CONFIG/en_vi.yml
```

**Note**:
Ở đây:
- $MODEL là dường dẫn tới vị trí lưu mô hình. Sau khi huấn luyện mô hình, thư mục chứa mô hình bao gồm mô hình huyến luyện, file config, file log, vocab.
- $CONFIG là đường dẫn tới vị trí chứa file config

## Bước 3: Dịch 

Mô hình dịch dựa trên thuật toán beam search và lưu bản dịch tại `$your_data_path/translate.en2vi.vi`.

```bash
python -m bin.main infer --model Transformer --model_dir $MODEL/en-vi.model --features_file $your_data_path/tst2012.en --predictions_file $your_data_path/translate.en2vi.vi
```

## Bước 4: Đánh giá chất lượng dựa trên điểm BLEU

Đánh giá điểm BLEU dựa trên multi-bleu

```bash
perl thrid-party/multi-bleu.perl $your_data_path/translate.en2vi.vi < $your_data_path/tst2012.vi
```

|        MODEL       | BLEU (Beam Search) |
| :-----------------:| :----------------: |
| Transformer (Base) |        28.54       |
