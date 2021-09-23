import re, os
import nltk
from nltk.corpus import wordnet
import dill as pickle
import pandas as pd
from torchtext import data

def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

# get_synonym  replace word with any synonym found among src
def get_synonym(word, SRC):
  syns = wordnet.synsets(word)
  for s in syns:
    for l in s.lemmas():
      if SRC.vocab.stoi[l.name()] != 0:
        return SRC.vocab.stoi[l.name()]
                
  return 0

class Tokenizer:
    def __init__(self, lang=None):
        if(lang is not None):
            self.nlp = spacy.load(lang)
            self.tokenizer_fn = self.nlp.tokenizer
        else:
            self.tokenizer_fn = lambda l: l.strip().split()
            
    def tokenize(self, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.tokenizer_fn(sentence) if tok.text != " "]


def read_data(src_file, trg_file):
    src_data = open(src_file).read().strip().split('\n')

    trg_data = open(trg_file).read().strip().split('\n')
 
    return src_data, trg_data


def read_file(file_dir):
    f = open(file_dir, 'r')
    data =  f.read().strip().split('\n')
    return data

def write_file(file_dir, content):
    f = open(file_dir, "w")
    f.write(content)
    f.close()

def create_fields(src_lang, trg_lang):
    
    #print("loading spacy tokenizers...")
    #
    #t_src = tokenize(src_lang)
    #t_trg = tokenize(trg_lang)
    t_src_tokenizer = t_trg_tokenizer = lambda x: x.strip().split()

    TRG = data.Field(lower=True, tokenize=t_trg_tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src_tokenizer)
        
    return SRC, TRG

def create_dataset(src_data, trg_data, max_strlen, batchsize, device, SRC, TRG, istrain=True):

    print("creating dataset and iterator... ")

    raw_data = {'src' : [line for line in src_data], 'trg': [line for line in trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    
    mask = (df['src'].str.count(' ') < max_strlen) & (df['trg'].str.count(' ') < max_strlen)
    df = df.loc[mask]

    df.to_csv("translate_transformer_temp.csv", index=False)
    
    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    train_iter = MyIterator(train, batch_size=batchsize, device=device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=istrain, shuffle=True)
    
    os.remove('translate_transformer_temp.csv')
    
    if istrain:
        SRC.build_vocab(train)
        TRG.build_vocab(train)

    return train_iter

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

def generate_language_token(lang: str):
    return '<{}>'.format(lang.strip())