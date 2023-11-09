import io, os
"""
dill extends Python's pickle module for serializing and de-serializing Python objects to the majority of the built-in Python types.
"""
import dill as pickle
import torch
from torch.utils.data import DataLoader
"""
The torchtext.data module provides the following:

  Ability to define a preprocessing pipeline
  Batching, padding, and numericalizing (including building a vocabulary object)
  Wrapper for dataset splits (train, validation, test)
  Loader a custom NLP dataset

BucketIterator:

  Defines an iterator that batches examples of similar lengths together.
  Minimizes amount of padding needed while producing freshly shuffled batches for each new epoch. See pool for the bucketing procedure used.

Field:
  Defines a datatype together with instructions for converting to Tensor.
  Field class models common text processing datatypes that can be represented by tensors. It 
  holds: 
  - a Vocab object that defines the set of possible values for elements of the field and 
  - their corresponding numerical representations. 
  The Field object also holds other parameters relating to how a datatype should be numericalized, 
  such as a tokenization method and the kind of Tensor that should be produced.
  If a Field is shared between two columns in a dataset (e.g., question and answer in a QA dataset), then they will have a shared vocabulary.

"""
from torchtext.data import BucketIterator, Dataset, Example, Field
"""
torchtext.datasets holds some dataset available of pytorch

TranslationDataset: Create a TranslationDataset given paths and fields:
  path – Common prefix of paths to the data files for both languages.
  exts – A tuple containing the extension to path for each language.
  fields – A tuple containing the fields that will be used for data in each language.
  keyword arguments (Remaining) – Passed to the constructor of data.Dataset.
"""
from torchtext.datasets import TranslationDataset, Multi30k, IWSLT, WMT14
from collections import Counter

import modules.constants as const
from utils.save import load_vocab_from_path

class DefaultLoader:
  def __init__(self, train_path_or_name, language_tuple=None, valid_path=None, eval_path=None, option=None):
    """Load training/eval data file pairing, process and create data iterator for training """
    self._language_tuple = language_tuple
    self._train_path = train_path_or_name
    self._eval_path = eval_path
    self._option = option

  @property
  def language_tuple(self):
    """DefaultLoader will use the default lang option @bleu_batch_iter <sos>, hence, None"""
    return None, None

  def tokenize(self, sentence):
    """
    Simply divide sentence into list of words
    """
    return sentence.strip().split()

  def detokenize(self, list_of_tokens):
    """Differentiate between [batch, len] and [len]; joining tokens back to strings"""
    if( len(list_of_tokens) == 0 or isinstance(list_of_tokens[0], str)):
      # [len], single sentence version
      return " ".join(list_of_tokens)
    else:
      # [batch, len], batch sentence version
      return [" ".join(tokens) for tokens in list_of_tokens]

  def _train_path_is_name(self):
    """
    Check if train file can be found
    """
    return os.path.isfile(self._train_path + self._language_tuple[0]) and os.path.isfile(self._train_path + self._language_tuple[1])

  def create_length_constraint(self, token_limit):
    """Filter an iterator if it pass a token limit"""
    return lambda x: len(x.src) <= token_limit and len(x.trg) <= token_limit

  """
  Build_field is used regardless of infer, train, or test
  """
  def build_field(self, **kwargs):
    """Build fields that will handle the conversion from token->idx and vice versa. To be overriden by MultiLoader."""
    return Field(**kwargs), Field(init_token=const.DEFAULT_SOS, eos_token=const.DEFAULT_EOS, **kwargs)

  """
  It's used to map idx->token strings -> string)
  The main point is set the src_field.vocab and trg_field.vocab
  """
  def build_vocab(self, fields, model_path=None, data=None, **kwargs):
    """Build the vocabulary object for torchtext Field. There are three flows:
      - if the model path is present, it will first try to load the pickled/dilled vocab object from path. This is accessed on continued training & standalone inference
      - if that failed and data is available, try to build the vocab from that data. This is accessed on first time training
      - if data is not available, search for set of two vocab files and read them into the fields. This is accessed on first time training
    TODO: expand on the vocab file option (loading pretrained vectors as well)
    """
    src_field, trg_field = fields
    if(model_path is None or not load_vocab_from_path(model_path, self._language_tuple, fields)):
      # the condition will try to load vocab pickled to model path.
      if(data is not None):
        print("Building vocab from received data.")
        # build the vocab using formatted data.
        src_field.build_vocab(data, **kwargs)
        trg_field.build_vocab(data, **kwargs)
      else:
        print("Building vocab from preloaded text file.")
        # load the vocab values from external location (a formatted text file). Initialize values as random
        external_vocab_location = self._option.get("external_vocab_location", None)
        src_ext, trg_ext = self._language_tuple
        # read the files and create a mock Counter object; which then is passed to vocab's class
        # see Field.build_vocab for the options used with vocab_cls
        """
        vocab_cls: Alias of torchtext.vocab.Vocab. It defines a vocabulary object that will be used to numericalize a field.

        Variables:	
        freqs – A collections.Counter object holding the frequencies of tokens in the data used to build the Vocab.
        stoi – A collections.defaultdict instance mapping token strings to numerical identifiers.
        itos – A list of token strings indexed by their numerical identifiers.
        => Just imagine it as a vocab class with methods
        """
        vocab_src = external_vocab_location + src_ext
        with io.open(vocab_src, "r", encoding="utf-8") as svf:
          mock_counter = Counter({w.strip():1 for w in svf.readlines()})
          special_tokens = [src_field.unk_token, src_field.pad_token, src_field.init_token, src_field.eos_token]
          src_field.vocab = src_field.vocab_cls(mock_counter, specials=special_tokens, min_freq=1, **kwargs)
        vocab_trg = external_vocab_location + trg_ext
        with io.open(vocab_trg, "r", encoding="utf-8") as tvf:
          mock_counter = Counter({w.strip():1 for w in tvf.readlines()})
          special_tokens = [trg_field.unk_token, trg_field.pad_token, trg_field.init_token, trg_field.eos_token]
          trg_field.vocab = trg_field.vocab_cls(mock_counter, specials=special_tokens, min_freq=1, **kwargs)
    else:
      print("Load vocab from path successful.")

  """
  It's used in train and test step only
  """
  def create_iterator(self, fields, model_path=None):
    """Create the iterator needed to load batches of data and bind them to existing fields
    NOTE: unlike the previous loader, this one inputs list of tokens instead of a string, which necessitate redefinining of translate_sentence pipe"""
    if(not self._train_path_is_name()):
      """
      We don't care it, since our task is Viet-Laos
      """
      # load the default torchtext dataset by name
      # TODO load additional arguments in the config
      dataset_cls = next( (s for s in [Multi30k, IWSLT, WMT14] if s.__name__ == self._train_path), None )
      if(dataset_cls is None):
        raise ValueError("The specified train path {:s}(+{:s}/{:s}) does neither point to a valid files path nor is a name of torchtext dataset class.".format(self._train_path, *self._language_tuple))
      src_suffix, trg_suffix = ext = self._language_tuple
#      print(ext, fields)
      self._train_data, self._valid_data, self._eval_data = dataset_cls.splits(exts=ext, fields=fields) #, split_ratio=self._option.get("train_test_split", const.DEFAULT_TRAIN_TEST_SPLIT)
    else:
      """
      This is where we should pay attention
      It's from Field classes -> TranslationDataset classes -> (creating vocab classes) -> BucketIterator classes
      """
      # create dataset from path. Also add all necessary constraints (e.g lengths trimming/excluding)
      src_suffix, trg_suffix = ext = self._language_tuple
      """
      filter_fn defines constraint for input vectors
      """
      filter_fn = self.create_length_constraint(self._option.get("train_max_length", const.DEFAULT_TRAIN_MAX_LENGTH))
      """
      View definition of TranslationDataset in the begining of this file
      """
      self._train_data = TranslationDataset(self._train_path, ext, fields, filter_pred=filter_fn)
      self._valid_data = self._eval_data = TranslationDataset(self._eval_path, ext, fields)
#    first_sample = self._train_data[0]; raise Exception("{} {}".format(first_sample.src, first_sample.trg))
    # whatever created, we now have the two set of data ready. add the necessary constraints/filtering/etc.
    train_data = self._train_data
    eval_data = self._eval_data
    # now we can execute build_vocab. This function will try to load vocab from model_path, and if fail, build the vocab from train_data
    build_vocab_kwargs = self._option.get("build_vocab_kwargs", {})
    self.build_vocab(fields, data=train_data, model_path=model_path, **build_vocab_kwargs)
#    raise Exception("{}".format(len(src_field.vocab)))
    # crafting iterators
    train_iter = BucketIterator(train_data, batch_size=self._option.get("batch_size", const.DEFAULT_BATCH_SIZE), device=self._option.get("device", const.DEFAULT_DEVICE) )
    eval_iter = BucketIterator(eval_data, batch_size=self._option.get("eval_batch_size", const.DEFAULT_EVAL_BATCH_SIZE), device=self._option.get("device", const.DEFAULT_DEVICE), train=False )
    return train_iter, eval_iter

