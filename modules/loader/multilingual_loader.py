import io, os
import dill as pickle
from collections import Counter

import torch
from torchtext.data import BucketIterator, Dataset, Example, Field, interleave_keys
import modules.constants as const
from utils.save import load_vocab_from_path
from utils.data import generate_language_token
from modules.loader.default_loader import DefaultLoader

class MultiDataset(Dataset):
    """
    Ensemble one or more corpuses from different languages.
    The corpuses use global source vocab and target vocab.

    Constructor Args:
        data_info: list of datasets info <See `train` argument in MultiLoader class>
        fields: A tuple containing src field and trg field.
    """
    @staticmethod
    def sort_key(ex):
        return interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, data_info, fields,  **kwargs):
        self.languages = set()

        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        examples = []
        for corpus, info in data_info:
            print("Loading corpus {} ...".format(corpus))

            src_lang = info["src_lang"]
            trg_lang = info["trg_lang"]
            src_path = os.path.expanduser('.'.join([info["path"], src_lang]))
            trg_path = os.path.expanduser('.'.join([info["path"], trg_lang]))
            self.languages.add(src_lang)
            self.languages.add(trg_lang)

            with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                 io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
                for src_line, trg_line in zip(src_file, trg_file):
                    src_line, trg_line = src_line.strip(), trg_line.strip()
                    if src_line != '' and trg_line != '':
                        # Append language-specific prefix token
                        src_line = ' '.join([generate_language_token(src_lang), src_line])
                        trg_line = ' '.join([generate_language_token(trg_lang), trg_line])
                        examples.append(Example.fromlist([src_line, trg_line], fields))
            print("Done!")

        super(MultiDataset, self).__init__(examples, fields, **kwargs)


class MultiLoader(DefaultLoader):
    def __init__(self, train, valid=None, option=None):
        """
        Load multiple training/eval parallel data files, process and create data iterator
        Constructor Args:
            train: a dictionary contains training data information
            valid (optional): a dictionary contains validation data information
            option (optional): a dictionary contains configurable parameters

            For example:
            train = {
                "corpus_1": {
                    "path": path/to/training/data,
                    "src_lang": src,
                    "trg_lang": trg
                },
                "corpus_2": {
                    ...
                }
            }
        """
        self._train_info = train
        self._valid_info = valid
        self._language_tuple = ('.src', '.trg')
        self._option = option
        
    @property
    def language_tuple(self):
        """Currently output valid data's tuple for bleu_valid_iter, which would use <{trg_lang}> during inference. Since <{src_lang}> had already been added to the valid data, return None instead."""
        return None, self._valid_info["trg_lang"]

    def _is_path(self, path, lang):
        """Check whether the path is a system path or a corpus name"""
        return os.path.isfile(path + '.' + lang)

    def build_field(self, **kwargs):
        return Field(**kwargs), Field(eos_token='<eos>', **kwargs)

    def build_vocab(self, fields, model_path=None, data=None, **kwargs):
        """Build the vocabulary object for torchtext Field. There are three flows:
        - if the model path is present, it will first try to load the pickled/dilled vocab object from path. This is accessed on continued training & standalone inference
        - if that failed and data is available, try to build the vocab from that data. This is accessed on first time training
        - if data is not available, search for set of two vocab files and read them into the fields. This is accessed on first time training
        TODO: expand on the vocab file option (loading pretrained vectors as well)
        """
        src_field, trg_field = fields
        if model_path is None or not load_vocab_from_path(model_path, self._language_tuple, fields):
            # the condition will try to load vocab pickled to model path.
            if data is not None:
                print("Building vocab from received data.")
                # build the vocab using formatted data.
                src_field.build_vocab(data, **kwargs)
                trg_field.build_vocab(data, **kwargs)
            else:
                # Not implemented mixing preloaded datasets and external datasets 
                raise ValueError("MultiLoader currently do not support preloaded text vocab")
        else:
            print("Load vocab from path successful.")

    def create_iterator(self, fields, model_path=None):
        """Create the iterator needed to load batches of data and bind them to existing fields"""
        # create dataset from path. Also add all necessary constraints (e.g lengths trimming/excluding)
        filter_fn = self.create_length_constraint(self._option.get("train_max_length", const.DEFAULT_TRAIN_MAX_LENGTH))
        self._train_data = MultiDataset(data_info=self._train_info.items(), fields=fields, filter_pred=filter_fn)
        
        # now we can execute build_vocab. This function will try to load vocab from model_path, and if fail, build the vocab from train_data
        build_vocab_kwargs = self._option.get("build_vocab_kwargs", {})
        build_vocab_kwargs["specials"] = build_vocab_kwargs.pop("specials", []) + list(self._train_data.languages)
        self.build_vocab(fields, data=self._train_data, model_path=model_path, **build_vocab_kwargs)

        # Create train iterator
        train_iter = BucketIterator(self._train_data, batch_size=self._option.get("batch_size", const.DEFAULT_BATCH_SIZE), device=self._option.get("device", const.DEFAULT_DEVICE))
    
        if self._valid_info is not None:
            self._valid_data = MultiDataset(data_info=[("valid", self._valid_info)], fields=fields)
            valid_iter = BucketIterator(self._valid_data, batch_size=self._option.get("eval_batch_size", const.DEFAULT_EVAL_BATCH_SIZE), device=self._option.get("device", const.DEFAULT_DEVICE), train=False)
        else:
            valid_iter = None      

        return train_iter, valid_iter
