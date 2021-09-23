import torch
from torch.autograd import Variable
from utils.data import get_synonym
from torch.nn.utils.rnn import pad_sequence
import abc

class DecodeStrategy(object):
    """
    Base, abstract class for generation strategies. Contain specific call to base model that use it

    """
    def __init__(self, model, max_len, device):
        self.model = model
        self.max_len = max_len
        self.device = device

    @property
    def SRC(self):
        return self.model.SRC

    @property
    def TRG(self):
        return self.model.TRG

    @abc.abstractmethod
    def translate_single(self, src_lang, trg_lang, sentences):
        """Translate a single sentence. Might be useful as backcompatibility"""
        raise NotImplementedError

    @abc.abstractmethod
    def translate_batch(self, src_lang, trg_lang, sentences):
        """Translate a batch of sentences.
        Args:
            sentences: The sentences, formatted as [batch_size] Tensor of str
        Returns: 
            The detokenized output, most commonly [batch_size] of str
        """

        raise NotImplementedError

    @abc.abstractmethod
    def replace_unknown(self, *args):
        """Replace unknown words from batched sentences"""
        raise NotImplementedError

    def preprocess_batch(self, lang, sentences, pad_token="<pad>"):
        """Feed a unprocessed batch into the torchtext.Field of source.
        Args:
            sentences: [batch_size] of str
            pad_token: the pad token used to pad the sentences
        Returns:
            the sentences in Tensor format, padded with pad_value"""
        processed_sent = list(map(self.SRC.preprocess, sentences)) # tokenizing
        tokenized_sent = [Torch.LongTensor([self._token_to_index(t) for t in s]) for s in processed_sent] # convert to tensors and indices
        sentences = Variable(pad_sequence(tokenized_sent, True, padding_value=self.SRC.vocab.stoi[pad_token])) # padding sentences
        return sentences

    def _token_to_index(self, tok):
        """Implementing get_synonym as default. Override if want to use default behavior (<unk> for unknown words, independent of wordnet)"""
        if self.SRC.vocab.stoi[tok] != self.SRC.vocab.stoi['<eos>']:
            return self.SRC.vocab.stoi[tok]
        return get_synonym(tok, self.SRC)
