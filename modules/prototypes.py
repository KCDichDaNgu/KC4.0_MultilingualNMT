import torch.nn as nn
from torchtext import data
import copy
import layers as layers

#class Embedder(nn.Module):
#    def __init__(self, vocab_size, d_model):
#        super().__init__()
#        self.vocab_size = vocab_size
#        self.d_model = d_model
#        
#        self.embed = nn.Embedding(vocab_size, d_model)
#        
#    def forward(self, x):
#        return self.embed(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        """An layer of the encoder. Contain a self-attention accepting padding mask
        Args:
            d_model: the inner dimension size of the layer
            heads: number of heads used in the attention
            dropout: applied dropout value during training
            """
        super().__init__()
        self.norm_1 = layers.Norm(d_model)
        self.norm_2 = layers.Norm(d_model)
        self.attn = layers.MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = layers.FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        """Run the encoding layer
        Args:
            x: the input (either embedding values or previous layer output), should be in shape [batch_size, src_len, d_model]
            src_mask: the padding mask, should be [batch_size, 1, src_len]
        Return:
            an output that have the same shape as input, [batch_size, src_len, d_model]
            the attention used [batch_size, src_len, src_len]
        """
        x2 = self.norm_1(x)
        # Self attention only
        x_sa, sa = self.attn(x2, x2, x2, src_mask)
        x = x + self.dropout_1(x_sa)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, sa

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        """An layer of the decoder. Contain a self-attention that accept no-peeking mask and a normal attention tha t accept padding mask
        Args:
            d_model: the inner dimension size of the layer
            heads: number of heads used in the attention
            dropout: applied dropout value during training
            """
        super().__init__()
        self.norm_1 = layers.Norm(d_model)
        self.norm_2 = layers.Norm(d_model)
        self.norm_3 = layers.Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = layers.MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = layers.MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = layers.FeedForward(d_model, dropout=dropout)

    def forward(self, x, memory, src_mask, trg_mask):
        """Run the decoding layer
        Args:
            x: the input (either embedding values or previous layer output), should be in shape [batch_size, tgt_len, d_model]
            memory: the outputs of the encoding section, used for normal attention. [batch_size, src_len, d_model]
            src_mask: the padding mask for the memory, [batch_size, 1, src_len]
            tgt_mask: the no-peeking mask for the decoder, [batch_size, tgt_len, tgt_len]
        Return:
            an output that have the same shape as input, [batch_size, tgt_len, d_model]
            the self-attention and normal attention received [batch_size, head, tgt_len, tgt_len] & [batch_size, head, tgt_len, src_len]
        """
        x2 = self.norm_1(x)
        # Self-attention
        x_sa, sa = self.attn_1(x2, x2, x2, trg_mask)
        x = x + self.dropout_1(x_sa)
        x2 = self.norm_2(x)
        # Normal multi-head attention
        x_na, na = self.attn_2(x2, memory, memory, src_mask)
        x = x + self.dropout_2(x_na)
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x, (sa, na)

def get_clones(module, N, keep_module=True):
    if(keep_module and N >= 1):
        # create N-1 copies in addition to the original
        return nn.ModuleList([module] + [copy.deepcopy(module) for i in range(N-1)]) 
    else:
        # create N new copy
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    """A wrapper that embed, positional encode, and self-attention encode the inputs.
    Args:
        vocab_size: the size of the vocab. Used for embedding
        d_model: the inner dim of the module
        N: number of layers used
        heads: number of heads used in the attention
        dropout: applied dropout value during training
        max_seq_length: the maximum length value used for this encoder. Needed for PositionalEncoder, due to caching
    """
    def __init__(self, vocab_size, d_model, N, heads, dropout, max_seq_length=200):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = layers.PositionalEncoder(d_model, dropout=dropout, max_seq_length=max_seq_length)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = layers.Norm(d_model)

        self._max_seq_length = max_seq_length

    def forward(self, src, src_mask, output_attention=False, seq_length_check=False):
        """Accepts a batch of indexed tokens, return the encoded values.
        Args:
            src: int Tensor of [batch_size, src_len]
            src_mask: the padding mask, [batch_size, 1, src_len]
            output_attention: if set, output a list containing used attention
            seq_length_check: if set, automatically trim the input if it goes past the expected sequence length.
        Returns:
            the encoded values [batch_size, src_len, d_model]
            if available, list of N (self-attention) calculated. They are in form of [batch_size, heads, src_len, src_len]
        """
        if(seq_length_check and src.shape[1] > self._max_seq_length):
            src = src[:, :self._max_seq_length]
            src_mask = src_mask[:, :, :self._max_seq_length]
        x = self.embed(src)
        x = self.pe(x)
        attentions = [None] * self.N
        for i in range(self.N):
            x, attn = self.layers[i](x, src_mask)
            attentions[i] = attn
        x = self.norm(x)
        return x if(not output_attention) else (x, attentions)

class Decoder(nn.Module):
    """A wrapper that receive the encoder outputs, run through the decoder process for a determined input
    Args:
        vocab_size: the size of the vocab. Used for embedding
        d_model: the inner dim of the module
        N: number of layers used
        heads: number of heads used in the attention
        dropout: applied dropout value during training
        max_seq_length: the maximum length value used for this encoder. Needed for PositionalEncoder, due to caching
    """
    def __init__(self, vocab_size, d_model, N, heads, dropout, max_seq_length=200):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = layers.PositionalEncoder(d_model, dropout=dropout, max_seq_length=max_seq_length)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = layers.Norm(d_model)

        self._max_seq_length = max_seq_length

    def forward(self, trg, memory, src_mask, trg_mask, output_attention=False):
        """Accepts a batch of indexed tokens and the encoding outputs, return the decoded values.
        Args:
            trg: input Tensor of [batch_size, trg_len]
            memory: output of Encoder [batch_size, src_len, d_model]
            src_mask: the padding mask, [batch_size, 1, src_len]
            trg_mask: the no-peeking mask, [batch_size, tgt_len, tgt_len]
            output_attention: if set, output a list containing used attention
        Returns:
            the decoded values [batch_size, tgt_len, d_model]
            if available, list of N (self-attention, attention) calculated. They are in form of [batch_size, heads, tgt_len, tgt/src_len]
        """
        x = self.embed(trg)
        x = self.pe(x)

        attentions = [None] * self.N
        for i in range(self.N):
            x, attn = self.layers[i](x, memory, src_mask, trg_mask)
            attentions[i] = attn
        x = self.norm(x)
        return x if(not output_attention) else (x, attentions)


class Config:
    """Deprecated"""
    def __init__(self):
        self.opt = {
            'train_src_data':'/workspace/khoai23/opennmt/data/iwslt_en_vi/train.en',
            'train_trg_data':'/workspace/khoai23/opennmt/data/iwslt_en_vi/train.vi',
            'valid_src_data':'/workspace/khoai23/opennmt/data/iwslt_en_vi/tst2013.en',
            'valid_trg_data':'/workspace/khoai23/opennmt/data/iwslt_en_vi/tst2013.vi',
            'src_lang':'en', # useless atm
            'trg_lang':'en',#'vi_spacy_model', # useless atm
            'max_strlen':160,
            'batchsize':1500,
            'device':'cuda',
            'd_model': 512,
            'n_layers': 6,
            'heads': 8,
            'dropout': 0.1,
            'lr':0.0001,
            'epochs':30,
            'printevery': 200,
            'k':5,
        }
