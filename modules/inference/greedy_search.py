##@title Beam của mình
import numpy as np
import torch
import math
import torch.nn.functional as functional
import torch.nn as nn
from torch.autograd import Variable

from modules.inference.decode_strategy import DecodeStrategy
from utils.misc import no_peeking_mask

class GreedySearch(DecodeStrategy):
    def __init__(self, model, max_len, device, replace_unk=None):
        """
        :param beam_size
        :param batch_size
        :param beam_offset
        """
        super(GreedySearch, self).__init__(model, max_len, device)
        # self.replace_unk = replace_unk
        # raise NotImplementedError("Replace unk was yeeted from base class DecodeStrategy. Fix first.")

    def initilize_value(self, sentences):
        """
        Calculate the required matrices during translation after the model is finished
        Input:
        :param src: Sentences

        Output: Initialize the first character includes outputs, e_outputs, log_scores
        """
        batch_size=len(sentences)
        init_tok = self.TRG.vocab.stoi['<sos>']
        src_mask = (sentences != self.SRC.vocab.stoi['<pad>']).unsqueeze(-2)
        eos_tok = self.TRG.vocab.stoi['<eos>']

        # Encoder
        e_output = self.model.encoder(sentences, src_mask)

        out = torch.LongTensor([[init_tok] for i in range(batch_size)])
        outputs = torch.zeros(batch_size, self.max_len).long()
        outputs[:, :1] = out

        outputs = outputs.to(self.device)
        is_finished = torch.LongTensor([[eos_tok] for i in range(batch_size)]).view(-1).to(self.device)
        return eos_tok, src_mask, is_finished, e_output, outputs

    def create_trg_mask(self, i, device):
        return no_peeking_mask(i, device)

    def current_predict(self, outputs, e_output, src_mask, trg_mask):
        model = self.model
        # out, attn = model.out(model.decoder(outputs, e_output, src_mask, trg_mask))
        decoder_output, attn = model.decoder(outputs, e_output, src_mask, trg_mask, output_attention=True)
            # total_time_decode += time.time()-decode_time
        out = model.out(decoder_output)

        out = functional.softmax(out, dim=-1)
        return out, attn

    def greedy_search(self, sentences, sampling_temp=0.0, keep_topk=1):
        batch_size=len(sentences)

        eos_tok, src_mask, is_finished, e_output, outputs = self.initilize_value(sentences)

        for i in range(1, self.max_len):
            out, attn = self.current_predict(outputs[:, :i], e_output, src_mask, self.create_trg_mask(i, self.device))
            topk_ix, topk_prob = self.sample_with_temperature(out[:, -1], sampling_temp, keep_topk)
            outputs[:, i] = topk_ix.view(-1)
            if torch.equal(outputs[:, i], is_finished):
                break
        
        # if self.replace_unk == True:
        #     outputs = self.replace_unknown(attn, sentences, outputs)

        # print("\n".join([' '.join([self.TRG.vocab.itos[tok] for tok in line[1:]]) for line in outputs]))
        # Write to file or Print to the console
        translated_sentences = []
        # Get the best sentences: idx = 0 + i*k
        for i in range(0, len(outputs)):
            is_eos = torch.nonzero(outputs[i]==eos_tok)
            if len(is_eos) == 0:
                # if there is no sequence end, remove
                sent = outputs[i, 1:]
            else:
                length = is_eos[0]
                sent = outputs[i, 1:length]
            translated_sentences.append([self.TRG.vocab.itos[tok] for tok in sent])

        return translated_sentences

    def sample_with_temperature(self, logits, sampling_temp, keep_topk):
        if sampling_temp == 0.0 or keep_topk == 1:
            # For temp=0.0, take the argmax to avoid divide-by-zero errors.
            # keep_topk=1 is also equivalent to argmax.
            topk_scores, topk_ids = logits.topk(1, dim=-1)
            if sampling_temp > 0:
                topk_scores /= sampling_temp
        else:
            logits = torch.div(logits, sampling_temp)

            if keep_topk > 0:
                top_values, top_indices = torch.topk(logits, keep_topk, dim=1)
                kth_best = top_values[:, -1].view([-1, 1])
                kth_best = kth_best.repeat([1, logits.shape[1]]).float()

                # Set all logits that are not in the top-k to -10000.
                # This puts the probabilities close to 0.
                ignore = torch.lt(logits, kth_best)
                logits = logits.masked_fill(ignore, -10000)

            dist = torch.distributions.Multinomial(
                logits=logits, total_count=1)
            topk_ids = torch.argmax(dist.sample(), dim=1, keepdim=True)
            topk_scores = logits.gather(dim=1, index=topk_ids)
        return topk_ids, topk_scores

    def translate_batch(self, sentences, src_size_limit, output_tokens=True, debug=False):
        # super(BeamSearch, self).__init__()
        sentences = self.preprocess_batch(sentences).to(self.device)
        return self.greedy_search(sentences, 0.2, 2)
        # print(self.initilize_value(sentences))
