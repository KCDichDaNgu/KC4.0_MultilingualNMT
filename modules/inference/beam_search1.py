import numpy as np
import torch
import math, time, operator
import torch.nn.functional as functional
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

from modules.inference.decode_strategy import DecodeStrategy
from utils.misc import no_peeking_mask

class BeamSearch1(DecodeStrategy):
    def __init__(self, model, max_len, device, beam_size=5, use_synonym_fn=False, replace_unk=None):
        """
        Args:
            model: the used model
            max_len: the maximum timestep to be used
            device: the device to perform calculation
            beam_size: the size of the beam itself
            use_synonym_fn: if set, use the get_synonym fn from wordnet to try replace <unk>
            replace_unk: a tuple of [layer, head] designation, to replace the unknown word by chosen attention
        """
        super(BeamSearch1, self).__init__(model, max_len, device)
        self.beam_size = beam_size
        self._use_synonym = use_synonym_fn
        self._replace_unk = replace_unk
        # print("Init BeamSearch ----------------")

    def trg_init_vars(self, src, batch_size, trg_init_token, trg_eos_token, single_src_mask):
        """
        Calculate the required matrices during translation after the model is finished
        Input:
        :param src: The batch of sentences

        Output: Initialize the first character includes outputs, e_outputs, log_scores
        """
        # Initialize target sequence (start with '<sos>' token) [batch_size x k x max_len]
        trg = torch.zeros(batch_size, self.beam_size, self.max_len, device=self.device).long()
        trg[:, :, 0] = trg_init_token

        # Precalc output from model's encoder
        e_out = self.model.encoder(src, single_src_mask) # [batch_size x S x d_model]
        # Output model prob
        trg_mask = no_peeking_mask(1, device=self.device)
        # [batch_size x 1]
        inp_decoder = trg[:, 0, 0].view(batch_size, 1)
        # [batch_size x 1 x vocab_size]
        prob = self.model.out(self.model.decoder(inp_decoder, e_out, single_src_mask, trg_mask))
        prob = functional.softmax(prob, dim=-1)
    
        # [batch_size x 1 x k]
        k_prob, k_index = torch.topk(prob, self.beam_size, dim=-1)
        trg[:, :, 1] = k_index.view(batch_size, self.beam_size)
        # Init log scores from k beams [batch_size x k x 1]
        log_scores = torch.log(k_prob.view(batch_size, self.beam_size, 1))
    
        # Repeat encoder's output k times for searching [(k * batch_size) x S x d_model]
        e_outs = torch.repeat_interleave(e_out, self.beam_size, dim=0)
        src_mask = torch.repeat_interleave(single_src_mask, self.beam_size, dim=0)

        # Create mask for checking eos
        sent_eos = torch.tensor([trg_eos_token for _ in range(self.beam_size)], device=self.device).view(1, self.beam_size)
    
        return sent_eos, log_scores, e_outs, e_out, src_mask, trg

    def compute_k_best(self, outputs, out, log_scores, i, debug=False):
        """
        Compute k words with the highest conditional probability
        Args:
            outputs: Array has k previous candidate output sequences. [batch_size*beam_size, max_len]
            i: the current timestep to execute. Int
            out: current output of the model at timestep. [batch_size*beam_size, vocab_size]
            log_scores: Conditional probability of past candidates (in outputs) [batch_size * beam_size]

        Returns: 
            new outputs has k best candidate output sequences
            log_scores for each of those candidate
        """
        row_b = len(out);  
        batch_size = row_b // self.beam_size
        eos_id = self.TRG.vocab.stoi['<eos>']

        probs, ix = out[:, -1].data.topk(self.beam_size)

        probs_rep = torch.Tensor([[1] + [1e-100] * (self.beam_size-1)]*row_b).view(row_b, self.beam_size).to(self.device)
        ix_rep = torch.LongTensor([[eos_id] + [-1]*(self.beam_size-1)]*row_b).view(row_b, self.beam_size).to(self.device)

        check_eos = torch.repeat_interleave((outputs[:, i-1] == eos_id).view(row_b, 1), self.beam_size, 1)

        probs = torch.where(check_eos, probs_rep, probs)
        ix = torch.where(check_eos, ix_rep, ix)

        log_probs = torch.log(probs).to(self.device) + log_scores.to(self.device) # CPU

        k_probs, k_ix = log_probs.view(batch_size, -1).topk(self.beam_size)
        if(debug):
            print("kprobs_after_select: ", log_probs, k_probs, k_ix)

        # Use cpu
        k_probs, k_ix = torch.Tensor(k_probs.cpu().data.numpy()), torch.LongTensor(k_ix.cpu().data.numpy())
        row = k_ix // self.beam_size + torch.LongTensor([[v*self.beam_size] for v in range(batch_size)])
        col = k_ix % self.beam_size
        if(debug):
            print("kprobs row/col", row, col, ix[row.view(-1), col.view(-1)])
            assert False

        outputs[:, :i] = outputs[row.view(-1), :i]
        outputs[:, i] = ix[row.view(-1), col.view(-1)]
        log_scores = k_probs.view(-1, 1)

        return outputs, log_scores

    def replace_unknown(self, outputs, sentences, attn, selector_tuple, unknown_token="<unk>"):
        """Replace the unknown words in the outputs with the highest valued attentionized words.
        Args:
            outputs: the output from decoding. [batchbeam] of list of str, with maximum values being 
            sentences: the original wordings of the sentences. [batch_size, src_len] of str
            attn: the attention received, in the form of list:  [layers units of (self-attention, attention) with shapes of [batchbeam, heads, tgt_len, tgt_len] & [batchbeam, heads, tgt_len, src_len] respectively]
            selector_tuple: (layer, head) used to select the attention
            unknown_token: token used for 
        Returns:
            the replaced version, in the same shape as outputs
            """
        layer_used, head_used = selector_tuple
        # used_attention = attn[layer_used][-1][:, head_used] # it should be [batchbeam, tgt_len, src_len], as we are using the attention to source
        inx = torch.arange(start=0,end=len(attn)-1, step=self.beam_size)
        used_attention = attn[inx]
        select_id_src = torch.argmax(used_attention, dim=-1).cpu().numpy() # [batchbeam, tgt_len] of best indices. Also convert to numpy version (remove sos not needed as it is attention of outputs)
        # print(select_id_src, len(select_id_src))
        beam_size = select_id_src.shape[0] // len(sentences) # used custom-calculated beam_size as we might not output the entirety of beams. See beam_search fn for details
        # print("beam: ", beam_size)
        # select per batchbeam. source batch id is found by dividing batchbeam id per beam; we are selecting [tgt_len] indices from [src_len] tokens; then concat at the first dimensions to retrieve [batch_beam, tgt_len] of replacement tokens
        # need itemgetter / map to retrieve from list
        # print([ operator.itemgetter(*src_idx)(sentences[bidx // beam_size]) for bidx, src_idx in enumerate(select_id_src)])
        # print([print(sentences[bidx // beam_size], src_idx) for bidx, src_idx in enumerate(select_id_src)])
        # replace_tokens = [ operator.itemgetter(*src_idx)(sentences[bidx // beam_size]) for bidx, src_idx in enumerate(select_id_src)]
        
        for i in range(len(outputs)):
            for j in range(len(outputs[i])):
                if outputs[i][j] == unknown_token:
                    outputs[i][j] = sentences[i][select_id_src[i][j]]

        # print(sentences[0][0], outputs[0][0])

                    # print(i)
        # zip together with sentences; then output { the token if not unk / the replacement if is }. Note that this will trim the orig version down to repl size.
        # replaced = [ [tok if tok != unknown_token else rpl for rpl, tok in zip(repl, orig)] for orig, repl in zipped ]
        
        # return replaced
        return outputs

    # def beam_search(self, src, max_len, device, k=4):
    def beam_search(self, src, src_tokens=None, n_best=1, debug=False):
        """
        Beam search for a single sentence
        Args:
        model : a Transformer instance
        src   : a batch (tokenized + numerized) sentence [batch_size x S]
        Returns:
        trg   : a batch (tokenized + numerized) sentence [batch_size x T]
        """
        src = src.to(self.device)
        trg_init_token = self.TRG.vocab.stoi["<sos>"]  
        trg_eos_token = self.TRG.vocab.stoi["<eos>"]
        single_src_mask = (src != self.SRC.vocab.stoi['<pad>']).unsqueeze(1).to(self.device)
        batch_size = src.size(0)

        sent_eos, log_scores, e_outs, e_out, src_mask, trg = self.trg_init_vars(src, batch_size, trg_init_token, trg_eos_token, single_src_mask)

        # The batch indexes
        batch_index = torch.arange(batch_size)
        finished_batches = torch.zeros(batch_size, device=self.device).long()

        log_attn = torch.zeros([self.beam_size*batch_size, self.max_len, len(src[0])])

        # Iteratively searching
        for i in range(2, self.max_len):
            trg_mask = no_peeking_mask(i, self.device)
      
            # Flatten trg tensor for feeding into model [(k * batch_size) x i]
            inp_decoder = trg[batch_index, :, :i].view(self.beam_size * len(batch_index), i)
            # Output model prob [(k * batch_size) x i x vocab_size]
            current_decode, attn = self.model.decoder(inp_decoder, e_outs, src_mask, trg_mask, output_attention=True)
            # print(len(attn[0]))
    
            prob = self.model.out(current_decode)
            prob = functional.softmax(prob, dim=-1)

            # Only care the last prob i-th
            # [(k * batch_size) x 1 x vocab_size]
            prob = prob[:, i-1, :].view(self.beam_size * len(batch_index), 1, -1)

            # Truncate prob to top k [(k * batch_size) x 1 x k]
            k_prob, k_index = prob.data.topk(self.beam_size, dim=-1)

            # Deflatten k_prob & k_index
            k_prob = k_prob.view(len(batch_index), self.beam_size, 1, self.beam_size)
            k_index = k_index.view(len(batch_index), self.beam_size, 1, self.beam_size)

            # Preserve eos beams
            # [batch_size x k] -> view -> [batch_size x k x 1 x 1] (broadcastable)
            eos_mask = (trg[batch_index, :, i-1] == trg_eos_token).view(len(batch_index), self.beam_size, 1, 1)
            k_prob.masked_fill_(eos_mask, 1.0)
            k_index.masked_fill_(eos_mask, trg_eos_token)

            # Find the best k cases
            # Compute log score at i-th timestep 
            # [batch_size x k x 1 x 1] + [batch_size x k x 1 x k] = [batch_size x k x 1 x k]
            combine_probs = log_scores[batch_index].unsqueeze(-1) + torch.log(k_prob) 
            # [batch_size x k x 1]
            log_scores[batch_index], positions = torch.topk(combine_probs.view(len(batch_index), self.beam_size * self.beam_size, 1), self.beam_size, dim=1)

            # The rows selected from top k
            rows = positions.view(len(batch_index), self.beam_size) // self.beam_size
            # The indexes in vocab respected to these rows
            cols = positions.view(len(batch_index), self.beam_size) % self.beam_size
      
            batch_sim = torch.arange(len(batch_index)).view(-1, 1)
            trg[batch_index, :, :] = trg[batch_index.view(-1, 1), rows, :]
            trg[batch_index, :, i] = k_index[batch_sim, rows, :, cols].view(len(batch_index), self.beam_size)

            # Update attn
            inx = torch.repeat_interleave(finished_batches, self.beam_size, dim=0)
            batch_attn = torch.nonzero(inx == 0).view(-1)
            # import copy
            # x = copy.deepcopy(attn[0][-1][:, 0].to("cpu"))
            # log_attn[batch_attn, :i, :] = x

            # if i == 7:
            #     print(log_attn[batch_attn, :i, :].shape, attn[0][-1][:, 0].shape)
            #     print(log_attn[batch_attn, :i, :])
            # Update which sentences finished all its beams
            mask = (trg[:, :, i] == sent_eos).all(1).view(-1).to(self.device)
            finished_batches.masked_fill_(mask, value=1)
            cnt = torch.sum(finished_batches).item()
            if cnt == batch_size:
                break
      
            # # Continue with remaining batches (if any)
            batch_index = torch.nonzero(finished_batches == 0).view(-1)
            e_outs = torch.repeat_interleave(e_out[batch_index], self.beam_size, dim=0)
            src_mask = torch.repeat_interleave(single_src_mask[batch_index], self.beam_size, dim=0)
        # End loop

        # Get the best beam
        log_scores = log_scores.view(batch_size, self.beam_size)
        results = []
        for t, j in enumerate(torch.argmax(log_scores, dim=-1)):
            sent = []
            for i in range(self.max_len):
                token_id = trg[t, j.item(), i].item()
                if token_id == trg_init_token:
                    continue
                if token_id == trg_eos_token:
                    break
                sent.append(self.TRG.vocab.itos[token_id])
            results.append(sent)

        # if(self._replace_unk and src_tokens is not None):
        #     # replace unknown words per translated sentences.
        #     # NOTE: lacking a src_tokens does not raise any warning. Add that in when logging module is available, to support error catching
        #     # print("Replace unk -----------------------")
        #     results = self.replace_unknown(results, src_tokens, log_attn, self._replace_unk)

        return results

    def translate_single_sentence(self, src, **kwargs):
        """Translate a single sentence. Currently unused."""
        raise NotImplementedError
        return self.translate_batch_sentence([src], **kwargs)

    def translate_batch_sentence(self, src, field_processed=False, src_size_limit=None, output_tokens=False, debug=False):
        """Translate a batch of sentences together. Currently disabling the synonym func.
        Args:
            src: the batch of sentences to be translated
            field_processed: bool, if the sentences had been already processed (i.e part of batched validation data)
            src_size_limit: if set, trim the input if it cross this value. Added due to current positional encoding support only <=200 tokens
            output_tokens: the output format. False will give a batch of sentences (str), while True will give batch of tokens (list of str)
            debug: enable to print external values
        Return:
            the result of translation, with format dictated by output_tokens
        """
        # start = time.time()

        self.model.eval()
        # create the indiced batch.
        processed_batch = self.preprocess_batch(src, field_processed=field_processed, src_size_limit=src_size_limit, output_tokens=True, debug=debug)
        # print("Time preprocess_batch: ", time.time()-start)

        sent_ids, sent_tokens = (processed_batch, None) if(field_processed) else processed_batch
        assert isinstance(sent_ids, torch.Tensor), "sent_ids is instead {}".format(type(sent_ids))

        translated_sentences = self.beam_search(sent_ids, src_tokens=sent_tokens, debug=debug)

        # print("Time for one batch: ", time.time()-batch_start)
        
        # if time.time()-batch_start > 2:
        #     [print("len src >2 : ++++++", len(i.split())) for i in src]
        #     [print("len translate >2: ++++++", len(i)) for i in translated_sentences]
        # else:
        #     [print("len src : ====", len(i.split())) for i in src]
        #     [print("len translate : ====", len(i)) for i in translated_sentences]
        # print("=====================================")

        # time.sleep(4) 
        if(debug):
            print("Time performed for batch {}: {:.2f}s".format(sent_ids.shape))

        if(not output_tokens):
            translated_sentences = [' '.join(tokens) for tokens in translated_sentences]

        return translated_sentences

    def preprocess_batch(self, sentences, field_processed=False, pad_token="<pad>", src_size_limit=None, output_tokens=False, debug=True):
        """Adding 
            src_size_limit: int, option to limit the length of src.
            field_processed: bool: if the sentences had been already processed (i.e part of batched validation data)
            output_tokens: if set, output a token version aside the id version, in [batch of [src_len]] str. Note that it won't work with field_processed
            """

        if(field_processed):
            # do nothing, as it had already performed tokenizing/stoi
            return sentences
        processed_sent = map(self.SRC.preprocess, sentences)
        if(src_size_limit):
            processed_sent = map(lambda x: x[:src_size_limit], processed_sent)
        processed_sent = list(processed_sent)
        tokenized_sent = [torch.LongTensor([self._token_to_index(t) for t in s]) for s in processed_sent] # convert to tensors, in indices format
        sentences = Variable(pad_sequence(tokenized_sent, True, padding_value=self.SRC.vocab.stoi[pad_token])) # padding sentences
        if(debug):
            print("Input batch after process: ", sentences.shape, sentences)

        if(output_tokens):
            return sentences, processed_sent
        else:
            return sentences

    def translate_batch(self, sentences, **kwargs):
        return self.translate_batch_sentence(sentences, **kwargs)

    def _token_to_index(self, tok):
        """Override to select, depending on the self._use_synonym param"""
        if(self._use_synonym):
            return super(BeamSearch1, self)._token_to_index(tok)
        else:
            return self.SRC.vocab.stoi[tok]
