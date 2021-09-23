import torch, time
import torch.nn.functional as functional
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

from modules.inference.beam_search import BeamSearch
from utils.data import generate_language_token
import modules.constants as const

def generate_subsequent_mask(sz, device):
    return torch.triu(
        torch.ones(sz, sz, dtype=torch.int, device=device)
    ).transpose(0, 1).unsqueeze(0)

class BeamSearch2(BeamSearch):
    """
    Same with BeamSearch2 class.
    Difference: remove the sentence which its beams terminated (reached <eos> token) from the time step loop.
    Update to reuse functions already coded in normal BeamSearch. Note that replacing unknown words & n_best is not available.
    """
    def _convert_to_sent(self, sent_id, eos_token_id):
        eos = torch.nonzero(sent_id == eos_token_id).view(-1)
        t = eos[0] if len(eos) > 0 else len(sent_id)
        return [self.TRG.vocab.itos[j] for j in sent_id[1 : t]]

    @torch.no_grad()
    def beam_search(self, src, src_lang=None, trg_lang=None, src_tokens=None, n_best=1, debug=False):
        """
        Beam search select k words with the highest conditional probability
         to be the first word of the k candidate output sequences.
        Args:
            src: The batch of sentences, already in [batch_size, tokens] of int
            src_tokens: src in str version, same size as above
            n_best: number of usable values per beam loaded (Not implemented)
            debug: if true, print some debug information during the search
        Return: 
            An array of translated sentences, in list-of-tokens format. TODO convert [batch_size, n_best, tgt_len] instead of [batch_size, tgt_len]
        """
        # Create some local variable
        src_field, trg_field = self.SRC, self.TRG
        sos_token = generate_language_token(trg_lang) if trg_lang is not None else const.DEFAULT_SOS
        init_token = trg_field.vocab.stoi[sos_token]
        eos_token_id = trg_field.vocab.stoi[const.DEFAULT_EOS]
        src = src.to(self.device)
        
        batch_size = src.size(0)
        model = self.model
        k = self.beam_size
        max_len = self.max_len
        device = self.device

        # Initialize target sequence (start with '<sos>' token) [batch_size x k x max_len]
        trg = torch.zeros(batch_size, k, max_len, device=device).long()
        trg[:, :, 0] = init_token

        # Precalc output from model's encoder 
        single_src_mask = (src != src_field.vocab.stoi['<pad>']).unsqueeze(1).to(device)
        e_out = model.encoder(src, single_src_mask) # [batch_size x S x d_model]

        # Output model prob
        trg_mask = generate_subsequent_mask(1, device=device)
        # [batch_size x 1]
        inp_decoder = trg[:, 0, 0].view(batch_size, 1)
        # [batch_size x 1 x vocab_size]
        prob = model.out(model.decoder(inp_decoder, e_out, single_src_mask, trg_mask))
        prob = functional.softmax(prob, dim=-1)
        
        # [batch_size x 1 x k]
        k_prob, k_index = torch.topk(prob, k, dim=-1)
        trg[:, :, 1] = k_index.view(batch_size, k)
        # Init log scores from k beams [batch_size x k x 1]
        log_scores = torch.log(k_prob.view(batch_size, k, 1))
        
        # Repeat encoder's output k times for searching [(k * batch_size) x S x d_model]
        e_outs = torch.repeat_interleave(e_out, k, dim=0)
        src_mask = torch.repeat_interleave(single_src_mask, k, dim=0)

        # Create mask for checking eos
        sent_eos = torch.tensor([eos_token_id for _ in range(k)], device=device).view(1, k)

        # The batch indexes
        batch_index = torch.arange(batch_size)
        finished_batches = torch.zeros(batch_size, device=device).long()

        # Iteratively searching
        for i in range(2, max_len):
            trg_mask = generate_subsequent_mask(i, device)
            
            # Flatten trg tensor for feeding into model [(k * batch_size) x i]
            inp_decoder = trg[batch_index, :, :i].view(k * len(batch_index), i)
            # Output model prob [(k * batch_size) x i x vocab_size]
            prob = model.out(model.decoder(inp_decoder, e_outs, src_mask, trg_mask))
            prob = functional.softmax(prob, dim=-1)

            # Only care the last prob i-th
            # [(k * batch_size) x 1 x vocab_size]
            prob = prob[:, i-1, :].view(k * len(batch_index), 1, -1)

            # Truncate prob to top k [(k * batch_size) x 1 x k]
            k_prob, k_index = prob.data.topk(k, dim=-1)

            # Deflatten k_prob & k_index
            k_prob = k_prob.view(len(batch_index), k, 1, k)
            k_index = k_index.view(len(batch_index), k, 1, k)

            # Preserve eos beams
            # [batch_size x k] -> view -> [batch_size x k x 1 x 1] (broadcastable)
            eos_mask = (trg[batch_index, :, i-1] == eos_token_id).view(len(batch_index), k, 1, 1)
            k_prob.masked_fill_(eos_mask, 1.0)
            k_index.masked_fill_(eos_mask, eos_token_id)

            # Find the best k cases
            # Compute log score at i-th timestep 
            # [batch_size x k x 1 x 1] + [batch_size x k x 1 x k] = [batch_size x k x 1 x k]
            combine_probs = log_scores[batch_index].unsqueeze(-1) + torch.log(k_prob) 
            # [batch_size x k x 1]
            log_scores[batch_index], positions = torch.topk(combine_probs.view(len(batch_index), k * k, 1), k, dim=1)

            # The rows selected from top k
            rows = positions.view(len(batch_index), k) // k
            # The indexes in vocab respected to these rows
            cols = positions.view(len(batch_index), k) % k
            
            batch_sim = torch.arange(len(batch_index)).view(-1, 1)
            trg[batch_index, :, :] = trg[batch_index.view(-1, 1), rows, :]
            trg[batch_index, :, i] = k_index[batch_sim, rows, :, cols].view(len(batch_index), k)
            
            # Update which sentences finished all its beams
            mask = (trg[:, :, i] == sent_eos).all(1).view(-1).to(device)
            finished_batches.masked_fill_(mask, value=1)    
            cnt = torch.sum(finished_batches).item()
            if cnt == batch_size:
                break
            
            # Continue with remaining batches (if any)
            batch_index = torch.nonzero(finished_batches == 0).view(-1)
            e_outs = torch.repeat_interleave(e_out[batch_index], k, dim=0)
            src_mask = torch.repeat_interleave(single_src_mask[batch_index], k, dim=0)
            # End loop

        # Get the best beam
        log_scores = log_scores.view(batch_size, k)
        results = [self._convert_to_sent(trg[t, j.item(), :], eos_token_id) for t, j in enumerate(torch.argmax(log_scores, dim=-1))]
        return results
