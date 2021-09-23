import numpy as np
import math
import torch
from torch.autograd import Variable
import torch.nn.functional as functional

from utils.data import multiple_replace, get_synonym

def no_peeking_mask(size, device):
    """
    Tạo mask được sử dụng trong decoder để lúc dự đoán trong quá trình huấn luyện
    mô hình không nhìn thấy được các từ ở tương lai
    """
    np_mask = np.triu(np.ones((1, size, size)),
k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    np_mask = np_mask.to(device)
    
    return np_mask

def create_masks(src, trg, src_pad, trg_pad, device):
    """ Tạo mask cho encoder, 
    để mô hình không bỏ qua thông tin của các kí tự PAD do chúng ta thêm vào 
    """
    src_mask = (src != src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = no_peeking_mask(size, device)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask & np_mask
        
    else:
        trg_mask = None
    return src_mask, trg_mask

def init_vars(src, model, SRC, TRG, device, k, max_len):
    """ Tính toán các ma trận cần thiết trong quá trình translation sau khi mô hình học xong
    """
    init_tok = TRG.vocab.stoi['<sos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)

    # tính sẵn output của encoder
    e_output = model.encoder(src, src_mask)

    outputs = torch.LongTensor([[init_tok]])

    outputs = outputs.to(device)

    trg_mask = no_peeking_mask(1, device)
    # dự đoán kí tự đầu tiên
    out = model.out(model.decoder(outputs,
    e_output, src_mask, trg_mask))
    out = functional.softmax(out, dim=-1)

    probs, ix = out[:, -1].data.topk(k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)

    outputs = torch.zeros(k, max_len).long()
    outputs = outputs.to(device)
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]

    e_outputs = torch.zeros(k, e_output.size(-2),e_output.size(-1))

    e_outputs = e_outputs.to(device)
    e_outputs[:, :] = e_output[0]

    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    # debug print
    
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)

    return outputs, log_scores

def beam_search(src, model, SRC, TRG, device, k, max_len, debug=False, output_list_of_tokens=False):

    outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG, device, k, max_len)
    eos_tok = TRG.vocab.stoi['<eos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    ind = None
    for i in range(2, max_len):
        if(debug):
            print("Current iteration to maxlen: {:d}".format(i))

        trg_mask = no_peeking_mask(i, device)

        out = model.out(model.decoder(outputs[:,:i], e_outputs, src_mask, trg_mask))

        out = functional.softmax(out, dim=-1)

        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, k)

        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).to(device)
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == k:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break

    # additional change to output list of tokens instead of string
    join_fn = (lambda x: x) if(output_list_of_tokens) else (lambda x: " ".join(x))

    if ind is None:
        length = (outputs[0]==eos_tok).nonzero()[0] if len((outputs[0]==eos_tok).nonzero()) > 0 else -1
        return join_fn([TRG.vocab.itos[tok] for tok in outputs[0, 1:length]])
    else:
        length = (outputs[ind]==eos_tok).nonzero()[0]
        return join_fn([TRG.vocab.itos[tok] for tok in outputs[ind, 1:length]])

def translate_sentence(raw_sentence, model, SRC, TRG, device, k, max_len, debug=False, output_list_of_tokens=False):
    """Dịch một câu sử dụng beamsearch
    """
    model.eval()
    indexed = []
    if(isinstance(raw_sentence, str)):
        # single sentence, require preprocessing
        sentence = SRC.preprocess(raw_sentence)
    else:
        # already tokenized (taken from iterators, etc.)
        sentence = raw_sentence
    
    for tok in sentence:
        if SRC.vocab.stoi[tok] != SRC.vocab.stoi['<eos>']:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC))
    
    output = Variable(torch.LongTensor([indexed]))
    
    output = output.to(device)
    
    output = beam_search(output, model, SRC, TRG, device, k, max_len, output_list_of_tokens=output_list_of_tokens)

    if(debug):
        print("{} -> {}".format(raw_sentence, output))

    return output

#    return  multiple_replace({' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':','}, sentence)
