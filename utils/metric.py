from torchtext.data.metrics import bleu_score

def bleu(valid_src_data, valid_trg_data, model, device, k, max_strlen):
    pred_sents = []
    for sentence in valid_src_data:
        pred_trg = model.translate_sentence(sentence, device, k, max_strlen)
        pred_sents.append(pred_trg)
    
    pred_sents = [self.TRG.preprocess(sent) for sent in pred_sents]
    trg_sents = [[sent.split()] for sent in valid_trg_data]
    
    return bleu_score(pred_sents, trg_sents)

def bleu_single(model, valid_dataset, debug=False):
    """Perform single sentence translation, then calculate bleu score. Update when batch beam search is online"""
    # need to join the sentence back per sample (the iterator is the one that had already been split to tokens)
    # THIS METRIC USE 2D vs 3D! AAAAAAHHHHHHH!!!!
    translate_pair = ( ([pair.trg], model.translate_sentence(pair.src, debug=debug)) for pair in valid_dataset)
#    raise Exception(next(translate_pair))
    labels, predictions = [list(l) for l in zip(*translate_pair)] # zip( *((l, p.split()) for l, p in translate_pair) )
    return bleu_score(predictions, labels)

def bleu_batch(model, valid_dataset, batch_size, debug=False):
    """Perform batch sentence translation in the same vein."""
    predictions = model.translate_batch_sentence([s.src for s in valid_dataset], output_tokens=True, batch_size=batch_size)
    labels = [[s.trg] for s in valid_dataset]
    return bleu_score(predictions, labels)


def _revert_trg(sent, eos): # revert batching process on sentence
    try:
        endloc = sent.index(eos)
        return sent[1:endloc]
    except ValueError:
        return sent[1:]

def bleu_batch_iter(model, valid_iter, src_lang=None, trg_lang=None, eos_token="<eos>", debug=False):
    """Perform batched translations; other metrics are the same. Note that the inputs/outputs had been preprocessed, but have [length, batch_size] shape as per BucketIterator"""
#    raise NotImplementedError("Error during calculation, currently unusable.")
 #   raise Exception([[model.SRC.vocab.itos[t] for t in batch] for batch in next(iter(valid_iter)).src.transpose(0, 1)])
    
    translated_batched_pair = (
        (
            pair.trg.transpose(0, 1), # transpose due to timestep-first batches
            model.decode_strategy.translate_batch_sentence(
                pair.src.transpose(0, 1),
                src_lang=src_lang,
                trg_lang=trg_lang,
                output_tokens=True, 
                field_processed=True, 
                replace_unk=False, # do not replace in this version
                debug=debug
            )
        ) 
        for pair in valid_iter 
    ) 
    flattened_pair = ( ([model.TRG.vocab.itos[i] for i in trg], pred) for batch_trg, batch_pred in translated_batched_pair for trg, pred in zip(batch_trg, batch_pred) )
    flat_labels, predictions = [list(l) for l in zip(*flattened_pair)]
    labels = [[_revert_trg(l, eos_token)] for l in flat_labels] # remove <sos> and <eos> also updim the trg for 3D requirements.
    return bleu_score(predictions, labels)
