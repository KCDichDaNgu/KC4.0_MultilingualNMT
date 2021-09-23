import torch
import os, re, io
import json
import dill as pickle
from shutil import copy2 as copy
MODEL_EXTENSION = ".pkl"
MODEL_FILE_FORMAT = "{:s}_{:d}{:s}" # model_prefix, epoch and extension
BEST_MODEL_FILE = ".model_score.txt"
MODEL_SERVE_FILE = ".serve.txt"
VOCAB_FILE_FORMAT = "{:s}{:s}{:s}"

def save_model_name(name, path, serve_config_path=MODEL_SERVE_FILE):
  with io.open(os.path.join(path, serve_config_path), "w", encoding="utf-8") as serve_config_file:
    serve_config_file.write(name)

def save_vocab_to_path(path, language_tuple, fields, name_prefix="vocab", check_saved_vocab=True):
  src_field, trg_field = fields
  src_ext, trg_ext = language_tuple
  src_vocab_path = os.path.join(path, VOCAB_FILE_FORMAT.format(name_prefix, src_ext, MODEL_EXTENSION))
  trg_vocab_path = os.path.join(path, VOCAB_FILE_FORMAT.format(name_prefix, trg_ext, MODEL_EXTENSION))
  if(check_saved_vocab and os.path.isfile(src_vocab_path) and os.path.isfile(trg_vocab_path)):# do nothing if already exist
    return
  with io.open(src_vocab_path , "wb") as src_vocab_file:
    pickle.dump(src_field.vocab, src_vocab_file)
  with io.open(trg_vocab_path , "wb") as trg_vocab_file:
    pickle.dump(trg_field.vocab, trg_vocab_file)

def load_vocab_from_path(path, language_tuple, fields, name_prefix="vocab"):
  """Load the vocabulary from path into respective fields. If files doesn't exist, return False; if loaded properly, return True"""
  src_field, trg_field = fields
  src_ext, trg_ext = language_tuple
  src_vocab_file_path = os.path.join(path, VOCAB_FILE_FORMAT.format(name_prefix, src_ext, MODEL_EXTENSION))
  trg_vocab_file_path = os.path.join(path, VOCAB_FILE_FORMAT.format(name_prefix, trg_ext, MODEL_EXTENSION))
  if(not os.path.isfile(src_vocab_file_path) or not os.path.isfile(trg_vocab_file_path)):
    # the vocab file wasn't dumped, return False
    return False
  with io.open(src_vocab_file_path, "rb") as src_vocab_file, io.open(trg_vocab_file_path, "rb") as trg_vocab_file:
    src_vocab = pickle.load(src_vocab_file)
    src_field.vocab = src_vocab
    trg_vocab = pickle.load(trg_vocab_file)
    trg_field.vocab = trg_vocab
  return True

def save_model_to_path(model, path, name_prefix="model", checkpoint_idx=0, save_vocab=True):
  save_path = os.path.join(path, MODEL_FILE_FORMAT.format(name_prefix, checkpoint_idx, MODEL_EXTENSION))
  torch.save(model.state_dict(), save_path)
  if(save_vocab):
    save_vocab_to_path(path, model.loader._language_tuple, model.fields)

def load_model_from_path(model, path, name_prefix="model", checkpoint_idx=0):
  # do not load vocab here, as the vocab structure will be decided in model.loader.build_vocab
  save_path = os.path.join(path, MODEL_FILE_FORMAT.format(name_prefix, checkpoint_idx, MODEL_EXTENSION))
  model.load_state_dict(torch.load(save_path))


def load_model(model, model_path):
  model.load_state_dict(torch.load(model_path))

def check_model_in_path(path, name_prefix="model", return_all_checkpoint=False):
  model_re = re.compile(r"{:s}_(\d+){:s}".format(name_prefix, MODEL_EXTENSION))
  if(not os.path.isdir(path)):
    return 0
  matches = [re.match(model_re, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
#  print(matches)
  indices = sorted([int(m.group(1)) for m in matches if m is not None])
  if(return_all_checkpoint):
    return indices
  elif(len(indices) == 0):
    return 0
  else:
    return indices[-1]

def save_and_clear_model(model, path, name_prefix="model", checkpoint_idx=0, maximum_saved_model=5):
  """Keep only last n models when saving. Explicitly save the model regardless of its checkpoint index, e.g if checkpoint_idx=0 & model 3 4 5 6 7 is in path, it will remove 3 and save 0 instead."""
  indices = check_model_in_path(path, name_prefix=name_prefix, return_all_checkpoint=True)
  if(maximum_saved_model <= len(indices)):
    # remove models until n-1 models are left
    for i in indices[:-(maximum_saved_model-1)]:
      os.remove(os.path.join(path, MODEL_FILE_FORMAT.format(name_prefix, i, MODEL_EXTENSION)))
  # perform save as normal
  save_model_to_path(model, path, name_prefix=name_prefix, checkpoint_idx=checkpoint_idx)

def load_model_score(path, score_file=BEST_MODEL_FILE):
  """Load the model score as a list from a json dump, organized from best to worst."""
  score_file_path = os.path.join(path, score_file)
  if(not os.path.isfile(score_file_path)):
    return []
  with io.open(score_file_path, "r") as jf:
    return json.load(jf)

def write_model_score(path, score_obj, score_file=BEST_MODEL_FILE):
  with io.open(os.path.join(path, score_file), "w") as jf:
    json.dump(score_obj, jf)

def save_model_best_to_path(model, path, score_obj, model_metric, best_model_prefix="best_model", maximum_saved_model=5, score_file=BEST_MODEL_FILE, save_after_update=True):
  worst_score = score_obj[-1] if len(score_obj) > 0 else -1.0
  if(model_metric > worst_score):
    # perform update, overriding a slot or create new if needed
    insert_loc = next((idx for idx, score in enumerate(score_obj) if model_metric > score), 0)
    # every model below it, up to {maximum_saved_model}, will be moved down an index
    for i in range(insert_loc, min(len(score_obj), maximum_saved_model)-1): # -1, due to the models are copied up to +1
      old_loc = save_path = os.path.join(path, MODEL_FILE_FORMAT.format(best_model_prefix, i, MODEL_EXTENSION))
      new_loc = save_path = os.path.join(path, MODEL_FILE_FORMAT.format(best_model_prefix, i+1, MODEL_EXTENSION))
      copy(old_loc, new_loc)
    # save the model to the selected loc
    save_model_to_path(model, path, name_prefix=best_model_prefix, checkpoint_idx=insert_loc)
    # update the score obj
    score_obj.insert(insert_loc, model_metric)
    score_obj = score_obj[:maximum_saved_model]
    # also update in disk, if enabled
    if(save_after_update):
      write_model_score(path, score_obj, score_file=score_file)
  # after routine had been done, return the obj
  return score_obj


