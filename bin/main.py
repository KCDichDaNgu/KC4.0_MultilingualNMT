import models
import argparse, os
from shutil import copy2 as copy
from modules.config import find_all_config

OVERRIDE_RUN_MODE = {"serve": "infer", "debug": "eval"}

def check_valid_file(path):
  if(os.path.isfile(path)):
    return path
  else:
    raise argparse.ArgumentError("This path {:s} is not a valid file, check again.".format(path))

def create_torchscript_model(model, model_dir, model_name):
  """Create a torchscript model using junk data. NOTE: same as tensorflow, is a limited model with no native python script."""
  import torch
  junk_input = torch.rand(2, 10)
  junk_output = torch.rand(2, 7)
  traced_model = torch.jit.trace(model, junk_input, junk_output)
  save_location = os.path.join(model_dir, model_name)
  traced_model.save(save_location)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Main argument parser")
  parser.add_argument("run_mode", choices=("train", "eval", "infer", "debug", "serve"), help="Main running mode of the program")
  parser.add_argument("--model", type=str, choices=models.AvailableModels.keys(), help="The type of model to be ran")
  parser.add_argument("--model_dir", type=str, required=True, help="Location of model")
  parser.add_argument("--config", type=str, nargs="+", default=None, help="Location of the config file")
  parser.add_argument("--no_keeping_config", action="store_false", help="If set, do not copy the config file to the model directory")
  # arguments for inference
  parser.add_argument("--features_file", type=str, help="Inference mode: Provide the location of features file")
  parser.add_argument("--predictions_file", type=str, help="Inference mode: Provide Location of output file which is predicted from features file")
  parser.add_argument("--src_lang", type=str, help="Inference mode: Provide language used by source file")
  parser.add_argument("--trg_lang", type=str, default=None, help="Inference mode: Choose language that is translated from source file. NOTE: only specify for multilingual model")
  parser.add_argument("--infer_batch_size", type=int, default=None, help="Specify the batch_size to run the model with. Default use the config value.")
  parser.add_argument("--checkpoint", type=str, default=None, help="All mode: specify to load the checkpoint into model.")
  parser.add_argument("--checkpoint_idx", type=int, default=0, help="All mode: specify the epoch of the checkpoint loaded. Only useful for training.")
  parser.add_argument("--serve_path", type=str, default=None, help="File to save TorchScript model into.")
  
  args = parser.parse_args()
  # create directory if not exist
  os.makedirs(args.model_dir, exist_ok=True)
  config_path = args.config
  if(config_path is None):
    config_path = find_all_config(args.model_dir)
    print("Config path not specified, load the configs in model directory which is {}".format(config_path))
  elif(args.no_keeping_config):
    # store false variable, mean true is default
    print("Config specified, copying all to model dir")
    for subpath in config_path:
      copy(subpath, args.model_dir)
    
  # load model. Specific run mode required converting
  run_mode = OVERRIDE_RUN_MODE.get(args.run_mode, args.run_mode)
  model = models.AvailableModels[args.model](config=config_path, model_dir=args.model_dir, mode=run_mode)
  model.load_checkpoint(args.model_dir, checkpoint=args.checkpoint, checkpoint_idx=args.checkpoint_idx)
  # run model
  run_mode = args.run_mode
  if(run_mode == "train"):
    model.run_train(model_dir=args.model_dir, config=config_path)
  elif(run_mode == "eval"):
    model.run_eval(model_dir=args.model_dir, config=config_path)
  elif(run_mode == "infer"):
    model.run_infer(args.features_file, args.predictions_file, src_lang=args.src_lang, trg_lang=args.trg_lang, config=config_path, batch_size=args.infer_batch_size)
  elif(run_mode == "debug"):
    raise NotImplementedError
    model.run_debug(model_dir=args.model_dir, config=config_path)
  elif(run_mode == "serve"):
    if(args.serve_path is None):
      raise parser.ArgumentError("In serving, --serve_path cannot be empty")
    model.prepare_serve(args.serve_path, model_dir=args.model_dir, config=config_path)
  else:
    raise ValueError("Run mode {:s} not implemented.".format(run_mode))
