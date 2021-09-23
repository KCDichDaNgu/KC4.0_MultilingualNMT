import os

#import utils.save as saver
#import models
#from models.transformer import Transformer
#from modules.config import find_all_config

class TransformerHandlerClass:
  def __init__(self):
    self.model = None
    self.device = None
    self.initialized = False

  def _find_checkpoint(self, model_dir, best_model_prefix="best_model", model_prefix="model", validate=True):
    """Attempt to retrieve the best model checkpoint from model_dir. Failing that, the model of the latest iteration.
    Args:
      model_dir: location to search for checkpoint. str
    Returns:
      single str denoting the checkpoint path """
    score_file_path = os.path.join(model_dir, saver.BEST_MODEL_FILE)
    if(os.path.isfile(score_file_path)): # score exist -> best model
      best_model_path = os.path.join(model_dir, saver.MODEL_FILE_FORMAT.format(best_model_prefix, 0, saver.MODEL_EXTENSION))
      if(validate):
        assert os.path.isfile(best_model_path), "Score file is available, but file {:s} is missing.".format(best_model_path)
      return best_model_path
    else: # score not exist -> latest model
      last_checkpoint_idx = saver.check_model_in_path(name_prefix=model_prefix)
      if(last_checkpoint_idx == 0):
        raise ValueError("No checkpoint found in folder {:s} with prefix {:s}.".format(model_dir, model_prefix))
      else:
        return os.path.join(model_dir, saver.MODEL_FILE_FORMAT.format(model_prefix, last_checkpoint_idx, saver.MODEL_EXTENSION))


  def initialize(self, ctx):
    manifest = ctx.manifest
    properties = ctx.system_properties

    self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
    self.model_dir = model_dir = properties.get("model_dir")

    # extract checkpoint location, config & model name
    model_serve_file = os.path.join(model_dir, saver.MODEL_SERVE_FILE)
    with io.open(model_serve_file, "r") as serve_config:
      model_name = serve_config.read().strip()
#    model_cls = models.AvailableModels[model_name]
    model_cls = Transformer # can't select due to nature of model file
    checkpoint_path = manifest['model'].get('serializedFile', self._find_checkpoint(model_dir)) # attempt to use the checkpoint fed from archiver; else use the best checkpoint found
    config_path = find_all_config(model_dir)

    # load model with inbuilt config + vocab & without pretraining data
    self.model = model = model_cls(config=config_path, model_dir=model_dir, mode="infer")
    model.load_checkpoint(args.model_dir, checkpoint=checkpoint_path) # TODO find_checkpoint might do some redundant thing here since load_checkpoint had already done searching for latest
    
    print("Model {:s} loaded successfully at location {:s}.".format(model_name, model_dir))
    self.initialized = True

  def handle(self, data):
    """The main bulk of handling. Process a batch of data received from client.
    Args: 
      data: the object received from client. Should contain something in [batch_size] of str
    Returns:
      the expected translation, [batch_size] of str
    """
    batch_sentences = data[0].get("data")
#    assert batch_sentences is not None, "data is {}".format(data)
    
    # make sure that sentences are detokenized before returning
    translated_sentences = self.model.translate_batch(batch_sentences, output_tokens=False)

    return translated_sentences

class BeamSearchHandlerClass:
  def __init__(self):
    self.model = None
    self.inferrer = None
    self.initialized = False

  def initialize(self, ctx):
    manifest = ctx.manifest
    properties = ctx.system_properties

    model_dir = properties['model_dir']
    ts_modelpath = manifest['model']['serializedFile'] 
    self.model = ts_model = torch.jit.load(os.path.join(model_dir, ts_modelpath))

    from modules.inference.beam_search import BeamSearch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    self.inferrer = BeamSearch(model, 160, device, beam_size=5)
    
    self.initialized = True

  def handle(self, data):
    batch_sentences = data[0].get("data")
#    assert batch_sentences is not None, "data is {}".format(data)
    
    translated_sentences = self.inferrer.translate_batch_sentence(data, output_tokens=False)
    return translated_sentences

RUNNING_MODEL = BeamSearchHandlerClass()

def handle(data, context):
  if(not RUNNING_MODEL.initialized): # Lazy init
    RUNNING_MODEL.initialize(context)

  if(data is None):
    return None

  return RUNNING_MODEL.handle(data)
