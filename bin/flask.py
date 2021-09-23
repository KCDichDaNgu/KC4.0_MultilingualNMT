"""Example of a translation client."""

from __future__ import print_function
import torch
import matplotlib
matplotlib.use('Agg')
import argparse

import re, os, yaml, io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

import models
from modules.config import find_all_config
from web.direct_translate import bind_blueprint as bind_blueprint_direct, about_blueprint, static_path, template_path

import flask
from flask_cors import CORS, cross_origin
import atexit

class LazyLoaderModel:
  def __init__(self):
    self._model = None
    self.model_cls = None
    self.model_cfg = None

  def _set_model(self, model_name, **config):
    self.model_cls = models.AvailableModels[model_name]
    self.model_cfg = config

  @property
  def model(self): # lazyload the model
    if(self._model is None):
      model_dir = self.model_cfg["model_dir"]
      self._model = self.model_cls(**self.model_cfg)
      self._model.load_checkpoint(model_dir)
    return self._model

  def clear(self):
    # delete the model. This is useful when --debug is on
    del self._model
    self._model = None

AppModel = LazyLoaderModel()
# app will use the static/template from opposing web folder
app = flask.Flask(__name__, static_folder=static_path, template_folder=template_path)

# test route,
@app.route('/echo', endpoint='mirror_reverse', methods=['POST'])
def mirror_reverse():
  requestContent = flask.request.get_json()
  data = requestContent["data"]
  data = data[::-1]
  return flask.jsonify(data=data)

@app.route('/translate', endpoint='translate', methods=['POST'])
def translate():
  requestContent = flask.request.get_json()
  inputs = requestContent["data"] # either get 'data' or 'input' namespace
#  target_lang = requestContent["direction"][:-2]
  target_lang = None
  outputs = AppModel.model.translate_batch(inputs, trg_lang=target_lang)
  # free cache?
  if(AppModel.model.device == "cuda"):
    torch.cuda.empty_cache()
  return flask.jsonify(data=outputs)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Translation client example")
  parser.add_argument("model_name", choices=models.AvailableModels.keys(), help="Name of the model to be loaded. Must conform to models.AvailableModels")
  parser.add_argument("--no_lazy", action="store_true", help="If specified, load the model immediately (as oppose to load at the first call)")
  parser.add_argument("--model_dir", required=True, type=str, help="Location to load the model. Must contain a config file")
  parser.add_argument("--host", type=str, default="0.0.0.0", help="model server host")
  parser.add_argument("--port", type=int, default=6019, help="model server port")
  parser.add_argument("--web", action="store_true", help="Set to enable web interface.")
  parser.add_argument("--flask_config", type=str, help="Load a config file containing arguments to build web interface")
  parser.add_argument("--timeout", type=float, default=10.0,
                      help="request timeout")
  parser.add_argument("--debug", action="store_true", help="If enable, application will output better debugging and update for every code change.")
  args = parser.parse_args()

  # NOTE: Here to build the model
  # set the model configuration for lazy loading
  configs = find_all_config(args.model_dir)
  AppModel._set_model(args.model_name, model_dir=args.model_dir, config=configs, mode="infer")
  if(args.no_lazy):
    # call first
    _ = AppModel.model()
  
  if(args.web):
    # NOTE: here to open a direct web interface 
    if(args.flask_config is not None):
      with io.open(args.flask_config, "r") as conf:
        webconfig = yaml.load(conf)
    else:
      webconfig = {}
    # register the blueprint binded to the model
    bound_blueprint_direct = bind_blueprint_direct(AppModel, webconfig)
    CORS(bound_blueprint_direct)
    app.register_blueprint(bound_blueprint_direct)

    CORS(about_blueprint)
    app.register_blueprint(about_blueprint)
    print(app.url_map)

  # run the flask app
  app.config['CORS_HEADERS'] = 'Content-Type'
  app.secret_key = os.urandom(24) #for secure session
  app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
  app.run(debug=args.debug, host=args.host, port=args.port)

  # hook to kill model when restart; this allow --debug to not have errant thread taking up gpu
  atexit.register(AppModel.clear)
