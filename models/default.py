class MockModel:
  """A model that only output string to show flow"""
  def __init__(self, *args, **kwargs):
    print("Mock model initialization, with args/kwargs: {} {}".format(args, kwargs))

  def run_train(self, **kwargs):
    print("Model in training, with args: {}".format(kwargs))

  def run_eval(self, **kwargs):
    print("Model in evaluation, with args: {}".format(kwargs))

  def run_debug(self, **kwargs):
    print("Model in debuging, with args: {}".format(kwargs))
