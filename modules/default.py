class MockLoader:
  def __init__(self, *args, **kwargs):
    """Only print stuff"""
    print("MockLoader initialized, args/kwargs {} {}".format(args, kwargs))

  def tokenize(self, inputs, **kwargs):
    print("MockLoader tokenize called, inputs/kwargs {} {}".format(inputs, kwargs))
    return inputs

  def detokenize(self, inputs, **kwargs):
    print("MockLoader detokenize called, inputs/kwargs {} {}".format(inputs, kwargs))
    return inputs
  
  def reverse_lookup(self, inputs, **kwargs):
    print("MockLoader reverse_lookup called, inputs/kwargs {} {}".format(inputs, kwargs))
    return inputs

  def lookup(self, inputs, **kwargs):
    print("MockLoader lookup called, inputs/kwargs {} {}".format(inputs, kwargs))
    return inputs

  def embed(self, inputs, **kwargs):
    print("MockLoader embed called, inputs/kwargs {} {}".format(inputs, kwargs))
    return inputs

class MockEncoder:
  def __init__(self, *args, **kwargs):
    """Only print stuff"""
    print("MockEncoder initialized, args/kwargs {} {}".format(args, kwargs))

  def encode(self, inputs, **kwargs):
    print("MockEncoder encode called, inputs/kwargs {} {}".format(inputs, kwargs))
    return inputs

  def __call__(self, inputs, num_layers=3, **kwargs):
    print("MockEncoder __call__ called, inputs/num_layers/kwargs {} {} {}".format(inputs, num_layers, kwargs))
    for i in range(num_layers):
      inputs = encode(inputs, **kwargs)
    return inputs

class MockDecoder:
  def __init__(self, *args, **kwargs):
    """Only print stuff"""
    print("MockDecoder initialized, args/kwargs {} {}".format(args, kwargs))

  def decode(self, inputs, **kwargs):
    print("MockDecoder decode called, inputs/kwargs {} {}".format(inputs, kwargs))
    return inputs

  def __call__(self, inputs, num_layers=3, **kwargs):
    print("MockDecoder __call__ called, inputs/num_layers/kwargs {} {} {}".format(inputs, num_layers, kwargs))
    for i in range(num_layers):
      inputs = decode(inputs, **kwargs)
    return inputs
