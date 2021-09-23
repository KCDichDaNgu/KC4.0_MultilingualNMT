import os
import flask
from flask_cors import cross_origin

class TextHandler:
  def __init__(self):
    from nltk import word_tokenize, sent_tokenize
    self.sent_tokenize = sent_tokenize
    self.word_tokenize = word_tokenize

  def preprocess(self, text):
    """Split text (str) to list of tokenized sentences [list of str] and joints [list of char] to rejoin these sentences to original format."""
    text = text.strip()
    # split by \n
    paragraphs = text.split('\n') if '\n' in text else [text]
    # sent-ize the paragraphs
    paragraphs = [self.sent_tokenize(p) for p in paragraphs]
    # enumerate the join tokens: ' ' for inside and '\n' for outside prgp
    flatten_sents, joints = [], []
    for sents in paragraphs:
      flatten_sents.extend(sents)
      joints.extend([' '] * (len(sents)-1))
      joints.append('\n')
    # ignore final endline
    joints[-1] = ''
    # tokenize words inside sentences
    tokenized_sents = [" ".join(self.word_tokenize(l)) for l in flatten_sents]
    return tokenized_sents, joints

  def postprocess(self, outputs, joints):
    """Rejoin data to original format. TODO detokenize individual outputs as well"""
    apnd = (o+j for o, j in zip(outputs, joints))
    return ''.join(apnd)

static_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static") 
template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'template')

print("Path: ", static_path, template_path)

about_blueprint = flask.Blueprint("about", __name__, template_folder=template_path, static_folder=static_path)
@about_blueprint.route('/about')
def about():
  return flask.render_template("about.html")

def bind_blueprint(lazymodel, config):
  """Dynamically bind the model to the routes."""

  direct_translate = flask.Blueprint('direct_translate', __name__, template_folder=template_path)
  texthandler = TextHandler()

  @direct_translate.route("/translate_paragraphs", methods=["POST"])
  @cross_origin()
  def translate_paragraphs():
    requestContent = flask.request.get_json()
    direction = requestContent["direction"]
    raw_data = requestContent["data"].strip()
#    paragraphs = rawdata.split('\n') if '\n' in rawdata else [rawdata]
#    inputs = texthandler.sent_tokenize(raw_data)
#    tokenized_inputs = [" ".join(texthandler.word_tokenize(l)) for l in inputs]
    tokenized_inputs, joints = texthandler.preprocess(raw_data)
    
    target_lang = None
    outputs = lazymodel.model.translate_batch_sentence(tokenized_inputs, trg_lang=target_lang)
#    joined_outputs = '\n'.join(outputs)
    joined_outputs = texthandler.postprocess(outputs, joints)
#    print(requestContent)
    return flask.jsonify(data={"data": joined_outputs, "status": True})
  
  @direct_translate.route('/translate_demo')
  @cross_origin()
  def translate_sentence():
    flask.session["current_page"] = "direct_translate" 
    return flask.render_template('translate_sentence.html')

  return direct_translate
