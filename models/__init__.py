from models.default import MockModel
from models.prototypes import Transformer as ProtoTransformer
from models.transformer import Transformer

AvailableModels = {
    "MockModel": MockModel, 
    "ProtoTransformer": ProtoTransformer, 
    "Transformer" : Transformer
}
