from .default_loader import DefaultLoader
from .multilingual_loader import MultiLoader

loaders = {"monoloader": DefaultLoader, "multiloader": MultiLoader}
