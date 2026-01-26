import importlib.metadata
from .train_bpe import train_bpe
from .tokenizer import Tokenizer
from .modules import *
from .utils import *

__version__ = importlib.metadata.version("cs336_basics")
