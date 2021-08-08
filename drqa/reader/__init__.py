
import os
from ..tokenizers import CoreNLPTokenizer
from .. import DATA_DIR


DEFAULTS = {
    'tokenizer': CoreNLPTokenizer,
    'model': os.path.join(DATA_DIR, 'reader/single.mdl'),
}


def default_it(key, value):
    global DEFAULTS
    DEFAULTS[key] = value

from .model import DocReader
from .predictor import Predictor
from . import vector
from . import config
from . import utils
from . import data

