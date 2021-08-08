
import os
from ..retriever import DocDB
from ..tokenizers import CoreNLPTokenizer
from ..retriever import TfidfDocRanker
from .. import DATA_DIR

DEFAULTS = {
    'tokenizer': CoreNLPTokenizer,
    'ranker': TfidfDocRanker,
    'db': DocDB,
    'reader_model': os.path.join(DATA_DIR, 'reader/multitask.mdl'),
}


def default_it(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


from .drqa import DrQA
