import os
DEFAULTS = {
    'corenlp_classpath': os.getenv('CLASSPATH')
}


def default_it(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


from .corenlp_tokenizer import CoreNLPTokenizer
from .regexp_tokenizer import RegexpTokenizer
from .simple_tokenizer import SimpleTokenizer

try:
    from .spacy_tokenizer import SpacyTokenizer
except ImportError:
    pass


def get_class(name):
    if name == 'spacy':
        return SpacyTokenizer
    if name == 'corenlp':
        return CoreNLPTokenizer
    if name == 'regexp':
        return RegexpTokenizer
    if name == 'simple':
        return SimpleTokenizer

    raise RuntimeError('Invalid tokenizer: %s' % name)


def fetch_annotators_for_arguments(args):
    annotators = set()
    if args.use_pos:
        annotators.add('pos')
    if args.use_lemma:
        annotators.add('lemma')
    if args.use_ner:
        annotators.add('ner')
    return annotators


def fetch_annotators_for_model(model):
    return fetch_annotators_for_arguments(model.args)
