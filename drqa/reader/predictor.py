

import logging

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize

from .vector import vectoriser, batchit
from . import DEFAULTS, utils
from .model import DocReader
from .. import tokenizers

logger = logging.getLogger(__name__)




PROCESS_TOK = None


def init(tokenizer_class, annotators):
    global PROCESS_TOK
    PROCESS_TOK = tokenizer_class(annotators=annotators)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)


def tokenize(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)





class Predictor(object):
    

    def __init__(self, model=None, tokenizer=None, normalize=True,
                 embedding_file=None, num_workers=None):
        
        logger.info('Initializing model...')
        self.model = DocReader.load(model or DEFAULTS['model'],
                                    normalize=normalize)

        if embedding_file:
            logger.info('Expanding dictionary...')
            words = utils.index_embedding_words(embedding_file)
            added = self.model.dict_expand(words)
            self.model.vec_embeddings_load(added, embedding_file)

        logger.info('Initializing tokenizer...')
        annotators = tokenizers.fetch_annotators_for_model(self.model)
        if not tokenizer:
            tokenizer_class = DEFAULTS['tokenizer']
        else:
            tokenizer_class = tokenizers.get_class(tokenizer)

        if num_workers is None or num_workers > 0:
            self.workers = ProcessPool(
                num_workers,
                initializer=init,
                initargs=(tokenizer_class, annotators),
            )
        else:
            self.workers = None
            self.tokenizer = tokenizer_class(annotators=annotators)

    def predict(self, document, question, candidates=None, top_n=1):
        
        results = self.predict_batch([(document, question, candidates,)], top_n)
        return results[0]

    def predict_batch(self, batch, top_n=1):
       
        documents, questions, candidates = [], [], []
        for b in batch:
            documents.append(b[0])
            questions.append(b[1])
            candidates.append(b[2] if len(b) == 3 else None)
        candidates = candidates if any(candidates) else None

        
        if self.workers:
            q_tokens = self.workers.map_async(tokenize, questions)
            d_tokens = self.workers.map_async(tokenize, documents)
            q_tokens = list(q_tokens.get())
            d_tokens = list(d_tokens.get())
        else:
            q_tokens = list(map(self.tokenizer.tokenize, questions))
            d_tokens = list(map(self.tokenizer.tokenize, documents))

        examples = []
        for i in range(len(questions)):
            examples.append({
                'id': i,
                'question': q_tokens[i].words(),
                'qlemma': q_tokens[i].lemmas(),
                'document': d_tokens[i].words(),
                'lemma': d_tokens[i].lemmas(),
                'pos': d_tokens[i].pos(),
                'ner': d_tokens[i].entities(),
            })

       
        if candidates:
            candidates = [{'input': d_tokens[i], 'cands': candidates[i]}
                          for i in range(len(candidates))]

        
        batch_exs = batchit([vectoriser(e, self.model) for e in examples])
        s, e, score = self.model.predict(batch_exs, candidates, top_n)

        results = []
        for i in range(len(s)):
            predictions = []
            for j in range(len(s[i])):
                span = d_tokens[i].slice(s[i][j], e[i][j] + 1).untokenize()
                predictions.append((span, score[i][j].item()))
            results.append(predictions)
        return results

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
