"""Using the complete DrQA model"""
import math
import torch
import regex
import heapq
import logging
import time

from multiprocessing.util import Finalize
from multiprocessing import Pool as ProcessPool

from ..reader.vector import batchit
from ..reader.data import ReaderDataset, SortedBatchSampler
from .. import tokenizers
from .. import reader
from . import DEFAULTS

logger = logging.getLogger(__name__)

#Using multiprocessing to access and tokenize text
PROCESS_TOK = None   #(For tokenizing)
PROCESS_DB = None    #(For accessing database)
PROCESS_CANDS = None #(For finding possible answer candidates)


def init(tokenizer_class, tokenizer_opts, db_class, db_opts, candidates=None):
    global PROCESS_TOK, PROCESS_DB, PROCESS_CANDS
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)
    PROCESS_CANDS = candidates


def get_the_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)


def text_tokenizer(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)

#Pipeline
class DrQA(object):


    def __init__(
            self,
            reader_model=None,
            embedding_file=None,
            tokenizer=None,
            fixed_candidates=None,
            batch_size=128,
            cuda=True,
            data_parallel=False,
            max_loaders=5,
            num_workers=None,
            db_config=None,
            ranker_config=None
    ):
        """Descriptions of the arguments used: 
         1. reader_model - The file containing the Document reader
         2. embedding_file - using available pretrained embeddings other than that of the existing DocReader resources
         3. fixed_candidates: if given, all predictions will be constrated to the set of candidates contained in the file. One entry per line.
         4. batch_size: batch size of each group of paragraph sorted according to length while preprocessing.
         5. cuda: to specify usage of gpu.
         6. data_parallel: to specify usage of multile gpus.
         7. max_loaders: maximum number of async data loading workers when reading.
              
         8. num_workers: number of parallel CPU processes being used
               
         9. db_config: config for doc db.
         10.ranker_config: config for ranker.
        """
       
        self.batch_size = batch_size
        self.max_loaders = max_loaders
        self.fixed_candidates = fixed_candidates is not None
        self.cuda = cuda

        logger.info('Initializing document ranker...')
        ranker_config = ranker_config or {}
        ranker_class = ranker_config.get('class', DEFAULTS['ranker'])
        ranker_opts = ranker_config.get('options', {})
        self.ranker = ranker_class(**ranker_opts)

        logger.info('Initializing document reader...')
        reader_model = reader_model or DEFAULTS['reader_model']
        self.reader = reader.DocReader.load(reader_model, normalize=False)
        if embedding_file:
            logger.info('Expanding dictionary...')
            words = reader.utils.index_embedding_words(embedding_file)
            added = self.reader.dict_expand(words)
            self.reader.vec_embeddings_load(added, embedding_file)
        if cuda:
            self.reader.cuda()
        if data_parallel:
            self.reader.parallelize()

        if not tokenizer:
            tok_class = DEFAULTS['tokenizer']
        else:
            tok_class = tokenizers.get_class(tokenizer)
        annotators = tokenizers.fetch_annotators_for_model(self.reader)
        tok_opts = {'annotators': annotators}

        # ElasticSearch is also used as backend if used as ranker
        if hasattr(self.ranker, 'es'):
            db_config = ranker_config
            db_class = ranker_class
            db_opts = ranker_opts
        else:
            db_config = db_config or {}
            db_class = db_config.get('class', DEFAULTS['db'])
            db_opts = db_config.get('options', {})

        logger.info('Initializing tokenizers and document retrievers...')
        self.num_workers = num_workers
        self.processes = ProcessPool(
            num_workers,
            initializer=init,
            initargs=(tok_class, tok_opts, db_class, db_opts, fixed_candidates)
        )

    def break_doc(self, doc):
        curr = []
        curr_len = 0
        for split in regex.split(r'\n+', doc):
            split = split.strip()
            if len(split) == 0:
                continue
            # Maybe group paragraphs together until we hit a length limit
            if len(curr) > 0 and curr_len + len(split) > self.GROUP_LENGTH:
                yield ' '.join(curr)
                curr = []
                curr_len = 0
            curr.append(split)
            curr_len += len(split)
        if len(curr) > 0:
            yield ' '.join(curr)

    def fetch_loader(self, data, num_loaders):
        dataset = ReaderDataset(data, self.reader)
        sampler = SortedBatchSampler(
            dataset.lengths(),
            self.batch_size,
            shuffle=False
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=num_loaders,
            collate_fn=batchit,
            pin_memory=self.cuda,
        )
        return loader

    def process(self, query, candidates=None, top_n=1, n_docs=5,
                return_context=False):
        predictions = self.process_batch(
            [query], [candidates] if candidates else None,
            top_n, n_docs, return_context
        )
        return predictions[0]

    def process_batch(self, queries, candidates=None, top_n=1, n_docs=5,
                      return_context=False):
        t0 = time.time()
        logger.info('Processing %d queries...' % len(queries))
        logger.info('Retrieving top %d docs...' % n_docs)

        if len(queries) == 1:
            ranked = [self.ranker.closest_docs(queries[0], k=n_docs)]
        else:
            ranked = self.ranker.batch_closest_docs(
                queries, k=n_docs, num_workers=self.num_workers
            )
        all_docids, all_doc_scores = zip(*ranked)

        flat_docids = list({d for docids in all_docids for d in docids})
        did2didx = {did: didx for didx, did in enumerate(flat_docids)}
        doc_texts = self.processes.map(get_the_text, flat_docids)

        flat_splits = []
        didx2sidx = []
        for text in doc_texts:
            splits = self.break_doc(text)
            didx2sidx.append([len(flat_splits), -1])
            for split in splits:
                flat_splits.append(split)
            didx2sidx[-1][1] = len(flat_splits)

        q_tokens = self.processes.map_async(text_tokenizer, queries)
        s_tokens = self.processes.map_async(text_tokenizer, flat_splits)
        q_tokens = q_tokens.get()
        s_tokens = s_tokens.get()

        examples = []
        for qidx in range(len(queries)):
            for rel_didx, did in enumerate(all_docids[qidx]):
                start, end = didx2sidx[did2didx[did]]
                for sidx in range(start, end):
                    if (len(q_tokens[qidx].words()) > 0 and
                            len(s_tokens[sidx].words()) > 0):
                        examples.append({
                            'id': (qidx, rel_didx, sidx),
                            'question': q_tokens[qidx].words(),
                            'qlemma': q_tokens[qidx].lemmas(),
                            'document': s_tokens[sidx].words(),
                            'lemma': s_tokens[sidx].lemmas(),
                            'pos': s_tokens[sidx].pos(),
                            'ner': s_tokens[sidx].entities(),
                        })

        logger.info('Reading %d paragraphs...' % len(examples))

        result_handles = []
        num_loaders = min(self.max_loaders, math.floor(len(examples) / 1e3))
        for batch in self.fetch_loader(examples, num_loaders):
            if candidates or self.fixed_candidates:
                batch_cands = []
                for ex_id in batch[-1]:
                    batch_cands.append({
                        'input': s_tokens[ex_id[2]],
                        'cands': candidates[ex_id[0]] if candidates else None
                    })
                handle = self.reader.predict(
                    batch, batch_cands, async_pool=self.processes
                )
            else:
                handle = self.reader.predict(batch, async_pool=self.processes)
            result_handles.append((handle, batch[-1], batch[0].size(0)))

        queues = [[] for _ in range(len(queries))]
        for result, ex_ids, batch_size in result_handles:
            s, e, score = result.get()
            for i in range(batch_size):
                if len(score[i]) > 0:
                    item = (score[i][0], ex_ids[i], s[i][0], e[i][0])
                    queue = queues[ex_ids[i][0]]
                    if len(queue) < top_n:
                        heapq.heappush(queue, item)
                    else:
                        heapq.heappushpop(queue, item)

        all_predictions = []
        for queue in queues:
            predictions = []
            while len(queue) > 0:
                score, (qidx, rel_didx, sidx), s, e = heapq.heappop(queue)
                prediction = {
                    'doc_id': all_docids[qidx][rel_didx],
                    'span': s_tokens[sidx].slice(s, e + 1).untokenize(),
                    'doc_score': float(all_doc_scores[qidx][rel_didx]),
                    'span_score': float(score),
                }
                if return_context:
                    prediction['context'] = {
                        'text': s_tokens[sidx].untokenize(),
                        'start': s_tokens[sidx].offsets()[s][0],
                        'end': s_tokens[sidx].offsets()[e][1],
                    }
                predictions.append(prediction)
            all_predictions.append(predictions[-1::-1])

        logger.info('Processed %d queries in %.4f (s)' %
                    (len(queries), time.time() - t0))

        return all_predictions
