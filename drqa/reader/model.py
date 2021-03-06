"""This file presents a DrQA document reader model."""
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
import copy
import numpy as np

from .config import override_model_arguments
from .rnn_reader import RnnDocReader

logger = logging.getLogger(__name__)


class DocReader(object):
    
    """This is a high level model which handles the intialization process of the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

#Initialization is carried on below
    def __init__(self, args, word_dict, feature_dict,
                 state_dict=None, normalize=True):
        
        # Book-keeping.
        self.args = args
        self.word_dict = word_dict
        self.args.vocab_size = len(word_dict)
        self.feature_dict = feature_dict
        self.args.num_features = len(feature_dict)
        self.updates = 0
        self.use_cuda = False
        self.parallel = False
        
        # Here we are building network. If normalize if false, scores are not normalized
        # 0-1 per paragraph (no softmax).
        if args.model_type == 'rnn':
            self.network = RnnDocReader(args, normalize)
        else:
            raise RuntimeError('Unsupported model: %s' % args.model_type)
        if state_dict:
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer('fixed_embedding', fixed_embedding)
            else:
                self.network.load_state_dict(state_dict)

    def dict_expand(self, words):
        
        """Here we are adding words to the DocReader dictionary if they do not exist. The
        underlying embedding matrix is also expanded (with random embeddings).
        Args:
            words: iterable of tokens to add to the dictionary.
        Output:
            added: set of tokens that were added.
        """
        to_add = {self.word_dict.normalize(w) for w in words
                  if w not in self.word_dict}
        
        # Adding words to dictionary and expanding the embedding layer

        
        if len(to_add) > 0:
            logger.info('Adding %d new words to dictionary...' % len(to_add))
            for w in to_add:
                self.word_dict.add(w)
            self.args.vocab_size = len(self.word_dict)
            logger.info('New vocab size: %d' % len(self.word_dict))

            old_embedding = self.network.embedding.weight.data
            self.network.embedding = torch.nn.Embedding(self.args.vocab_size,
                                                        self.args.embedding_dim,
                                                        padding_idx=0)
            new_embedding = self.network.embedding.weight.data
            new_embedding[:old_embedding.size(0)] = old_embedding
        return to_add

    def vec_embeddings_load(self, words, embedding_file):
        
        
         """Loading pretrained embeddings for a given list of words, if they exist.
        Arguments:
            words: iterable of tokens. Only those that are indexed in the
              dictionary are kept.
            embedding_file: path to text file of embeddings, space separated.
        """
        
        
        words = {w for w in words if w in self.word_dict}
        logger.info('Loading pre-trained embeddings for %d words from %s' %
                    (len(words), embedding_file))
        embedding = self.network.embedding.weight.data

        # When normalized, some words are duplicated. (Average the embeddings).

        
        
        vec_counts = {}
        with open(embedding_file) as f:
            line = f.readline().rstrip().split(' ')
            if len(line) != 2:
                f.seek(0)
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        logging.warning(
                            'WARN: Duplicate embedding found for %s' % w
                        )
                        vec_counts[w] = vec_counts[w] + 1
                        embedding[self.word_dict[w]].add_(vec)

        for w, c in vec_counts.items():
            embedding[self.word_dict[w]].div_(c)

        logger.info('Loaded %d embeddings (%.2f%%)' %
                    (len(vec_counts), 100 * len(vec_counts) / len(words)))

    def embed_tune(self, words):
        
        
        """Unfixing the embeddings of a list of words. This is only applicable if
        only few embeddings are being tuned (tune_partial = N).
        
        Shuffles the N specified words to the front of the dictionary, and saves
        the original vectors of the other N + 1:vocab words in a fixed buffer.
        Arguments:
            words: iterable of tokens contained in dictionary.
        """
        
        
        
        
        words = {w for w in words if w in self.word_dict}

        if len(words) == 0:
            logger.warning('Tried to tune embeddings, but no words given!')
            return

        if len(words) == len(self.word_dict):
            logger.warning('Tuning ALL embeddings in dictionary')
            return
        # Mixing vectors with words

        
        embedding = self.network.embedding.weight.data
        for idx, swap_word in enumerate(words, self.word_dict.START):
            
         # Getting current word + embedding for this index
            curr_word = self.word_dict[idx]
            curr_emb = embedding[idx].clone()
            old_idx = self.word_dict[swap_word]
            
          # Swapping embeddings + dictionary indices

            embedding[idx].copy_(embedding[old_idx])
            embedding[old_idx].copy_(curr_emb)
            self.word_dict[swap_word] = idx
            self.word_dict[idx] = swap_word
            self.word_dict[curr_word] = old_idx
            self.word_dict[old_idx] = curr_word
        self.network.register_buffer(
            'fixed_embedding', embedding[idx + 1:].clone()
        )

    def init_optimizer(self, state_dict=None):
        """Initializing an optimizer for the network's free parameters.
        Args:
            state_dict: network parameters
        """
        if self.args.fix_embeddings:
            for p in self.network.embedding.parameters():
                p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(parameters, self.args.learning_rate,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamax':
            self.optimizer = optim.Adamax(parameters,
                                          weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError('Unsupported optimizer: %s' %
                               self.args.optimizer)
            
         #Learning is done here
        
    def update(self, ex):
        """Forward a batch of examples; step the optimizer to update weights."""
        if not self.optimizer:
            raise RuntimeError('No optimizer set.')

       #Training mode is done here
        self.network.train()

        # Giving Transfer to GPU

        if self.use_cuda:
            inputs = [e if e is None else e.cuda(non_blocking=True)
                      for e in ex[:5]]
            target_s = ex[5].cuda(non_blocking=True)
            target_e = ex[6].cuda(non_blocking=True)
        else:
            inputs = [e if e is None else e for e in ex[:5]]
            target_s = ex[5]
            target_e = ex[6]
            
       # Running forward
    
        score_s, score_e = self.network(*inputs)

        # Computing losses and accuracies

        loss = F.nll_loss(score_s, target_s) + F.nll_loss(score_e, target_e)
        #  Removing gradients and run backward

        self.optimizer.zero_grad()
        loss.backward()

        
        torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                       self.args.grad_clipping)

         # Updating parameters

        self.optimizer.step()
        self.updates += 1

       
        self.para_reset()

        return loss.item(), ex[0].size(0)

    def para_reset(self):
        
        """Resetting any partially fixed parameters to original states."""

         # Resetting fixed embeddings to original value

        if self.args.tune_partial > 0:
            if self.parallel:
                embedding = self.network.module.embedding.weight.data
                fixed_embedding = self.network.module.fixed_embedding
            else:
                embedding = self.network.embedding.weight.data
                fixed_embedding = self.network.fixed_embedding

         # Embeddings to fix are the last indices

            offset = embedding.size(0) - fixed_embedding.size(0)
            if offset >= 0:
                embedding[offset:] = fixed_embedding

    
    def predict(self, ex, candidates=None, top_n=1, async_pool=None):
        
    """Forwarding a batch of examples only to get predictions.
        Arguments:
            ex: the batch
            candidates: batch * variable length list of string answer options.
              The model will only consider exact spans contained in this list.
            top_n: Number of predictions to return per batch element.
            async_pool: If provided, non-gpu post-processing will be offloaded
              to this CPU process pool.
        Output:
            pred_s: batch * top_n predicted start indices
            pred_e: batch * top_n predicted end indices
            pred_score: batch * top_n prediction scores
        If async_pool is given, these will be AsyncResult handles.
        """
       
        self.network.eval()

        if self.use_cuda:
            inputs = [e if e is None else e.cuda(non_blocking=True)
                      for e in ex[:5]]
        else:
            inputs = [e for e in ex[:5]]

      
        with torch.no_grad():
            score_s, score_e = self.network(*inputs)
        # Decode predictions
        score_s = score_s.data.cpu()
        score_e = score_e.data.cpu()
        if candidates:
            args = (score_s, score_e, candidates, top_n, self.args.max_len)
            if async_pool:
                return async_pool.apply_async(self.candidates_translate, args)
            else:
                return self.candidates_translate(*args)
        else:
            args = (score_s, score_e, top_n, self.args.max_len)
            if async_pool:
                return async_pool.apply_async(self.decode, args)
            else:
                return self.decode(*args)

    @staticmethod
    def decode(score_s, score_e, top_n=1, max_len=None):
         """Taking argmax of constrained score_s * score_e.
        Args:
            score_s: independent start predictions
            score_e: independent end predictions
            top_n: number of top scored pairs to take
            max_len: max span length to consider
        """
        
        pred_s = []
        pred_e = []
        pred_score = []
        max_len = max_len or score_s.size(1)
        for i in range(score_s.size(0)):
        # Outer product of scores to get full p_s * p_e matrix

            scores = torch.ger(score_s[i], score_e[i])

            scores.triu_().tril_(max_len - 1)

            scores = scores.numpy()
            scores_flat = scores.flatten()
            if top_n == 1:
                idx_sort = [np.argmax(scores_flat)]
            elif len(scores_flat) < top_n:
                idx_sort = np.argsort(-scores_flat)
            else:
                idx = np.argpartition(-scores_flat, top_n)[0:top_n]
                idx_sort = idx[np.argsort(-scores_flat[idx])]
            s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
            pred_s.append(s_idx)
            pred_e.append(e_idx)
            pred_score.append(scores_flat[idx_sort])
        return pred_s, pred_e, pred_score

    @staticmethod
    def candidates_translate(score_s, score_e, candidates, top_n=1, max_len=None):
       
        pred_s = []
        pred_e = []
        pred_score = []
        for i in range(score_s.size(0)):
            tokens = candidates[i]['input']
            cands = candidates[i]['cands']

            if not cands:
                from ..pipeline.drqa import PROCESS_CANDS
                cands = PROCESS_CANDS
            if not cands:
                raise RuntimeError('No candidates given.')
            # Score all valid candidates found in text.
            # Brute force get all ngrams and compare against the candidate list.
            max_len = max_len or len(tokens)
            scores, s_idx, e_idx = [], [], []
            for s, e in tokens.ngrams(n=max_len, as_strings=False):
                span = tokens.slice(s, e).untokenize()
                if span in cands or span.lower() in cands:
                    scores.append(score_s[i][s] * score_e[i][e - 1])
                    s_idx.append(s)
                    e_idx.append(e - 1)
                    
                # No candidates present
            if len(scores) == 0:
                pred_s.append([])
                pred_e.append([])
                pred_score.append([])
            else:
                scores = np.array(scores)
                s_idx = np.array(s_idx)
                e_idx = np.array(e_idx)

                idx_sort = np.argsort(-scores)[0:top_n]
                pred_s.append(s_idx[idx_sort])
                pred_e.append(e_idx[idx_sort])
                pred_score.append(scores[idx_sort])
        return pred_s, pred_e, pred_score

       # Saving and loading the model

    def save(self, filename):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        state_dict = copy.copy(network.state_dict())
        if 'fixed_embedding' in state_dict:
            state_dict.pop('fixed_embedding')
        params = {
            'state_dict': state_dict,
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    def checkpoint(self, filename, epoch):
        if self.parallel:
            network = self.network.module
        else:
            network = self.network
        params = {
            'state_dict': network.state_dict(),
            'word_dict': self.word_dict,
            'feature_dict': self.feature_dict,
            'args': self.args,
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
        }
        try:
            torch.save(params, filename)
        except BaseException:
            logger.warning('WARN: Saving failed... continuing anyway.')

    @staticmethod
    def load(filename, new_args=None, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        args = saved_params['args']
        if new_args:
            args = override_model_arguments(args, new_args)
        return DocReader(args, word_dict, feature_dict, state_dict, normalize)

    @staticmethod
    def load_checkpoint(filename, normalize=True):
        logger.info('Loading model %s' % filename)
        saved_params = torch.load(
            filename, map_location=lambda storage, loc: storage
        )
        word_dict = saved_params['word_dict']
        feature_dict = saved_params['feature_dict']
        state_dict = saved_params['state_dict']
        epoch = saved_params['epoch']
        optimizer = saved_params['optimizer']
        args = saved_params['args']
        model = DocReader(args, word_dict, feature_dict, state_dict, normalize)
        model.init_optimizer(optimizer)
        return model, epoch

    

    def cuda(self):
        self.use_cuda = True
        self.network = self.network.cuda()

    def cpu(self):
        self.use_cuda = False
        self.network = self.network.cpu()

    def parallelize(self):
        self.parallel = True
        self.network = torch.nn.DataParallel(self.network)
