"""The training script for the main DrQA reader is in this file."""

import argparse
import torch
import numpy as np
import json
import os
import sys
import subprocess
import logging


from drqa.reader import utils, vector, config, data
from drqa.reader import DocReader
from drqa import DATA_DIR as DRQA_DATA

logger = logging.getLogger()


# Training args are as follows

# Defaults
DATA_DIR = os.path.join(DRQA_DATA, 'datasets')
MODEL_DIR = '/tmp/drqa-models/'
EMBED_DIR = os.path.join(DRQA_DATA, 'embeddings')

def stringtobool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    parser.register('type', 'bool', stringtobool)

    # Providing runtime which will be useful during runtime
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--no-cuda', type='bool', default=False,
                         help='Train on CPU, even if GPUs are available.')
    runtime.add_argument('--gpu', type=int, default=-1,
                         help='Run on a specific GPU')
    runtime.add_argument('--data-workers', type=int, default=5,
                         help='Number of subprocesses for data loading')
    runtime.add_argument('--parallel', type='bool', default=False,
                         help='Use DataParallel on all available GPUs')
    runtime.add_argument('--random-seed', type=int, default=1013,
                         help=('Random seed for all numpy/torch/cuda '
                               'operations (for reproducibility)'))
    runtime.add_argument('--num-epochs', type=int, default=40,
                         help='Train data iterations')
    runtime.add_argument('--batch-size', type=int, default=32,
                         help='Batch size for training')
    runtime.add_argument('--test-batch-size', type=int, default=128,
                         help='Batch size during validation/testing')

    # Operations on files
    files = parser.add_argument_group('Filesystem')
    files.add_argument('--model-dir', type=str, default=MODEL_DIR,
                       help='Directory for saved models/checkpoints/logs')
    files.add_argument('--model-name', type=str, default='',
                       help='Unique model identifier (.mdl, .txt, .checkpoint)')
    files.add_argument('--data-dir', type=str, default=DATA_DIR,
                       help='Directory of training/validation data')
    files.add_argument('--train-file', type=str,
                       default='SQuAD-v1.1-train-processed-corenlp.txt',
                       help='Preprocessed train file')
    files.add_argument('--dev-file', type=str,
                       default='SQuAD-v1.1-dev-processed-corenlp.txt',
                       help='Preprocessed dev file')
    files.add_argument('--dev-json', type=str, default='SQuAD-v1.1-dev.json',
                       help=('Unprocessed dev file to run validation '
                             'while training on'))
    files.add_argument('--embed-dir', type=str, default=EMBED_DIR,
                       help='Directory of pre-trained embedding files')
    files.add_argument('--embedding-file', type=str,
                       default='glove.840B.300d.txt',
                       help='Space-separated pretrained embeddings file')

    # (Saving + loading) is carried on in this section
    save_load = parser.add_argument_group('Saving/Loading')
    save_load.add_argument('--checkpoint', type='bool', default=False,
                           help='Save model + optimizer state after each epoch')
    save_load.add_argument('--pretrained', type=str, default='',
                           help='Path to a pretrained model to warm-start with')
    save_load.add_argument('--expand-dictionary', type='bool', default=False,
                           help='Expand dictionary of pretrained model to ' +
                                'include training/dev words of new data')
    # Section for preprocessing of data
    preprocess = parser.add_argument_group('Preprocessing')
    preprocess.add_argument('--uncased-question', type='bool', default=False,
                            help='Question words will be lower-cased')
    preprocess.add_argument('--uncased-doc', type='bool', default=False,
                            help='Document words will be lower-cased')
    preprocess.add_argument('--restrict-vocab', type='bool', default=True,
                            help='Only use pre-trained words in embedding_file')

    # General
    general = parser.add_argument_group('General')
    general.add_argument('--official-eval', type='bool', default=True,
                         help='Validate with official SQuAD eval')
    general.add_argument('--valid-metric', type=str, default='f1',
                         help='The evaluation metric used for model selection')
    general.add_argument('--display-iter', type=int, default=25,
                         help='Log state after every <display_iter> epochs')
    general.add_argument('--sort-by-len', type='bool', default=True,
                         help='Sort batches by length for speed')


def default_its(args):
    """Confirming proper intialization of commandline arguments."""
    # Checking for the existence of critical files
    args.dev_json = os.path.join(args.data_dir, args.dev_json)
    if not os.path.isfile(args.dev_json):
        raise IOError('No such file: %s' % args.dev_json)
    args.train_file = os.path.join(args.data_dir, args.train_file)
    if not os.path.isfile(args.train_file):
        raise IOError('No such file: %s' % args.train_file)
    args.dev_file = os.path.join(args.data_dir, args.dev_file)
    if not os.path.isfile(args.dev_file):
        raise IOError('No such file: %s' % args.dev_file)
    if args.embedding_file:
        args.embedding_file = os.path.join(args.embed_dir, args.embedding_file)
        if not os.path.isfile(args.embedding_file):
            raise IOError('No such file: %s' % args.embedding_file)

    # Setting model directory
    subprocess.call(['mkdir', '-p', args.model_dir])

    # Seting name for the model
    if not args.model_name:
        import uuid
        import time
        args.model_name = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]

    # Setting (log + model) file names
    args.log_file = os.path.join(args.model_dir, args.model_name + '.txt')
    args.model_file = os.path.join(args.model_dir, args.model_name + '.mdl')

    # Options for embeddings
    if args.embedding_file:
        with open(args.embedding_file) as f:
            line = f.readline().rstrip().split(' ')
            if len(line) == 2:
                dim = int(line[1])
            else:
                dim = len(f.readline().rstrip().split(' ')) - 1
        args.embedding_dim = dim
    elif not args.embedding_dim:
        raise RuntimeError('Either embedding_file or embedding_dim '
                           'needs to be specified.')

    # Assuring the consistency of tune_partial and fix_embeddings.
    if args.tune_partial > 0 and args.fix_embeddings:
        logger.warning('WARN: fix_embeddings set to False as tune_partial > 0.')
        args.fix_embeddings = False

    # Assuring the consistency of embedding_file and fix_embeddings.
    if args.fix_embeddings:
        if not (args.embedding_file or args.pretrained):
            logger.warning('WARN: fix_embeddings set to False '
                           'as embeddings are random.')
            args.fix_embeddings = False
    return args


# Initalization from scratch.



def init_from_scratch(args, train_exs, dev_exs):
    """new data, New model,new dictionary."""
    # Building a feature dict from the annotations in the given data.
    logger.info('-' * 100)
    logger.info('Generate features')
    feature_dict = utils.build_feature_dict(args, train_exs)
    logger.info('Num features = %d' % len(feature_dict))
    logger.info(feature_dict)

    # Building a dictionary from the words+data questions with splits in train/dev
    logger.info('-' * 100)
    logger.info('Build dictionary')
    word_dict = utils.make_words_dict(args, train_exs + dev_exs)
    logger.info('Num words = %d' % len(word_dict))

    # Model Initialization
    model = DocReader(config.fetch_model_arguments(args), word_dict, feature_dict)

    # Loading pretrained embeddings for dcitionary words.
    if args.embedding_file:
        model.vec_embeddings_load(word_dict.tokens(), args.embedding_file)

    return model


# Loop for training.


def train(args, data_loader, model, global_stats):
    """Going through single epoch of model training with the provided data loader."""
    # Initializing timers and meters
    train_loss = utils.AverageMeter()
    epoch_time = utils.Timer()

    # Run single epoch
    for idx, ex in enumerate(data_loader):
        train_loss.update(*model.update(ex))

        if idx % args.display_iter == 0:
            logger.info('train: Epoch = %d | iter = %d/%d | ' %
                        (global_stats['epoch'], idx, len(data_loader)) +
                        'loss = %.2f | elapsed time = %.2f (s)' %
                        (train_loss.avg, global_stats['timer'].time()))
            train_loss.reset()

    logger.info('train: Epoch %d done. Time for epoch = %.2f (s)' %
                (global_stats['epoch'], epoch_time.time()))

    # Checkpoint
    if args.checkpoint:
        model.checkpoint(args.model_file + '.checkpoint',
                         global_stats['epoch'] + 1)


def unoffic_validate(args, data_loader, model, global_stats, mode):
    eval_time = utils.Timer()
    start_acc = utils.AverageMeter()
    end_acc = utils.AverageMeter()
    exact_match = utils.AverageMeter()

   
    examples = 0
    for ex in data_loader:
        batch_size = ex[0].size(0)
        pred_s, pred_e, _ = model.predict(ex)
        target_s, target_e = ex[-3:-1]

        
        accuracies = eval_accuracies(pred_s, target_s, pred_e, target_e)
        start_acc.update(accuracies[0], batch_size)
        end_acc.update(accuracies[1], batch_size)
        exact_match.update(accuracies[2], batch_size)

        
        examples += batch_size
        if mode == 'train' and examples >= 1e4:
            break

    logger.info('%s valid unofficial: Epoch = %d | start = %.2f | ' %
                (mode, global_stats['epoch'], start_acc.avg) +
                'end = %.2f | exact = %.2f | examples = %d | ' %
                (end_acc.avg, exact_match.avg, examples) +
                'valid time = %.2f (s)' % eval_time.time())

    return {'exact_match': exact_match.avg}


def offic_validate(args, data_loader, model, global_stats,
                      offsets, texts, answers):
   
    eval_time = utils.Timer()
    f1 = utils.AverageMeter()
    exact_match = utils.AverageMeter()

    examples = 0
    for ex in data_loader:
        ex_id, batch_size = ex[-1], ex[0].size(0)
        pred_s, pred_e, _ = model.predict(ex)

        for i in range(batch_size):
            s_offset = offsets[ex_id[i]][pred_s[i][0]][0]
            e_offset = offsets[ex_id[i]][pred_e[i][0]][1]
            prediction = texts[ex_id[i]][s_offset:e_offset]

            ground_truths = answers[ex_id[i]]
            exact_match.update(utils.metric_max_over_ground_truths(
                utils.exact_match_score, prediction, ground_truths))
            f1.update(utils.metric_max_over_ground_truths(
                utils.f1_score, prediction, ground_truths))

        examples += batch_size

    logger.info('dev valid official: Epoch = %d | EM = %.2f | ' %
                (global_stats['epoch'], exact_match.avg * 100) +
                'F1 = %.2f | examples = %d | valid time = %.2f (s)' %
                (f1.avg * 100, examples, eval_time.time()))

    return {'exact_match': exact_match.avg * 100, 'f1': f1.avg * 100}


def eval_accuracies(pred_s, target_s, pred_e, target_e):
    """An unofficial evalutation helper.
    Compute exact start/end/complete match accuracies for a batch.
    """
    if torch.is_tensor(target_s):
        target_s = [[e.item()] for e in target_s]
        target_e = [[e.item()] for e in target_e]

    batch_size = len(pred_s)
    start = utils.AverageMeter()
    end = utils.AverageMeter()
    em = utils.AverageMeter()
    for i in range(batch_size):
        if pred_s[i] in target_s[i]:
            start.update(1)
        else:
            start.update(0)

        if pred_e[i] in target_e[i]:
            end.update(1)
        else:
            end.update(0)

        if any([1 for _s, _e in zip(target_s[i], target_e[i])
                if _s == pred_s[i] and _e == pred_e[i]]):
            em.update(1)
        else:
            em.update(0)
    return start.avg * 100, end.avg * 100, em.avg * 100



def main(args):
    
    logger.info('-' * 100)
    logger.info('Load data files')
    train_exs = utils.load_the_data(args, args.train_file, skip_no_answer=True)
    logger.info('Num train examples = %d' % len(train_exs))
    dev_exs = utils.load_the_data(args, args.dev_file)
    logger.info('Num dev examples = %d' % len(dev_exs))

    if args.official_eval:
        dev_texts = utils.text_loader(args.dev_json)
        dev_offsets = {ex['id']: ex['offsets'] for ex in dev_exs}
        dev_answers = utils.load_answers(args.dev_json)

    # MODEL
    logger.info('-' * 100)
    start_epoch = 0
    if args.checkpoint and os.path.isfile(args.model_file + '.checkpoint'):
        # Just resume training, no modifications.
        logger.info('Found a checkpoint...')
        checkpoint_file = args.model_file + '.checkpoint'
        model, start_epoch = DocReader.load_checkpoint(checkpoint_file, args)
    else:
        # Training starts fresh. But the model state is either pretrained or
        # newly (randomly) initialized.
        if args.pretrained:
            logger.info('Using pretrained model...')
            model = DocReader.load(args.pretrained, args)
            if args.dict_expand:
                logger.info('Expanding dictionary for new data...')
                # Add words in training + dev examples
                words = utils.words_loader(args, train_exs + dev_exs)
                added = model.dict_expand(words)
                # Load pretrained embeddings for added words
                if args.embedding_file:
                    model.vec_embeddings_load(added, args.embedding_file)

        else:
            logger.info('Training model from scratch...')
            model = init_from_scratch(args, train_exs, dev_exs)

        # Set up partial tuning of embeddings
        if args.tune_partial > 0:
            logger.info('-' * 100)
            logger.info('Counting %d most frequent question words' %
                        args.tune_partial)
            top_words = utils.top_question_words(
                args, train_exs, model.word_dict
            )
            for word in top_words[:5]:
                logger.info(word)
            logger.info('...')
            for word in top_words[-6:-1]:
                logger.info(word)
            model.tune_embeddings([w[0] for w in top_words])

        # Set up optimizer
        model.init_optimizer()

    # Use the GPU?
    if args.cuda:
        model.cuda()

    # Use multiple GPUs?
    if args.parallel:
        model.parallelize()

    # DATA ITERATORS
    # Two datasets: train and dev. If we sort by length it's faster.
    logger.info('-' * 100)
    logger.info('Make data loaders')
    train_dataset = data.ReaderDataset(train_exs, model, single_answer=True)
    if args.sort_by_len:
        train_sampler = data.SortedBatchSampler(train_dataset.lengths(),
                                                args.batch_size,
                                                shuffle=True)
    else:
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchit,
        pin_memory=args.cuda,
    )
    dev_dataset = data.ReaderDataset(dev_exs, model, single_answer=False)
    if args.sort_by_len:
        dev_sampler = data.SortedBatchSampler(dev_dataset.lengths(),
                                              args.test_batch_size,
                                              shuffle=False)
    else:
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.test_batch_size,
        sampler=dev_sampler,
        num_workers=args.data_workers,
        collate_fn=vector.batchit,
        pin_memory=args.cuda,
    )

    # PRINT CONFIG
    logger.info('-' * 100)
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    # TRAIN/VALID LOOP
    logger.info('-' * 100)
    logger.info('Starting training...')
    stats = {'timer': utils.Timer(), 'epoch': 0, 'best_valid': 0}
    for epoch in range(start_epoch, args.num_epochs):
        stats['epoch'] = epoch

        # Train
        train(args, train_loader, model, stats)

        # Validate unofficial (train)
        unoffic_validate(args, train_loader, model, stats, mode='train')

        # Validate unofficial (dev)
        result = unoffic_validate(args, dev_loader, model, stats, mode='dev')

        # Validate official
        if args.official_eval:
            result = offic_validate(args, dev_loader, model, stats,
                                       dev_offsets, dev_texts, dev_answers)

        # Save best valid
        if result[args.valid_metric] > stats['best_valid']:
            logger.info('Best valid: %s = %.2f (epoch %d, %d updates)' %
                        (args.valid_metric, result[args.valid_metric],
                         stats['epoch'], model.updates))
            model.save(args.model_file)
            stats['best_valid'] = result[args.valid_metric]


if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'DrQA Document Reader',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    config.add_model_arguments(parser)
    args = parser.parse_args()
    default_its(args)

    # Set cuda
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.set_device(args.gpu)

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if args.log_file:
        if args.checkpoint:
            logfile = logging.FileHandler(args.log_file, 'a')
        else:
            logfile = logging.FileHandler(args.log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))

    # Run!
    main(args)
