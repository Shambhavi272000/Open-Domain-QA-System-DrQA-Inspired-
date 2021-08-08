# Document Reader

## Preprocessing

`preprocess.py` takes a SQuAD-formatted dataset and outputs a preprocessed, training-ready file. Specifically, it handles tokenization, mapping character offsets to token offsets, and any additional featurization such as lemmatization, part-of-speech tagging, and named entity recognition.

To preprocess SQuAD (assuming both input and output files are in `data/datasets`):

```bash
python scripts/reader/preprocess.py data/datasets data/datasets --split SQuAD-v1.1-train
```
```bash
python scripts/reader/preprocess.py data/datasets data/datasets --split SQuAD-v1.1-dev
```
- _You need to have [SQuAD](../../README.md#qa-datasets) train-v1.1.json and dev-v1.1.json in data/datasets (here renamed as SQuAD-v1.1-<train/dev>.json)_

## Training

`train.py` is the main train script for the Document Reader. Requierments for training: 
- _ [glove embeddings](#note-on-word-embeddings) downloaded to data/embeddings/glove.840B.300d.txt._
- _Follow the preprocessing above._


Commands to run:

```bash
python scripts/reader/train.py --embedding-file glove.840B.300d.txt --tune-partial 1000
```


The training has many options that you can tune:


### Note on Word Embeddings

Using pre-trained word embeddings is very important for performance.Downloading the embeddings files and storing them under `data/embeddings/<file>.txt`. The code expects space separated plain text files (\<token\> \<d1\> ... \<dN\>).

- [GloVe: Common Crawl (cased)](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip)
- [FastText: Wikipedia (uncased)](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip)

## Predicting

`predict.py` uses a trained Document Reader model to make predictions for an input dataset.

Required arguments:
```
dataset               SQuAD-like dataset to evaluate on (format B).
```



Evaluation is done with the official_eval.py script from the SQuAD creators. It is available at `scripts/reader/official_eval.py` after running `./download.sh`.

```bash
python scripts/reader/official_eval.py /path/to/format/B/dataset.json /path/to/predictions/with/--official/flag/set.json
```

## Interactive

The Document Reader can also be used interactively (like the [full pipeline](../../README.md#quick-start-demo)).

```bash
python scripts/reader/interactive.py --model /path/to/model
```

```
>>> text = "Mary had a little lamb, whose fleece was white as snow. And everywhere that Mary went the lamb was sure to go."
>>> question = "What color is Mary's lamb?"
>>> process(text, question)

+------+-------+---------+
| Rank |  Span |  Score  |
+------+-------+---------+
|  1   | white | 0.78002 |
+------+-------+---------+
```
