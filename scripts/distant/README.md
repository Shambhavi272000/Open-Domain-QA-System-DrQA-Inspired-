# Distant Supervision

## Generating Data

Datasets like SQuAD provide both supporting contexts and exact answer spans along with a question, but datasets having this extent of supervision are not commonly found.

The other QA datasets considered in this project only contain question/answer pairs. 

Distant supervision is a way of automatically generating (noisy) full training examples (context, span, question) from these partial relations (QA pairs only) by using some heuristics.

`generate.py` runs a pipeline for generating distantly supervised datasets for DrQA. For doing so, run:

```bash
python generate.py /path/to/dataset/dir dataset /path/to/output/dir
```

The input dataset files must be in [format A](../../README.md#format-a).

The generated datasets are already in the preprocessed format required for the [Document Reader training](../reader/README.md#training). To combine different distantly supervised datasets, simply concatenate the files.

## Controlling Quality

Paragraphs are skipped if:

1. They are too long or short.
2. They don't contain a token match with the answer.
3. They don't contain token matches with named entities found in the question (using both NER recognizers from NLTK and the default for the `--tokenizer` option).
4. The overlap between the context and the question is too low.

## Checking Results

To visualize the generated data, run:

```bash
python check_data.py /path/to/generated/file
```

