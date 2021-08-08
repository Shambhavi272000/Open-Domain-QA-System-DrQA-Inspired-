# CHALLENGES FACED IN THE TASK OF PROVIDING LONG ELABORATIVE ANSWERS BY NLP MODELS

* Open-domain long-form question answering (LFQA) is a challenge in natural language processing (NLP).
* It involves retrieving documents relevant to a given question and using them to generate an *elaborate paragraph-length answer.
* Considerable amount of work has already been done in providing short, factoid answers to open-ended questions, but the area of LFQA is less worked upon.
* If we are able to prodcue long answers, it can serve as a major testbed to measure the semantic understanding of the model while generating the text.

* The current evaluation models and current benchmarks are not up to the mark for making progress of the LFQA models.
* The only large-scale, publicly available dataset for long-form question answering - EL15, has major shortcomings to evaluate LFQA models. 
* Inspired by an approach put forward by Google research in their paper - “Hurdles to Progress in Long-form Question Answering” , I present to you a possible approach
  for producing long answers for questions. 
  
* The major drawback of using ELI5 dataset are the following : 

1. Little evidence that models actually use the retrievals on which they condition
2. That trivial baselines (e.g., input copying) beat modern systems, like RAG / BART+DPR
3. There is a significant train/validation overlap in the dataset.
