The following read me contains information about a well performing model to generate short answers for factoid questions using Wikipedia articles as a knowledge base.

A proposed solution for providing long answers for a given question by referring Wikipedia articles or any other trusted knowledge base, despite the hurdles in the task, is present in the folder <B>LFQA.</B> 

# DrQA-Inspired-Open-Domain-QA-System

## INTRODUCTION:
This repository contains an understandable Pytorch implementation of open domain question answering system using Wikipedia as the knowledge source.
Open Domain question answering is a prime challenge in natural language processing (NLP) that involves extracting documents relevant to a given question and using them to generate appropriate answers.
This job can be accomplished by combining two different tasks:

   1.Document retrieval:  Selecting relevant documents from a large corpus of articles as per the query asked.  
   2.Document Reader:   Extracting answers to the asked question from a single document or a small collection of documents.
## DATA:
The work has been tested on three different kinds of data: 

 * Wikipedia- a knowledge source for finding answers
 
 * SQuAD dataset- Main resource to train document reader
 
 * Three other QA datasets (CuratedTREC, WebQuestions and WikiMovies)- To evaluate multitask learning ability of the system and distant supervision. 

The details of all datasets are as following: 
![image](https://user-images.githubusercontent.com/67188688/128638225-be200f58-a32b-4de3-a8fb-6ad40b53581b.png)


### 1. Wikipedia - Knowledge Source

1.The main dataset used for the task in this repository is Wikipedia English dump for the knowledge background for answering questions.

2.From all wikipedia pages, wikiextractor module was used to extract only the plain textual content.

3.The portions like images,lists, tables,outline pages are removed.

4.By a custom pre-processing function, the ambiguities not caught by WikiExtractor are removed and the documents are returned as a dictionary with key being the document “id” (article title) and values being the article text.

5.These preprocessed and cleaned documents are then stores in an sqlite database.

6. After these pre processing steps, we are left with 5,075,182 articles consisting of 9,008,962 unique uncased token types.

### SQuAD Dataset

1. The Stanford Question Answering Dataset(SQuAD) is a Wikipedia based dataset containing 87k training examples and 10k development examples. 
2. The unique feature of this dataset is it's large hidden test case only accessible to creators. 
3. Each instance of this dataste consists of :
     * Wikipedia article extracted paragraph
     * Paragraph associated human generated question 
     * Answer span from the paragraph

Example from SQuAD:  

QUESTION: How many provinces did the Ottoman empire contain in the 17th century?

ANSWER: 32

ARTICLE CHOSEN: Ottoman Empire

PARAGRAPH CHOSEN: ... At the beginning of the 17th century the empire contained 32 provinces and numerous vassal states. Some of these were later absorbed into the Ottoman Empire, while others were granted various types of autonomy during the course of centuries.

4. Generally, the given model had to predict the answer span from the paragraph and it is tested against the given span. The evaluation metrics used are <b> Exact string match (EM) </b> and <b> F1 score </b>.
5. We use SQuAD in two different evaluating procedures:
     * <b>Evaluating Document Reader-</b> Answering questions by using the relevant paragraphs given.
     * <b>Evaluating on Open-domain QA task-</b> By using only the QA pairs without access to the associated paragraphs(Whole Wikipedia is a resource).

### OPEN-DOMAIN QA EVALUATION RESOURCES

Instances of SQuAD datasets are very specific because a human annotator was asked to frame a question from a given paragraph. 
Other datasets,which have been developed differently and not just based on wikipedia, used for Open-domain QA evaluation are:

*CuratedTREC
*WebQuestions
*WikiMovies

### CREATING TRAINING DATA FOR DISTANTLY SUPERVISED DATA

The QA datasets like SQuAD, which have associated document for each question answering pair,can't be used for training our document reader. Hence, to train the document reader, a distantly supervised dataset was created by following a procedure. For each question answer pair: 

1. Document retriever is run with the question to extract top 5 wikipedia articles.
 
2. The paragraph from the retrieved pages having the following properties are discarded:

       * All paragraphs from articles without an exact match of the answer
       
       * Paragraphs shorter than 25 characters
       
       * Paragraphs longer than 1500 characters
       
       * Paragraphs not containing named entities if present in the question.
       
3. From the rest of the paragraphs, all positions that match an answer are scored using unigram and bigram overlap between question and a 20 token window.
 
4. The top 5 paragraphs having highest overlap are kept on top.

## MODEL

In the proposed method:

1.**The document retriever model** The document retriever technique used is a non-machine learning system which focuses on reading only the articles of purpose and discarding the ones out of context. The articles and questions are compared after calculating the TF-IDF weighted bag of vectors scores. The local word order information, which proves to be a very important feature for understanding the semantics of a sentence, is also preserved by using n-grams(especifically bigram counts) which are hashed using the feature hashing(unsigned murmur hash).

2.**The Document reader model** is a recurrent neural network model which is an attention based deep neural network that learns to read real documents and answer complex questions with minimal prior knowledge of language structure. 

The working of the document reader is described as follows:

 Feature engineering : ENCODING:

### * Paragraph encoding :
 For a paragraph containing m words, m tokens are generated as a sequence of feature vectors p˜i ∈ R d and then it is passed as input to the RNNs.   
 
                        
Here, contextual information of each paragraph token is extracted by concatenating each layer’s hidden units together at the end of a multi-layer bidirectional long short-term memory network (LSTM). 
The sequence vector hence obtained, P˜i contains Word embeddings using the 300-D Glove word embeddings(femb(pi) = E(pi) ), 
Exact match score(fexact match(pi) = I(pi ∈ q)) which shows whether pi exactly matches to any word in the query, Token features (ftoken(pi) = (POS(pi), NER(pi), TF(pi))) to elaborate contextual properties of each token.POS(pi) gives tags to tokens which indicate the part of speech the word belongs to like adjective,verb,noun etc.NER(pi) tags classify named entities into defined categories such as person names, organizations, locations etc. Aligned question embedding(falign(pi) =∑ j ai,jE(qj )) is the attention score between each para token and question words to take into account similar but non-identical words like crow and bird. The attention score is calculated as follows: 

                 a(i,j) = exp (α(E(pi)) · α(E(qj )))  ⁄  ∑ exp (α(E(pi)) • α(E(qj )))〗

### * Question encoding :
After obtaining the word embeddings of each word in the question only one recurrent neural network is applied to combine resulting hidden units into one single vector. 

                                        {q1, . . . , ql} → q 
The single vector is calculated as sum of product of each token vector and a weighted term bj. 

                                         q = ∑j bjqi 
The term bj incorporates the importance of each question words in the following way in which w is a weight to be learnt :

                                         bj = exp(w · qj )/ ∑j exp(w · qj') 
                                         
### * Prediction Task :
Once the relevant article is selected by the retriever, now the task is reduced to just deciding which part of the para has our desired answer. In terms of tokens, the task now is to predict the span of tokens which contains the correct answer.This is done by two different RNN classifiers which predict the start and end tokens. The probability distribution is done for each token being the start or end token by computing the similarity score of a token and the question vector.  
 
For start token:

Pstart(i) ∝ exp (piWsq)

For end token: 

Pend(i) ∝ exp (piWeq)

## RESULTS: 

The results were determined based on three different experiments: 
1. Evaluating the document retriever 
2. Evaluating the document reader
3. Their combination: DrQA: Open domain QA on complete Wikipedia.

### EVALUATION OF DOCUMENT RETRIEVER MODEL

* It is evaluated on all four datasets. 
* The performance of our system is compared with : Standard Wikipedia search engine, retrieval
  with Okapi BM25(by using cosine distance in the word embeddings space by encoding questions and articles as bag-of-embeddings). Our model outperforms both of these methods on  all the datasets.
* The evaluation metric is the ratio of questions in the dataset for which an answer is found in at least one of the top 5 pages returned by our retriever model. 
* The comparison with wikipedia search engine, which contains the percent of the questions with answers found in each dataset, is as follows: 
  ![image](https://user-images.githubusercontent.com/67188688/128637226-c2dc0c0e-7794-4d62-94b8-62b1dc18c5be.png)
  
 
### EVALUATION OF DOCUMENT READER MODEL

<b>Implementation of document reader for SQuAD</b>

 *MODEL:3-layer bidirectional LSTMs with h = 128 hidden units for both paragraph and question encoding*
 
Steps followed: 

* Stanford CoreNLP toolkit for tokenization, generating lemma, POS and named entity tags.
* All training examples are sorted length wise (of the paragraph associated with each instance).
* They are then divided into 32 minibatches. 
* Adamax was used as the optimization function. 
* Dropout with p = 0.3 is applied to word embeddings and all the hidden units of LSTMs. 

<b>Results</b>

* The single model is able to achieve the highest performance on SQuAD dataset: 70% exact match and 79% F1 score on the test set.
![image](https://user-images.githubusercontent.com/67188688/128637881-f6ae8d34-76b4-4836-8d4f-954311c0a7a3.png)

* It is also observed that dropping any feature reduces the F1 score accuracy evidently, as shown below:
![image](https://user-images.githubusercontent.com/67188688/128637737-9408d69b-8738-4d8b-8e02-7df6a8f83a64.png)

* Thus, all five features are equally important to obtain the maximum result. Still, without the aligned question embedding feature (only word embedding and a few manual features), our system is still able to achieve F1 over 77%.

### EVALUATION OF FULL DrQA SYSTEM: 

* Finally, the performance of the full system DrQA for answering open-domain questions is assessed using the four datasets.
* Three versions of DrQA are evaluated :
  1. SQuAD: A single Document Reader model is trained on the SQuAD training set only and used on all evaluation sets. 
  2. Fine-tune (DS): A Document Reader model is pre-trained on SQuAD and then fine-tuned
for each dataset independently using its distant supervision (DS) training set.
  3. Multitask (DS): A single Document Reader model is jointly trained on the SQuAD training set and all the DS sources. 
 
*Full Wikipedia results are as following. 
![image](https://user-images.githubusercontent.com/67188688/128638151-0b78ad5c-e0ca-4cc1-aa51-bb0b960ae37c.png)


Top-1 exact-match accuracy (in %, using SQuAD eval script). +Finetune (DS): Document Reader models trained on SQuAD and fine-tuned on each DS training set independently. +Multitask (DS): Document Reader single model trained on SQuAD and all the distant supervision (DS) training sets jointly.

* The single model trained only on SQuAD is outperformed on all four of the datasets by the multitask
model that uses distant supervision.

* However performance when training on SQuAD alone is not far behind, indicating that task transfer is occurring.

* <b>The best single model that we can find is our overall goal, and that is the Multitask (DS) system.</b>

* The unique method used for evaluating the full wikipedia is that a streamlined model is applied which does not use CoreNLP parsed ftoken features or lemmas for fexact match. 



## THE INFORMATION ON THE GITHUB STRUCTURE, FILE DESCRIPTIONS AND HOW TO RUN THE MODEL, PLEASE REFER- 








   
 
