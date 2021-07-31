# DrQA-Inspired-Open-Domain-QA-System

## INTRODUCTION:
This repository contains an understandable Pytorch implementation of open domain question answering system using Wikipedia as the knowledge source.
Open Domain question answering is a prime challenge in natural language processing (NLP) that involves extracting documents relevant to a given question and using them to generate appropriate answers.
This job can be accomplished by combining two different tasks:

   1.Document retrieval:  Selecting relevant documents from a large corpus of articles as per the query asked.  
   2.Document Reader:   Extracting answers to the asked question from a single document or a small collection of documents.
## DATA PREPROCESSING:

1.The main dataset used for the task in this repository is Wikipedia English dump for the knowledge background for answering questions.
2.From all wikipedia pages, wikiextractor module was used to extract only the plain textual content.
3.The portions like images,lists, tables,outline pages are removed.
4.By a custom pre-processing function, the ambiguities not caught by WikiExtractor are removed and the documents are returned as a dictionary with key being the document “id” (article title) and values being the article text.
5.These preprocessed and cleaned documents are then stores in an sqlite database. 

## MODEL

In the proposed method:
1.**The document retriever model** The document retriever technique used is a non-machine learning system which focuses on reading only the articles of purpose and discarding the ones out of context. The articles and questions are compared after calculating the TF-IDF weighted bag of vectors scores. The local word order information, which proves to be a very important feature for understanding the semantics of a sentence, is also preserved by using n-grams(especifically bigram counts) which are hashed using the feature hashing(unsigned murmur hash).

2.**The Document reader model** is a recurrent neural network model which is an attention based deep neural network that learns to read real documents and answer complex questions with minimal prior knowledge of language structure. 

The working of the document reader is described as follows:

 Feature engineering : ENCODING:

## * Paragraph encoding :
 For a paragraph containing m words, m tokens are generated as a sequence of feature vectors p˜i ∈ R d and then it is passed as input to the RNNs.   
 
                        
Here, contextual information of each paragraph token is extracted by concatenating each layer’s hidden units together at the end of a multi-layer bidirectional long short-term memory network (LSTM). 
The sequence vector hence obtained, P˜i contains Word embeddings using the 300-D Glove word embeddings(femb(pi) = E(pi) ), 
Exact match score(fexact match(pi) = I(pi ∈ q)) which shows whether pi exactly matches to any word in the query, Token features (ftoken(pi) = (POS(pi), NER(pi), TF(pi))) to elaborate contextual properties of each token.POS(pi) gives tags to tokens which indicate the part of speech the word belongs to like adjective,verb,noun etc.NER(pi) tags classify named entities into defined categories such as person names, organizations, locations etc. Aligned question embedding(falign(pi) =∑ j ai,jE(qj )) is the attention score between each para token and question words to take into account similar but non-identical words like crow and bird. The attention score is calculated as follows: 

                 a(i,j) = exp (α(E(pi)) · α(E(qj )))  ⁄  ∑ exp (α(E(pi)) • α(E(qj )))〗

## * Question encoding :
After obtaining the word embeddings of each word in the question only one recurrent neural network is applied to combine resulting hidden units into one single vector. 

                                        {q1, . . . , ql} → q 
The single vector is calculated as sum of product of each token vector and a weighted term bj. 

                                         q = ∑j bjqi 
The term bj incorporates the importance of each question words in the following way in which w is a weight to be learnt :

                                         bj = exp(w · qj )/ ∑j exp(w · qj') 
                                         
## * Prediction Task :
Once the relevant article is selected by the retriever, now the task is reduced to just deciding which part of the para has our desired answer. In terms of tokens, the task now is to predict the span of tokens which contains the correct answer.This is done by two different RNN classifiers which predict the start and end tokens. The probability distribution is done for each token being the start or end token by computing the similarity score of a token and the question vector.  
 
For start token:

Pstart(i) ∝ exp (piWsq)

For end token: 

Pend(i) ∝ exp (piWeq)



   
 
