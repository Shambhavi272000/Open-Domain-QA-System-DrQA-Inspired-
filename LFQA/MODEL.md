# POSSIBLE MODEL FOR PERFORMING LFQA: 

Imagine that we are taken with a sudden desire to understand how the fruit of a tropical tree gets transformed into chocolate bars, or want to understand the role of fever in the human body's immune response: how would we go about finding that information? For such tasks, short answers won't do. We need descriptive answers to accomplish the task. 

A new system that relies on sparse attention and contrastive retriever learning on ELI5 LFQA dataset comes close to providing satisfactory elaborative answers for a given question. 

The task involves integrating the retrieval component of open-domain QA, which involves searching a large external knowledge source for documents relevant to a given question, with a text generation component to produce paragraph-length answers.

The ELI5 task (Fan et al., 2019) asks models to generate paragraph-length answers to open-ended questions in English that often rely on world knowledge (e.g., how do jellyfish function without brains or nervous systems?). We have a two component model: 

<b> Retriever - </b> A dense retriever (“contrastive REALM” or C-REALM), which returns documents related to an input question.

<b> Generator - </b>  e the Routing Transformer (RT) from Roy et al. (2020), which is the current state-of-the-art in long-form language modeling. 


## RETRIEVER : Contrastive-REALM 

“Contrastive REALM” or C-REALM returns documents related to an input question.

Let me break down the term as per my understanding-

<B> CONTRASTIVE LEARNING: </B>
The goal of contrastive representation learning is to learn such an embedding space in which similar sample pairs stay close to each other while dissimilar ones are far apart.When working with unsupervised data, contrastive learning is one of the most powerful approaches in self-supervised learning.

<B> REALM </B>

*WHY A NEW ARCHITECTURE? 

We are not using  pre-trained models like BERT and RoBERTa which are capable of memorizing a surprising amount of world knowledge. The pointer here is that the these models memorize knowledge implicitly — i.e., world knowledge is captured in an abstract way in the model weights — making it difficult to determine what knowledge has been stored and where it is kept in the model.Also, as we try to increase the amount of knowledge to be stored, we have to simultaenously increase the size of the model which will make them slow or expensive. 

When these models, take for example BERT, is pretraine using MLM, it attains the world knowledge in a way and stores it in its model weights that is implicit knowledge. For example when trying to produce the output for the following example during pre-training, the BERT will memorize the information of where Einstein was born, but we will not know where it has been stored and how. 

Einstein was a __ born scientist. (answer: German)

<i>*THE NEW ARCHITECTURE-</i> 

In contrast to standard language representation models, REALM augments the language representation model with a knowledge retriever that first retrieves another piece of text from an external document collection as the supporting knowledge — in our experiments, we use the Wikipedia text corpus — and then feeds this supporting text as well as the original text into a language representation model as shown below- 
![image](https://user-images.githubusercontent.com/67188688/128645555-8b7a2106-f59b-4d72-9dce-dcc5052b4781.png)

The retriever works well because of the follwing principle, which is very close to human learning process: 

1. When the returned answer is correct, the model is encouraged.
2. When the returned answer is incorrect, the model is discouraged.

REALM utilizes the BERT model to learn good representations for a question and uses SCANN to retrieve Wikipedia articles that have a high topical similarity with the question representation. This is then trained end-to-end to maximize the log-likelihood on the QA task.

In REALM, the selection of the best document is carried out by using maximum inner product search (MIPS). 

The MIPS models do the following:
1. They first encode all of the documents in the corpus, such that each document has a corresponding document vector.
2. For every input, a query vector is encoded. 
3. In MIPS, given a query, the document in the collection that has the maximum inner product value between its document vector and the query vector is retrieved, as shown in the following figure:

![image](https://user-images.githubusercontent.com/67188688/128645704-74ba378d-365f-4862-b6ec-1d3199c91266.png)

The performance of c-REALM is improvd from REALM as it uses a contrastive loss function, which means to represent the question in the minibatch in a way which ensures it is grounded closer to the "ground-truth" answers and away from the irrelevant ones. 

### WHY DO WE NEED IT? 


For searching exactly matching queries in a database, SQL queries were sufficient. But, when it comes to producing results on the basis of semantic properties of a query, the task becomes extremely difficult if we only rely on criterias such as exact match,common number of terms etc. 

For example , the term "Animal Husbandry" is more related to "cows" than "husband", even though it has a matching term with the latter. 

This task has been much easier with the advent of Machine Learning and NLP as now we can understand the semnatics of language and answer abstract queries, which we are trying to do in the given task. Now that we can convert texts into matrices and vectors, all we have to do is compute the embedding of the given query and find the literary works whose embeddings are closest to the query’s.

In this task, a embedding-based search is a technique that is effective at answering queries that rely on semantic understanding rather than simple indexable properties. 
<b>In this technique, machine learning models are trained to map the queries and database items to a common vector embedding space, such that the distance between embeddings carries semantic meaning, i.e., similar items are closer together.</b>

To help speed this process up, Google research presented an effective MIPS technique by using vector similarity search library <b>(ScaNN)</b>.



### ALGORITHM 
Consider a corpus of long-form questions and answers, represented by (qi, ai). The retriever uses qi as a
query to retrieve K documents (ri,j ) from a knowledge corpus (Wikipedia), which is enabled by an encoder network that projects both questions and candidate documents to a 128-d shared embedding space. 




## GENERATOR MODEL- 

The generator model, which produces its generated answers based to the documents retrieved by the c-REALM. 
The generator model is actually a ROUTING TRANSFORMER (RT) which utilizes a sparse attention model which uses local attention as well as mini-batch k-means clustering so that long-range dependencies can be kept intact, which is crucial as we need to produce lengthy answers. 

By this method, a word can compute attention with another relevant word present <i>anywhere</i> in the text unlike the general algorithms where a word can only attend to words in close proximity or immediate locality. 
The RT model basically tries to reduce the unnecessary computation of each token attending to every other token as most of them do not contribute to the results. Hence, the new routing mechanism is the combination of global and local attention, to get the best of both worlds. 

The routing transformer also uses k-means to route similar queries into the same cluster for attention,which means that by btach k-means clustering, each token now has to attend to only a set of most relevant tokens. This method reduces the complexity of attention in the Transformer model from n2 to n1.5, where n is the sequence length, which enables it to scale to long sequences.

