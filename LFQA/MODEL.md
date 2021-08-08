# POSSIBLE MODEL FOR PERFORMING LFQA: 

Imagine that we are taken with a sudden desire to understand how the fruit of a tropical tree gets transformed into chocolate bars, or want to understand the role of fever in the human body's immune response: how would we go about finding that information? For such tasks, short answers won't do. We need descriptive answers to accomplish the task. 

A new system that relies on sparse attention and contrastive retriever learning on ELI5 LFQA dataset comes close to providing satisfactory elaborative answers for a given question. 

The task involves integrating the retrieval component of open-domain QA, which involves searching a large external knowledge source for documents relevant to a given question, with a text generation component to produce paragraph-length answers.

The ELI5 task (Fan et al., 2019) asks models to generate paragraph-length answers to open-ended questions in English that often rely on world knowledge (e.g., how do jellyfish function without brains or nervous systems?). We have a two component model: 

<b> Retriever - </b> A dense retriever (“contrastive REALM” or C-REALM), which returns documents related to an input question.

<b> Retriever - </b> 

