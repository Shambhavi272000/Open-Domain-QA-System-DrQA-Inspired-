# REPRODUCING CLAIMED RESULTS AND USING THE MODEL 

* DrQA requires Linux/OSX and Python 3.5 or higher. It also requires installing PyTorch version 1.0. Its other dependencies are listed in requirements.txt. CUDA is strongly recommended for speed, but not necessary.

* Commands to clone the repository and install DrQA: 

```
git clone https://github.com/facebookresearch/DrQA.git
cd DrQA; pip install -r requirements.txt; python setup.py develop
```
* The tokenizer used is CoreNLP tokenizer, and it can be downloaded directly using: 
```
./install_corenlp.sh
```

# DOWNLOADING TRAINED MODEL AND DATA 

To download all trained models and data for Wikipedia question answering, run:
```
./download.sh
```
This downloads a 7.5GB tarball (25GB untarred), hence it takes several minutes. 

<B> TO RUN SPECIFIC MODELS, THE PATH CAN BE PROVIDED IN CODE LIKE: </B>
```
import drqa.reader

drqa.reader.set_default('model', '/path/to/model')

reader = drqa.reader.Predictor()  # Default model loaded for prediction
```

## DATASETS USED: 

*SQuAD: train, dev

*WebQuestions: train, test, entities

*WikiMovies: train/test/entities (Rehosted in expected format from https://research.fb.com/downloads/babi/)

*CuratedTrec: train/test (Rehosted in expected format from https://github.com/brmson/dataset-factoid-curated)


## FORMAT OF DATA REQUIERED 

The retriever/eval.py, pipeline/eval.py, and distant/generate.py scripts expect the datasets as a .txt file where each line is a JSON encoded QA pair, like: 
```
'{"question": "q1", "answer": ["a11", ..., "a1i"]}'

...

'{"question": "qN", "answer": ["aN1", ..., "aNi"]}'
```

The code to convert the data from SQuAD and WebQuestions is present in scripts/convert. However, it is already done in download.sh. 

The reader directory scripts expect the datasets as a .json file where the data is arranged like SQuAD:

![image](https://user-images.githubusercontent.com/67188688/128646473-751673ba-cb41-416f-876e-9bcfd80f5d9b.png)

 
# RUNNING THE DRQA AND SPECIFIC COMPONENTS OF DRQA USING THE INTERACTIVE SYSTEM
  
  ## DOCUMENT READER: 
 
  To interactively ask questions about texts given to a trained model, please run: 
  
  ```
  python scripts/reader/interactive.py 
  ```
 <b> To Train the Document reader, please refer :   </b>
  
  ## DOCUMENT RETRIEVER: 
 
  To interactively ask questions from Wikipedia , please run: 
  ```
  python scripts/retriever/interactive.py 
  ```
 <b> To Train the Document retriever, please refer :   </b>
  
  ## COMPLETE DrQA MODEL: 
  
  The full system is linked together in drqa.pipeline.DrQA.

  To interactively ask questions using the full DrQA, please run:
  
  ```
  python scripts/pipeline/interactive.py
  ```
  
 ## USING THE INTERACTIVE MODE: 
 
 Using this mode, one can directly start asking open-domain questions to wikipedia. 
 To do so, in the command line one has to write the queries in the follwing manner: 
 
  ```
  process('How does a jellyfish function without a brain or a nervous system?')
  
  ```

 
  
    




