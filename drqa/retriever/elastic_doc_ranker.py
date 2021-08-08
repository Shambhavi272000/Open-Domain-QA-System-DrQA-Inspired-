import logging
import scipy.sparse as sp

from multiprocessing.pool import ThreadPool
from functools import partial
from elasticsearch import Elasticsearch

from . import utils
from . import DEFAULTS
from .. import tokenizers

logger = logging.getLogger(__name__)


class ElasticDocRanker(object):
    

    def __init__(self, elastic_url=None, elastic_index=None, elastic_fields=None, elastic_field_doc_name=None, strict=True, elastic_field_content=None):
        
        elastic_url = elastic_url or DEFAULTS['elastic_url']
        logger.info('Connecting to %s' % elastic_url)
        self.es = Elasticsearch(hosts=elastic_url)
        self.elastic_index = elastic_index
        self.elastic_fields = elastic_fields
        self.elastic_field_doc_name = elastic_field_doc_name
        self.elastic_field_content = elastic_field_content
        self.strict = strict

   

    def fetch_index_doc(self, doc_id):
       
        field_index = self.elastic_field_doc_name
        if isinstance(field_index, list):
            field_index = '.'.join(field_index)
        result = self.es.search(index=self.elastic_index, body={'query':{'match': 
            {field_index: doc_id}}})
        return result['hits']['hits'][0]['_id']
        

    def fetch_id_of_doc(self, doc_index):
        
        result = self.es.search(index=self.elastic_index, body={'query': { 'match': {"_id": doc_index}}})
        source = result['hits']['hits'][0]['_source']
        return utils.get_field(source, self.elastic_field_doc_name)

    def closest_docs(self, query, k=1):
       
        results = self.es.search(index=self.elastic_index, body={'size':k ,'query':
                                        {'multi_match': {
                                            'query': query,
                                            'type': 'most_fields',
                                            'fields': self.elastic_fields}}})
        hits = results['hits']['hits']
        doc_ids = [utils.get_field(row['_source'], self.elastic_field_doc_name) for row in hits]
        doc_scores = [row['_score'] for row in hits]
        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=1, num_workers=None):
       
        with ThreadPool(num_workers) as threads:
            closest_docs = partial(self.closest_docs, k=k)
            results = threads.map(closest_docs, queries)
        return results

    
    def __enter__(self):
        return self

    def close(self):
       
        self.es = None

    def fetch_id_of_docs(self):
        
        results = self.es.search(index= self.elastic_index, body={
            "query": {"match_all": {}}})
        doc_ids = [utils.get_field(result['_source'], self.elastic_field_doc_name) for result in results['hits']['hits']]
        return doc_ids

    def get_doc_text(self, doc_id):
        
        idx = self.fetch_index_doc(doc_id)
        result = self.es.get(index=self.elastic_index, doc_type='_doc', id=idx)
        return result if result is None else result['_source'][self.elastic_field_content]

