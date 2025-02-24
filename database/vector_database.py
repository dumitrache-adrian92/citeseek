import os

from elasticsearch import Elasticsearch


class VectorDatabase:
    def __init__(self, db_url: str):
        self.es = Elasticsearch(db_url)
        print('Connected to Elasticsearch!')
        print(self.es.info())

    def insert_document_in_index(self, document: dict, index_name: str):
        """
        Inserts a single document into a specified Elasticsearch index with its 
        title and embedding.
    
        :param document: A dictionary containing `the document to be indexed.
                         Must include a 'title' key (str) and an 'abstract' key (str).
        :type document: dict
        :param index_name: The name of the Elasticsearch index where the document 
                           should be stored.
        :type index_name: str
        :return: None
        """
        self.es.index(index=index_name, document={
            **document,
        })

    def insert_documents_in_index(self, documents: list[dict], index_name: str):
        """
        Inserts a list of documents into a specified Elasticsearch index with their
        titles and embeddings.

        :param documents: A list of dictionaries containing the documents to be indexed.
                          Each dictionary must include a 'title' key (str) and an 'abstract' key (str).
        :type documents: list[dict]
        :param index_name: The name of the Elasticsearch index where the documents
                           should be stored.
        :type index_name: str
        :return: None
        """
        operations = []
        
        for document in documents:
            operations.append({
                'index': {
                    '_index': index_name
                }
            })

            operations.append({
                **document,
            })

        self.es.bulk(operations=operations)

    def create_index(self, index_name: str) -> None:
        """
        Creates a new Elasticsearch index with a specified name and maps it to use dense vectors 
        for embedding storage. Deletes the index first if it already exists.
    
        :param index_name: The name of the Elasticsearch index to create or recreate.
        :type index_name: str
        :return: None
        :rtype: NoneType
        """
        self.es.indices.delete(index=index_name, ignore_unavailable=True)
        self.es.indices.create(index=index_name, mappings={
            'properties': {
                'embedding': {
                    'type': 'dense_vector',
                    'similarity': 'cosine'
                }
            }
        })

    def check_already_indexed(self, title: str, index_name: str) -> bool:
        """
        Checks if a document with a given title is already indexed in the specified index.

        This function performs a search query on the Elasticsearch instance to determine
        whether a document with the given title exists in the index. If the search finds at
        least one result, it returns True, indicating the document is already indexed.
        Otherwise, it returns False.

        :param title: The title of the document to check for existence in the index.
        :type title: str
        :param index_name: The name of the Elasticsearch index to search within.
        :type index_name: str
        :return: A boolean value indicating whether the document is already indexed.
        :rtype: bool
        """
        rsp = self.es.search(index=index_name, body={
            "query": {
                "match": {
                    "title": title
                }
            }
        })

        return rsp['hits']['total']['value'] > 0

    def knn_search(self, query_vector: list[float], index_name: str, k: int = 10) -> list[dict]:
        """
        Performs a k-nearest neighbors search on the specified index using the given query vector.

        :param query_vector: The query vector to use for the k-NN search.
        :type query_vector: list[float]
        :param index_name: The name of the Elasticsearch index to search within.
        :type index_name: str
        :param k: The number of nearest neighbors to return.
        :type k: int
        """
        rsp = self.es.search(
            index=index_name,
            knn={
                'field': 'embedding',
                'query_vector': query_vector,
                'num_candidates': 10000,
                'k': k,
            }
        )

        return rsp['hits']['hits']
