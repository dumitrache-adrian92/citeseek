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

        Args:
            document (dict): A dictionary containing the document to be indexed.
                             Must include a 'title' key (str) and an 'abstract' key (str).
            index_name (str): The name of the Elasticsearch index where the document
                              should be stored.

        Returns:
            None
        """
        self.es.index(index=index_name, document={
            **document,
        })

    def insert_documents_in_index(self, documents: list[dict], index_name: str):
        """
        Inserts a list of documents into a specified Elasticsearch index with their
        titles and embeddings.

        Args:
            documents (list[dict]): A list of dictionaries containing the documents to be indexed.
                                    Each dictionary must include a 'title' key (str) and an 'abstract' key (str).
            index_name (str): The name of the Elasticsearch index where the documents should be stored.

        Returns:
            None
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

        return self.es.bulk(operations=operations)

    def create_index(self, index_name: str, mappings: dict) -> None:
        """
            Creates a new Elasticsearch index with a specified name and mapping configuration
            for embedding storage. Deletes the index first if it already exists.

            Args:
                index_name (str): The name of the Elasticsearch index to create or recreate.
                mappings (dict): The mapping configuration for the index.

            Returns:
                None
            """
        self.es.indices.delete(index=index_name, ignore_unavailable=True)
        self.es.indices.create(index=index_name, mappings=mappings)

    def check_already_indexed(self, title: str, index_name: str) -> bool:
        """
        Checks if a document with a given title is already indexed in the specified index.

        This function performs a search query on the Elasticsearch instance to determine
        whether a document with the given title exists in the index. If the search finds at
        least one result, it returns True, indicating the document is already indexed.
        Otherwise, it returns False.

        Args:
            title (str): The title of the document to check for existence in the index.
            index_name (str): The name of the Elasticsearch index to search within.

        Returns:
            bool: A boolean value indicating whether the document is already indexed.
        """
        rsp = self.es.search(index=index_name, body={
            "query": {
                "match_phrase": {
                    "title": title
                }
            }
        })

        return rsp['hits']['total']['value'] > 0

    def knn_search(self, query_vector: list[float], index_name: str, field: str, k: int = 10) -> list[dict]:
        """
        Performs a k-nearest neighbors search on the specified index using the given query vector.

        Args:
            query_vector (list[float]): The query vector to use for the k-NN search.
            index_name (str): The name of the Elasticsearch index to search within.
            field (str): The field in the index to use for the k-NN.
            k (int): The number of nearest neighbors to return.

        Returns:
            list[dict]: A list of the k nearest neighbors to the query vector.
        """
        rsp = self.es.search(
            index=index_name,
            knn={
                'field': field,
                'query_vector': query_vector,
                'num_candidates': 10000,
                'k': k,
            }
        )

        return rsp['hits']['hits']
