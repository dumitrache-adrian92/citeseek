import os

from datasets import load_dataset
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


class PaperDatabase:
    """
    Represents a system that integrates Elasticsearch and SentenceTransformer for managing
    and indexing papers with vector embeddings. This class is designed for use in
    environments where Elasticsearch is used to store and search for dense vectors.

    The class initializes connections to Elasticsearch when created, and provides methods
    to generate embeddings

    :ivar es: Instance of Elasticsearch used for storing and searching indexed data.
    :type es: Elasticsearch
    :ivar model: Pre-trained SentenceTransformer model used for generating text embeddings (MiniLM L6 v2).
    :type model: SentenceTransformer
    :ivar s2orc_api_key: API key for the S2ORC dataset, retrieved from environment variables.
    :type s2orc_api_key: Optional[str]
    :ivar tokenizer: Tokenizer associated with the SentenceTransformer model.
    :type tokenizer: PreTrainedTokenizer
    """

    def __init__(self, db_url):
        self.es = Elasticsearch(db_url)
        print('Connected to Elasticsearch!')
        print(self.es.info())
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.s2orc_api_key = os.getenv("S2ORC_API_KEY")
        self.tokenizer = self.model.tokenizer

        if not self.s2orc_api_key:
            raise ValueError("S2ORC_API_KEY environment variable not set")

        print('API key looks good!')

        if self.model.device.type != 'cuda':
            print('Model loaded and not on GPU!')
        else:
            print('Model loaded and on GPU!')

    def insert_document_in_index(self, document, index_name):
        """
        Inserts a single document into a specified Elasticsearch index with its 
        title and embedding.
    
        :param document: A dictionary containing the document to be indexed. 
                         Must include a 'title' key (str) and an 'abstract' key (str).
        :type document: dict
        :param index_name: The name of the Elasticsearch index where the document 
                           should be stored.
        :type index_name: str
        :return: None
        """
        self.es.index(index=index_name, document={
            'title': document['title'],
            'embedding': self.get_embedding(document['abstract'])
        })

    def get_embedding(self, text):
        """
        Generates a vector embedding for the given text using the pre-trained 
        SentenceTransformer model.
    
        :param text: The input text to generate the embedding for.
        :type text: str
        :return: A dense vector embedding representing the input text.
        :rtype: numpy.ndarray
        """
        return self.model.encode(text)

    def create_index(self, index_name):
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
                }
            }
        })

    def reindex(self, index_name):
        print("Loading dataset")

        ds = load_dataset("sentence-transformers/s2orc",
                          "title-abstract-pair",
                          split="train")

        print("Dataset loaded")

        for i, document in enumerate(ds):
            if i % 1000 == 0:
                print(f"Inserting document {i}")
            self.insert_document_in_index(document, index_name)

        print("Finished inserting batches")

    def check_already_indexed(self, title):
        rsp = self.es.search(index="paper_abstracts", body={
            "query": {
                "match": {
                    "title": title
                }
            }
        })

        return rsp['hits']['total']['value'] > 0
