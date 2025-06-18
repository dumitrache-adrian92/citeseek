import logging
import os
import re
import torch
from enum import Enum

from elasticsearch import Elasticsearch
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk import sent_tokenize
from paper_text_extractor.paper_text_extractor import get_paper_text
from pathlib import Path
from typing import TypedDict, List
from langgraph.graph import START, StateGraph

from missing_citation_retriever.reranker_strategy import (RerankerStrategy,
                                                          OpenAIRerankerStrategy,
                                                          GeminiRerankerStrategy)
from missing_citation_retriever.gemini_citing_sentence_classifier import GeminiCitingSentenceClassifier

citation_regex = r' ?(?:\[\d+(?:-\d+|(?:, ?\d+(-\d+)?)*)+\]|\([^)]*et al\.(?:, ?\d{4})?\)|[A-Za-z]+ et al\.) ?'


class RerankerProvider(str, Enum):
    """Available reranker providers for the MissingCitationRetriever."""
    OPENAI = "openai"
    GEMINI = "gemini"

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Return a list of all available provider names."""
        return [provider.value for provider in cls]


def _format_query_instruction(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def _contains_reference(sentence: str) -> bool:
    # Matches any citation in the form of [1], [1-3], [1, 3], [1, 3-5], [1, 3, 5-7], (Dumitrache et al., 2025), etc.

    return bool(re.search(citation_regex, sentence))


def _remove_reference_markers(sentence: str) -> str:
    # Removes citation markers like [1], [1-3], [1, 3], (Dumitrache et al., 2025), etc.
    return re.sub(citation_regex, '', sentence).strip()


class MissingCitationRetriever:
    """
    A class to retrieve missing citations for sentences in academic papers.

    This class uses a combination of a vector store for document retrieval,
    a text classifier for citing sentence classification, and a language model
    for re-ranking retrieved documents to identify and suggest missing citations
    in academic papers.

    Args:
        url (str): The URL of the Elasticsearch instance.
        embeddings_index (str): The name of the index to use for storing the embeddings.
        paper_index (str): The name of the index to use for storing the papers.

    Attributes:
        _embeddings (HuggingFaceEmbeddings): The embeddings model used for vector representation.
        _QUERY (str): The query instruction for retrieving relevant papers.
        _vector_store (ElasticsearchStore): The Elasticsearch store for vector embeddings.
        _es (Elasticsearch): The Elasticsearch client.
        _text_splitter (RecursiveCharacterTextSplitter): The text splitter for document chunking.
        _citing_sentence_classifier (pipeline): The HuggingFace pipeline for sentence classification.
        _reranker_strategy (RerankerStrategy): The strategy for re-ranking retrieved documents.
        _graph (StateGraph): The state graph for processing sentences.

    Methods:
        index_papers(titles, abstracts):
            Indexes a list of papers and their text embeddings in the vector store.

        check_paper(path):
            Analyzes a paper to identify sentences that likely contain missing citations
            and retrieves relevant papers for those sentences.

        classify_sentences(sentences):
            Classifies a list of sentences to determine if they contain citations.
    """

    class State(TypedDict):
        sentence: str
        retrieved: List[Document]
        reordered: List[str]

    def __init__(self,
                 url='http://localhost:9200',
                 embeddings_index='paper_embeddings',
                 paper_index='papers',
                 k=50,
                 silent: bool = False,
                 provider: str = 'gemini'):
        self.embeddings_index = embeddings_index
        self.paper_index = paper_index
        self.k = k

        if silent:
            logging.basicConfig(level=logging.WARNING)
        else:
            logging.basicConfig(level=logging.INFO)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True}
        self._embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large-instruct',
                                                 model_kwargs=model_kwargs,
                                                 encode_kwargs=encode_kwargs)
        logging.info(f"Loaded embedding model {self._embeddings.model_name} on device {device}")

        self._QUERY = "Given a sentence where a paper is cited, find the abstract of the paper it cites."
        self._vector_store = ElasticsearchStore(self.embeddings_index,
                                                embedding=self._embeddings,
                                                es_url=url)
        self._es = Elasticsearch(url)
        if not self._es.ping():
            raise ConnectionError("Elasticsearch instance not reachable.")

        self._text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100, add_start_index=True)

        self._citing_sentence_classifier = GeminiCitingSentenceClassifier()

        # Initialize the reranker strategy based on the provider
        provider = provider.lower()
        if provider == RerankerProvider.OPENAI.value:
            logging.info("Using OpenAI GPT for reranking")

            if not os.getenv('OPENAI_API_KEY'):
                raise ValueError("OPENAI_API_KEY is not set.")

            self._reranker_strategy = OpenAIRerankerStrategy(top_n=5, model='gpt-4o-mini')
        elif provider == RerankerProvider.GEMINI.value:
            logging.info("Using Gemini for reranking")

            if not os.getenv('GEMINI_API_KEY'):
                raise ValueError("GEMINI_API_KEY is not set.")

            self._reranker_strategy = GeminiRerankerStrategy(
                top_n=5,
                model='gemini-2.0-flash-001',
                api_key=os.getenv('GEMINI_API_KEY'),
                max_tokens=8192,
                context_size=8192
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}."
                             f"Use one of {RerankerProvider.get_available_providers()}.")

        graph_builder = StateGraph(self.State).add_sequence([self._retrieve, self._reorder])
        graph_builder.add_edge(START, "_retrieve")
        self._graph = graph_builder.compile()

    def index_papers(self, titles: list[str], abstracts: list[str]):
        """
        Indexes a list of papers and their text embeddings in the vector store.

        Args:
            titles (list[str]): A list of paper titles.
            abstracts (list[str]): A list of paper abstracts.
        Returns:
            None
        """
        ids = self._insert_documents_in_index([{'title': title, 'abstract': abstract}
                                               for title, abstract in zip(titles, abstracts)],
                                              self.paper_index)

        docs: list[Document] = [Document(page_content="\n\n".join([title, abstract]),
                                         metadata={'id': id_})
                                for title, abstract, id_ in zip(titles, abstracts, ids)]

        split_docs = self._text_splitter.split_documents(docs)

        self._vector_store.add_documents(split_docs)

    def check_paper(self, path: Path):
        """
        Analyzes a paper to identify sentences that likely contain missing citations and retrieves relevant papers for
        those sentences.

        Args:
            path (Path): The file path to the paper to be analyzed.

        Returns:
            List[dict]: A list of dictionaries where each dictionary contains a sentence and recommended papers.
        """
        raw_text = get_paper_text(path, remove_references=True, remove_abstract=True)

        paragraphs = [
            [
                re.sub("\n", " ", _remove_reference_markers(sent).replace("-\n", ""))
                for sent in paragraph
            ]
            for paragraph in [sent_tokenize(p) for p in raw_text.split('\n\n')]
            if len(paragraph) > 0
        ]
        paragraphs = [p for p in paragraphs if len(p) >= 3]

        already_citing = list(map(lambda paragraph: list(map(_contains_reference, paragraph)), paragraphs))

        predictions = self._citing_sentence_classifier.process_batch_parallel(paragraphs)

        # Find sentences that should contain citations according to the classifier, but don't
        sentences = []
        for paragraph, paragraph_citing, paragraph_predictions in zip(paragraphs, already_citing, predictions):
            for sentence, is_citing, should_cite in zip(paragraph, paragraph_citing, paragraph_predictions):
                if not is_citing and should_cite:
                    sentences.append(sentence)

        responses = [self._graph.invoke({'sentence': sentence}) for sentence in sentences]

        return [{r['sentence']: r['reordered']} for r in responses]

    def _insert_documents_in_index(self, documents: list[dict], index_name: str) -> list[str]:
        """
        Inserts a list of documents into a specified Elasticsearch index.

        Used to store clear paper text that chunks can point back to.

        Args:
            documents (list[dict]): A list of dictionaries containing the documents to be indexed.
            index_name (str): The name of the Elasticsearch index where the documents should be stored.

        Returns:
            list[str]: IDs of the inserted documents
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

        return [doc['index']['_id'] for doc in self._es.bulk(operations=operations)['items']]

    def _retrieve(self, state: State):
        """
        Retrieves papers relevant to the given sentence through a similarity search.

        Args:
            state (State): The current state of the system.

        Returns:
            dict: The updated state of the system.
        """
        instruction = _format_query_instruction(self._QUERY, state['sentence'])
        retrieved_snippets = self._vector_store.similarity_search(instruction,
                                                                  k=self.k,
                                                                  fetch_k=10000)

        # Retrieve complete documents using the IDs
        doc_ids = [snippet.metadata['id'] for snippet in retrieved_snippets]
        docs_response = self._es.mget(index=self.paper_index, body={'ids': doc_ids})
        docs = [doc['_source'] for doc in docs_response['docs'] if doc['found']]

        return {'retrieved': [Document(page_content=doc['title'] + '\n\n' + doc['abstract'],
                                       metadata={'title': doc['title']})
                              for doc in docs]}

    def _reorder(self, state: State):
        """
        Reorders retrieved papers based on the relevance to the given sentence using a language model reranker.

        Args:
            state (State): The current state of the system.

        Returns:
            dict: The updated state of the system.
        """
        instruction = _format_query_instruction(self._QUERY, state['sentence'])

        return {'reordered': self._reranker_strategy.rerank(state['retrieved'], instruction)}
