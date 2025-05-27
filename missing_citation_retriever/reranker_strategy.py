from abc import ABC, abstractmethod
from langchain_core.documents import Document
from typing import List
from langchain_community.document_compressors import RankLLMRerank
from rank_llm.data import Request, Query, Candidate
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import SafeGenai


class RerankerStrategy(ABC):
    """
    Abstract base class for reranker strategies.
    Defines the interface that all concrete reranker strategies must implement.
    """

    @abstractmethod
    def rerank(self, documents: List[Document], query: str) -> List[str]:
        """
        Rerank the retrieved documents based on the query.

        Args:
            documents (List[Document]): The documents to rerank
            query (str): The query to use for reranking

        Returns:
            List[str]: A list of reranked document titles
        """
        pass


class OpenAIRerankerStrategy(RerankerStrategy):
    """
    A reranker strategy that uses OpenAI's models through LangChain's RankLLMRerank.
    """

    def __init__(self, top_n: int = 5, model: str = 'gpt-4o-mini'):
        """
        Initialize the OpenAI reranker.

        Args:
            top_n (int): Number of top documents to return
            model (str): The OpenAI model to use
        """
        self._reranker = RankLLMRerank(top_n=top_n, model='gpt', gpt_model=model)

    def rerank(self, documents: List[Document], query: str) -> List[str]:
        """
        Rerank documents using OpenAI's model.

        Args:
            documents (List[Document]): The documents to rerank
            query (str): The query to use for reranking

        Returns:
            List[str]: A list of reranked document titles
        """
        reranked_docs = self._reranker.compress_documents(documents, query)
        return [doc.metadata['title'] for doc in reranked_docs]


class GeminiRerankerStrategy(RerankerStrategy):
    """
    A reranker strategy that uses Google's Gemini models through rank_llm.
    """

    def __init__(self, top_n: int = 5, model: str = 'gemini-2.0-flash-001',
                 api_key: str = None, max_tokens: int = 8192, context_size: int = 8192):
        """
        Initialize the Gemini reranker.

        Args:
            top_n (int): Number of top documents to return
            model (str): The Gemini model to use
            api_key (str): The API key for Gemini
            max_tokens (int): Maximum number of tokens
            context_size (int): Context size for the model
        """
        model_coordinator = SafeGenai(top_n=top_n, model=model,
                                      keys=api_key,
                                      max_tokens=max_tokens,
                                      context_size=context_size)
        self._reranker = Reranker(model_coordinator=model_coordinator)

    def rerank(self, documents: List[Document], query: str) -> List[str]:
        """
        Rerank documents using Gemini model.

        Args:
            documents (List[Document]): The documents to rerank
            query (str): The query to use for reranking

        Returns:
            List[str]: A list of reranked document titles
        """
        rerank_request = Request(
            query=Query(text=query, qid=1),
            candidates=[
                Candidate(
                    docid=doc.metadata['title'],
                    score=0.0,
                    doc={'passage': doc.page_content}
                ) for doc in documents
            ]
        )
        reranked_docs = self._reranker.rerank(rerank_request)
        return [doc.docid for doc in reranked_docs.candidates]
