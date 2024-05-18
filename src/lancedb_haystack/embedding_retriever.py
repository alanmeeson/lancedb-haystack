# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from haystack import DeserializationError, Document, component, default_from_dict, default_to_dict

from lancedb_haystack.document_store import LanceDBDocumentStore
from lancedb_haystack.filters import _convert_filters_to_where_clause_and_params


@component
class LanceDBEmbeddingRetriever:
    """
    A component for retrieving documents from an LanceDBDocumentStore using embeddings and vector similarity.
    """

    NAME = "lancedb_haystack.embedding_retriever.LanceDBEmbeddingRetriever"

    def __init__(
        self, document_store: LanceDBDocumentStore, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = 10
    ):
        """
        Create the LanceDBEmbeddingRetriever component.

        :param document_store: An instance of LanceDBDocumentStore.
        :param filters: A dictionary with filters to narrow down the search space. Defaults to `None`.
        :param top_k: The maximum number of documents to retrieve. Defaults to `10`.

        :raises ValueError: If the specified top_k is not > 0.
        :raises ValueError: If the provided document store is not an LanceDBDocumentStore
        """
        if not isinstance(document_store, LanceDBDocumentStore):
            err = "document_store must be an instance of LanceDBDocumentStore"
            raise ValueError(err)

        self._document_store = document_store

        if top_k and (top_k <= 0):
            err = f"top_k must be greater than 0. Currently, top_k is {top_k}"
            raise ValueError(err)

        self._filters = filters if filters else {}
        self._top_k = top_k

    @component.output_types(documents=List[Document])
    def run(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ):
        """
        Run the LanceDBEmbeddingRetriever on the given input data.

        :param query_embedding: Embedding of the query.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The maximum number of documents to return.
        :return: The retrieved documents.
        """
        filters = filters if filters else self._filters
        top_k = top_k if top_k else self._top_k

        # Perform the Embedding query
        if self._document_store._table_name in self._document_store.db.table_names():
            table = self._document_store.db.open_table(self._document_store._table_name)
        else:
            return []

        # Perform the FTS query
        query_builder = table.search(query_embedding)

        if filters:
            filter_query = _convert_filters_to_where_clause_and_params(filters)
            query_builder = query_builder.where(filter_query, prefilter=True)

        if top_k:
            query_builder = query_builder.limit(top_k)

        res = query_builder.to_list()

        docs = []
        for doc_dict in res:
            doc_dict["score"] = doc_dict.pop("_distance")
            doc_dict["embedding"] = doc_dict.pop("vector")

            doc = Document.from_dict(doc_dict)
            docs.append(doc)

        return {"documents": docs}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        docstore = self._document_store.to_dict()
        return default_to_dict(self, document_store=docstore, filters=self._filters, top_k=self._top_k)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LanceDBEmbeddingRetriever":
        """
        Deserialize this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})
        if "document_store" not in init_params:
            err = "Missing 'document_store' in serialization data"
            raise DeserializationError(err)
        if "type" not in init_params["document_store"]:
            err = "Missing 'type' in document store's serialization data"
            raise DeserializationError(err)
        data["init_parameters"]["document_store"] = LanceDBDocumentStore.from_dict(
            data["init_parameters"]["document_store"]
        )
        return default_from_dict(cls, data)
