# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

from haystack import DeserializationError, Document, component, default_from_dict, default_to_dict

from lancedb_haystack.document_store import LanceDBDocumentStore


@component
class LanceDBFTSRetriever:
    """
    A component for retrieving documents from an LanceDBDocumentStore using the FTS.
    """

    NAME = "lancedb_haystack.fts_retriever.LanceDBFTSRetriever"

    def __init__(
        self, document_store: LanceDBDocumentStore, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = 10
    ):
        """
        Create an LanceDBFTSRetriever component. Usually you pass some basic configuration
        parameters to the constructor.

        :param document_store: A Document Store object used to retrieve documents
        :param filters: A dictionary with filters to narrow down the search space (default is None).
        :param top_k: The maximum number of documents to retrieve (default is 10).
        :raises ValueError: If the specified top_k is not > 0.
        :raises ValueError: If the provided document store is not an LanceDBDocumentStore
        """
        if not isinstance(document_store, LanceDBDocumentStore):
            err = "document_store must be an instance of LanceDBDocumentStore"
            raise ValueError(err)

        self._document_store = document_store

        if top_k and top_k <= 0:
            err = f"top_k must be greater than 0. Currently, the top_k is {top_k}"
            raise ValueError(err)

        self._filters = filters if filters else {}
        self._top_k = top_k

    @component.output_types(documents=List[Document])
    def run(self, query: str, filters: Optional[Dict[str, Any]] = None, top_k: Optional[int] = 10):
        """
        Run the LanceDBFTSRetriever on the given input data.

        :param query: The query string for the Retriever.
        :param filters: A dictionary with filters to narrow down the search space.
        :param top_k: The maximum number of documents to return.
        :return: The retrieved documents.

        :raises ValueError: If the specified DocumentStore is not found or is not a LanceDBFTSRetriever instance.
        """

        filters = filters if filters else self._filters
        top_k = top_k if top_k else self._top_k

        if not query:
            err = "Query should be a non-empty string"
            raise ValueError(err)

        docs = self._document_store.perform_query(query=query, filters=filters, top_k=top_k)

        return {"documents": docs}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        docstore = self._document_store.to_dict()
        return default_to_dict(
            self,
            document_store=docstore,
            filters=self._filters,
            top_k=self._top_k,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LanceDBFTSRetriever":
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
