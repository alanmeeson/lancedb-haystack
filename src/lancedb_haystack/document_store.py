#
# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, Iterable, List, Optional

import lancedb
from haystack import Document, default_from_dict, default_to_dict
from haystack.document_stores.types import DocumentStore, DuplicatePolicy

from lancedb_haystack.filters import _convert_filters_to_where_clause_and_params, _in

logger = logging.getLogger(__name__)


class LanceDBDocumentStore(DocumentStore):
    """
    Stores data in LanceDB, and leverages its inbuilt search features.
    """

    def __init__(self, database: str, table_name: str):
        """
        Initializes the DocumentStore.

        :param database: The path to the database file to be opened.
        """
        self._database = database
        self._table_name = table_name
        self.db = lancedb.connect(database)

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        if self._table_name in self.db.table_names():
            table = self.db.open_table(self._table_name)
            count = table.count_rows()
        else:
            count = 0

        return count

    def filter_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Returns the documents that match the filters provided.

        Filters are defined as nested dictionaries that can be of two types:
        - Comparison
        - Logic

        Comparison dictionaries must contain the keys:

        - `field`
        - `operator`
        - `value`

        Logic dictionaries must contain the keys:

        - `operator`
        - `conditions`

        The `conditions` key must be a list of dictionaries, either of type Comparison or Logic.

        The `operator` value in Comparison dictionaries must be one of:

        - `==`
        - `!=`
        - `>`
        - `>=`
        - `<`
        - `<=`
        - `in`
        - `not in`

        The `operator` values in Logic dictionaries must be one of:

        - `NOT`
        - `OR`
        - `AND`


        A simple filter:
        ```python
        filters = {"field": "meta.type", "operator": "==", "value": "article"}
        ```

        A more complex filter:
        ```python
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.type", "operator": "==", "value": "article"},
                {"field": "meta.date", "operator": ">=", "value": 1420066800},
                {"field": "meta.date", "operator": "<", "value": 1609455600},
                {"field": "meta.rating", "operator": ">=", "value": 3},
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "meta.genre", "operator": "in", "value": ["economy", "politics"]},
                        {"field": "meta.publisher", "operator": "==", "value": "nytimes"},
                    ],
                },
            ],
        }

        :param filters: the filters to apply to the document list.
        :return: a list of Documents that match the given filters.
        """

        if self._table_name in self.db.table_names():
            table = self.db.open_table(self._table_name)
        else:
            return []

        if filters:
            query = _convert_filters_to_where_clause_and_params(filters)
            res = table.search().where(query).to_list()
        else:
            res = table.search().to_list()

        docs = []
        for doc_dict in res:
            if "score" not in doc_dict:
                doc_dict["score"] = None

            doc_dict["embedding"] = doc_dict.pop("vector")
            doc = Document.from_dict(doc_dict)
            docs.append(doc)

        return docs

    def write_documents(self, documents: List[Document], policy: DuplicatePolicy = DuplicatePolicy.NONE) -> int:
        """
        Writes (or overwrites) documents into the store.

        :param documents: a list of documents.
        :param policy: documents with the same ID count as duplicates. When duplicates are met,
            the store can:
             - skip: keep the existing document and ignore the new one.
             - overwrite: remove the old document and write the new one.
             - fail: an error is raised
        :raises DuplicateDocumentError: Exception trigger on duplicate document if `policy=DuplicatePolicy.FAIL`
        :return: None
        """
        if (
            not isinstance(documents, Iterable)
            or isinstance(documents, str)
            or any(not isinstance(doc, Document) for doc in documents)
        ):
            err = "Please provide a list of Documents."
            raise ValueError(err)

        doc_dicts = []
        for doc in documents:
            doc_dict = doc.to_dict(flatten=False)
            doc_dict["vector"] = doc_dict.pop("embedding")
            del doc_dict["score"]
            doc_dicts.append(doc_dict)

        if self._table_name not in self.db.table_names():
            table = self.db.create_table(self._table_name, doc_dicts)
            table.create_fts_index("content")
        else:
            table = self.db.open_table(self._table_name)

            if policy == DuplicatePolicy.NONE:
                policy = DuplicatePolicy.OVERWRITE

            if policy == DuplicatePolicy.OVERWRITE:
                table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(doc_dicts)

            elif policy == DuplicatePolicy.SKIP:
                table.merge_insert("id").when_not_matched_insert_all().execute(doc_dicts)

            elif policy == DuplicatePolicy.FAIL:
                err = "DuplicatePolicy.FAIL is not supported"
                raise ValueError(err)

        # LanceDB merge_insert doesn't say how many documents are written
        return 0

    def delete_documents(self, object_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.
        Fails with `MissingDocumentError` if no document with this id is present in the store.

        :param object_ids: the object_ids to delete
        """

        if self._table_name in self.db.table_names():
            table = self.db.open_table(self._table_name)
            where_clause = _in("id", object_ids)
            table.delete(where_clause)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this store to a dictionary.
        """
        data = default_to_dict(self, database=self._database, table_name=self._table_name)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LanceDBDocumentStore":
        """
        Deserializes the store from a dictionary.
        """
        return default_from_dict(cls, data)
