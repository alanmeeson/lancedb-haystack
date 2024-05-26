#
# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, Iterable, List, Optional, Union

import lancedb
import pyarrow as pa
from haystack import Document, default_from_dict, default_to_dict
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy

from lancedb_haystack.conversion.lancedb_to_python import convert_lancedb_to_document
from lancedb_haystack.conversion.python_to_lancedb import convert_document_to_lancedb
from lancedb_haystack.filters import convert_filters_to_where_clause, in_
from lancedb_haystack.schema.serialization import dict_to_pyarrow_struct, pyarrow_struct_to_dict

logger = logging.getLogger(__name__)


class LanceDBDocumentStore(DocumentStore):
    """
    Stores data in LanceDB, and leverages its inbuilt search features.
    """

    def __init__(
        self,
        database: str,
        table_name: str,
        metadata_schema: Optional[pa.StructType] = None,
        embedding_dims: Optional[int] = None,
    ):
        """
        Initializes the DocumentStore.

        If metadata_schema and embedding_dims are used, these will be used to construct the table schema.
        If they are not, then the schema will be inferred from the first document written. Note: this will limit the
        metadata which can be included in the DocumentStore to only those fields which are included in the first
        Document.
        If this is an existing DocumentStore, which already has a schema in it, said schema will override anything you
        pass in here.

        :param database: The path to the database file to be opened.
        :param table_name: The name of the table in the lancedb to use.
        :param metadata_schema: The schema for the metadata to use if creating the table.
        :param embedding_dims: The size of the embedding vector to use if creating the table.
        """
        self._database = database
        self._table_name = table_name
        self._metadata_schema = metadata_schema
        self._embedding_dims = embedding_dims
        self.db = lancedb.connect(database)

    def table_exists(self) -> bool:
        """Return True if the table already exists in the LanceDB backing this DocumentStore"""
        return self._table_name in self.db.table_names()

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        If the table doesn't exist yet, returns 0.
        """
        if self.table_exists():
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

        # If the table hasn't been created yet, just return an empty list.
        if not self.table_exists():
            return []

        # If the table does exist, open it and perform the search
        table = self.db.open_table(self._table_name)

        if filters:
            # If we have filters, construct the filtering clause and use it.
            query = convert_filters_to_where_clause(filters)
            res = table.search().where(query).limit(0).to_list()
        else:
            # Note: we have the limit(0) here and above so that we return _all_ documents.  Otherwise LanceDB defaults
            # to a limit of 10.
            res = table.search().limit(0).to_list()

        # Convert the results from LanceDB into Haystack Documents.
        # Note: we use the table.schema here as that will be the authoritative schema.
        docs = [convert_lancedb_to_document(doc_dict, table.schema) for doc_dict in res]

        return docs

    def perform_query(
        self,
        query: Optional[Union[str, List[float]]] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[Document]:

        if top_k and top_k <= 0:
            err = f"If specifying top_k, must be greater than 0. Currently, the top_k is {top_k}"
            raise ValueError(err)

        # If the table hasn't been created yet, just return an empty list.
        if not self.table_exists():
            return []

        # If the table does exist, open it and perform the search
        table = self.db.open_table(self._table_name)

        if query:
            query_builder = table.search(query)
        else:
            query_builder = table.search()

        if filters:
            # If we have filters, construct the filtering clause and use it.
            query = convert_filters_to_where_clause(filters)
            query_builder = query_builder.where(query)

        if top_k:
            query_builder = query_builder.limit(top_k)
        else:
            # Note: we have the limit(0) here and above so that we return _all_ documents.  Otherwise LanceDB defaults
            # to a limit of 10.
            query_builder = query_builder.limit(0)

        res = query_builder.to_list()

        # Convert the results from LanceDB into Haystack Documents.
        # Note: we use the table.schema here as that will be the authoritative schema.
        docs = [convert_lancedb_to_document(doc_dict, table.schema) for doc_dict in res]

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

        # Connect to the table and figure out the schema
        if self.table_exists():
            # If table already exists then we use it, and use the schema from it
            table = self.db.open_table(self._table_name)
            schema = table.schema
        else:
            # If the table doesn't already exist, then we use the metadata schema provided in the constructor
            schema = _create_schema(self._metadata_schema, self._embedding_dims)
            table = self.db.create_table(name=self._table_name, schema=schema, on_bad_vectors="fill", fill_value=0)

        # TODO: add something here that would handle the inferring schema from first document.

        # Convert the documents ready to insert
        doc_dicts = [convert_document_to_lancedb(doc, schema) for doc in documents]

        # Actually do the insert
        if policy == DuplicatePolicy.OVERWRITE or policy == DuplicatePolicy.NONE:  # noqa: PLR1714
            table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(doc_dicts)
            unique_new_ids = {doc["id"] for doc in doc_dicts}
            num_modified = len(unique_new_ids)

        elif policy == DuplicatePolicy.SKIP:
            unique_new_ids = {doc["id"] for doc in doc_dicts}
            existing_ids = {
                res["id"] for res in table.search().where(in_("id", list(unique_new_ids))).select(["id"]).to_list()
            }
            num_modified = len(unique_new_ids - existing_ids)
            table.merge_insert("id").when_not_matched_insert_all().execute(doc_dicts)

        elif policy == DuplicatePolicy.FAIL:
            unique_new_ids = {doc["id"] for doc in doc_dicts}
            existing_ids = {
                res["id"] for res in table.search().where(in_("id", list(unique_new_ids))).select(["id"]).to_list()
            }

            if len(existing_ids) > 0:
                raise DuplicateDocumentError()
            else:
                table.merge_insert("id").when_not_matched_insert_all().execute(doc_dicts)
                num_modified = len(unique_new_ids)

        table.create_fts_index("content", replace=True)

        return num_modified

    def delete_documents(self, object_ids: List[str]) -> None:
        """
        Deletes all documents with a matching document_ids from the document store.
        Fails with `MissingDocumentError` if no document with this id is present in the store.

        :param object_ids: the object_ids to delete
        """

        if self.table_exists():
            table = self.db.open_table(self._table_name)
            where_clause = in_("id", object_ids)
            table.delete(where_clause)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this store to a dictionary.
        """
        data = default_to_dict(
            self,
            database=self._database,
            table_name=self._table_name,
            metadata_schema=pyarrow_struct_to_dict(self._metadata_schema) if self._metadata_schema else None,
            embedding_dims=self._embedding_dims,
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LanceDBDocumentStore":
        """
        Deserializes the store from a dictionary.
        """
        metadata_schema = data["init_parameters"].get("metadata_schema")
        if metadata_schema:
            metadata_schema = dict_to_pyarrow_struct(metadata_schema)
            data["init_parameters"]["metadata_schema"] = metadata_schema

        return default_from_dict(cls, data)


# -------------------------------------------
# Functions for wrangling the schema


def _create_schema(metadata_schema: pa.StructType, embedding_dims: Optional[int]) -> pa.Schema:
    if metadata_schema is None:
        err = "Trying to create new schema when metadata_schema is not specified."
        raise ValueError(err)

    if embedding_dims is None:
        err = "Trying to create new schema when embedding_dims is not specified."
        raise ValueError(err)

    if embedding_dims <= 0:
        err = "Trying to create new schema with a negative or zero embedding length."
        raise ValueError(err)

    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("vector", pa.list_(pa.float32(), list_size=embedding_dims)),
            pa.field("content", pa.string()),
            pa.field("dataframe", pa.string()),  # Using a string, so we can jam the dataframe in as json.
            pa.field("blob", pa.binary()),
            pa.field("meta", _prepare_metadata_schema(metadata_schema)),
            pa.field("_isempty", _create_isempty_section(["blob", "content", "dataframe", "id", "meta", "vector"])),
        ]
    )


def _create_isempty_section(field_names) -> pa.StructType:
    """Creates the _isempty struct for the given list of fields.

    Haystack expects it's DocumentStores to return Documents which have only the fields they had when written.
    Unfortunately, LanceDB expects all fields to exist in all records, and not all types have easy 'None' analogues.
    To solve this we have a struct of boolean flags to indicate if a given field should be considered to be emtpy.
    """
    _isempty_type = pa.struct([pa.field(field_name, pa.bool_()) for field_name in field_names])
    return _isempty_type


def _prepare_metadata_schema(struct: pa.StructType) -> pa.StructType:
    """Take a pyarrow.StructType describing the metadata section and prepare it for use with LanceDB.

    This covers a couple of steps to address limitations:
    1. sorting the fields into alphabetical order.  If we don't do this, then LanceDB tends to complain when we give it
       a python dict, as those fields tend to be iterated in alphabetical order.
    2. Add the _isempty section to each StructType in the specification.  This lets us know if the field is meant to be
       empty in a given instance.
    """

    # Extract a list of all the fields, and recursively process and sub-structures
    fields = []
    for idx in range(struct.num_fields):
        field = struct.field(idx)

        if isinstance(field.type, pa.StructType):
            field = pa.field(field.name, _prepare_metadata_schema(field.type))

        fields.append(field)

    # Sort the fields by the field name
    fields.sort(key=lambda x: x.name)

    # Add the "_isempty" field
    is_empty_type = _create_isempty_section([f.name for f in fields])
    fields.insert(0, pa.field("_isempty", is_empty_type))

    return pa.struct(fields)
