#
# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Any, Dict, Iterable, List, Optional
import datetime
import lancedb
import pyarrow as pa
from collections import OrderedDict
from haystack import Document, default_from_dict, default_to_dict
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy

from lancedb_haystack.filters import _convert_filters_to_where_clause_and_params, _in
from lancedb_haystack.type_utils import pyarrow_struct_to_dict, dict_to_pyarrow_struct

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

        :param database: The path to the database file to be opened.
        :param table_name: The name of the table in the lancedb to use.
        :param metadata_schema: The schema for the metadata to use if creating the table.
        :param embedding_dims: The size of the embedding vector to use if creating the table.
        """
        self._database = database
        self._table_name = table_name
        self._metadata_schema = prepare_metadata_schema(metadata_schema)
        self._embedding_dims = embedding_dims
        self.db = lancedb.connect(database)

    def _create_schema(self):
        if self._metadata_schema is None:
            err = "Trying to create new schema when metadata_schema is not specified."
            raise ValueError(err)

        if self._embedding_dims is None:
            err = "Trying to create new schema when embedding_dims is not specified."
            raise ValueError(err)

        return pa.schema(
            [
                pa.field("id", pa.string(), nullable=False),
                pa.field("vector", pa.list_(pa.float64(), list_size=self._embedding_dims)),  #lancedb.vector(self._embedding_dims, pa.float32())),
                pa.field("_isempty_vector", pa.bool_()),
                pa.field("content", pa.string()),
                pa.field("dataframe", pa.string()),  # Using a string so we can jam the dataframe in as json.
                pa.field("blob", pa.binary()),
                pa.field("meta", self._metadata_schema),
                # We skip score as this gets created by searches, and if we add it things break.
                # pa.field('score', pa.float32()),
            ]
        )

    def _table_exists(self) -> bool:
        return self._table_name in self.db.table_names()

    def count_documents(self) -> int:
        """
        Returns how many documents are present in the document store.
        """
        if self._table_exists():
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

        if not self._table_exists():
            return []

        table = self.db.open_table(self._table_name)

        if filters:
            query = _convert_filters_to_where_clause_and_params(filters)
            res = table.search().where(query).limit(0).to_list()
        else:
            res = table.search().limit(0).to_list()

        docs = [
            convert_lancedb_to_document(doc_dict, table.schema)
            for doc_dict in res
        ]

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

        if not self._table_exists():
            schema = self._create_schema()
            table = self.db.create_table(name=self._table_name, schema=schema, on_bad_vectors="fill", fill_value=0)
        else:
            table = self.db.open_table(self._table_name)
            schema = table.schema

        doc_dicts = [convert_document_to_lancedb(doc, schema) for doc in documents]

        if policy == DuplicatePolicy.NONE:
            policy = DuplicatePolicy.OVERWRITE

        if policy == DuplicatePolicy.OVERWRITE:
            table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(doc_dicts)
            unique_new_ids = {doc["id"] for doc in doc_dicts}
            num_modified = len(unique_new_ids)

        elif policy == DuplicatePolicy.SKIP:
            unique_new_ids = {doc["id"] for doc in doc_dicts}
            existing_ids = {
                res["id"] for res in table.search().where(_in("id", list(unique_new_ids))).select(["id"]).to_list()
            }
            num_modified = len(unique_new_ids - existing_ids)
            table.merge_insert("id").when_not_matched_insert_all().execute(doc_dicts)

        elif policy == DuplicatePolicy.FAIL:
            unique_new_ids = list({doc["id"] for doc in doc_dicts})
            existing_ids = {
                res["id"] for res in table.search().where(_in("id", unique_new_ids)).select(["id"]).to_list()
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

        if self._table_exists():
            table = self.db.open_table(self._table_name)
            where_clause = _in("id", object_ids)
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
        metadata_schema = data['init_parameters'].get('metadata_schema')
        if metadata_schema:
            metadata_schema = dict_to_pyarrow_struct(metadata_schema)
            data['init_parameters']['metadata_schema'] = metadata_schema

        return default_from_dict(cls, data)


def prepare_metadata_schema(struct: pa.StructType) -> pa.StructType:
    """Sort the fields in a struct into alphabetical order so that it doesn't complain when given a dict."""

    fields = []

    for idx in range(struct.num_fields):
        field = struct.field(idx)

        if isinstance(field, pa.StructType):
            field = prepare_metadata_schema(field)

        fields.append(field)

    fields.sort(key=lambda x: x.name)
    _isempty = [pa.field(f.name, pa.bool_()) for f in fields]
    fields.insert(0, pa.field("_isempty", pa.struct(_isempty)))

    return pa.struct(fields)


def convert_document_to_lancedb(document: Document, schema: pa.Schema) -> dict:
    """Converts a Haystack Document to a format ready to store in lancedb"""

    embed_dims = schema.field("vector").type.list_size
    meta_schema = schema.field('meta')
    doc_dict = document.to_dict(flatten=False)

    # convert embedding to vector and fill if missing
    embedding = doc_dict.pop("embedding")
    doc_dict["vector"] = embedding if embedding else [0] * embed_dims
    doc_dict["_isempty_vector"] = False if embedding else True

    # remove score - the retrievers will add this if necessary
    del doc_dict["score"]

    # fill missing metadata fields
    meta_dict = {}
    _isempty = {}
    for field_idx in range(meta_schema.type.num_fields):
        field = meta_schema.type.field(field_idx)
        field_name = field.name
        if field_name in doc_dict["meta"]:
            if str(field.type).startswith('timestamp'):
                meta_dict[field_name] = pa.scalar(
                    datetime.datetime.fromisoformat(doc_dict["meta"][field_name]),
                    type=field.type
                )
            else:
                meta_dict[field_name] = doc_dict["meta"][field_name]
        else:
            meta_dict[field_name] = None

        _isempty[field_name] = False if field_name in doc_dict["meta"] else True

    meta_dict["_isempty"] = _isempty
    doc_dict["meta"] = meta_dict

    # fill missing metadata fields
    for field_name in schema.names:
        value = doc_dict[field_name]
        doc_dict[field_name] = value if value else None

    return doc_dict


def convert_lancedb_to_document(result, schema) -> Document:
    """Convert a result lancedb into a document"""

    # Add score if it's not present; and take it from _distance if appropriate
    if "_distance" in result:
        result["score"] = result.pop("_distance")
    elif "score" not in result:
        result["score"] = None

    # filter the embedding
    embedding = result.pop("vector")
    if not result['_isempty_vector']:
        result["embedding"] = embedding

    del result['_isempty_vector']

    # filter the metadata
    meta_dict = _convert_metadata_lancedb(result['meta'], schema.field('meta').type)
    result['meta'] = meta_dict

    doc = Document.from_dict(result)
    return doc


def _convert_metadata_lancedb(metadata_dict: dict, meta_struct_type: pa.StructType) -> dict:
    """Converts the metadata section of """
    meta_dict = {}
    for k, v in metadata_dict.items():
        if (k != '_isempty') and not metadata_dict['_isempty'][k]:
            # only add if the field isn't flagged as empty
            if isinstance(v, datetime.datetime):
                v = v.isoformat()
            elif isinstance(v, dict):
                nested_type = meta_struct_type.field(k).type
                v = _convert_metadata_lancedb(v, nested_type)

            meta_dict[k] = v

    return meta_dict
