# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
from typing import List

import pyarrow as pa
import pytest
from haystack.dataclasses import Document
from haystack.document_stores.errors import DuplicateDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.testing.document_store import DocumentStoreBaseTests

from lancedb_haystack.document_store import LanceDBDocumentStore


class TestLanceDBDocumentStore(DocumentStoreBaseTests):
    """
    Common test cases will be provided by `DocumentStoreBaseTests` but
    you can add more to this class.
    """

    @pytest.fixture
    def document_store(self, tmp_path) -> LanceDBDocumentStore:
        """
        This is the most basic requirement for the child class: provide
        an instance of this document store so the base class can use it.
        """
        schema = pa.struct(
            [
                ("name", pa.string()),
                ("page", pa.string()),
                ("chapter", pa.string()),
                ("number", pa.int32()),
                ("date", pa.timestamp("s")),
                ("no_embedding", pa.bool_()),
            ]
        )
        return LanceDBDocumentStore(
            database=tmp_path, table_name="test_table", metadata_schema=schema, embedding_dims=768
        )

    def assert_documents_are_equal(self, received: List[Document], expected: List[Document]):
        """
        Assert that two lists of Documents are equal.

        Over-ridden to avoid direct equality comparisons on the floating point embeddings
        """
        eps = pow(10, -6)
        are_equal = []
        for a, b in zip(received, expected):
            if type(a) != type(b):
                is_equal = False
            else:
                a_dict = a.to_dict()
                b_dict = b.to_dict()
                a_embedding = a_dict.pop("embedding")
                b_embedding = b_dict.pop("embedding")

                if (a_embedding is None) and (b_embedding is None):
                    embeddings_equal = True
                elif (
                    (a_embedding is not None)
                    and (b_embedding is not None)
                    and all(abs(a - b) < eps for a, b in zip(a_embedding, b_embedding))
                ):
                    embeddings_equal = True
                else:
                    embeddings_equal = False

                is_equal = (a_dict == b_dict) and embeddings_equal

            are_equal.append(is_equal)

        assert all(are_equal)

    def test_write_documents(self, document_store: DocumentStore):
        """
        Test write_documents() fails when trying to write Document with same id
        using DuplicatePolicy.FAIL.
        """
        doc = Document(content="test doc")
        assert document_store.write_documents([doc], policy=DuplicatePolicy.FAIL) == 1
        with pytest.raises(DuplicateDocumentError):
            document_store.write_documents(documents=[doc], policy=DuplicatePolicy.FAIL)
        self.assert_documents_are_equal(document_store.filter_documents(), [doc])

    def assert_documents_are_equivalent(self, received: List[Document], expected: List[Document]):
        """
        Assert that two lists of Documents are equivalent; or rather, equal but not necessarily the same order.
        """
        recv = received.sort(key=lambda x: x.id)
        exp = expected.sort(key=lambda x: x.id)
        assert recv == exp

    def test_not_operator(self, document_store, filterable_docs):
        document_store.write_documents(filterable_docs)
        result = document_store.filter_documents(
            filters={
                "operator": "NOT",
                "conditions": [
                    {"field": "meta.number", "operator": "==", "value": 100},
                    {"field": "meta.name", "operator": "==", "value": "name_0"},
                ],
            }
        )

        self.assert_documents_are_equivalent(
            result, [d for d in filterable_docs if not (d.meta.get("number") == 100 and d.meta.get("name") == "name_0")]
        )
