# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
import pyarrow as pa
from haystack.dataclasses import Document

from lancedb_haystack.document_store import LanceDBDocumentStore
from lancedb_haystack.embedding_retriever import LanceDBEmbeddingRetriever


def test_init_default(tmp_path):
    store = LanceDBDocumentStore(tmp_path, table_name="test_table")

    retriever = LanceDBEmbeddingRetriever(document_store=store)
    assert retriever._document_store == store
    assert retriever._filters == {}
    assert retriever._top_k == 10


def test_to_dict(tmp_path):
    path = str(tmp_path)
    store = LanceDBDocumentStore(path, "test_table")
    retriever = LanceDBEmbeddingRetriever(document_store=store)
    res = retriever.to_dict()

    assert res == {
        "type": "lancedb_haystack.embedding_retriever.LanceDBEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "database": path,
                    "table_name": "test_table",
                    "embedding_dims": None,
                    "metadata_schema": None,
                },
                "type": "lancedb_haystack.document_store.LanceDBDocumentStore",
            },
            "filters": {},
            "top_k": 10,
        },
    }


def test_from_dict(tmp_path):
    path = str(tmp_path)
    data = {
        "type": "lancedb_haystack.embedding_retriever.LanceDBEmbeddingRetriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {"database": path, "table_name": "test_table"},
                "type": "lancedb_haystack.document_store.LanceDBDocumentStore",
            },
            "filters": {},
            "top_k": 10,
        },
    }

    retriever = LanceDBEmbeddingRetriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10


def test_run(tmp_path):
    path = str(tmp_path)
    store = LanceDBDocumentStore(
        path, "test_table", metadata_schema=pa.struct([pa.field("foo", pa.string())]), embedding_dims=2
    )
    retriever = LanceDBEmbeddingRetriever(document_store=store)
    store.write_documents([Document(content="Test doc", embedding=[0.5, 0.7], meta={"foo": "a"})])
    res = retriever.run(query_embedding=[0.5, 0.7])

    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
