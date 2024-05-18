# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
from haystack.dataclasses import Document

from lancedb_haystack.document_store import LanceDBDocumentStore
from lancedb_haystack.embedding_retriever import LanceDBEmbeddingRetriever

# TODO: see if there's a cleaner way of testing an optional package that won't be available on all environments


def test_init_default(tmp_dir):
    store = LanceDBDocumentStore(tmp_dir)

    retriever = LanceDBEmbeddingRetriever(document_store=store)
    assert retriever._document_store == store
    assert retriever._filters == {}
    assert retriever._top_k == 10


def test_to_dict(tmp_dir):
    path = tmp_dir
    store = LanceDBDocumentStore(path, "test_table")
    retriever = LanceDBEmbeddingRetriever(document_store=store)
    res = retriever.to_dict()

    assert res == {
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


def test_from_dict(tmp_dir):
    path = tmp_dir
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


def test_run(tmp_dir):
    path = tmp_dir
    store = LanceDBDocumentStore(path, "test_table")
    retriever = LanceDBEmbeddingRetriever(document_store=store)
    store.write_documents([Document(content="Test doc", embedding=[0.5, 0.7])])
    res = retriever.run(query_embedding=[0.5, 0.7])

    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc"
