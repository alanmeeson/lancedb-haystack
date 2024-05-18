# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock, patch

import lancedb
from haystack.dataclasses import Document

from lancedb_haystack.document_store import LanceDBDocumentStore
from lancedb_haystack.fts_retriever import LanceDBFTSRetriever


def test_init_default():
    mock_store = Mock(spec=LanceDBDocumentStore)
    mock_store.db = Mock(spec=lancedb.DBConnection)
    retriever = LanceDBFTSRetriever(document_store=mock_store)
    assert retriever._document_store == mock_store
    assert retriever._filters == {}
    assert retriever._top_k == 10


@patch("lancedb_haystack.document_store.LanceDBDocumentStore")
def test_to_dict(tmp_dir):
    path = tmp_dir
    document_store = LanceDBDocumentStore(path, "test_table")
    retriever = LanceDBFTSRetriever(document_store=document_store)
    res = retriever.to_dict()

    assert res == {
        "type": "lancedb_haystack.fts_retriever.LanceDBFTSRetriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {"database": path, "table_name": "test_table"},
                "type": "lancedb_haystack.document_store.LanceDBDocumentStore",
            },
            "filters": {},
            "top_k": 10,
        },
    }


@patch("lancedb_haystack.document_store.LanceDBDocumentStore")
def test_from_dict(tmp_dir):
    path = tmp_dir
    data = {
        "type": "lancedb_haystack.fts_retriever.LanceDBFTSRetriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {"database": path, "table_name": "test_table"},
                "type": "lancedb_haystack.document_store.LanceDBDocumentStore",
            },
            "filters": {},
            "top_k": 10,
        },
    }
    retriever = LanceDBFTSRetriever.from_dict(data)
    assert retriever._document_store
    assert retriever._filters == {}
    assert retriever._top_k == 10


def test_run(tmp_dir):
    path = tmp_dir
    store = LanceDBDocumentStore(path)
    retriever = LanceDBFTSRetriever(document_store=store)
    store.write_documents([Document(content="Test doc expecting some query")])
    res = retriever.run(query="some query")
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc expecting some query"
