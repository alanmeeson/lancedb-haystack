# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0
import pyarrow as pa
from haystack.dataclasses import Document

from lancedb_haystack.document_store import LanceDBDocumentStore
from lancedb_haystack.fts_retriever import LanceDBFTSRetriever


def test_to_dict(tmp_path):
    path = str(tmp_path)
    document_store = LanceDBDocumentStore(
        path,
        "test_table",
        metadata_schema=pa.struct([pa.field("a", pa.string()), pa.field("b", pa.int32())]),
        embedding_dims=384,
    )
    retriever = LanceDBFTSRetriever(document_store=document_store)
    res = retriever.to_dict()

    assert res == {
        "type": "lancedb_haystack.fts_retriever.LanceDBFTSRetriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "database": path,
                    "table_name": "test_table",
                    "embedding_dims": 384,
                    "metadata_schema": {
                        "type": "struct",
                        "children": [
                            {"name": "a", "type": "string", "nullable": True, "metadata": None},
                            {"name": "b", "type": "int32", "nullable": True, "metadata": None},
                        ],
                    },
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
        "type": "lancedb_haystack.fts_retriever.LanceDBFTSRetriever",
        "init_parameters": {
            "document_store": {
                "init_parameters": {
                    "database": path,
                    "table_name": "test_table",
                    "embedding_dims": 384,
                    "metadata_schema": {
                        "type": "struct",
                        "children": [
                            {"name": "a", "type": "string", "nullable": True, "metadata": None},
                            {"name": "b", "type": "int32", "nullable": True, "metadata": None},
                        ],
                    },
                },
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


def test_run(tmp_path):
    path = str(tmp_path)
    store = LanceDBDocumentStore(
        path, "test_table", metadata_schema=pa.struct([pa.field("a", pa.int32())]), embedding_dims=2
    )
    retriever = LanceDBFTSRetriever(document_store=store)
    store.write_documents([Document(content="Test doc expecting some query", meta={"a": 1})])
    res = retriever.run(query="some query")
    assert len(res) == 1
    assert len(res["documents"]) == 1
    assert res["documents"][0].content == "Test doc expecting some query"
    assert res["documents"][0].meta["a"] == 1
