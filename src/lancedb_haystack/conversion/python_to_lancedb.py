import datetime
from typing import Any, Union

import pyarrow as pa
from haystack import Document


def convert_document_to_lancedb(document: Document, schema: pa.Schema) -> dict:
    """Converts a Haystack Document to a format ready to store in lancedb

    :param document: the Haystack document to prepare for insertion into LanceDB
    :param schema: the lancedb table schema
    :return: a dict
    """

    embed_dims = schema.field("vector").type.list_size
    meta_schema = schema.field("meta").type
    doc_dict = document.to_dict(flatten=False)
    lance_dict = {}
    _isempty = {}

    # remove score - the retrievers will add this if necessary
    del doc_dict["score"]

    # convert embedding to vector and fill if missing
    embedding = doc_dict.pop("embedding")
    lance_dict["vector"] = embedding if embedding else [0] * embed_dims
    _isempty["vector"] = False if embedding else True

    # convert the metadata
    lance_dict["meta"] = convert_struct(doc_dict["meta"], meta_schema)
    _isempty["meta"] = False

    # fill missing fields
    field_names_to_do = list(set(schema.names) - {"_isempty", "meta", "vector"})
    for field_name in field_names_to_do:
        if field_name in doc_dict:
            lance_dict[field_name] = doc_dict[field_name]
            _isempty[field_name] = False
        else:
            lance_dict[field_name] = None
            _isempty[field_name] = True

    lance_dict["_isempty"] = _isempty

    return lance_dict


def convert_field(value: Any, field_type: pa.DataType):
    """Converts the value of a field from it's representation in haystack to the one used for LanceDB

    :param value: the value to convert
    :param field_type: The pyarrow type of the value, so we know how to convert it.
    :return: the converted value
    """
    field_str = str(field_type)
    if field_str.startswith("timestamp"):
        return convert_timestamp(value, field_type)
    elif field_str.startswith("struct"):
        return convert_struct(value, field_type)
    else:
        return value


def convert_struct(value: dict, field_type: pa.StructType) -> dict:
    """Convert a dict into the format expected by LanceDB

    :param value: the value to convert
    :param field_type: The pyarrow type of the value, so we know how to convert it.
    :return: the converted value
    """
    fields = [field_type.field(idx) for idx in range(field_type.num_fields) if field_type.field(idx).name != "_isempty"]

    struct_dict = {}
    _isempty = {}
    for field in fields:
        field_name = field.name

        if field_name in value:
            struct_dict[field_name] = convert_field(value[field_name], field.type)
            _isempty[field_name] = False
        else:
            struct_dict[field_name] = None
            _isempty[field_name] = True

    struct_dict["_isempty"] = _isempty
    return struct_dict


def convert_timestamp(value: Union[datetime.datetime, str], field_type: pa.DataType) -> pa.Scalar:
    """Convert datetime or iso string into the format expected by LanceDB.

    :param value: the value to convert
    :param field_type: The pyarrow type of the value, so we know how to convert it.
    :return: the converted value
    """
    if isinstance(value, datetime.datetime):
        return_value = value
    elif isinstance(value, str):
        return_value = datetime.datetime.fromisoformat(value)

    return pa.scalar(return_value, field_type)
