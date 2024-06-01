import datetime
from typing import Any

import pyarrow as pa
from haystack import Document


def convert_lancedb_to_document(result: dict, schema: pa.Schema) -> Document:
    """Convert a lancedb result into a document

    :param result: the result from the LanceDB query
    :param schema: the lancedb table schema
    :return: a haystack Document
    """

    is_empty = result.pop("_isempty")
    fields = [schema.field(field_name) for field_name in schema.names if field_name != "_isempty"]

    doc_dict = {
        field.name: convert_field(result[field.name], field.type) for field in fields if not is_empty[field.name]
    }

    # Score will either be provided as 'score' (if using FTS), or '_distance' (if vector search), or be missing
    # (if filter).
    if "score" in result:
        doc_dict["score"] = result["score"]
    elif "_distance" in result:
        doc_dict["score"] = result["_distance"]
    elif "score" not in result:
        doc_dict["score"] = None

    # Remap the vector
    if "vector" in doc_dict:
        doc_dict["embedding"] = doc_dict.pop("vector")

    # Add empty metadata if it's missing
    if not doc_dict.get("meta"):
        doc_dict["meta"] = {}

    doc = Document.from_dict(doc_dict)
    return doc


def convert_field(value: Any, field_type: pa.DataType) -> Any:
    """Converts the value of a field from it's representation in LanceDB to the one used for Haystack

    :param value: the value to convert
    :param field_type: The pyarrow type of the value, so we know how to convert it.
    :return: the converted value
    """
    type_str = str(field_type)
    if type_str.startswith("timestamp"):
        return convert_timestamp(value, field_type)
    elif type_str.startswith("struct"):
        return convert_struct(value, field_type)
    else:
        return value


def convert_struct(value: dict, field_type: pa.StructType) -> dict:
    """Converts the metadata section of the LanceDB representation of a Document to a Haystack Document.

    This involves filtering out empty fields, as well as handling some type conversions to ensure that it complies with
    the expected Haystack DocumentStore behaviour.

    :param value: the value to convert
    :param field_type: The pyarrow type of the value, so we know how to convert it.
    :return: the converted value
    """
    is_empty = value.pop("_isempty")
    fields = [field_type.field(idx) for idx in range(field_type.num_fields) if field_type.field(idx).name != "_isempty"]

    meta_dict = {
        field.name: convert_field(value[field.name], field.type) for field in fields if not is_empty[field.name]
    }

    return meta_dict


def convert_timestamp(value: datetime.datetime, field_type: pa.DataType):  # noqa: ARG001
    """Convert timestamp values to isoformat as expected by Haystack

    :param value: the value to convert
    :param field_type: The pyarrow type of the value, in this case, is ignored.
    :return: the converted value
    """
    return value.isoformat()
