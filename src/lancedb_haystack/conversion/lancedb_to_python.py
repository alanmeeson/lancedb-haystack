import datetime
from typing import Any

import pyarrow as pa
from haystack import Document


def convert_lancedb_to_document(result: dict, schema: pa.Schema) -> Document:
    """Convert a result lancedb into a document"""

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

    # TODO: consider refactoring this,  it feels like duplication even though the root of the tree is special.
    # For the fields which are mentioned in the _isempty section, only include if they're not empty. (including vector)

    # doc_dict.update({
    #    field_name: convert_field(result[field_name], schema.field(field_name).type)
    #    for field_name, is_empty in result['_isempty'].items() if not is_empty
    # })

    # TODO: consider if I can remove this,  it should probably never actually do anything
    # catch anything that isn't in the _isempty
    # dealt_with_fields = set(result['_isempty'].keys()) | {'score', '_distance'}
    # left_over_fields = set(result.keys()) - dealt_with_fields
    # for field_name in left_over_fields:
    #    doc_dict[field_name] = result[field_name]

    # recursively process the metadata field
    # if doc_dict['meta']:
    #    meta_dict = convert_field(doc_dict['meta'], schema.field('meta').type)
    #    doc_dict['meta'] = meta_dict

    doc = Document.from_dict(doc_dict)
    return doc


def convert_field(value: Any, field_type: pa.DataType) -> Any:
    """ """
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
    """
    is_empty = value.pop("_isempty")
    fields = [field_type.field(idx) for idx in range(field_type.num_fields) if field_type.field(idx).name != "_isempty"]

    meta_dict = {
        field.name: convert_field(value[field.name], field.type) for field in fields if not is_empty[field.name]
    }

    return meta_dict


def convert_timestamp(value: datetime.datetime, field_type: pa.DataType):  # noqa: ARG001
    """Convert timestamp values to isoformat as expected by Haystack"""
    return value.isoformat()
