import pyarrow as pa


def field_to_dict(field: pa.Field) -> dict:
    """
    Convert a PyArrow field to a dictionary representation, including nested fields.

    :param field: The PyArrow field to convert.
    :return: The dictionary representation of the field.
    """
    field_dict = {"name": field.name, "type": str(field.type), "nullable": field.nullable, "metadata": field.metadata}

    if pa.types.is_struct(field.type):
        field_dict["type"] = "struct"
        field_dict["children"] = [field_to_dict(child) for child in field.type]

    return field_dict


def pyarrow_struct_to_dict(struct_type: pa.StructType) -> dict:
    """
    Convert a PyArrow StructType to a dictionary representation, including nested fields.

    :param struct_type: The PyArrow StructType to convert.
    :return: The dictionary representation of the StructType.
    """
    struct_dict = {
        "type": "struct",
        "children": [field_to_dict(child) for child in struct_type if child.name != "_isempty"],
    }

    return struct_dict


def pyarrow_schema_to_dict(schema: pa.Schema) -> dict:
    """
    Convert a PyArrow schema to a JSON representation.

    :param schema: The PyArrow schema to convert.
    :return: The JSON representation of the schema.
    """
    schema_dict = {"fields": [field_to_dict(field) for field in schema]}
    return schema_dict


def dict_to_field(field_dict: dict) -> pa.Field:
    """
    Convert a dictionary representation of a field back to a PyArrow field, including nested fields.

    :param field_dict: The dictionary representation of the field.
    :return: The reconstructed PyArrow field.
    """
    field_name = field_dict["name"]
    field_type_str = field_dict["type"]
    field_nullable = field_dict["nullable"]
    field_metadata = field_dict["metadata"]

    if field_type_str == "struct":
        field_type = dict_to_pyarrow_struct(field_dict)
    else:
        field_type = getattr(pa, field_type_str)()

    return pa.field(field_name, field_type, field_nullable, field_metadata)


def dict_to_pyarrow_struct(struct_dict: dict) -> pa.StructType:
    """
    Convert a dict representation of a struct type back to a PyArrow StructType.

    :param struct_dict: The dict representation of the struct.
    :return: The reconstructed PyArrow StructType.
    """
    children = [dict_to_field(child) for child in struct_dict["children"]]
    return pa.struct(children)


def dict_to_pyarrow_schema(schema_dict: dict) -> pa.Schema:
    """
    Convert a dict representation of a schema back to a PyArrow schema.

    :param schema_dict: The dict representation of the schema.
    :return: The reconstructed PyArrow schema.
    """
    fields = [dict_to_field(field) for field in schema_dict["fields"]]
    return pa.schema(fields)
