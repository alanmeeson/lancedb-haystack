# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from typing import Any, Dict

import pandas as pd
from haystack.errors import FilterError


def convert_filters_to_where_clause(filters: Dict[str, Any]) -> str:
    """Convert Haystack filters to a WHERE clause and a tuple of params to query PostgreSQL.

    :param filters: the filters to convert. See: https://docs.haystack.deepset.ai/docs/metadata-filtering
    :return: a string containing the LanceDB where clause.
    """
    if "field" in filters:
        query = _parse_comparison_condition(filters)
    else:
        query = _parse_logical_condition(filters)

    return query


def _parse_logical_condition(condition: Dict[str, Any]) -> str:
    """Compose the sub-queries of a logical step"""
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise FilterError(msg)

    operator = condition["operator"]
    if operator not in ["AND", "OR", "NOT"]:
        msg = f"Unknown logical operator '{operator}'. Valid operators are: 'AND', 'OR', 'NOT'"
        raise FilterError(msg)

    # logical conditions can be nested, so we need to parse them recursively
    conditions = []
    for c in condition["conditions"]:
        if "field" in c:
            query = _parse_comparison_condition(c)
        else:
            query = _parse_logical_condition(c)
        conditions.append(query)

    if operator == "AND":
        sql_query = f"({' AND '.join(conditions)})"
    elif operator == "OR":
        sql_query = f"({' OR '.join(conditions)})"
    elif operator == "NOT":
        sql_query = f"NOT ({' AND '.join(conditions)})"
    else:
        msg = f"Unknown logical operator '{operator}'"
        raise FilterError(msg)

    return sql_query


def _parse_comparison_condition(condition: Dict[str, Any]) -> str:
    """Identifies and applies the right comparison function.

    :param condition: The condition term to convert
    :return: a string containing the comparison for use in a LanceDB where clause
    """
    field: str = condition["field"]
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise FilterError(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise FilterError(msg)
    operator: str = condition["operator"]
    if operator not in COMPARISON_OPERATORS:
        msg = f"Unknown comparison operator '{operator}'. Valid operators are: {list(COMPARISON_OPERATORS.keys())}"
        raise FilterError(msg)

    value: Any = condition["value"]
    if isinstance(value, pd.DataFrame):
        # DataFrames are stored as JSONB and we query them as such
        value = value.to_json()
        # Note: not using Jsonb, but just json
        # field = f"({field})::jsonb"
        # TODO: investigate how to handle tables in LanceDB

    query = COMPARISON_OPERATORS[operator](field, value)
    return query


def equal(field: str, value: Any) -> str:
    """Construct a query string for comparing equality between a field and a value.

    :param field: the field to compare
    :param value: the value to compare it to.
    :return: a comparison string.
    """
    if value is None:
        return is_null(field)
    if isinstance(value, str):
        query = f"{field} = '{value}'"
    else:
        query = f"{field} = {value}"

    return query


def not_equal(field: str, value: Any) -> str:
    """Construct a query string for filtering when a field and a value are not equal

    :param field: the field to compare
    :param value: the value to compare it to.
    :return: a comparison string.
    """
    if value is None:
        query = is_not_null(field)
    elif isinstance(value, str):
        query = f"({is_null(field)} OR {field} != '{value}')"
    else:
        query = f"({is_null(field)} OR {field} != {value})"
    return query


def greater_than(field: str, value: Any) -> str:
    """Construct a query string for filtering when a field is greater than the value.

    Note: 'greater_than' comparisons to None always evaluate to false.

    :param field: the field to compare
    :param value: the value to compare it to.
    :return: a comparison string.
    :raises FilterError: if value is a DataFrame, string, list or dict which are not supported for this comparison.
    """
    if value is None:
        # Greater than comparisons with "None" as value are always false.
        return "False == True"
    elif isinstance(value, pd.DataFrame):
        msg = "Can't compare DataFrames using operators '>', '>=', '<', '<='."
        raise FilterError(msg)
    elif isinstance(value, str):
        try:
            datetime.fromisoformat(value)
            value = f"timestamp '{value}'"
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc

    if type(value) in [list, dict]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)

    return f"({is_not_null(field)} AND {field} > {value})"


def greater_than_equal(field: str, value: Any) -> str:
    """Construct a query string for filtering when a field is greater than or equal to the value.

    Note: 'greater_than_equal' comparisons to None always evaluate to false.

    :param field: the field to compare
    :param value: the value to compare it to.
    :return: a comparison string.
    :raises FilterError: if value is a DataFrame, string, list or dict which are not supported for this comparison.
    """
    if value is None:
        # Greater than equal comparisons with "None" as value are always false.
        return "False == True"
    elif isinstance(value, pd.DataFrame):
        msg = "Can't compare DataFrames using operators '>', '>=', '<', '<='."
        raise FilterError(msg)
    elif isinstance(value, str):
        try:
            datetime.fromisoformat(value)
            value = f"timestamp '{value}'"
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, dict]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)

    return f"({is_not_null(field)} AND {field} >= {value})"


def less_than(field: str, value: Any) -> str:
    """Construct a query string for filtering when a field is less than to the value.

    Note: 'less_than' comparisons to None always evaluate to false.

    :param field: the field to compare
    :param value: the value to compare it to.
    :return: a comparison string.
    :raises FilterError: if value is a DataFrame, string, list or dict which are not supported for this comparison.
    """
    if value is None:
        # Less than comparisons with "None" as value are always false.
        return "False == True"
    elif isinstance(value, pd.DataFrame):
        msg = "Can't compare DataFrames using operators '>', '>=', '<', '<='."
        raise FilterError(msg)
    elif isinstance(value, str):
        try:
            datetime.fromisoformat(value)
            value = f"timestamp '{value}'"
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, dict]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)

    return f"({is_not_null(field)} AND {field} < {value})"


def less_than_equal(field: str, value: Any) -> str:
    """Construct a query string for filtering when a field is less than or equal to the value.

    Note: 'less_than_equal' comparisons to None always evaluate to false.

    :param field: the field to compare
    :param value: the value to compare it to.
    :return: a comparison string.
    :raises FilterError: if value is a DataFrame, string, list or dict which are not supported for this comparison.
    """
    if value is None:
        # Less than equal comparisons with "None" as value are always false.
        return "False == True"
    elif isinstance(value, pd.DataFrame):
        msg = "Can't compare DataFrames using operators '>', '>=', '<', '<='."
        raise FilterError(msg)
    elif isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
        value = f"timestamp '{value}'"
    if type(value) in [list, dict]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)

    return f"({is_not_null(field)} AND {field} <= {value})"


def not_in(field: str, value: Any) -> str:
    """Construct a query string for filtering when a field's value is not in a provided list.

    :param field: the field to filter on
    :param value: the list of values.
    :return: a comparison string.
    :raises FilterError: if value is not a list.
    """
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'not in' comparator in Pinecone"
        raise FilterError(msg)

    vals = ", ".join([f"'{val}'" if isinstance(val, str) else str(val) for val in value])
    return f"{is_null(field)} OR {field} NOT IN ({vals}))"


def in_(field: str, value: Any) -> str:
    """Construct a query string for filtering when a field's value is in a provided list.

    :param field: the field to filter on
    :param value: the list of values.
    :return: a comparison string.
    :raises FilterError: if value is not a list.
    """
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' comparator in Pinecone"
        raise FilterError(msg)

    # ValueError: LanceError(IO): Received literal Utf8("10") and could not convert to literal of type 'Int32'
    vals = ", ".join([f"'{val}'" if isinstance(val, str) else str(val) for val in value])
    query = f"{is_not_null(field)} AND {field} IN ({vals})"

    return query


def is_null(field: str) -> str:
    """Construct Filter term for the field being either empty or Null

    :param field: the field to check for being null
    :return: the filter string
    """
    if field == "vector":
        # IF it's the vector field, check the _isempty_vector field too
        query = "(_isempty_vector == True OR vector is NULL)"
    elif field.startswith("meta"):
        # If it's a meta field in which we track whether things are empty, also check the _isempty entry
        field_prefix, field_name = field.rsplit(".", 1)
        query = f"({field_prefix}._isempty.{field_name} == True OR {field} is NULL)"
    else:
        query = f"{field} IS NULL"

    return query


def is_not_null(field: str) -> str:
    """Construct Filter term for the field being neither empty nor Null

    :param field: the field to check for being null
    :return: the filter string
    """
    if field == "vector":
        # IF it's the vector field, check the _isempty_vector field too
        query = "(_isempty_vector == False AND vector IS NOT NULL)"
    elif field.startswith("meta"):
        # If it's a meta field in which we track whether things are empty, also check the _isempty entry
        field_prefix, field_name = field.rsplit(".", 1)
        query = f"({field_prefix}._isempty.{field_name} == False AND {field} IS NOT NULL)"
    else:
        query = f"{field} IS NOT NULL"

    return query


COMPARISON_OPERATORS = {
    "==": equal,
    "!=": not_equal,
    ">": greater_than,
    ">=": greater_than_equal,
    "<": less_than,
    "<=": less_than_equal,
    "in": in_,
    "not in": not_in,
}
