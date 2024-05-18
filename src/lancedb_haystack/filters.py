# SPDX-FileCopyrightText: 2024-present Alan Meeson <am@carefullycalculated.co.uk>
#
# SPDX-License-Identifier: Apache-2.0

# Note: S608 warning for SQL injection vector is disabled, as SQL query construction is necessary for building the
# queries for the filters.
# TODO: find a better way of doing this that doesn't require string construction.
from datetime import datetime
from typing import Any, Dict

from haystack.errors import FilterError
from pandas import DataFrame

NO_VALUE = "no_value"


def _convert_filters_to_where_clause_and_params(filters: Dict[str, Any]) -> str:
    """
    Convert Haystack filters to a WHERE clause and a tuple of params to query PostgreSQL.
    """
    if "field" in filters:
        query = _parse_comparison_condition(filters)
    else:
        query = _parse_logical_condition(filters)

    return query


def _parse_logical_condition(condition: Dict[str, Any]) -> str:
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
    if isinstance(value, DataFrame):
        # DataFrames are stored as JSONB and we query them as such
        value = value.to_json()
        # Note: not using Jsonb, but just json
        # field = f"({field})::jsonb"
        # TODO: investigate how to handle tables in LanceDB

    query = COMPARISON_OPERATORS[operator](field, value)
    return query


def _equal(field: str, value: Any) -> str:
    if value is None:
        # NO_VALUE is a placeholder that will be removed in _convert_filters_to_where_clause_and_params
        return f"{field} IS NULL"
    if isinstance(value, str):
        query = f"{field} = '{value}'"
    else:
        query = f"{field} = {value}"

    return query


def _not_equal(field: str, value: Any) -> str:
    # we use IS DISTINCT FROM to correctly handle NULL values
    # (not handled by !=)
    if isinstance(value, str):
        query = f"{field} IS NOT '{value}'"
    else:
        query = f"{field} IS NOT {value}"
    return query


def _greater_than(field: str, value: Any) -> str:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, dict]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)

    return f"{field} > {value}"


def _greater_than_equal(field: str, value: Any) -> str:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, dict]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)

    return f"{field} >= {value}"


def _less_than(field: str, value: Any) -> str:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, dict]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)

    return f"{field} < {value}"


def _less_than_equal(field: str, value: Any) -> str:
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise FilterError(msg) from exc
    if type(value) in [list, dict]:
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise FilterError(msg)

    return f"{field} <= {value}"


def _not_in(field: str, value: Any) -> str:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'not in' comparator in Pinecone"
        raise FilterError(msg)

    vals = ", ".join([f"'{val}'" for val in value])
    return f"{field} IS NULL OR {field} NOT IN ({vals}))"


def _in(field: str, value: Any) -> str:
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' comparator in Pinecone"
        raise FilterError(msg)

    # TODO: consider replacing with IN
    vals = ", ".join([f"'{val}'" for val in value])
    return f"{field} IN ({vals})"


COMPARISON_OPERATORS = {
    "==": _equal,
    "!=": _not_equal,
    ">": _greater_than,
    ">=": _greater_than_equal,
    "<": _less_than,
    "<=": _less_than_equal,
    "in": _in,
    "not in": _not_in,
}
