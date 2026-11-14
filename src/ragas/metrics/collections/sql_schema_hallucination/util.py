"""SQL schema hallucination utility functions."""

from __future__ import annotations

import typing as t

try:
    import sqlparse
    from sqlparse.sql import (
        Function,
        Identifier,
        IdentifierList,
        Parenthesis,
        TokenList,
        Where,
    )
    from sqlparse.tokens import DML, Keyword
except ImportError:  # pragma: no cover - surfaced at metric init with guidance
    sqlparse = None  # type: ignore[assignment]

Schema = t.Mapping[str, t.Sequence[str]]

# Keywords whose following identifiers name tables rather than columns.
_TABLE_CONTEXT = {"FROM", "INTO", "UPDATE"}
# Keywords that return us to a column context.
_COLUMN_CONTEXT = {"SELECT", "WHERE", "ON", "GROUP BY", "ORDER BY", "HAVING", "SET"}


class SQLReferences(t.NamedTuple):
    tables: t.Set[str]
    # Columns qualified by a table name, as (table, column).
    qualified: t.Set[t.Tuple[str, str]]
    # Unqualified columns paired with the tables visible in their query scope,
    # so they are validated only against those tables and never leak across
    # subquery / CTE boundaries.
    unqualified: t.Set[t.Tuple[t.FrozenSet[str], str]]

    @property
    def columns(self) -> t.Set[t.Tuple[t.Optional[str], str]]:
        """Flat view of all column references for introspection and tests."""
        flat: t.Set[t.Tuple[t.Optional[str], str]] = {
            (table, col) for table, col in self.qualified
        }
        flat |= {(None, col) for _, col in self.unqualified}
        return flat


class _Scope:
    """References collected within one query scope (a SELECT or DML statement).

    Subqueries and CTE bodies get their own scope so their tables and aliases
    never affect validation in the enclosing query.
    """

    def __init__(self, parent: t.Optional["_Scope"] = None) -> None:
        self.parent = parent
        self.tables: t.Set[str] = set()
        self.alias_to_table: t.Dict[str, str] = {}
        self.virtual: t.Set[str] = set()  # CTE and derived-table names
        self.raw_columns: t.Set[t.Tuple[t.Optional[str], str]] = set()
        self.aliases: t.Set[str] = set()  # column aliases to ignore

    def lookup_alias(self, name: str) -> t.Optional[str]:
        """Resolve a table alias, walking outer scopes for correlated queries."""
        scope: t.Optional["_Scope"] = self
        while scope is not None:
            if name in scope.alias_to_table:
                return scope.alias_to_table[name]
            scope = scope.parent
        return None

    def is_virtual(self, name: str) -> bool:
        scope: t.Optional["_Scope"] = self
        while scope is not None:
            if name in scope.virtual:
                return True
            scope = scope.parent
        return False


def _clean(name: t.Optional[str]) -> t.Optional[str]:
    return name.strip('"`[]') if name else None


def _tokens(token: TokenList) -> t.Iterator[t.Any]:
    """Yield the meaningful children of a token, flattening an IdentifierList."""
    if isinstance(token, IdentifierList):
        yield from token.get_identifiers()  # type: ignore[attr-defined]
    else:
        yield token


def _has_select(token: TokenList) -> bool:
    return any(getattr(tok, "ttype", None) is DML for tok in token.flatten())


def _collect_ctes(statement: TokenList) -> t.Set[str]:
    ctes: t.Set[str] = set()
    after_with = False
    for token in statement.tokens:
        if token.is_whitespace:
            continue
        if token.normalized == "WITH":
            after_with = True
            continue
        if after_with:
            if isinstance(token, (Identifier, IdentifierList)):
                for child in _tokens(token):
                    name = _clean(child.get_real_name())
                    if name:
                        ctes.add(name)
            after_with = False
    return ctes


def _walk(node: TokenList, context: str, scope: _Scope, scopes: t.List[_Scope]) -> None:
    for token in node.tokens:
        if token.is_whitespace:
            continue

        if token.ttype is DML:
            context = "TABLE" if token.normalized.upper() == "UPDATE" else "COLUMN"
            continue
        if token.ttype is Keyword:
            upper = token.normalized.upper()
            if upper in _TABLE_CONTEXT or upper.endswith("JOIN"):
                context = "TABLE"
            elif upper in _COLUMN_CONTEXT:
                context = "COLUMN"
            continue

        if isinstance(token, Where):
            _walk(token, "COLUMN", scope, scopes)
            continue
        if isinstance(token, Parenthesis):
            _descend_parenthesis(token, context, scope, scopes)
            continue

        if isinstance(token, (Identifier, IdentifierList)):
            for child in _tokens(token):
                _handle(child, context, scope, scopes)
            continue

        if isinstance(token, Function):
            _handle(token, context, scope, scopes)
            continue

        # Recurse into grouped tokens like Comparison or Operation so nested
        # identifiers (e.g. a WHERE predicate) are still classified.
        if isinstance(token, TokenList):
            _walk(token, context, scope, scopes)


def _descend_parenthesis(
    paren: Parenthesis, context: str, scope: _Scope, scopes: t.List[_Scope]
) -> None:
    """A subquery gets its own scope; any other parenthesis stays in-scope."""
    if _has_select(paren):
        child = _Scope(parent=scope)
        scopes.append(child)
        _walk(paren, "COLUMN", child, scopes)
    else:
        _walk(paren, context, scope, scopes)


def _handle(
    identifier: t.Any, context: str, scope: _Scope, scopes: t.List[_Scope]
) -> None:
    # A function call: its name is not a column, but its arguments may be. In a
    # table context ``table (col, col)`` also parses as a function (e.g. the
    # target of INSERT INTO), so the name is the table there.
    if isinstance(identifier, Function):
        name = _clean(identifier.get_real_name())
        if context == "TABLE" and name:
            scope.tables.add(name)
        paren = next(
            (tok for tok in identifier.tokens if isinstance(tok, Parenthesis)), None
        )
        if paren is not None:
            _descend_parenthesis(paren, "COLUMN", scope, scopes)
        return
    if not isinstance(identifier, Identifier):
        return

    paren = next(
        (tok for tok in identifier.tokens if isinstance(tok, Parenthesis)), None
    )
    if paren is not None and _has_select(paren):
        # Derived table: (SELECT ...) alias. The alias is virtual in this scope
        # and the subquery is walked as its own scope.
        alias = _clean(identifier.get_alias())
        if alias:
            scope.virtual.add(alias)
            scope.aliases.add(alias)
        _descend_parenthesis(paren, "COLUMN", scope, scopes)
        return

    alias = _clean(identifier.get_alias())
    if alias:
        scope.aliases.add(alias)

    real = _clean(identifier.get_real_name())
    if not real:
        return

    if context == "TABLE":
        scope.tables.add(real)
        if alias:
            scope.alias_to_table[alias] = real
    else:
        qualifier = _clean(identifier.get_parent_name())
        scope.raw_columns.add((qualifier, real))


def _resolve(
    scope: _Scope,
) -> t.Tuple[
    t.Set[str], t.Set[t.Tuple[str, str]], t.Set[t.Tuple[t.FrozenSet[str], str]]
]:
    tables = scope.tables - scope.virtual
    scope_tables = frozenset(tables)

    qualified: t.Set[t.Tuple[str, str]] = set()
    unqualified: t.Set[t.Tuple[t.FrozenSet[str], str]] = set()
    for qualifier, name in scope.raw_columns:
        if name in scope.aliases:
            continue
        if qualifier is None:
            unqualified.add((scope_tables, name))
            continue
        if scope.is_virtual(qualifier):
            continue
        resolved = scope.lookup_alias(qualifier) or qualifier
        if scope.is_virtual(resolved):
            continue
        qualified.add((resolved, name))

    return tables, qualified, unqualified


def extract_references(sql: str) -> SQLReferences:
    """Extract table and column references from a SQL query.

    Handles multiple statements, joins, functions, subqueries, CTEs and simple
    DML. Subqueries and CTE bodies form their own scope, so an unqualified
    column is validated only against the tables of its own query. In-query
    aliases and CTE / derived-table names are resolved and never counted as
    references. ``*`` wildcards and function names are ignored.

    Note: schema-qualified names deeper than ``table.column`` (e.g.
    ``db.schema.table``) are not resolved, matching the flat ``{table: columns}``
    schema contract.
    """
    assert sqlparse is not None, "sqlparse is required for SQLSchemaHallucination"

    tables: t.Set[str] = set()
    qualified: t.Set[t.Tuple[str, str]] = set()
    unqualified: t.Set[t.Tuple[t.FrozenSet[str], str]] = set()
    for statement in sqlparse.parse(sql):
        root = _Scope()
        root.virtual |= _collect_ctes(statement)
        scopes: t.List[_Scope] = [root]
        _walk(statement, "COLUMN", root, scopes)
        for scope in scopes:
            scope_tables, scope_qualified, scope_unqualified = _resolve(scope)
            tables |= scope_tables
            qualified |= scope_qualified
            unqualified |= scope_unqualified

    return SQLReferences(tables=tables, qualified=qualified, unqualified=unqualified)


def _normalize_schema(schema: Schema) -> t.Dict[str, t.Set[str]]:
    return {
        table.lower(): {col.lower() for col in cols} for table, cols in schema.items()
    }


def score_references(
    references: SQLReferences,
    schema: Schema,
) -> t.Tuple[int, int, t.List[str]]:
    """Count references that exist in the schema.

    Unqualified columns are validated against the columns of the tables their
    own query scope references, so a column that only exists in an unrelated
    table (or in a subquery) is still flagged. Returns (valid, total,
    hallucinated).
    """
    normalized = _normalize_schema(schema)
    all_columns = {col for cols in normalized.values() for col in cols}

    valid = 0
    total = 0
    hallucinated: t.List[str] = []

    for table in references.tables:
        total += 1
        if table.lower() in normalized:
            valid += 1
        else:
            hallucinated.append(table)

    for table, column in references.qualified:
        total += 1
        if table.lower() in normalized and column.lower() in normalized[table.lower()]:
            valid += 1
        else:
            hallucinated.append(f"{table}.{column}")

    for scope_tables, column in references.unqualified:
        total += 1
        referenced = set()
        for table in scope_tables:
            referenced |= normalized.get(table.lower(), set())
        # No referenced table is in the schema; fall back to the whole schema so
        # a bad table name does not also flag every column as hallucinated.
        if not referenced:
            referenced = all_columns
        if column.lower() in referenced:
            valid += 1
        else:
            hallucinated.append(column)

    return valid, total, hallucinated
