"""Deterministic SQL clause summaries, preserving query scope and dependencies."""
from __future__ import annotations

import sqlglot
from sqlglot import ErrorLevel, exp
from sqlglot.optimizer.scope import Scope, traverse_scope


def decompose_sql(sql: str, dialect: str | None = None) -> list[dict]:
    """Extract perspective-ready clause groups for each AST query scope.

    IDs are deterministic within a candidate, not equivalent across candidates.
    Predicates remain intact so AND/OR and clause boundaries retain their meaning.
    """
    dialect = dialect.lower() if dialect else None
    statements = [node for node in sqlglot.parse(sql, read=dialect) if node is not None]
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise ValueError("Semantic verification requires one SQL query")
    tree = statements[0]
    scopes = list(traverse_scope(tree))
    ids = {id(scope): f"q{i}" for i, scope in enumerate(reversed(scopes))}

    def render(node):
        return node.sql(dialect=dialect, comments=False, unsupported_level=ErrorLevel.RAISE)

    blocks = []

    def section(name, values):
        block["sections"][name] = values

    for scope in reversed(scopes):
        node = scope.expression
        parent = ids[id(scope.parent)] if scope.parent else "root"
        block = {"scope_id": ids[id(scope)], "parent_scope_id": parent,
                 "kind": node.key.upper(), "sections": {}}
        blocks.append(block)
        section("SELECT", [render(e) for e in node.expressions] if isinstance(node, exp.Select) else [])
        section("SOURCES", [f"{alias}: {ids[id(source)] if isinstance(source, Scope) else render(source)}"
                            for alias, (_, source) in scope.selected_sources.items()])
        section("JOINS", [render(join) for join in node.args.get("joins", [])])
        nodes = list(scope.walk())
        windows = [e for e in nodes if isinstance(e, exp.Window)]
        window_aggregates = {id(e) for w in windows for e in w.find_all(exp.AggFunc)}
        section("AGGREGATION", list(dict.fromkeys(
            render(e) for e in nodes if isinstance(e, exp.AggFunc) and id(e) not in window_aggregates
        )))
        for heading, argument in (("FILTERS (WHERE)", "where"), ("GROUP BY", "group"),
                                  ("HAVING", "having"), ("WINDOW DEFINITIONS", "windows"),
                                  ("QUALIFY", "qualify"), ("DISTINCT", "distinct"),
                                  ("ORDER BY", "order"), ("LIMIT", "limit"), ("OFFSET", "offset")):
            value = node.args.get(argument)
            if isinstance(value, bool):
                continue  # UNION DISTINCT/ALL is described under SET OPERATION.
            values = value if isinstance(value, list) else [value] if value is not None else []
            section(heading, [render(e) for e in values])
        section("WINDOW EXPRESSIONS", list(dict.fromkeys(render(e) for e in windows)))
        section("EXTERNAL REFERENCES (binding requires schema)", [render(c) for c in scope.external_columns if c.table])
        section("SET OPERATION", [])
        if isinstance(node, exp.SetOperation):
            section("SET OPERATION", [node.key.upper() + (" DISTINCT" if node.args.get("distinct") else " ALL"),
                                      "inputs: " + ", ".join(ids[id(s)] for s in scope.union_scopes)])
    return blocks


def summarize_sql(sql: str, dialect: str | None = None) -> str:
    blocks = decompose_sql(sql, dialect)
    return format_sql_blocks(blocks, dialect)


def format_sql_blocks(blocks: list[dict], dialect: str | None = None) -> str:
    lines = [f"SQLGlot {sqlglot.__version__}; dialect: {dialect.lower() if dialect else 'unspecified (SQLGlot parser)'}"]
    for block in blocks:
        lines.extend(["", f"QUERY {block['scope_id']} ({block['kind']}, parent: {block['parent_scope_id']})"])
        for heading, values in block["sections"].items():
            lines.append(heading + ":")
            lines.extend("    " + value for value in (values or ["<none>"]))
    return "\n".join(lines)
