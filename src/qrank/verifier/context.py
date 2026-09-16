"""Dataset-specific, reference-free context for semantic verification."""
from __future__ import annotations

import argparse
import csv
import sqlite3
from contextlib import closing
import json
import re
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Mapping

from .models import VerificationError, stable_json


@dataclass(frozen=True)
class VerificationContext:
    task_type: str
    question: str
    evidence: str = ""
    context: str = ""
    db_id: str = ""
    difficulty: str = ""
    sql_dialect: str | None = None


def detect_task_type(row: Mapping[str, Any], task_type: str = "auto") -> str:
    if task_type != "auto":
        if task_type not in {"text-to-sql", "prepbench"}:
            raise ValueError(f"Unsupported task type: {task_type!r}")
        return task_type
    metadata = row.get("metadata") or {}
    declared = row.get("task_type") or metadata.get("task_type")
    if declared:
        return detect_task_type(row, str(declared)) if declared != "auto" else "text-to-sql"
    setting = row.get("setting")
    if isinstance(setting, str) and "prep" in setting.lower() and "code" in setting.lower():
        return "prepbench"
    if any(re.search(r"\bdef\s+solve\s*\(\s*tables\b", c.get("text", ""))
           for c in row.get("candidates", [])):
        return "prepbench"
    return "text-to-sql"


def build_context(
    row: Mapping[str, Any], *, task_type: str = "auto",
    schemas: Mapping[str, str] | None = None,
    descriptions: Mapping[str, str] | None = None,
    prepbench_root: str | Path | None = None,
    sql_dialect: str | None = None,
) -> VerificationContext:
    """Only task fields and optional input CSVs are exposed; never gold outputs.

    PrepBench CSV layout: <root>/<case_id>/inputs/*.csv (falling back to
    question_id). Read headers and at most five records per file. Values are
    kept as text; no dtype inference or candidate execution is performed.
    """
    kind = detect_task_type(row, task_type)
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        raise ValueError("question must be a nonempty string")
    metadata = row.get("metadata") or {}
    context = row.get("context") or ""
    if not isinstance(context, str):
        context = json.dumps(context, ensure_ascii=False, sort_keys=True)
    db_id = ""
    dialect = None
    if kind == "text-to-sql":
        db_id = str(metadata.get("db_id") or row.get("group_id") or "")
        if schemas and db_id not in schemas:
            raise ValueError(f"No schema found for database {db_id!r}")
        schema = (schemas or {}).get(db_id, "")
        dialect = sql_dialect or row.get("sql_dialect") or metadata.get("sql_dialect")
        context = "\n\n".join(part for part in (
            "Task: text-to-SQL. The candidate is a SQL query whose result must "
            "answer the question against the supplied database schema.",
            f"SQL dialect: {dialect}." if dialect else "The SQL dialect is unspecified unless supplied below.",
            schema,
            (descriptions or {}).get(db_id, ""),
            context,
        ) if part)
    else:
        parts = [
            "Task: PrepBench data preparation using Python/pandas. "
            "Candidate interface: solve(tables: dict[str, pd.DataFrame]) -> "
            "dict[str, pd.DataFrame]. Input keys are filenames; returned keys are "
            "the requested output filenames. pd denotes pandas. "
            "Input dtypes, imports, and runtime globals are unspecified unless "
            "supplied below or established by the candidate code."
        ]
        if context:
            parts.append(context)
        if prepbench_root is not None:
            root = Path(prepbench_root).resolve()
            case_id = row.get("case_id", row["question_id"])
            case = (root / str(case_id) / "inputs").resolve()
            if not case.is_relative_to(root):
                raise ValueError("PrepBench case ID must stay within --prepbench-root")
            files = sorted(case.glob("*.csv"))
            if not files:
                raise ValueError(f"No input CSV files found in {case}")
            for path in files:
                if not path.resolve().is_relative_to(case):
                    raise ValueError(f"Input CSV must stay within {case}: {path}")
                with path.open(newline="", encoding="utf-8-sig") as stream:
                    reader = csv.reader(stream)
                    columns = next(reader, [])
                    sample = list(islice(reader, 5))
                parts.append("INPUT TABLE (sample values are CSV text, not runtime dtypes):\n" +
                             json.dumps({"filename": path.name, "columns": columns,
                                         "sample_rows": sample}, ensure_ascii=False))
        elif not context:
            parts.append("Input table schemas, samples, and runtime dtypes were not provided. "
                         "Use U for material questions that depend on missing information.")
        context = "\n\n".join(parts)
    return VerificationContext(kind, str(row["question"]), str(row.get("evidence") or ""),
                               context, db_id, str(metadata.get("difficulty") or ""), dialect)


def load_database_schemas(args: argparse.Namespace) -> dict[str, str]:
    """Read only schema metadata; never execute candidate SQL or read gold SQL."""
    base = Path(__file__).resolve().parents[3] / "dataset/base"
    schema_file = Path(args.schemas) if args.schemas else None
    if schema_file is None and not getattr(args, "db_root", None) and base.exists():
        for filename in ("schema.json", "dev_tables.json"):
            matches = sorted(base.rglob(filename))
            if len(matches) > 1:
                raise VerificationError(f"Multiple {filename} files found; specify --schemas")
            if matches:
                schema_file = matches[0]
                break
    if schema_file:
        tables = json.loads(schema_file.read_text(encoding="utf-8"))
        if isinstance(tables, dict):
            schemas = {}
            for db_id, definitions in tables.items():
                if isinstance(definitions, str):
                    schema = definitions
                elif isinstance(definitions, list) and all(isinstance(s, str) for s in definitions):
                    schema = "\n\n".join(definitions)
                else:
                    raise VerificationError(f"Invalid schema definitions for {db_id!r} in {schema_file}")
                if not schema.strip():
                    raise VerificationError(f"Empty schema for {db_id!r} in {schema_file}")
                schemas[str(db_id)] = schema
            if not schemas:
                raise VerificationError(f"No database schemas found in {schema_file}")
            return schemas
        if not isinstance(tables, list) or not tables or any(
            not isinstance(t, dict) or "db_id" not in t for t in tables
        ):
            raise VerificationError(f"Expected a database-to-DDL mapping or BIRD tables list in {schema_file}")
        schemas = {str(t["db_id"]): stable_json(t) for t in tables}
        if len(schemas) != len(tables):
            raise VerificationError(f"Duplicate database IDs in {schema_file}")
        return schemas
    root = Path(args.db_root) if getattr(args, "db_root", None) else base
    if getattr(args, "db_root", None) and not root.is_dir():
        raise VerificationError(f"Database directory does not exist: {root}")
    schemas = {}
    for path in sorted(root.rglob("*.sqlite")):
        if path.stem in schemas:
            raise VerificationError(f"Duplicate database {path.stem!r}; narrow --db-root")
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
            definitions = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type IN ('table', 'view') "
                "AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL ORDER BY name"
            ).fetchall()
            schemas[path.stem] = "\n\n".join(row[0] for row in definitions)
    if getattr(args, "db_root", None) and not schemas:
        raise VerificationError(f"No .sqlite databases found under {root}")
    return schemas


def load_database_descriptions(args: argparse.Namespace) -> dict[str, str]:
    """Load documented BIRD columns/values, never query results or reference SQL.

    Only <db_id>/database_description/*.csv files are considered. An explicit
    descriptions root can accompany --schemas; --db-root includes them by default.
    """
    directory = getattr(args, "descriptions_root", None) or getattr(args, "db_root", None)
    if not directory:
        return {}
    root = Path(directory)
    if not root.is_dir():
        raise VerificationError(f"Descriptions directory does not exist: {root}")
    descriptions: dict[str, list[str]] = {}
    seen = set()
    for path in sorted(root.rglob("database_description/*.csv")):
        db_id = path.parent.parent.name
        identity = (db_id, path.stem)
        if identity in seen:
            raise VerificationError(f"Duplicate table description {identity}; narrow the descriptions root")
        seen.add(identity)
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise VerificationError(f"Description CSV has no headers: {path}")
            rows = list(reader)
        descriptions.setdefault(db_id, []).append(json.dumps({
            "source": str(path.resolve()), "table": path.stem, "documented_columns_and_values": rows,
        }, ensure_ascii=False, sort_keys=True))
    return {db_id: "DOCUMENTED DATABASE DESCRIPTIONS (with provenance):\n" + "\n".join(parts)
            for db_id, parts in descriptions.items()}
