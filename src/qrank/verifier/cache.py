"""Persistent cache for semantic verification responses."""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path
from typing import Any

from qrank.llms import LLMBackend

from .models import stable_json
from .prompts import PROMPT_VERSION


class SQLiteCache:
    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with closing(sqlite3.connect(self.path)) as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS verifier_cache (
                    cache_key TEXT PRIMARY KEY,
                    raw_text TEXT NOT NULL,
                    parsed_json TEXT NOT NULL,
                    model TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    created_utc TEXT NOT NULL
                )
            """)
            con.commit()

    def _con(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(self.path, timeout=60)
            self._local.con = con
        return con

    def get(self, key: str) -> tuple[str, dict[str, Any]] | None:
        row = self._con().execute(
            "SELECT raw_text, parsed_json FROM verifier_cache WHERE cache_key=?", (key,)
        ).fetchone()
        if not row:
            return None
        return str(row[0]), json.loads(row[1])

    def put(self, key: str, raw_text: str, parsed: dict[str, Any], backend: LLMBackend,
            *, prompt_version: str = PROMPT_VERSION) -> None:
        con = self._con()
        con.execute(
            """INSERT OR REPLACE INTO verifier_cache
               (cache_key,raw_text,parsed_json,model,backend,prompt_version,created_utc)
               VALUES (?,?,?,?,?,?,?)""",
            (
                key,
                raw_text,
                stable_json(parsed),
                backend.model,
                backend.backend_name,
                prompt_version,
                dt.datetime.now(dt.timezone.utc).isoformat(),
            ),
        )
        con.commit()
