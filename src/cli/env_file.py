"""Read, edit and atomically write a .env file without losing what is in it.

The file is held as ordered lines. Comments, blank lines, keys this module
does not manage and anything it cannot parse are kept verbatim; updating a key
rewrites only the value span, so `export`, spacing and inline comments survive.

Values are parsed the way python-dotenv (what pydantic-settings reads .env
with) parses them, one assignment per line. Multi-line quoted values are not
supported: such a line is kept, but is not treated as an assignment.
"""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

_ASSIGNMENT = re.compile(
    r"^(?P<lead>\s*(?:export\s+)?)(?P<key>[A-Za-z_][A-Za-z0-9_.]*)\s*=[ \t]*"
)
# Values made only of these characters need no quotes.
_BARE = re.compile(r"^[A-Za-z0-9_./:@,+=%&?~-]*$")
_INLINE_COMMENT = re.compile(r"\s+#")
_DOUBLE_ESCAPES = {"\\": "\\", '"': '"', "'": "'", "n": "\n", "t": "\t", "r": "\r"}


@dataclass(frozen=True)
class Assignment:
    key: str
    value: str
    line: str
    # Where the value, quotes included, sits in `line`.
    start: int
    end: int


@dataclass(frozen=True)
class Other:
    """A comment, blank line, or anything else that is not KEY=VALUE."""

    line: str


Entry = Assignment | Other


class UnwritableValue(ValueError):
    pass


def quote(value: str) -> str:
    """Render a value so python-dotenv reads it back unchanged."""
    if "${" in value:
        # python-dotenv expands ${VAR} in every value, quoted or not, with no escape.
        raise UnwritableValue("a .env value cannot contain '${'")
    if "\n" in value or "\r" in value:
        raise UnwritableValue("a .env value must fit on one line")
    if _BARE.match(value):
        return value
    # In single quotes only \\ and \' are escapes.
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _closing_quote(text: str, quote_char: str) -> int:
    """Index of the unescaped quote closing `text[0]`, or -1."""
    i = 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == quote_char:
            return i
        i += 1
    return -1


def _unescape(body: str, quote_char: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(body):
        ch = body[i]
        nxt = body[i + 1] if i + 1 < len(body) else ""
        if ch == "\\" and quote_char == "'" and nxt in ("\\", "'"):
            out.append(nxt)
            i += 2
        elif ch == "\\" and quote_char == '"' and nxt in _DOUBLE_ESCAPES:
            out.append(_DOUBLE_ESCAPES[nxt])
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def parse_line(line: str) -> Entry:
    match = _ASSIGNMENT.match(line)
    if match is None:
        return Other(line)
    start = match.end()
    rest = line[start:]

    if rest[:1] in ("'", '"'):
        close = _closing_quote(rest, rest[0])
        if close == -1:
            return Other(line)
        value = _unescape(rest[1:close], rest[0])
        return Assignment(match["key"], value, line, start, start + close + 1)

    comment = _INLINE_COMMENT.search(rest)
    raw = rest[: comment.start()] if comment else rest
    value = raw.rstrip()
    return Assignment(match["key"], value, line, start, start + len(value))


class EnvFile:
    def __init__(self, entries: list[Entry], trailing_newline: bool = True) -> None:
        self.entries = entries
        self.trailing_newline = trailing_newline

    @classmethod
    def parse(cls, text: str) -> EnvFile:
        lines = text.splitlines()
        return cls([parse_line(line) for line in lines], text == "" or text.endswith("\n"))

    @classmethod
    def read(cls, path: Path) -> EnvFile:
        """Parse `path`; a missing file is an empty one."""
        try:
            return cls.parse(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls([])

    def keys(self) -> list[str]:
        seen = dict.fromkeys(e.key for e in self.entries if isinstance(e, Assignment))
        return list(seen)

    def get(self, key: str) -> str | None:
        """The effective value: like python-dotenv, the last assignment wins."""
        found = None
        for entry in self.entries:
            if isinstance(entry, Assignment) and entry.key == key:
                found = entry.value
        return found

    def set(self, key: str, value: str) -> None:
        """Update every assignment of `key` in place, or append one."""
        rendered = quote(value)
        updated = False
        for i, entry in enumerate(self.entries):
            if isinstance(entry, Assignment) and entry.key == key:
                line = entry.line[: entry.start] + rendered + entry.line[entry.end :]
                self.entries[i] = replace(
                    entry, value=value, line=line, end=entry.start + len(rendered)
                )
                updated = True
        if not updated:
            line = f"{key}={rendered}"
            self.entries.append(Assignment(key, value, line, len(key) + 1, len(line)))

    def render(self) -> str:
        text = "\n".join(entry.line for entry in self.entries)
        return text + "\n" if text and self.trailing_newline else text

    def write(self, path: Path) -> None:
        write_atomic(path, self.render())


def write_atomic(path: Path, text: str, mode: int = 0o600) -> None:
    """Replace `path` with `text` in one step; by default only the owner can read it."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
