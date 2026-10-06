#!/usr/bin/env python3
"""Anti-slop fence for the Python half of this repo.

A port of the oxlint anti-slop ruleset (MIT, https://github.com/dmmulroy/anti-slop,
vendored for the TSX half in web/tools/anti-slop/) to Python idioms. Same premise:
reject patterns that *hide* uncertainty instead of resolving it at a boundary.

    uv run python tools/anti_slop.py src tests
    uv run python tools/anti_slop.py --all src        # include opt-in rules
    uv run python tools/anti_slop.py --list-rules

A finding can be waived inline, but only with a reason:

    value = getattr(raw, "usage", None)  # anti-slop: allow no-getattr-fallback - third-party optional field

Stdlib only, so it runs anywhere the project runs.
"""
from __future__ import annotations

import argparse
import ast
import io
import re
import tokenize
import sys
from dataclasses import dataclass
from pathlib import Path

# Rules that are off unless --all is passed: real code has legitimate uses and
# turning them on by default produces noise rather than a fence.
OPT_IN = frozenset({"no-defensive-isinstance", "no-unsafe-dictionary-type"})

RULES: dict[str, str] = {
    "no-any-parameters": "A parameter typed Any accepts anything and proves nothing. Type it, or parse it at the boundary.",
    "no-any-returns": "An Any return pushes the unknown onto every caller. Return a type the caller can act on.",
    "no-unsafe-dictionary-type": "dict[str, Any] (or a bare dict) is a bag, not a contract. Use a dataclass or a precise value type.",
    "no-getattr-fallback": "getattr with a literal name and a default reaches around the type instead of establishing it.",
    "no-conditional-empty-dict-spread": "This spread hides key omission behind an empty dict. Build the dict in statements and set the key when present.",
    "no-module-mocking": "Patching a module by string path couples the test to an import path rather than a seam. Inject the dependency instead.",
    "no-shape-in-symbol-names": 'Rename for the domain role; "shape" describes structure rather than ownership.',
    "no-cast-or-type-ignore": "cast() and `type: ignore` assert what was not proven. Narrow with a real check or fix the type.",
    "no-silent-except": "An except body of pass/... swallows the failure. Handle it, or let it propagate.",
    "no-defensive-isinstance": "An isinstance check narrows a representation without establishing its contract. Parse at the I/O boundary, then branch on the domain value.",
}

_WAIVER = re.compile(r"#\s*anti-slop:\s*allow\s+([a-z0-9-]+)\s*[-—]\s*(\S.*)$")


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    col: int
    rule: str

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: {self.rule}: {RULES[self.rule]}"


def _is_any(node: ast.expr | None) -> bool:
    """True for `Any`, `typing.Any`, and `t.Any`."""
    if isinstance(node, ast.Name):
        return node.id == "Any"
    if isinstance(node, ast.Attribute):
        return node.attr == "Any"
    return False


def _is_unsafe_dict(node: ast.expr | None) -> bool:
    """True for a bare `dict`/`Dict` annotation or one whose value type is Any."""
    if isinstance(node, ast.Name) and node.id in ("dict", "Dict"):
        return True
    if not isinstance(node, ast.Subscript):
        return False
    base = node.value
    named = isinstance(base, ast.Name) and base.id in ("dict", "Dict")
    if not named:
        return False
    key_value = node.slice
    return isinstance(key_value, ast.Tuple) and any(_is_any(e) for e in key_value.elts)


def _annotation_findings(node: ast.expr | None) -> bool:
    """Whether an annotation is Any or an unsafe dict, at any nesting depth."""
    if node is None:
        return False
    if _is_any(node) or _is_unsafe_dict(node):
        return True
    if isinstance(node, ast.Subscript):
        inner = node.slice
        parts = inner.elts if isinstance(inner, ast.Tuple) else [inner]
        return any(_annotation_findings(p) for p in parts)
    return False


class Visitor(ast.NodeVisitor):
    def __init__(self, path: Path, enabled: frozenset[str]) -> None:
        self.path = path
        self.enabled = enabled
        self.findings: list[Finding] = []

    def _report(self, rule: str, node: ast.AST) -> None:
        if rule not in self.enabled:
            return
        self.findings.append(Finding(self.path, node.lineno, node.col_offset + 1, rule))

    # ---------- signatures ----------

    def _check_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._check_name(node.name, node)
        args = node.args
        for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
            if arg is None:
                continue
            if _is_any(arg.annotation):
                self._report("no-any-parameters", arg)
            elif _annotation_findings(arg.annotation):
                self._report("no-unsafe-dictionary-type", arg)
            self._check_name(arg.arg, arg)

        if _is_any(node.returns):
            self._report("no-any-returns", node.returns)
        elif _annotation_findings(node.returns):
            self._report("no-unsafe-dictionary-type", node.returns)
        self.generic_visit(node)

    visit_FunctionDef = _check_function
    visit_AsyncFunctionDef = _check_function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._check_name(node.name, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if _is_any(node.annotation):
            self._report("no-any-parameters", node)
        elif _annotation_findings(node.annotation):
            self._report("no-unsafe-dictionary-type", node)
        self.generic_visit(node)

    # ---------- names ----------

    def _check_name(self, name: str, node: ast.AST) -> None:
        if "shape" in name.lower():
            self._report("no-shape-in-symbol-names", node)

    # ---------- expressions ----------

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        name = _called_name(func)

        # Reflect.get analogue: getattr(obj, "literal", default)
        if name == "getattr" and len(node.args) == 3:
            target = node.args[1]
            if isinstance(target, ast.Constant) and isinstance(target.value, str):
                self._report("no-getattr-fallback", node)

        if name == "cast":
            self._report("no-cast-or-type-ignore", node)

        # Patching by import path rather than injecting a seam.
        patching = name in ("setattr", "setitem", "patch", "object") and node.args
        if patching and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            receiver = func.value if isinstance(func, ast.Attribute) else None
            if _called_name(receiver) in ("monkeypatch", "mock", "patch") or name == "patch":
                self._report("no-module-mocking", node)

        if name == "isinstance":
            self._report("no-defensive-isinstance", node)

        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        # `**({...} if cond else {})` — a None key is a ** unpacking.
        for key, value in zip(node.keys, node.values):
            if key is not None or not isinstance(value, ast.IfExp):
                continue
            branches = (value.body, value.orelse)
            if any(isinstance(b, ast.Dict) and not b.keys for b in branches):
                self._report("no-conditional-empty-dict-spread", value)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        body = [s for s in node.body if not isinstance(s, ast.Expr) or not isinstance(s.value, ast.Constant)]
        if not body or all(isinstance(s, ast.Pass) for s in body):
            self._report("no-silent-except", node)
        self.generic_visit(node)


def _called_name(func: ast.expr | None) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _type_ignore_comments(source: str) -> list[tuple[int, int]]:
    """Positions of `type: ignore` in comments only, never in string literals."""
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    return [
        (token.start[0], token.start[1] + 1)
        for token in tokens
        if token.type == tokenize.COMMENT and "type: ignore" in token.string
    ]


def _waivers(source: str) -> dict[int, set[str]]:
    """Line number -> rules waived on that line, for waivers carrying a reason."""
    out: dict[int, set[str]] = {}
    for number, text in enumerate(source.splitlines(), start=1):
        match = _WAIVER.search(text)
        if match:
            out.setdefault(number, set()).add(match.group(1))
    return out


def check(path: Path, enabled: frozenset[str]) -> list[Finding]:
    source = path.read_text(encoding="utf-8")
    visitor = Visitor(path, enabled)
    visitor.visit(ast.parse(source, filename=str(path)))

    findings = list(visitor.findings)
    if "no-cast-or-type-ignore" in enabled:
        findings.extend(
            Finding(path, line, col, "no-cast-or-type-ignore")
            for line, col in _type_ignore_comments(source)
        )

    waived = _waivers(source)
    return sorted(
        (f for f in findings if f.rule not in waived.get(f.line, ())),
        key=lambda f: (str(f.path), f.line, f.col),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path, help="files or directories to check")
    parser.add_argument("--all", action="store_true", help=f"also run opt-in rules: {', '.join(sorted(OPT_IN))}")
    parser.add_argument("--list-rules", action="store_true")
    args = parser.parse_args()

    if args.list_rules:
        for rule, message in sorted(RULES.items()):
            flag = " (opt-in)" if rule in OPT_IN else ""
            print(f"{rule}{flag}\n    {message}\n")
        return 0

    if not args.paths:
        parser.error("give at least one path")

    enabled = frozenset(RULES) if args.all else frozenset(RULES) - OPT_IN

    files: list[Path] = []
    for path in args.paths:
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])

    findings = [f for file in files if "__pycache__" not in file.parts for f in check(file, enabled)]
    for finding in findings:
        print(finding.render())

    if findings:
        counts: dict[str, int] = {}
        for finding in findings:
            counts[finding.rule] = counts.get(finding.rule, 0) + 1
        print(f"\n{len(findings)} finding(s) across {len(files)} file(s):", file=sys.stderr)
        for rule, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {count:>3}  {rule}", file=sys.stderr)
        return 1

    print(f"clean: {len(files)} file(s), {len(enabled)} rule(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
