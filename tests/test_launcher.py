"""Launcher decisions that need no running processes."""
import os
from pathlib import Path

import pytest

from src.cli.__main__ import _with_default_command
from src.cli.launcher import build_stale


def _touch(path: Path, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    os.utime(path, (mtime, mtime))


@pytest.fixture
def web(tmp_path):
    _touch(tmp_path / "app" / "page.tsx", 1000)
    _touch(tmp_path / "app" / "api" / "health" / "route.ts", 1000)
    _touch(tmp_path / "package-lock.json", 1000)
    return tmp_path


def test_no_build_yet_is_stale(web):
    assert build_stale(web)


def test_build_newer_than_every_source_is_fresh(web):
    _touch(web / ".next" / "BUILD_ID", 2000)
    assert not build_stale(web)


def test_edited_source_in_a_nested_folder_is_stale(web):
    _touch(web / ".next" / "BUILD_ID", 2000)
    os.utime(web / "app" / "api" / "health" / "route.ts", (3000, 3000))
    assert build_stale(web)


def test_newer_lockfile_is_stale(web):
    _touch(web / ".next" / "BUILD_ID", 2000)
    os.utime(web / "package-lock.json", (3000, 3000))
    assert build_stale(web)


def test_other_files_in_next_do_not_count(web):
    _touch(web / ".next" / "BUILD_ID", 2000)
    _touch(web / ".next" / "cache" / "later", 3000)
    assert not build_stale(web)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], ["start"]),
        (["--dev"], ["start", "--dev"]),
        (["login"], ["login"]),
        (["doctor", "--offline"], ["doctor", "--offline"]),
        (["--help"], ["--help"]),
    ],
)
def test_start_is_the_default_command(argv, expected):
    assert _with_default_command(argv) == expected
