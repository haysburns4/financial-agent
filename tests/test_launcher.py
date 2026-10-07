"""Launcher decisions that need no running processes."""
import os
import socket
import sys
from pathlib import Path

import pytest

from src.cli import launcher as launcher_module
from src.cli.__main__ import _with_default_command
from src.cli.checks import LocalServerTarget
from src.cli.launcher import Launcher, build_stale, local_llm_command
from src.cli.supervisor import Supervisor
from tests.cli_fakes import FakeSystem, ScriptedPrompter


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


# ---------- --with-local-llm ----------

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
FAKE_SERVER = Path(__file__).parent / "fixtures" / "fake_mlx_server.py"


class NoStartSupervisor(Supervisor):
    def start(self, spec, echo=True):
        raise AssertionError(f"should not have started {spec.name}")


def _launcher(tmp_path, system, supervisor=None, environ=None):
    io = ScriptedPrompter()
    launcher = Launcher(
        tmp_path, io, system, None, environ or {"HF_HUB_CACHE": str(tmp_path / "hub")},
        supervisor or NoStartSupervisor(), lambda url: True, interactive=False,
    )
    return launcher, io


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_local_llm_command_passes_the_model_and_port_verbatim():
    target = LocalServerTarget("http://localhost:9001/v1", "localhost", 9001, MODEL)
    assert local_llm_command(target) == (
        "uv", "run", "mlx_lm.server", "--model", MODEL, "--host", "127.0.0.1", "--port", "9001",
    )


def test_a_busy_port_is_taken_for_a_running_server(tmp_path):
    launcher, io = _launcher(tmp_path, FakeSystem(modules=("mlx_lm",), busy_ports=(8080,)))

    assert launcher._start_local_llm({"LLM_PROVIDER": "local"})
    assert any("already in use" in line and "not starting another" in line for line in io.output)


def test_a_remote_base_url_is_not_started_here(tmp_path):
    launcher, io = _launcher(tmp_path, FakeSystem(modules=("mlx_lm",)))

    assert launcher._start_local_llm({"LLM_PROVIDER": "local", "LOCAL_LLM_BASE_URL": "http://gpu-box:8080/v1"})
    assert any("not this machine" in line for line in io.output)


def test_missing_mlx_lm_stops_the_launch(tmp_path):
    launcher, io = _launcher(tmp_path, FakeSystem(modules=()))

    assert not launcher._start_local_llm({"LLM_PROVIDER": "local"})
    assert any("mlx-lm is not installed" in line for line in io.output)


def test_waits_for_the_model_to_load_then_stops_the_server_with_everything_else(tmp_path, monkeypatch):
    port = _free_port()
    monkeypatch.setattr(
        launcher_module, "local_llm_command",
        lambda target: (sys.executable, str(FAKE_SERVER), str(target.port), target.model, "1.0"),
    )
    supervisor = Supervisor(grace=2.0)
    launcher, io = _launcher(tmp_path, FakeSystem(modules=("mlx_lm",)), supervisor)
    values = {"LLM_PROVIDER": "local", "LOCAL_LLM_BASE_URL": f"http://127.0.0.1:{port}/v1"}

    try:
        assert launcher._start_local_llm(values)
        assert any("this can take" in line for line in io.output)
        assert any(line.startswith("MLX server ready") for line in io.output)
        (child,) = supervisor.children
        assert child.running and "request" in child.tail  # the warm-up completion reached it
    finally:
        supervisor.stop()
    assert not child.running
