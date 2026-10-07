"""The supervisor never leaves a child (or grandchild) behind."""
import io
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.cli.supervisor import ProcessSpec, Supervisor

ROOT = Path(__file__).resolve().parents[1]

# Writes "<own pid> <grandchild pid>" to argv[1], then idles. The grandchild
# shares its process group, as next-server does under `npm run start`.
WITH_GRANDCHILD = """
import os, subprocess, sys, time
g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
tmp = sys.argv[1] + ".tmp"
open(tmp, "w").write(f"{os.getpid()} {g.pid}")
os.replace(tmp, sys.argv[1])
time.sleep(60)
"""

# Ignores SIGTERM, so only the SIGKILL fallback stops it.
STUBBORN = """
import os, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
tmp = sys.argv[1] + ".tmp"
open(tmp, "w").write(str(os.getpid()))
os.replace(tmp, sys.argv[1])
time.sleep(60)
"""

RUNNER = """
import signal, sys
from pathlib import Path
from src.cli.__main__ import _interrupt
from src.cli.supervisor import ProcessSpec, Supervisor

signal.signal(signal.SIGHUP, _interrupt)
signal.signal(signal.SIGTERM, _interrupt)

tmp = Path(sys.argv[1])
supervisor = Supervisor(grace=1.0)
supervisor.start(ProcessSpec("web", (sys.executable, "-c", sys.argv[2], str(tmp / "web.pid")), tmp))
supervisor.start(ProcessSpec("api", (sys.executable, "-c", sys.argv[3], str(tmp / "api.pid")), tmp))
sys.exit(supervisor.wait())
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def _pids(path: Path) -> list[int]:
    return [int(p) for p in path.read_text().split()]


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP], ids=["ctrl-c", "kill", "hangup"])
def test_interrupt_stops_both_children_and_their_groups(tmp_path, sig):
    runner = subprocess.Popen(
        [sys.executable, "-c", RUNNER, str(tmp_path), WITH_GRANDCHILD, STUBBORN],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        assert _wait_for(lambda: (tmp_path / "web.pid").exists() and (tmp_path / "api.pid").exists())
        pids = _pids(tmp_path / "web.pid") + _pids(tmp_path / "api.pid")
        assert all(_alive(pid) for pid in pids)

        runner.send_signal(sig)
        output, _ = runner.communicate(timeout=10)

        assert runner.returncode == 130, output
        assert "Stopping" in output
        # The grandchild is reaped by init once orphaned, so give it a moment.
        assert _wait_for(lambda: not any(_alive(pid) for pid in pids)), pids
    finally:
        runner.kill()


def test_one_child_dying_stops_the_other_and_shows_its_last_lines(tmp_path):
    out = io.StringIO()
    supervisor = Supervisor(out=out, grace=2.0, poll_interval=0.05)
    survivor = supervisor.start(
        ProcessSpec("web", (sys.executable, "-c", "import time; time.sleep(60)"), tmp_path)
    )
    supervisor.start(
        ProcessSpec(
            "api",
            (sys.executable, "-c", "import sys, time; print('boom: bad config', flush=True); time.sleep(0.3); sys.exit(3)"),
            tmp_path,
        )
    )

    code = supervisor.wait()

    assert code == 3
    assert not survivor.running
    text = out.getvalue()
    assert "api exited unexpectedly with code 3" in text
    assert text.rstrip().endswith("[api] boom: bad config")


def test_output_is_prefixed_per_child(tmp_path):
    out = io.StringIO()
    supervisor = Supervisor(out=out, color=False)
    code = supervisor.run_to_completion(ProcessSpec("web", (sys.executable, "-c", "print('compiled')"), tmp_path))
    assert code == 0
    assert out.getvalue() == "[web] compiled\n"
    assert supervisor.children == []


def test_quiet_child_keeps_its_tail(tmp_path):
    out = io.StringIO()
    supervisor = Supervisor(out=out)
    child = supervisor.start(
        ProcessSpec("api", (sys.executable, "-c", "print('starting up')"), tmp_path), echo=False
    )
    child.process.wait()
    supervisor.stop()
    assert out.getvalue() == ""
    assert list(child.tail) == ["starting up"]


def test_stop_escalates_to_sigkill(tmp_path):
    pid_file = tmp_path / "pid"
    supervisor = Supervisor(out=io.StringIO(), grace=0.3)
    child = supervisor.start(ProcessSpec("api", (sys.executable, "-c", STUBBORN, str(pid_file)), tmp_path))
    assert _wait_for(pid_file.exists)

    started = time.monotonic()
    supervisor.stop()

    assert child.process.returncode == -signal.SIGKILL
    assert time.monotonic() - started < 3
