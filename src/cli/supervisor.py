"""Run the API and web UI as children: stream their output, stop them together.

Each child gets its own session (so its own process group): `uv run` and
`npm run` both fork the real server, and signalling the group reaches it.
That also keeps the terminal's Ctrl-C away from the children; the supervisor
catches it and shuts them down itself — SIGTERM, a grace period, then SIGKILL.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

TAIL_LINES = 30


@dataclass(frozen=True)
class ProcessSpec:
    name: str
    command: tuple[str, ...]
    cwd: Path
    # ANSI SGR colour for the [name] prefix, e.g. "36" for cyan.
    color: str = "0"
    env: Mapping[str, str] | None = None


class Child:
    def __init__(self, spec: ProcessSpec, process: subprocess.Popen[str]) -> None:
        self.spec = spec
        self.process = process
        self.tail: deque[str] = deque(maxlen=TAIL_LINES)
        # Set while this child's output should be printed; it is kept in `tail` either way.
        self.echo = threading.Event()
        self.reader: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self.process.poll() is None


class Supervisor:
    def __init__(
        self,
        out: TextIO = sys.stdout,
        grace: float = 10.0,
        poll_interval: float = 0.2,
        color: bool | None = None,
    ) -> None:
        self.out = out
        self.grace = grace
        self.poll_interval = poll_interval
        self.color = out.isatty() if color is None else color
        self.children: list[Child] = []
        self._lock = threading.Lock()
        # Set once the terminal is gone (SIGHUP): keep draining the children's
        # pipes, but stop writing, so a dead terminal cannot abort shutdown.
        self._output_gone = False

    # ---------- starting ----------

    def start(self, spec: ProcessSpec, echo: bool = True) -> Child:
        process = subprocess.Popen(
            list(spec.command),
            cwd=spec.cwd,
            env=dict(spec.env) if spec.env is not None else None,
            stdin=subprocess.DEVNULL,  # the terminal belongs to our own prompts
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )
        child = Child(spec, process)
        if echo:
            child.echo.set()
        child.reader = threading.Thread(target=self._pump, args=(child,), daemon=True)
        child.reader.start()
        self.children.append(child)
        return child

    def run_to_completion(self, spec: ProcessSpec) -> int:
        """Run a one-off command (a build, an install) with prefixed output."""
        child = self.start(spec)
        code = child.process.wait()
        self._drain(child)
        self.children.remove(child)
        return code

    def echo_all(self) -> None:
        for child in self.children:
            child.echo.set()

    # ---------- output ----------

    def _pump(self, child: Child) -> None:
        stream = child.process.stdout
        if stream is None:
            return
        for raw in stream:
            line = raw.rstrip("\n")
            child.tail.append(line)
            if child.echo.is_set():
                self._emit(child.spec, line)

    def _emit(self, spec: ProcessSpec, line: str) -> None:
        prefix = f"[{spec.name}]"
        if self.color:
            prefix = f"\033[{spec.color}m{prefix}\033[0m"
        self._write(f"{prefix} {line}\n")

    def say(self, message: str) -> None:
        self._write(f"{message}\n")

    def _write(self, text: str) -> None:
        with self._lock:
            if self._output_gone:
                return
            try:
                self.out.write(text)
                self.out.flush()
            except OSError:
                self._output_gone = True

    def print_tail(self, child: Child) -> None:
        self._drain(child)
        for line in child.tail:
            self._emit(child.spec, line)

    def _drain(self, child: Child) -> None:
        if child.reader is not None:
            child.reader.join(timeout=2)

    # ---------- supervising ----------

    def wait(self) -> int:
        """Block until Ctrl-C (returns 130) or a child exits (returns its code, at least 1)."""
        try:
            while True:
                for child in self.children:
                    code = child.process.poll()
                    if code is not None:
                        return self._died(child, code)
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            self.say("\nStopping…")
            self.stop()
            return 130

    def _died(self, child: Child, code: int) -> int:
        # The survivors' shutdown chatter would bury the lines that matter.
        for other in self.children:
            other.echo.clear()
        self.stop()
        self.say(f"\n{child.spec.name} exited unexpectedly with code {code}. Last {len(child.tail)} lines:")
        self.print_tail(child)
        return code or 1

    def stop(self) -> None:
        """SIGTERM every child's group, wait up to `grace`, then SIGKILL what is left.

        A second Ctrl-C skips the wait.
        """
        children = list(self.children)
        for child in children:
            _signal_group(child, signal.SIGTERM)
        deadline = time.monotonic() + self.grace
        try:
            for child in children:
                try:
                    child.process.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    break  # out of time; SIGKILL below
        except KeyboardInterrupt:
            self.say("Forcing shutdown…")
        # Also reaches grandchildren that outlived their group leader.
        for child in children:
            _signal_group(child, signal.SIGKILL)
        for child in children:
            child.process.wait()
            self._drain(child)


def _signal_group(child: Child, sig: signal.Signals) -> None:
    try:
        os.killpg(child.process.pid, sig)
    except (ProcessLookupError, PermissionError):
        # Gone, or (macOS) only an unreaped zombie leader is left.
        return
