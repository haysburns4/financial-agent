"""`doctor`: run every check without prompting, and say what to fix."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from src.cli import checks
from src.cli.checks import CheckResult, NetworkChecks, Status, System
from src.cli.env_file import EnvFile
from src.cli.spec import BY_NAME, SPEC, providers_in_use


def env_values(env: EnvFile) -> dict[str, str]:
    return {key: value for key in env.keys() if (value := env.get(key)) is not None}


def effective_values(env: Mapping[str, str], environ: Mapping[str, str]) -> dict[str, str]:
    """What Settings will see: shell exports override .env."""
    values = dict(env)
    values.update({spec.name: environ[spec.name] for spec in SPEC if spec.name in environ})
    return values


def local_checks(root: Path, env: Mapping[str, str], environ: Mapping[str, str], system: System) -> list[CheckResult]:
    """Everything that needs neither the network nor a complete config."""
    values = effective_values(env, environ)
    web = root / "web"
    results = checks.check_shadowing(env, environ)
    results.append(checks.check_web_env(web, values))
    results.append(checks.check_node(system))
    results.append(checks.check_node_modules(web))
    api_port = values.get("API_PORT") or BY_NAME["API_PORT"].default or ""
    ports = [int(api_port)] if api_port.isdigit() else []
    results.extend(checks.check_port(port, system) for port in [*ports, checks.WEB_PORT])
    return results


def run_doctor(
    root: Path, environ: Mapping[str, str], system: System, network: NetworkChecks | None
) -> list[CheckResult]:
    path = root / ".env"
    if path.exists():
        env = env_values(EnvFile.read(path))
        results = [checks.check_env_permissions(path)]
    else:
        # Settings can come from exported variables alone (CI, containers);
        # check_settings below fails if that leaves anything required unset.
        env = {}
        results = [
            checks.warn(".env", "not found; using exported variables only", f"{checks.SETUP_HINT}, or `cp .env.example .env`")
        ]
    values = effective_values(env, environ)
    results.extend(checks.check_settings(values))
    if not values.get("TOKEN_ENCRYPTION_KEY"):
        results.append(
            checks.warn(
                "TOKEN_ENCRYPTION_KEY",
                "not set; the E-Trade login will not survive a restart",
                checks.SETUP_HINT,
            )
        )
    package = checks.check_provider_package(providers_in_use(values), system)
    if package is not None:
        results.append(package)
    results.extend(local_checks(root, env, environ, system))
    if network is not None:
        results.extend(checks.check_network(values, network))
    return results


def exit_code(results: Sequence[CheckResult]) -> int:
    return 1 if any(r.status is Status.FAIL for r in results) else 0
