"""What differs between local LLM servers, and between model families.

`LLM_PROVIDER=local` talks to any OpenAI-compatible server. Two things vary
underneath that API, and they vary independently, so they are two tables:

- the server (LOCAL_LLM_SERVER): which output-cap parameter it takes, which
  thinking switches it honours, where it reports its context window, and how
  to start it;
- the model family, matched from the model id (or LOCAL_LLM_MODEL_FAMILY):
  how to ask it not to think, the tags it thinks inside, and how its
  tokenizer treats digits.

Plain data, standard library only: src/cli reads these before src.config can
be built. Claims marked UNVERIFIED come from docs or third-party reports, not
a probe; correct them here when a live server says otherwise.
"""
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

ServerKind = Literal["mlx", "ollama", "llamacpp", "lmstudio", "vllm", "openai_compat"]
# Where a server reports the context window it actually allocated.
WindowSource = Literal["ollama_ps", "llamacpp_props", "vllm_models", "lmstudio_models"]


@dataclass(frozen=True)
class ServerProfile:
    kind: ServerKind
    label: str
    default_base_url: str
    # The request field that caps output tokens.
    max_tokens_param: Literal["max_completion_tokens", "max_tokens"]
    # Forwards `chat_template_kwargs` to the model's chat template.
    template_kwargs: bool
    # Honours the `reasoning_effort` request field.
    reasoning_effort: bool
    window_source: WindowSource | None
    # How to start it, with {model} and {port} filled in.
    start_hint: str
    # How to give the model a larger context window, if the server sets it.
    window_hint: str | None = None
    # `./start --with-local-llm` can run it (it is a dependency of this repo).
    launchable: bool = False


SERVERS: Mapping[str, ServerProfile] = {
    profile.kind: profile
    for profile in (
        # Probed against mlx-lm 0.32: takes max_completion_tokens, forwards
        # template kwargs, returns thinking in a separate `reasoning` field.
        # No endpoint reports the window; it is the model's own.
        ServerProfile(
            "mlx", "mlx_lm.server", "http://localhost:8080/v1", "max_completion_tokens",
            template_kwargs=True, reasoning_effort=False, window_source=None,
            start_hint="`uv run mlx_lm.server --model {model} --port {port}`",
            launchable=True,
        ),
        # docs.ollama.com/api/openai-compatibility: max_tokens (not
        # max_completion_tokens), reasoning_effort ("none" turns boolean
        # thinking off), no way to set num_ctx over this API. /api/ps reports
        # each loaded model's context_length. Template kwargs: UNVERIFIED, so
        # not relied on.
        ServerProfile(
            "ollama", "Ollama", "http://localhost:11434/v1", "max_tokens",
            template_kwargs=False, reasoning_effort=True, window_source="ollama_ps",
            start_hint="`ollama serve`, then `ollama pull {model}`",
            window_hint="set OLLAMA_CONTEXT_LENGTH for `ollama serve`, or PARAMETER num_ctx in a Modelfile",
        ),
        # llama.cpp tools/server README: chat_template_kwargs, tools with
        # --jinja (on by default), /props default_generation_settings.n_ctx.
        # Only max_tokens is documented. reasoning_effort: UNVERIFIED.
        ServerProfile(
            "llamacpp", "llama.cpp server", "http://localhost:8080/v1", "max_tokens",
            template_kwargs=True, reasoning_effort=False, window_source="llamacpp_props",
            start_hint="`llama-server -hf {model} --jinja --port {port}`",
            window_hint="pass a larger --ctx-size (-c) to llama-server",
        ),
        # LM Studio's /api/v0/models reports loaded_context_length for loaded
        # models (UNVERIFIED: third-party reports, not its docs). Template
        # kwargs and reasoning_effort: UNVERIFIED, so not relied on.
        ServerProfile(
            "lmstudio", "LM Studio", "http://localhost:1234/v1", "max_tokens",
            template_kwargs=False, reasoning_effort=False, window_source="lmstudio_models",
            start_hint="`lms server start --port {port}`, then `lms load {model}`",
            window_hint="raise the context length in the model's load settings",
        ),
        # vLLM: chat_template_kwargs, max_completion_tokens, and max_model_len
        # on each /v1/models entry. UNVERIFIED against a live server.
        ServerProfile(
            "vllm", "vLLM", "http://localhost:8000/v1", "max_completion_tokens",
            template_kwargs=True, reasoning_effort=True, window_source="vllm_models",
            start_hint="`vllm serve {model} --port {port}`",
            window_hint="pass a larger --max-model-len to vllm serve",
        ),
        # Anything else: assume only the oldest common subset of the API.
        ServerProfile(
            "openai_compat", "OpenAI-compatible server", "http://localhost:8080/v1", "max_tokens",
            template_kwargs=False, reasoning_effort=False, window_source=None,
            start_hint="start your server so it answers at this URL",
        ),
    )
}
SERVER_KINDS: tuple[str, ...] = tuple(SERVERS)
DEFAULT_SERVER: ServerKind = "mlx"


@dataclass(frozen=True)
class ModelProfile:
    family: str
    # Searched in the lower-cased model id; the first family to match wins.
    patterns: tuple[str, ...]
    # Ways to ask for no thinking, tried in this order against what the
    # server honours: chat template kwargs, `reasoning_effort`, then a
    # directive appended to the system prompt.
    template_kwargs: Mapping[str, object] = field(default_factory=dict)
    reasoning_effort: str | None = None
    no_think_directive: str | None = None
    # Tags the model may think inside, in the answer text; None if it never
    # does (or the server always separates it).
    think_tags: tuple[str, str] | None = None
    # The chat template opens the thinking block itself, so the reply holds
    # only the closing tag.
    think_opens_implicitly: bool = False
    # 1 if the tokenizer splits numbers into single digits, 3 if it groups them.
    digits_per_token: int = 1


THINK = ("<think>", "</think>")
MODELS: tuple[ModelProfile, ...] = (
    # Hybrid thinking: the template's enable_thinking flag, or "/no_think".
    ModelProfile(
        "qwen3", (r"qwen3",), {"enable_thinking": False}, reasoning_effort="none",
        no_think_directive="/no_think", think_tags=THINK,
    ),
    # Always think; their templates open the block.
    ModelProfile("deepseek-r1", (r"deepseek-r1", r"qwq"), think_tags=THINK, think_opens_implicitly=True),
    # Reasoning is a separate channel the server parses; it cannot be off.
    ModelProfile("gpt-oss", (r"gpt-oss",), reasoning_effort="low", digits_per_token=3),
    ModelProfile("qwen", (r"qwen",)),
    ModelProfile("llama", (r"llama",), digits_per_token=3),
    ModelProfile("gemma", (r"gemma",)),
    ModelProfile("mistral", (r"mistral", r"ministral", r"devstral", r"codestral")),
    ModelProfile("phi", (r"\bphi",), think_tags=THINK, digits_per_token=3),
)
# Unknown models: strip think tags if any appear, and count tokens high.
GENERIC = ModelProfile("generic", (), think_tags=THINK)
MODEL_FAMILIES: tuple[str, ...] = (*(m.family for m in MODELS), GENERIC.family)


def server_profile(kind: str | None) -> ServerProfile:
    """The profile for `kind`; unknown or blank means the default server."""
    return SERVERS.get((kind or DEFAULT_SERVER).strip().lower(), SERVERS[DEFAULT_SERVER])


def model_profile(model: str, family: str | None = None) -> ModelProfile:
    """The named `family`, else the first whose pattern is in `model`, else generic."""
    if family:
        named = family.strip().lower()
        return next((m for m in (*MODELS, GENERIC) if m.family == named), GENERIC)
    lowered = model.lower()
    return next((m for m in MODELS if any(re.search(p, lowered) for p in m.patterns)), GENERIC)


@dataclass(frozen=True)
class ThinkingOff:
    """How one request asks one model, on one server, not to think."""

    template_kwargs: Mapping[str, object] = field(default_factory=dict)
    reasoning_effort: str | None = None
    system_suffix: str = ""


def thinking_off(server: ServerProfile, model: ModelProfile) -> ThinkingOff:
    """The first switch the model has that the server honours."""
    if model.template_kwargs and server.template_kwargs:
        return ThinkingOff(template_kwargs=model.template_kwargs)
    if model.reasoning_effort and server.reasoning_effort:
        return ThinkingOff(reasoning_effort=model.reasoning_effort)
    if model.no_think_directive:
        return ThinkingOff(system_suffix=f"\n\n{model.no_think_directive}")
    return ThinkingOff()


def server_root(base_url: str) -> str:
    """`base_url` without its trailing /v1: where native endpoints live."""
    parts = urlsplit(base_url.rstrip("/"))
    path = parts.path[: -len("/v1")] if parts.path.endswith("/v1") else parts.path
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def start_hint(server: ServerProfile, base_url: str, model: str) -> str:
    """How to start `server` so it answers at `base_url` with `model`."""
    command = server.start_hint.format(model=model, port=_port(base_url) or _port(server.default_base_url))
    launch = "`./start --with-local-llm` or " if server.launchable else ""
    return f"start it with {launch}{command}, or point LOCAL_LLM_BASE_URL at where it runs"


def _port(url: str) -> int | None:
    """The port `url` names, or its scheme's; None if it is not a valid one."""
    parts = urlsplit(url)
    try:
        return parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return None
