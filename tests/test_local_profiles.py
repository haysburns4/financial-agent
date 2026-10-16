"""What differs between local servers and model families: resolution, the
thinking switch each pairing gets, and stripping thinking from answer text."""
import pytest

from src.llm.local_backend import ThinkFilter
from src.llm.local_profiles import (
    GENERIC,
    SERVERS,
    ThinkingOff,
    model_profile,
    server_profile,
    server_root,
    thinking_off,
)


@pytest.mark.parametrize(
    ("model", "family"),
    [
        ("mlx-community/Qwen3-14B-4bit", "qwen3"),
        ("qwen3:14b", "qwen3"),  # an Ollama tag
        ("Qwen/Qwen2.5-7B-Instruct", "qwen"),
        ("deepseek-r1:8b", "deepseek-r1"),
        ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "deepseek-r1"),  # not llama: it thinks
        ("Qwen/QwQ-32B", "deepseek-r1"),
        ("openai/gpt-oss-20b", "gpt-oss"),
        ("meta-llama/Llama-3.1-8B-Instruct", "llama"),
        ("gemma3:12b", "gemma"),
        ("mistralai/Ministral-8B-Instruct-2410", "mistral"),
        ("microsoft/phi-4", "phi"),
        ("dolphin-2.9", "generic"),  # "phi" inside a word is not Phi
        ("some-new-model", "generic"),
    ],
)
def test_the_family_is_matched_from_the_model_id(model, family):
    assert model_profile(model).family == family


def test_a_named_family_overrides_the_match():
    assert model_profile("my-finetune", "qwen3").family == "qwen3"
    assert model_profile("my-finetune", " Llama ").family == "llama"
    assert model_profile("mlx-community/Qwen3-14B-4bit", "no-such-family") is GENERIC


def test_an_unknown_or_blank_server_is_the_default():
    assert server_profile(None).kind == "mlx"
    assert server_profile("").kind == "mlx"
    assert server_profile(" Ollama ").kind == "ollama"
    assert server_profile("tgi").kind == "mlx"


@pytest.mark.parametrize(
    ("server", "model", "expected"),
    [
        # The default: identical to the request this backend always sent.
        ("mlx", "qwen3:14b", ThinkingOff(template_kwargs={"enable_thinking": False})),
        ("llamacpp", "qwen3:14b", ThinkingOff(template_kwargs={"enable_thinking": False})),
        ("ollama", "qwen3:14b", ThinkingOff(reasoning_effort="none")),
        # Honours neither switch: Qwen3's own soft switch still works.
        ("lmstudio", "qwen3:14b", ThinkingOff(system_suffix="\n\n/no_think")),
        ("vllm", "openai/gpt-oss-20b", ThinkingOff(reasoning_effort="low")),
        ("mlx", "openai/gpt-oss-20b", ThinkingOff()),
        ("ollama", "llama3.1:8b", ThinkingOff()),
        ("ollama", "deepseek-r1:8b", ThinkingOff()),  # cannot be switched off
    ],
)
def test_thinking_is_switched_off_the_first_way_both_sides_support(server, model, expected):
    assert thinking_off(SERVERS[server], model_profile(model)) == expected


@pytest.mark.parametrize(
    ("base_url", "root"),
    [
        ("http://localhost:11434/v1", "http://localhost:11434"),
        ("http://localhost:11434/v1/", "http://localhost:11434"),
        ("http://gpu-box/llm/v1", "http://gpu-box/llm"),
        ("http://localhost:8080", "http://localhost:8080"),
    ],
)
def test_native_endpoints_live_at_the_server_root(base_url, root):
    assert server_root(base_url) == root


def test_only_mlx_is_ours_to_launch():
    # It is the one server this repo depends on (mlx-lm in pyproject).
    assert [s.kind for s in SERVERS.values() if s.launchable] == ["mlx"]


# ---------- stripping thinking from the answer ----------


def _run(chunks: list[str], opens_implicitly: bool = False) -> list[str]:
    think = ThinkFilter(("<think>", "</think>"), opens_implicitly)
    released = [think.feed(c) for c in chunks]
    return [*released, think.flush()]


def test_a_think_block_is_removed_even_when_its_tags_are_split():
    released = _run(["<thi", "nk>The user wants", " 4.</th", "ink>\n\nFour", "."])
    assert "".join(released) == "\n\nFour."
    assert released[0] == ""  # held back: it might have been (and was) a tag


def test_text_around_a_block_is_kept_and_released_as_it_settles():
    released = _run(["Hi <", "b>", " there <think>x</think> done"])
    assert released[:2] == ["Hi ", "<b>"]  # "<" waited one chunk, then was not a tag
    assert "".join(released) == "Hi <b> there  done"


def test_an_answer_without_thinking_passes_straight_through():
    assert _run(["Four", "."]) == ["Four", ".", ""]


def test_a_block_opened_by_the_template_hides_everything_before_its_close():
    released = _run(["Let me add", " 2 and 2.</think>", "Four."], opens_implicitly=True)
    assert released == ["", "", "Four.", ""]


def test_a_template_opened_block_that_never_closes_was_the_answer():
    # The model skipped thinking; nothing it said can be dropped.
    assert "".join(_run(["Four", "."], opens_implicitly=True)) == "Four."


def test_an_explicit_block_left_open_at_the_end_is_dropped():
    # The reply was cut off mid-thought: there is no answer to show.
    assert "".join(_run(["<think>still going"])) == ""
