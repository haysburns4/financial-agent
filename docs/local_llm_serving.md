# Local LLM serving

`LLM_PROVIDER=local` works with any OpenAI-compatible server. Define which one in
`LOCAL_LLM_SERVER`. By mapping to the respective `ServerProfile`, the app will send the request fields it takes, turn thinking off, and fit prompts to the context window the server reports.

## MLX

The default, on Apple Silicon. Run it with `./start --with-local-llm`.

```
LLM_PROVIDER=local
LOCAL_LLM_SERVER=mlx
LOCAL_LLM_BASE_URL=http://localhost:8080/v1
LOCAL_LLM_MODEL=mlx-community/Qwen3-14B-4bit
```

## Ollama

Run `ollama pull qwen3:14b`, then start the server. Its default context is a few
thousand tokens, so give it room or prompts get cut:
`OLLAMA_CONTEXT_LENGTH=40960 ollama serve`.

```
LLM_PROVIDER=local
LOCAL_LLM_SERVER=ollama
LOCAL_LLM_BASE_URL=http://localhost:11434/v1
LOCAL_LLM_MODEL=qwen3:14b
```

## llama.cpp

Start it with `llama-server -hf Qwen/Qwen3-14B-GGUF:Q4_K_M --jinja -c 40960 --port 8080`.

```
LLM_PROVIDER=local
LOCAL_LLM_SERVER=llamacpp
LOCAL_LLM_BASE_URL=http://localhost:8080/v1
LOCAL_LLM_MODEL=Qwen/Qwen3-14B-GGUF:Q4_K_M
```

## Other servers

`LOCAL_LLM_SERVER` also takes `lmstudio`, `vllm`, or `openai_compat` for anything
else. vLLM's default port, 8000, is the same as the API's, so run it with `--port 8001`.

## A model per task

`LOCAL_LLM_MODEL_CHAT` and `LOCAL_LLM_MODEL_SYNTHESIZER` run the two tasks on
different models from one server; unset, both use `LOCAL_LLM_MODEL`.

## Checking a setup

`./start doctor` and `/health` report the server, its models, and its context
window. To check a new setup end to end, with a real tool-calling chat turn:

```
RUN_LIVE_LOCAL=1 uv run pytest tests/test_local_live.py
```
