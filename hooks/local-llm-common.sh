#!/usr/bin/env bash
# Shared by the hooks: whether a local LLM server answers, at LOCAL_LLM_BASE_URL if set,
# otherwise at the same default addresses the MCP server searches (Ollama, then
# llama-server). Each check waits at most a second, so a hook never stalls a prompt.
local_llm_reachable() {
  if [ -n "${LOCAL_LLM_BASE_URL:-}" ]; then
    curl -sf --max-time 1 "$LOCAL_LLM_BASE_URL/v1/models" >/dev/null 2>&1
  else
    curl -sf --max-time 1 http://localhost:11434/api/version >/dev/null 2>&1 \
      || curl -sf --max-time 1 http://localhost:8080/v1/models >/dev/null 2>&1
  fi
}
