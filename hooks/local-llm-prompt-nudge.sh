#!/usr/bin/env bash
# UserPromptSubmit hint: when a prompt looks like work the local_llm_* tools are made for
# (reviewing or explaining files, reviewing git changes, digesting test or build output),
# adds one line to Claude's context naming the matching tool. Claude Code loads MCP tools
# as deferred names, which Claude rarely reaches for on its own; this puts the right one in
# front of it at the moment it matters, the way CodeGraph's own prompt hook does.
#
# Cheap by design: the prompt's words are matched first, so most prompts return at once
# without a network call; only a match checks that a local LLM server answers. Silent
# (returns {}) when nothing matches, the server is down, or jq is missing.

set -uo pipefail

none() { printf '{}'; exit 0; }
command -v jq >/dev/null 2>&1 || none

prompt=$(jq -r '.prompt // empty' 2>/dev/null | tr '[:upper:]' '[:lower:]')
[ -z "$prompt" ] && none

has() { printf '%s' "$prompt" | grep -Eq "$1"; }

# Most specific first; the first match wins.
if has '(review|check|look over|audit).*(diff|changes|branch|pull request|\bpr\b|commit)|(diff|changes|branch|pull request|\bpr\b).*(review|look over|audit)'; then
  hint="local_llm_review_diff runs git diff itself and reviews the result, so the diff never enters your context. Use it for changes you didn't write or no longer have in view (a branch against main, a checked-out PR, a long session's uncommitted work)."
elif has '\blogs?\b|test (output|run|results|failures?)|failing tests?|tests? (fail|are failing)|build (output|log|fails?|failure|error)|\bci\b|stack ?trace|traceback|npm test|pytest|go test|cargo test'; then
  hint="for long command output, redirect it to a file (e.g. \`npm test > /tmp/test.log 2>&1\`) and call local_llm_digest_file on it: it returns the outcome and each failure with its line numbers, so you only read those lines."
elif has '(review|audit|look over|check).*(file|code|module|class|function|\.[a-z]{1,4}\b)'; then
  hint="for a file of more than a few dozen lines, local_llm_review_file reviews it server-side and cites line numbers, at a fraction of the context a Read costs."
elif has '(explain|walk me through|what does|how does).*(file|code|module|work|\.[a-z]{1,4}\b)'; then
  hint="for a file of more than a few dozen lines, local_llm_explain_file explains it server-side, at a fraction of the context a Read costs."
elif has '(how|whether|do|does).*(relate|consistent|agree|depend)|across (these|the|several|multiple) files|compare .*files'; then
  hint="local_llm_analyze_files reads several files (or glob patterns) server-side and answers questions across them, such as how they relate or whether they agree."
else
  none
fi

# shellcheck source=local-llm-common.sh
. "$(dirname "$0")/local-llm-common.sh"
local_llm_reachable || none

context="local-llm: $hint These are deferred MCP tools: search for the tool by name to load it. Skip it for small files, or files you're about to edit (Read those)."
jq -n --arg context "$context" '{hookSpecificOutput: {hookEventName: "UserPromptSubmit", additionalContext: $context}}'
