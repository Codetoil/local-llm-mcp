#!/bin/bash
set -euo pipefail

LABEL="sh.brew.ollama"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
PB=/usr/libexec/PlistBuddy
DOMAIN="gui/$(id -u)"

# Edit this list: KEY=VALUE, one per line
VARS="
OLLAMA_KEEP_ALIVE=15m
OLLAMA_CONTEXT_LENGTH=131072
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0
"

[ -f "$PLIST" ] || { echo "Plist not found: $PLIST" >&2; exit 1; }

# Save the original before touching it; it becomes $PLIST.bak only if something changes.
backup=$(mktemp)
cp "$PLIST" "$backup"
trap 'echo "failed; the original plist is saved at $backup" >&2' ERR

# Make sure the EnvironmentVariables dict exists
$PB -c "Print :EnvironmentVariables" "$PLIST" >/dev/null 2>&1 \
  || $PB -c "Add :EnvironmentVariables dict" "$PLIST"

changed=0
while IFS='=' read -r key value; do
  [ -z "$key" ] && continue
  current=$($PB -c "Print :EnvironmentVariables:$key" "$PLIST" 2>/dev/null || true)
  if [ "$current" != "$value" ]; then
    if [ -z "$current" ]; then
      $PB -c "Add :EnvironmentVariables:$key string $value" "$PLIST"
    else
      $PB -c "Set :EnvironmentVariables:$key $value" "$PLIST"
    fi
    echo "set $key=$value"
    changed=1
  fi
done <<< "$VARS"

if [ "$changed" = 1 ]; then
  mv "$backup" "$PLIST.bak"
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
  launchctl bootstrap "$DOMAIN" "$PLIST"
  echo "service reloaded"
else
  rm -f "$backup"
  echo "plist already up to date"
fi

