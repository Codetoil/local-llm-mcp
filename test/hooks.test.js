import { test, describe, before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "child_process";
import fs from "fs/promises";
import os from "os";
import path from "path";
import { fileURLToPath } from "url";
import { startMockLlamaServer } from "./helpers/mockLlamaServer.js";

// The plugin's shell hooks, run as real processes with JSON on stdin, the way Claude Code
// runs them. LOCAL_LLM_BASE_URL points them at a mock server (reachable) or a closed port
// (unreachable), so they never probe the real default ports.
const hooksDir = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "hooks");
let mock;
let closedUrl;
let tmpDir;

// Asynchronous on purpose: a synchronous spawn would block this process's event loop, so the
// in-process mock server couldn't answer the hook's curl and every check would time out.
function runHook(name, input, baseUrl) {
  return new Promise((resolve, reject) => {
    const child = spawn(path.join(hooksDir, name), { env: { ...process.env, LOCAL_LLM_BASE_URL: baseUrl } });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => (stdout += chunk));
    child.stderr.on("data", (chunk) => (stderr += chunk));
    child.on("error", reject);
    child.on("close", (code) => {
      if (code !== 0) return reject(new Error(`hook exited ${code}: ${stderr}`));
      resolve(JSON.parse(stdout));
    });
    child.stdin.end(JSON.stringify(input));
  });
}

before(async () => {
  mock = await startMockLlamaServer();
  const gone = await startMockLlamaServer();
  closedUrl = gone.url;
  await gone.close();
  tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "local-llm-mcp-hooks-test-"));
});

after(async () => {
  await mock?.close();
  if (tmpDir) await fs.rm(tmpDir, { recursive: true, force: true });
});

describe("prompt nudge hook (UserPromptSubmit)", () => {
  const nudge = (prompt, baseUrl = mock.url) => runHook("local-llm-prompt-nudge.sh", { prompt }, baseUrl);
  const hintFor = async (prompt) => (await nudge(prompt)).hookSpecificOutput?.additionalContext ?? null;

  const cases = [
    ["Can you review the changes on this branch before I open a PR?", "local_llm_review_diff"],
    ["look over my uncommitted changes", "local_llm_review_diff"],
    ["Review the diff against main", "local_llm_review_diff"],
    ["The tests are failing, can you figure out why?", "local_llm_digest_file"],
    ["run npm test and fix what breaks", "local_llm_digest_file"],
    ["why does the CI build fail?", "local_llm_digest_file"],
    ["here's a stack trace from production", "local_llm_digest_file"],
    ["Please review index.js for security issues", "local_llm_review_file"],
    ["audit the auth module", "local_llm_review_file"],
    ["Explain how this file works: src/router.ts", "local_llm_explain_file"],
    ["what does the parser module do?", "local_llm_explain_file"],
    ["Do README.md and AGENTS.md agree with each other?", "local_llm_analyze_files"],
    ["compare these config files", "local_llm_analyze_files"],
  ];
  for (const [prompt, tool] of cases) {
    test(`"${prompt}" -> ${tool}`, async () => {
      const hint = await hintFor(prompt);
      assert.ok(hint, "expected a hint");
      assert.match(hint, new RegExp(`\\b${tool}\\b`));
      assert.match(hint, /deferred MCP tools/);
    });
  }

  for (const prompt of ["rename the variable x to count", "add a dark mode toggle", "commit and push", "hi"]) {
    test(`no hint for "${prompt}"`, async () => {
      assert.deepEqual(await nudge(prompt), {});
    });
  }

  test("silent when no local LLM server answers, even for a matching prompt", async () => {
    assert.deepEqual(await nudge("review the changes on this branch", closedUrl), {});
  });

  test("silent for input without a prompt", async () => {
    assert.deepEqual(await runHook("local-llm-prompt-nudge.sh", {}, mock.url), {});
  });

  test("uses the UserPromptSubmit output shape", async () => {
    const output = await nudge("the build fails, here is the log");
    assert.equal(output.hookSpecificOutput.hookEventName, "UserPromptSubmit");
    assert.equal(typeof output.hookSpecificOutput.additionalContext, "string");
  });
});

describe("Read gate hook (PreToolUse on Read)", () => {
  const sessionId = `hooks-test-${process.pid}-${Date.now()}`;
  const stateFile = path.join("/tmp/claude-local-llm-gate", `${sessionId}.seen`);
  const gate = (filePath, baseUrl = mock.url, session = sessionId) =>
    runHook("local-llm-read-gate.sh", { session_id: session, tool_input: { file_path: filePath } }, baseUrl);

  after(async () => {
    await fs.rm(stateFile, { force: true });
    await fs.rm(path.join("/tmp/claude-local-llm-gate", `${sessionId}-down.seen`), { force: true });
  });

  test("denies the first Read of a larger file, then lets the retry through", async () => {
    const big = path.join(tmpDir, "big.js");
    await fs.writeFile(big, "x".repeat(1000));
    const first = await gate(big);
    assert.equal(first.hookSpecificOutput.permissionDecision, "deny");
    assert.match(first.hookSpecificOutput.permissionDecisionReason, /local_llm_review_file/);
    assert.deepEqual(await gate(big), {});
  });

  test("lets small files through", async () => {
    const small = path.join(tmpDir, "small.js");
    await fs.writeFile(small, "const a = 1;\n");
    assert.deepEqual(await gate(small), {});
  });

  test("fails open when no local LLM server answers", async () => {
    const big = path.join(tmpDir, "big-down.js");
    await fs.writeFile(big, "x".repeat(1000));
    assert.deepEqual(await gate(big, closedUrl, `${sessionId}-down`), {});
  });
});
