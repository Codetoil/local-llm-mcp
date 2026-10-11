import { test, describe, before, after, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import fs from "fs/promises";
import os from "os";
import path from "path";
import { startMockOllamaServer } from "./helpers/mockOllamaServer.js";
import { TOOL_MODEL_PREFERENCES, TOOL_REASONING_OVERRIDES } from "../prompts.js";

// Its own file/process because index.js reads LOCAL_LLM_BASE_URL once at import time, and
// these tests need it pointed at an Ollama-shaped mock rather than the llama-server one.
let mock;
let LocalLlmServer;
let tmpDir;
let originalCwd;

const DEFAULT_INSTALLED = [
  { name: "qwen3:8b", capabilities: ["completion", "tools", "thinking"], maxContext: 40960 },
  { name: "llama3.2:latest", capabilities: ["completion", "tools"], maxContext: 131072 },
  { name: "nomic-embed-text:latest", capabilities: ["embedding"], maxContext: 2048 },
];

before(async () => {
  mock = await startMockOllamaServer();
  process.env.LOCAL_LLM_BASE_URL = mock.url;
  ({ LocalLlmServer } = await import("../index.js"));
  tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "local-llm-mcp-ollama-test-"));
  // Same reason as server.test.js: keep getCodeGraphContext() away from this repo's own index.
  originalCwd = process.cwd();
  process.chdir(tmpDir);
});

after(async () => {
  if (originalCwd) process.chdir(originalCwd);
  await mock?.close();
  if (tmpDir) await fs.rm(tmpDir, { recursive: true, force: true });
});

beforeEach(() => {
  mock.state.requests.length = 0;
  mock.state.installed = DEFAULT_INSTALLED;
  mock.state.loaded = [];
  mock.state.defaultContext = 4096;
  mock.state.loads = 0;
  mock.state.pullable = {};
  mock.state.decision = () => ({ effort: { type: "choice", choice: "high", probabilities: { low: 0.2, high: 0.8 } } });
  mock.state.doneReason = "stop";
});

const requestsTo = (url) => mock.state.requests.filter((r) => r.url === url);
// Chat requests that ask for an answer, as opposed to the empty ones that only load a model.
const chatRequests = () => requestsTo("/api/chat").filter((r) => r.body.messages?.length);

describe("backend detection", () => {
  afterEach(() => {
    delete process.env.LOCAL_LLM_BACKEND;
  });

  test("detects Ollama from /api/version and caches the answer", async () => {
    const server = new LocalLlmServer();
    assert.equal(await server.detectBackend(), "ollama");
    assert.equal(await server.detectBackend(), "ollama");
    assert.equal(requestsTo("/api/version").length, 1);
  });

  test("LOCAL_LLM_BACKEND=openai skips the probe and uses the OpenAI-compatible endpoints", async () => {
    process.env.LOCAL_LLM_BACKEND = "openai";
    const server = new LocalLlmServer();
    assert.equal(await server.detectBackend(), "openai");
    assert.equal(requestsTo("/api/version").length, 0);
  });
});

describe("resolveModel (Ollama)", () => {
  afterEach(() => {
    delete TOOL_MODEL_PREFERENCES.write_tests;
  });

  test("prefers the loaded model over installed ones listed first", async () => {
    mock.state.loaded = [{ name: "llama3.2:latest", context_length: 32768 }];
    const server = new LocalLlmServer();
    const resolved = await server.resolveModel(undefined, "generate_code");
    assert.equal(resolved.id, "llama3.2:latest");
    assert.equal(resolved.family, "llama");
  });

  test("with nothing loaded, falls back to the first installed chat model", async () => {
    const server = new LocalLlmServer();
    const resolved = await server.resolveModel(undefined, "generate_code");
    assert.equal(resolved.id, "qwen3:8b");
  });

  test("never picks an embedding-only model for a chat tool, even when it's the loaded one", async () => {
    mock.state.loaded = [{ name: "nomic-embed-text:latest", context_length: 2048 }];
    const server = new LocalLlmServer();
    const resolved = await server.resolveModel(undefined, "review_file");
    assert.equal(resolved.id, "qwen3:8b");
  });

  test("picks an embedding model for semantic_similarity", async () => {
    mock.state.loaded = [{ name: "qwen3:8b", context_length: 32768 }];
    const server = new LocalLlmServer();
    const resolved = await server.resolveModel(undefined, "semantic_similarity");
    assert.equal(resolved.id, "nomic-embed-text:latest");
  });

  test("explains how to get an embedding model when none is installed", async () => {
    mock.state.installed = DEFAULT_INSTALLED.filter((model) => !model.capabilities.includes("embedding"));
    const server = new LocalLlmServer();
    await assert.rejects(() => server.resolveModel(undefined, "semantic_similarity"), /ollama pull nomic-embed-text/);
  });

  test("TOOL_MODEL_PREFERENCES still applies across installed models", async () => {
    TOOL_MODEL_PREFERENCES.write_tests = ["llama"];
    const server = new LocalLlmServer();
    const resolved = await server.resolveModel(undefined, "write_tests");
    assert.equal(resolved.id, "llama3.2:latest");
  });
});

describe("callLocalLlm (Ollama)", () => {
  afterEach(() => {
    delete TOOL_REASONING_OVERRIDES.generate_code;
    delete process.env.LOCAL_LLM_CHAT_MODEL;
  });

  test("LOCAL_LLM_CHAT_MODEL is used even when another model is loaded", async () => {
    process.env.LOCAL_LLM_CHAT_MODEL = "llama3.2:latest";
    mock.state.loaded = [{ name: "qwen3:8b", context_length: 32768 }];
    const server = new LocalLlmServer();
    await server.generateCode({ prompt: "x", language: "python" });
    assert.equal(chatRequests()[0].body.model, "llama3.2:latest");
  });

  test("a misspelled model surfaces Ollama's own explanation, not just the status code", async () => {
    const server = new LocalLlmServer();
    await assert.rejects(
      () => server.generateCode({ model: "gemma4:e4b-typo", prompt: "x", language: "python" }),
      /status code 404 \(model 'gemma4:e4b-typo' not found\)/
    );
  });

  test("uses the native /api/chat with num_ctx and num_predict, not /v1/chat/completions", async () => {
    const server = new LocalLlmServer();
    const result = await server.generateCode({ prompt: "add two numbers", language: "javascript" });
    assert.match(result.content[0].text, /add two numbers/);

    assert.equal(requestsTo("/v1/chat/completions").length, 0);
    const [chat] = chatRequests();
    assert.equal(chat.body.model, "qwen3:8b");
    assert.equal(chat.body.stream, false);
    assert.equal(chat.body.options.num_ctx, 32768);
    assert.ok(chat.body.options.num_predict > 0);
    assert.equal(chat.body.messages[0].role, "system");
    assert.equal(chat.body.think, undefined);
    assert.equal(chat.body.chat_template_kwargs, undefined);
  });

  test("maps a reasoning override to Ollama's own think flag", async () => {
    TOOL_REASONING_OVERRIDES.generate_code = false;
    const server = new LocalLlmServer();
    await server.generateCode({ prompt: "x", language: "python" });
    const [chat] = chatRequests();
    assert.equal(chat.body.think, false);
  });

  test("appends the generation-limit warning on done_reason length", async () => {
    mock.state.doneReason = "length";
    const server = new LocalLlmServer();
    const result = await server.generateCode({ prompt: "x", language: "python" });
    assert.match(result.content[0].text, /truncated at the \d+-token generation limit/);
  });

  test("warns when Ollama evaluated far fewer prompt tokens than the input needs", async () => {
    const original = mock.state.promptEvalCount;
    mock.state.promptEvalCount = () => 5; // way below ~chars/4 for any real prompt
    try {
      const server = new LocalLlmServer();
      const result = await server.generateCode({ model: "llama3.2:latest", prompt: "x".repeat(2000), language: "python" });
      assert.match(result.content[0].text, /didn't fit the model's 32768-token context window/);
      assert.match(result.content[0].text, /raise LOCAL_LLM_MIN_CTX above 32768/);
    } finally {
      mock.state.promptEvalCount = original;
    }
  });

  test("the truncation warning says when the window is already the model's maximum", async () => {
    mock.state.installed = [{ name: "tiny:1b", capabilities: ["completion"], maxContext: 8192 }];
    const original = mock.state.promptEvalCount;
    mock.state.promptEvalCount = () => 5;
    try {
      const server = new LocalLlmServer();
      const result = await server.generateCode({ prompt: "x".repeat(2000), language: "python" });
      assert.match(result.content[0].text, /didn't fit the model's 8192-token context window/);
      assert.match(result.content[0].text, /largest window this model supports/);
    } finally {
      mock.state.promptEvalCount = original;
    }
  });

  test("an unloaded model is loaded with an empty request, then raised to LOCAL_LLM_MIN_CTX", async () => {
    const server = new LocalLlmServer();
    await server.generateCode({ prompt: "x", language: "python" });
    const [preload, chat] = requestsTo("/api/chat");
    assert.deepEqual(preload.body.messages, []);
    assert.equal(chat.body.options.num_ctx, 32768);
    assert.deepEqual(mock.state.loaded, [{ name: "qwen3:8b", context_length: 32768 }]);
  });

  test("keeps a larger configured default (OLLAMA_CONTEXT_LENGTH), loading the model only once", async () => {
    mock.state.defaultContext = 131072;
    const server = new LocalLlmServer();
    await server.generateCode({ model: "llama3.2:latest", prompt: "x", language: "python" });
    await server.generateCode({ model: "llama3.2:latest", prompt: "y", language: "python" });
    assert.deepEqual(chatRequests().map((r) => r.body.options.num_ctx), [131072, 131072]);
    assert.equal(mock.state.loads, 1);
  });

  test("keeps the window of a model already running above the minimum, without reloading", async () => {
    mock.state.loaded = [{ name: "llama3.2:latest", context_length: 65536 }];
    const server = new LocalLlmServer();
    await server.generateCode({ model: "llama3.2:latest", prompt: "x", language: "python" });
    assert.equal(chatRequests()[0].body.options.num_ctx, 65536);
    assert.equal(requestsTo("/api/chat").length, 1); // no empty preload request
    assert.equal(mock.state.loads, 0);
  });

  test("raises a model running below the minimum", async () => {
    mock.state.loaded = [{ name: "llama3.2:latest", context_length: 1024 }];
    const server = new LocalLlmServer();
    await server.generateCode({ model: "llama3.2:latest", prompt: "x", language: "python" });
    assert.equal(chatRequests()[0].body.options.num_ctx, 32768);
  });

  test("caps the minimum at the model's maximum, so a small model isn't reloaded every call", async () => {
    mock.state.installed = [{ name: "tiny:1b", capabilities: ["completion"], maxContext: 8192 }];
    const server = new LocalLlmServer();
    await server.generateCode({ prompt: "x", language: "python" });
    await server.generateCode({ prompt: "y", language: "python" });
    assert.deepEqual(chatRequests().map((r) => r.body.options.num_ctx), [8192, 8192]);
    assert.equal(mock.state.loads, 1);
  });

  test("no truncation warning on a normally sized prompt", async () => {
    const server = new LocalLlmServer();
    const result = await server.generateCode({ prompt: "x", language: "python" });
    assert.doesNotMatch(result.content[0].text, /WARNING/);
  });

  test("records Ollama's eval counts as token usage", async () => {
    const server = new LocalLlmServer();
    await server.generateCode({ prompt: "x", language: "python" });
    const [chat] = chatRequests();
    const expectedPrompt = mock.state.promptEvalCount(chat.body);
    assert.equal(server.tokenStats.promptTokens, expectedPrompt);
    assert.equal(server.tokenStats.completionTokens, 20);
    assert.equal(server.tokenStats.totalTokens, expectedPrompt + 20);
  });

  test("file-aware tools go through the same native path", async () => {
    const filePath = path.join(tmpDir, "sample.js");
    await fs.writeFile(filePath, "export const a = 1;\n");
    const server = new LocalLlmServer();
    const result = await server.reviewFile({ file_path: filePath });
    assert.match(result.content[0].text, /export const a = 1;/);
    assert.equal(chatRequests().length, 1);
  });
});

describe("other tools (Ollama)", () => {
  test("serverInfo reports the loaded model and the context window it's running at", async () => {
    mock.state.loaded = [{ name: "llama3.2:latest", context_length: 32768 }];
    const server = new LocalLlmServer();
    const info = JSON.parse((await server.serverInfo()).content[0].text);
    assert.equal(info.backend, "ollama");
    assert.equal(info.ollama_version, "0.40.2");
    assert.equal(info.model_id, "llama3.2:latest");
    assert.equal(info.model_loaded, true);
    assert.equal(info.context_size, 32768);
    assert.equal(info.max_context_size, 131072);
    assert.equal(info.min_context_size, 32768);
  });

  test("serverInfo with nothing loaded names the model that would be used, running window unknown", async () => {
    const server = new LocalLlmServer();
    const info = JSON.parse((await server.serverInfo()).content[0].text);
    assert.equal(info.model_id, "qwen3:8b");
    assert.equal(info.model_loaded, false);
    assert.equal(info.context_size, null);
    assert.equal(info.max_context_size, 40960);
  });

  test("tokenize explains it isn't available instead of failing on a 404", async () => {
    const server = new LocalLlmServer();
    await assert.rejects(() => server.tokenize({ text: "hello" }), /isn't supported with Ollama/);
    assert.equal(requestsTo("/tokenize").length, 0);
  });

  test("semanticSimilarity embeds with the embedding model via /v1/embeddings", async () => {
    mock.state.loaded = [{ name: "qwen3:8b", context_length: 32768 }];
    const server = new LocalLlmServer();
    const result = await server.semanticSimilarity({ query: "abc", candidates: ["abd", "a much longer text"] });
    const { results } = JSON.parse(result.content[0].text);
    assert.equal(results.length, 2);
    const [embed] = requestsTo("/v1/embeddings");
    assert.equal(embed.body.model, "nomic-embed-text:latest");
  });
});

describe("auto-pull of configured models (Ollama)", () => {
  afterEach(() => {
    delete process.env.LOCAL_LLM_CHAT_MODEL;
    delete process.env.LOCAL_LLM_EMBEDDING_MODEL;
    delete process.env.LOCAL_LLM_AUTO_PULL;
  });

  test("pulls a configured chat model that isn't installed, then uses it", async () => {
    process.env.LOCAL_LLM_CHAT_MODEL = "gemma4:e4b";
    mock.state.pullable = { "gemma4:e4b": { capabilities: ["completion"], maxContext: 131072 } };
    const server = new LocalLlmServer();
    await server.generateCode({ prompt: "x", language: "python" });
    assert.deepEqual(requestsTo("/api/pull").map((r) => r.body.model), ["gemma4:e4b"]);
    assert.equal(chatRequests()[0].body.model, "gemma4:e4b");
  });

  test("pulls a configured embedding model on first semantic_similarity call", async () => {
    process.env.LOCAL_LLM_EMBEDDING_MODEL = "mxbai-embed-large";
    mock.state.pullable = { "mxbai-embed-large:latest": { capabilities: ["embedding"], maxContext: 512 } };
    const server = new LocalLlmServer();
    await server.semanticSimilarity({ query: "a", candidates: ["b"] });
    assert.equal(requestsTo("/api/pull").length, 1);
    assert.equal(requestsTo("/v1/embeddings")[0].body.model, "mxbai-embed-large");
  });

  test("an untagged configured name matches the installed ':latest' model, so nothing is pulled", async () => {
    process.env.LOCAL_LLM_EMBEDDING_MODEL = "nomic-embed-text";
    const server = new LocalLlmServer();
    await server.semanticSimilarity({ query: "a", candidates: ["b"] });
    assert.equal(requestsTo("/api/pull").length, 0);
  });

  test("checks the installed list once per configured model, not on every call", async () => {
    process.env.LOCAL_LLM_CHAT_MODEL = "llama3.2:latest";
    const server = new LocalLlmServer();
    await server.generateCode({ prompt: "x", language: "python" });
    const afterFirstCall = requestsTo("/api/tags").length;
    await server.generateCode({ prompt: "y", language: "python" });
    assert.equal(requestsTo("/api/tags").length, afterFirstCall);
  });

  test("concurrent first calls share a single pull", async () => {
    process.env.LOCAL_LLM_CHAT_MODEL = "gemma4:e4b";
    mock.state.pullable = { "gemma4:e4b": { capabilities: ["completion"], maxContext: 131072 } };
    const server = new LocalLlmServer();
    await Promise.all([
      server.generateCode({ prompt: "x", language: "python" }),
      server.generateCode({ prompt: "y", language: "python" }),
    ]);
    assert.equal(requestsTo("/api/pull").length, 1);
  });

  test("a configured name Ollama can't pull fails with Ollama's reason", async () => {
    process.env.LOCAL_LLM_CHAT_MODEL = "gemma4:e4b-typo";
    const server = new LocalLlmServer();
    await assert.rejects(
      () => server.generateCode({ prompt: "x", language: "python" }),
      /gemma4:e4b-typo is configured but not installed, and pulling it failed: pull model manifest: file does not exist/
    );
  });

  test("LOCAL_LLM_AUTO_PULL=0 never pulls", async () => {
    process.env.LOCAL_LLM_AUTO_PULL = "0";
    process.env.LOCAL_LLM_CHAT_MODEL = "gemma4:e4b";
    mock.state.pullable = { "gemma4:e4b": { capabilities: ["completion"], maxContext: 131072 } };
    const server = new LocalLlmServer();
    await assert.rejects(() => server.generateCode({ prompt: "x", language: "python" }), /not found/);
    assert.equal(requestsTo("/api/pull").length, 0);
  });

  test("never pulls a model named by the calling agent's per-call model argument", async () => {
    mock.state.pullable = { "gemma4:e4b": { capabilities: ["completion"], maxContext: 131072 } };
    const server = new LocalLlmServer();
    await assert.rejects(() => server.generateCode({ model: "gemma4:e4b", prompt: "x", language: "python" }), /not found/);
    assert.equal(requestsTo("/api/pull").length, 0);
  });
});

describe("thinking decisions (Ollama)", () => {
  const LAYA = { name: "laya:latest", capabilities: ["decision"], maxContext: 512 };
  const answer = (choice, p) => () => ({
    effort: { type: "choice", choice, probabilities: { low: choice === "low" ? p : 1 - p, high: choice === "high" ? p : 1 - p } },
  });

  beforeEach(() => {
    mock.state.installed = [...DEFAULT_INSTALLED, LAYA];
  });

  afterEach(() => {
    delete process.env.LOCAL_LLM_DECISION_MODEL;
    delete TOOL_REASONING_OVERRIDES.generate_code;
  });

  test("a confident 'high' turns thinking on, using the installed decision model", async () => {
    const server = new LocalLlmServer();
    await server.generateCode({ prompt: "find the race condition in this job queue", language: "go" });
    const [decision] = requestsTo("/v1/systemone");
    assert.equal(decision.body.model, "laya:latest");
    assert.equal(decision.body.state.tool, "generate_code");
    assert.match(decision.body.state.request, /race condition/);
    assert.ok(decision.body.questions.effort.criteria.low && decision.body.questions.effort.criteria.high);
    assert.equal(chatRequests()[0].body.think, true);
  });

  test("a confident 'low' turns thinking off", async () => {
    mock.state.decision = answer("low", 0.78);
    const server = new LocalLlmServer();
    await server.generateCode({ prompt: "is_even(n)", language: "python" });
    assert.equal(chatRequests()[0].body.think, false);
  });

  test("an unsure answer leaves the chat model's own default", async () => {
    mock.state.decision = answer("high", 0.55);
    const server = new LocalLlmServer();
    await server.generateCode({ prompt: "x", language: "python" });
    assert.equal(chatRequests()[0].body.think, undefined);
    assert.equal(server.thinkingDecisions.unsure, 1);
  });

  test("sends the instruction and code size, never the code itself", async () => {
    const filePath = path.join(tmpDir, "big.js");
    await fs.writeFile(filePath, Array.from({ length: 300 }, (_, i) => `const secretValue${i} = ${i};`).join("\n"));
    const server = new LocalLlmServer();
    await server.reviewFile({ file_path: filePath, focus: "concurrency bugs" });
    const { state } = requestsTo("/v1/systemone")[0].body;
    assert.deepEqual(state, { tool: "review_file", request: "concurrency bugs", code_lines: 300 });
  });

  test("a request without instructions says so instead of sending an empty string", async () => {
    const filePath = path.join(tmpDir, "small.js");
    await fs.writeFile(filePath, "export const a = 1;\n");
    const server = new LocalLlmServer();
    await server.reviewFile({ file_path: filePath });
    assert.equal(requestsTo("/v1/systemone")[0].body.state.request, "(no instructions beyond the tool itself)");
  });

  test("no decision model installed: no decision, model default", async () => {
    mock.state.installed = DEFAULT_INSTALLED;
    const server = new LocalLlmServer();
    await server.generateCode({ prompt: "x", language: "python" });
    assert.equal(requestsTo("/v1/systemone").length, 0);
    assert.equal(chatRequests()[0].body.think, undefined);
  });

  test("a chat model that can't think is never asked to", async () => {
    const server = new LocalLlmServer();
    await server.generateCode({ model: "llama3.2:latest", prompt: "x", language: "python" });
    assert.equal(requestsTo("/v1/systemone").length, 0);
    assert.equal(chatRequests()[0].body.think, undefined);
  });

  test("a fixed TOOL_REASONING_OVERRIDES entry wins, without asking", async () => {
    TOOL_REASONING_OVERRIDES.generate_code = false;
    const server = new LocalLlmServer();
    await server.generateCode({ prompt: "find the race condition", language: "go" });
    assert.equal(requestsTo("/v1/systemone").length, 0);
    assert.equal(chatRequests()[0].body.think, false);
  });

  test("LOCAL_LLM_DECISION_MODEL=off disables it", async () => {
    process.env.LOCAL_LLM_DECISION_MODEL = "off";
    const server = new LocalLlmServer();
    await server.generateCode({ prompt: "x", language: "python" });
    assert.equal(requestsTo("/v1/systemone").length, 0);
  });

  test("a configured decision model that isn't installed is pulled, then used", async () => {
    mock.state.installed = DEFAULT_INSTALLED;
    mock.state.pullable = { "laya:421m-typed-decisions-mlx-fp16": { capabilities: ["decision"], maxContext: 1024 } };
    process.env.LOCAL_LLM_DECISION_MODEL = "laya:421m-typed-decisions-mlx-fp16";
    const server = new LocalLlmServer();
    await server.generateCode({ prompt: "x", language: "python" });
    assert.equal(requestsTo("/api/pull")[0].body.model, "laya:421m-typed-decisions-mlx-fp16");
    assert.equal(requestsTo("/v1/systemone")[0].body.model, "laya:421m-typed-decisions-mlx-fp16");
  });

  test("a failing decision model never fails the call", async () => {
    mock.state.decision = null;
    const server = new LocalLlmServer();
    const result = await server.generateCode({ prompt: "x", language: "python" });
    assert.match(result.content[0].text, /echo:/);
    assert.equal(chatRequests()[0].body.think, undefined);
    assert.equal(server.thinkingDecisions.failed, 1);
  });

  test("session stats count the decisions", async () => {
    const server = new LocalLlmServer();
    await server.generateCode({ prompt: "x", language: "python" });
    mock.state.decision = answer("low", 0.9);
    await server.generateCode({ prompt: "y", language: "python" });
    const stats = JSON.parse(server.sessionStats().content[0].text);
    assert.deepEqual(stats.thinkingDecisions, { model: "laya:latest", decided: 2, thinkOn: 1, thinkOff: 1, unsure: 0, failed: 0 });
  });
});
