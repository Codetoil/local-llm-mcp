#!/usr/bin/env node

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import axios from "axios";
import fs from "fs/promises";
import { realpathSync } from "fs";
import path from "path";
import { execFile } from "child_process";
import { promisify } from "util";
import { buildMessages, resolveFamily, TOOL_MODEL_PREFERENCES, TOOL_REASONING_OVERRIDES } from "./prompts.js";
import packageJson from "./package.json" with { type: "json" };

const execFileAsync = promisify(execFile);

const LOCAL_LLM_BASE_URL = process.env.LOCAL_LLM_BASE_URL || "http://localhost:8080";
// Defensive ceiling on generated tokens per call. With a reasoning model and no cap, a
// single tool call can spend an unbounded number of hidden <think> tokens before ever
// producing a visible answer -- this bounds worst-case latency without being tight enough
// to routinely truncate legitimate output. Override via LOCAL_LLM_MAX_TOKENS if needed.
const MAX_TOKENS = Number(process.env.LOCAL_LLM_MAX_TOKENS) || 16384;
// Minimum context window for Ollama calls (num_ctx). Ollama silently drops whatever prompt
// doesn't fit the window -- exactly the large file contents the file-aware tools send -- and
// its stock default window is small. A model already running with at least this much (e.g.
// from OLLAMA_CONTEXT_LENGTH) keeps its window; see ollamaContextWindow. Capped at the
// model's maximum. LOCAL_LLM_MIN_CTX=0 never sends num_ctx, leaving it entirely to Ollama.
// Only used with the Ollama backend; other servers fix the context size at server start.
const MIN_CTX = process.env.LOCAL_LLM_MIN_CTX === undefined ? 32768 : Number(process.env.LOCAL_LLM_MIN_CTX);
// Ordinary text tokenizes at roughly 3-5 characters per token. A prompt that comes back with
// far more characters per evaluated token than that was cut down to fit num_ctx.
const TRUNCATED_CHARS_PER_TOKEN = 8;
const MODEL_CACHE_TTL_MS = 30_000;
// Pulling a configured model can mean a multi-GB download; allow far longer than a chat call.
const PULL_TIMEOUT_MS = 60 * 60_000;
const GLOB_METACHARACTERS = /[*?[{]/;
const MAX_GLOB_MATCHES = 50;
const CODEGRAPH_CONTEXT_CHAR_LIMIT = 4000;
const CODEGRAPH_TIMEOUT_MS = 5000;

const SERVER_INSTRUCTIONS = `Offloads code-shaped work (generation, review, refactoring, explanation, test-writing) to a local LLM, saving your own context budget.

For anything already on disk, prefer the file-aware tools (local_llm_review_file, local_llm_explain_file, local_llm_analyze_files, local_llm_generate_code_with_context) over reading the file yourself and pasting it into the string-based tools -- the file-aware tools read server-side, at roughly 50 tokens of your context versus thousands for a full read. local_llm_analyze_files also works well for cross-file consistency checks (e.g. "do these docs agree with each other"), not just dependency/relationship analysis.

This server has no internet access and no knowledge of anything outside the files you point it at. Don't use it for questions needing current external knowledge (library APIs, changelogs, GitHub issues, current docs) -- it will confabulate specifics rather than refuse. Verify its output before trusting it, the same way you would a fast but occasionally-wrong reviewer.`;

const MODEL_ARG_SCHEMA = {
  type: "string",
  description:
    "Optional model override. If omitted, the model currently loaded by the local LLM server is auto-detected.",
};

function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Prefixes each line with its 1-based line number and a tab, matching the format the
// orchestrating agent's own Read tool returns. Without this, the local model has no way to
// know real line numbers and will confabulate them when a review/analysis response cites a
// location -- used only for review_file/explain_file/analyze_files, not
// generate_code_with_context, since generation output shouldn't risk copying the numbering.
function numberLines(content) {
  return content
    .split("\n")
    .map((line, i) => `${i + 1}\t${line}`)
    .join("\n");
}

// Expands any glob patterns (entries containing *, ?, [, or {) into absolute file paths;
// plain literal paths pass through unchanged. Relative patterns resolve against
// process.cwd(). Throws if the total expansion exceeds MAX_GLOB_MATCHES, so a broad
// pattern like "**/*.js" can't silently balloon a request into hundreds of files.
async function expandFilePaths(patterns) {
  const expanded = [];
  for (const pattern of patterns) {
    if (!GLOB_METACHARACTERS.test(pattern)) {
      expanded.push(pattern);
      continue;
    }
    for await (const match of fs.glob(pattern)) {
      expanded.push(path.resolve(match));
    }
  }

  if (expanded.length > MAX_GLOB_MATCHES) {
    throw new Error(
      `Glob pattern(s) matched ${expanded.length} files, which exceeds the limit of ${MAX_GLOB_MATCHES}. Narrow your pattern(s).`
    );
  }

  return expanded;
}

// Best-effort CodeGraph enrichment: if the target project has a .codegraph/ index, runs
// `codegraph explore <query>` and folds its output (call paths, blast radius) into the
// prompt sent to the local model. Returns null on any failure -- no .codegraph/ directory,
// the `codegraph` binary missing from PATH, a timeout, or a non-zero exit -- since this is
// strictly additive and must never block a tool response or require CodeGraph to be
// installed. The binary name is overridable via CODEGRAPH_BIN for testing.
async function getCodeGraphContext(query, cwd = process.cwd()) {
  try {
    await fs.access(path.join(cwd, ".codegraph"));
  } catch {
    return null;
  }

  try {
    const bin = process.env.CODEGRAPH_BIN || "codegraph";
    const { stdout } = await execFileAsync(bin, ["explore", query], {
      cwd,
      timeout: CODEGRAPH_TIMEOUT_MS,
    });
    return stdout.slice(0, CODEGRAPH_CONTEXT_CHAR_LIMIT);
  } catch {
    return null;
  }
}

class LocalLlmServer {
  constructor() {
    this.server = new Server(
      {
        name: "local-llm-mcp",
        version: packageJson.version,
      },
      {
        capabilities: {
          tools: {},
        },
        instructions: SERVER_INSTRUCTIONS,
      }
    );

    this.modelsCache = null;
    this.modelsCacheAt = 0;
    this.maxContextCache = new Map();
    this.confirmedModels = new Set();
    this.pulls = new Map();
    this.tokenStats = { calls: 0, promptTokens: 0, completionTokens: 0, totalTokens: 0, perTool: {} };

    this.setupToolHandlers();

    this.server.onerror = (error) => console.error("[MCP Error]", error);
  }

  setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: "local_llm_generate_code",
          description: "Generate code using your local LLM server. Use this for writing new functions, classes, or code snippets. Provide detailed requirements and context. If you have reference files to base the generated code on, use local_llm_generate_code_with_context instead so they're read server-side.",
          inputSchema: {
            type: "object",
            properties: {
              prompt: {
                type: "string",
                description: "Detailed description of the code to generate, including requirements, language, and context",
              },
              language: {
                type: "string",
                description: "Programming language (e.g., javascript, python, rust)",
              },
              model: MODEL_ARG_SCHEMA,
            },
            required: ["prompt", "language"],
          },
        },
        {
          name: "local_llm_explain_code",
          description: "Explain how code works using your local LLM server. Use this to understand complex code sections, algorithms, or patterns already in the conversation. If the code is in a file instead, use local_llm_explain_file so it's read server-side.",
          inputSchema: {
            type: "object",
            properties: {
              code: {
                type: "string",
                description: "The code to explain",
              },
              context: {
                type: "string",
                description: "Additional context about what you want to understand",
              },
              model: MODEL_ARG_SCHEMA,
            },
            required: ["code"],
          },
        },
        {
          name: "local_llm_review_code",
          description: "Review code for issues, bugs, or improvements using your local LLM server. Use this for code already in the conversation. If the code is in a file instead, use local_llm_review_file so it's read server-side.",
          inputSchema: {
            type: "object",
            properties: {
              code: {
                type: "string",
                description: "The code to review",
              },
              focus: {
                type: "string",
                description: "What to focus on (e.g., 'performance', 'security', 'best practices', 'bugs')",
                default: "general code quality",
              },
              model: MODEL_ARG_SCHEMA,
            },
            required: ["code"],
          },
        },
        {
          name: "local_llm_refactor_code",
          description: "Refactor code to improve quality, readability, or structure using your local LLM server.",
          inputSchema: {
            type: "object",
            properties: {
              code: {
                type: "string",
                description: "The code to refactor",
              },
              goal: {
                type: "string",
                description: "Refactoring goal (e.g., 'improve readability', 'reduce complexity', 'modernize syntax')",
              },
              model: MODEL_ARG_SCHEMA,
            },
            required: ["code", "goal"],
          },
        },
        {
          name: "local_llm_fix_code",
          description: "Fix bugs or errors in code using your local LLM server. Provide the broken code and error details.",
          inputSchema: {
            type: "object",
            properties: {
              code: {
                type: "string",
                description: "The code with issues",
              },
              error: {
                type: "string",
                description: "Error message or description of the problem",
              },
              model: MODEL_ARG_SCHEMA,
            },
            required: ["code", "error"],
          },
        },
        {
          name: "local_llm_write_tests",
          description: "Generate unit tests for code using your local LLM server.",
          inputSchema: {
            type: "object",
            properties: {
              code: {
                type: "string",
                description: "The code to write tests for",
              },
              framework: {
                type: "string",
                description: "Testing framework to use (e.g., 'jest', 'pytest', 'mocha')",
              },
              model: MODEL_ARG_SCHEMA,
            },
            required: ["code", "framework"],
          },
        },
        {
          name: "local_llm_general_task",
          description: "Execute any general coding task grounded in code/text you provide, using your local LLM server. Use this for tasks that don't fit other categories. This server has no internet access -- don't use it for questions needing current external knowledge (library APIs, docs, GitHub issues), since it will confabulate rather than refuse.",
          inputSchema: {
            type: "object",
            properties: {
              task: {
                type: "string",
                description: "Detailed description of the task to perform",
              },
              context: {
                type: "string",
                description: "Any relevant context, code, or background information",
              },
              model: MODEL_ARG_SCHEMA,
            },
            required: ["task"],
          },
        },
        {
          name: "local_llm_review_file",
          description: "Review a file by path using your local LLM server. Prefer this over reading the file yourself and passing it to local_llm_review_code -- the MCP server reads the file directly, at a fraction of the token cost of a full read. Findings cite real line numbers (the file is sent to the model pre-numbered), so file:line locations in the response can be trusted without re-deriving them yourself -- though the findings themselves still warrant the usual verification.",
          inputSchema: {
            type: "object",
            properties: {
              file_path: {
                type: "string",
                description: "Absolute path to the file to review",
              },
              focus: {
                type: "string",
                description: "What to focus on (e.g., 'performance', 'security', 'best practices', 'bugs')",
                default: "general code quality",
              },
              model: MODEL_ARG_SCHEMA,
            },
            required: ["file_path"],
          },
        },
        {
          name: "local_llm_explain_file",
          description: "Explain a file by path using your local LLM server. Prefer this over reading the file yourself and passing it to local_llm_explain_code -- the MCP server reads the file directly, at a fraction of the token cost of a full read. The file is sent to the model pre-numbered, so it can reference real line numbers in its explanation.",
          inputSchema: {
            type: "object",
            properties: {
              file_path: {
                type: "string",
                description: "Absolute path to the file to explain",
              },
              context: {
                type: "string",
                description: "Additional context about what you want to understand",
              },
              model: MODEL_ARG_SCHEMA,
            },
            required: ["file_path"],
          },
        },
        {
          name: "local_llm_analyze_files",
          description: "Analyze multiple files together using your local LLM server, reading them server-side instead of you reading each one. Useful for understanding relationships/dependencies between files, or for cross-file consistency checks (e.g. do these docs agree with each other, does this doc match the current code) -- prefer this over reading each file yourself first. Each file is sent to the model pre-numbered, so findings can cite real file:line locations -- verify a citation before acting on it if precision matters, the same way you'd verify any of its other output.",
          inputSchema: {
            type: "object",
            properties: {
              file_paths: {
                type: "array",
                items: { type: "string" },
                description: "Array of absolute paths to files to analyze together",
              },
              task: {
                type: "string",
                description: "What analysis to perform (e.g., 'find dependencies', 'check consistency', 'summarize architecture')",
              },
              model: MODEL_ARG_SCHEMA,
            },
            required: ["file_paths", "task"],
          },
        },
        {
          name: "local_llm_generate_code_with_context",
          description: "Generate code using your local LLM server with context from existing files. Reads reference files server-side to understand patterns -- prefer this over reading the reference files yourself and pasting them into local_llm_generate_code.",
          inputSchema: {
            type: "object",
            properties: {
              prompt: {
                type: "string",
                description: "Detailed description of the code to generate",
              },
              language: {
                type: "string",
                description: "Programming language (e.g., javascript, python, rust)",
              },
              context_files: {
                type: "array",
                items: { type: "string" },
                description: "Array of file paths to use as context/examples",
              },
              model: MODEL_ARG_SCHEMA,
            },
            required: ["prompt", "language"],
          },
        },
        {
          name: "local_llm_server_info",
          description: "Report which model the local LLM server currently has loaded, its context size, and its capabilities. Use this to check what's actually running before assuming a model or capability.",
          inputSchema: {
            type: "object",
            properties: {},
          },
        },
        {
          name: "local_llm_session_stats",
          description: "Report cumulative token usage sent to/received from the local LLM server so far in this session, with a per-tool breakdown. Use this to see how many tokens have been offloaded from the orchestrating agent's own context.",
          inputSchema: {
            type: "object",
            properties: {},
          },
        },
        {
          name: "local_llm_tokenize",
          description: "Count how many tokens a piece of text would consume according to the local LLM server's currently loaded tokenizer. Useful for checking context-window fit before sending large content. Needs llama-server; not available with Ollama.",
          inputSchema: {
            type: "object",
            properties: {
              text: {
                type: "string",
                description: "The text to tokenize",
              },
              include_tokens: {
                type: "boolean",
                description: "If true, also return the raw token ID array (omitted by default to keep output small)",
                default: false,
              },
            },
            required: ["text"],
          },
        },
        {
          name: "local_llm_semantic_similarity",
          description: "Rank a list of candidate texts by semantic similarity to a query, using the local LLM server's embedding model. Returns similarity scores only, never raw embedding vectors, to keep output small.",
          inputSchema: {
            type: "object",
            properties: {
              query: {
                type: "string",
                description: "The text to compare candidates against",
              },
              candidates: {
                type: "array",
                items: { type: "string" },
                description: "Texts to rank by similarity to the query",
              },
              model: MODEL_ARG_SCHEMA,
            },
            required: ["query", "candidates"],
          },
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;

      try {
        switch (name) {
          case "local_llm_generate_code":
            return await this.generateCode(args);
          case "local_llm_explain_code":
            return await this.explainCode(args);
          case "local_llm_review_code":
            return await this.reviewCode(args);
          case "local_llm_refactor_code":
            return await this.refactorCode(args);
          case "local_llm_fix_code":
            return await this.fixCode(args);
          case "local_llm_write_tests":
            return await this.writeTests(args);
          case "local_llm_general_task":
            return await this.generalTask(args);
          case "local_llm_review_file":
            return await this.reviewFile(args);
          case "local_llm_explain_file":
            return await this.explainFile(args);
          case "local_llm_analyze_files":
            return await this.analyzeFiles(args);
          case "local_llm_generate_code_with_context":
            return await this.generateCodeWithContext(args);
          case "local_llm_server_info":
            return await this.serverInfo();
          case "local_llm_session_stats":
            return this.sessionStats();
          case "local_llm_tokenize":
            return await this.tokenize(args);
          case "local_llm_semantic_similarity":
            return await this.semanticSimilarity(args);
          default:
            throw new Error(`Unknown tool: ${name}`);
        }
      } catch (error) {
        return {
          content: [
            {
              type: "text",
              text: `Error: ${error.message}`,
            },
          ],
        };
      }
    });
  }

  connectionErrorMessage() {
    return `Cannot connect to local LLM server. Make sure it's running at ${LOCAL_LLM_BASE_URL} and is either Ollama or OpenAI-compatible (llama-server, vLLM, LM Studio, etc.).`;
  }

  // "ollama" or "openai". Ollama also speaks the OpenAI-compatible /v1 API, but only its
  // native /api/chat can set a per-request context window (see MIN_CTX), so it gets its own
  // request path. Detected once by probing Ollama's /api/version, which OpenAI-compatible
  // servers 404 on; LOCAL_LLM_BACKEND=ollama|openai skips the probe.
  async detectBackend() {
    if (this.backend) return this.backend;

    const override = process.env.LOCAL_LLM_BACKEND;
    if (override === "ollama" || override === "openai") {
      this.backend = override;
      return this.backend;
    }

    try {
      const response = await axios.get(`${LOCAL_LLM_BASE_URL}/api/version`, { timeout: 5000 });
      this.backend = typeof response.data?.version === "string" ? "ollama" : "openai";
    } catch (error) {
      // No HTTP response at all (server down, timeout): don't cache a guess -- the request
      // that follows surfaces the real connection error, and the next call probes again.
      if (!error.response) return "openai";
      this.backend = "openai";
    }
    return this.backend;
  }

  // Ollama's /v1/models lists every *installed* model rather than what's loaded, so taking
  // the first entry would be arbitrary. Loaded models (/api/ps) go first, then the rest of
  // /api/tags (Ollama loads those on demand), each carrying its capabilities so resolveModel
  // can keep chat tools off embedding-only models and vice versa.
  async fetchOllamaModels() {
    const [ps, tags] = await Promise.all([
      axios.get(`${LOCAL_LLM_BASE_URL}/api/ps`, { timeout: 5000 }),
      axios.get(`${LOCAL_LLM_BASE_URL}/api/tags`, { timeout: 5000 }),
    ]);
    const loaded = ps.data?.models ?? [];
    const installed = tags.data?.models ?? [];
    const capabilitiesByName = new Map(installed.map((model) => [model.name, model.capabilities]));
    const names = [...new Set([...loaded, ...installed].map((model) => model.name))];
    return names.map((id) => ({ id, capabilities: capabilitiesByName.get(id) ?? null }));
  }

  async fetchAvailableModels() {
    const now = Date.now();
    if (this.modelsCache && now - this.modelsCacheAt < MODEL_CACHE_TTL_MS) {
      return this.modelsCache;
    }

    try {
      let models;
      if ((await this.detectBackend()) === "ollama") {
        models = await this.fetchOllamaModels();
      } else {
        const response = await axios.get(`${LOCAL_LLM_BASE_URL}/v1/models`, {
          timeout: 5000,
        });
        models = response.data?.data ?? [];
      }
      if (models.length === 0) {
        throw new Error("local LLM server reported no loaded model");
      }

      this.modelsCache = models;
      this.modelsCacheAt = now;
      return models;
    } catch (error) {
      this.modelsCache = null;
      if (error.code === "ECONNREFUSED") {
        throw new Error(this.connectionErrorMessage(), { cause: error });
      }
      throw new Error(`Failed to discover model from local LLM server: ${error.message}`, { cause: error });
    }
  }

  // Picks a model when the caller didn't specify one. LOCAL_LLM_CHAT_MODEL /
  // LOCAL_LLM_EMBEDDING_MODEL, when set, always win -- they're the user's own choice, not a
  // model hardcoded here, and they keep the pick stable when other models get loaded or
  // pulled. Otherwise, with a single model loaded (the common case), always resolves to it. Behind a multi-model router (e.g. llama-swap),
  // consults TOOL_MODEL_PREFERENCES for the given tool before falling back to whichever
  // model /v1/models lists first. Models that report capabilities (Ollama) are first narrowed
  // to the ones that can serve this tool -- embeddings for semantic_similarity, chat otherwise.
  async resolveModel(explicitModel, toolName) {
    if (explicitModel) {
      return { id: explicitModel, family: resolveFamily(explicitModel) };
    }

    const required = toolName === "semantic_similarity" ? "embedding" : "completion";
    const configured =
      required === "embedding" ? process.env.LOCAL_LLM_EMBEDDING_MODEL : process.env.LOCAL_LLM_CHAT_MODEL;
    if (configured) {
      if ((await this.detectBackend()) === "ollama") await this.ensureOllamaModel(configured);
      return { id: configured, family: resolveFamily(configured) };
    }

    const models = (await this.fetchAvailableModels()).filter(
      (model) => !model.capabilities || model.capabilities.includes(required)
    );
    if (models.length === 0) {
      const hint = required === "embedding" ? " Install one, e.g. `ollama pull nomic-embed-text`." : "";
      throw new Error(`No available model supports ${required}, which ${toolName} needs.${hint}`);
    }

    const preferredFamilies = TOOL_MODEL_PREFERENCES[toolName];
    if (preferredFamilies) {
      for (const family of preferredFamilies) {
        const match = models.find((model) => resolveFamily(model.id) === family);
        if (match) return { id: match.id, family };
      }
    }

    const [first] = models;
    return { id: first.id, family: resolveFamily(first.id) };
  }

  // Pulls a configured model (LOCAL_LLM_CHAT_MODEL / LOCAL_LLM_EMBEDDING_MODEL) that isn't
  // installed yet, so first use just works. Only ever for those user-configured names -- never
  // for a per-call `model` argument, which would let the calling agent start arbitrary
  // downloads. LOCAL_LLM_AUTO_PULL=0 turns it off. Once a name is known to be installed it isn't
  // checked again, and concurrent calls share a single pull.
  async ensureOllamaModel(name) {
    if (process.env.LOCAL_LLM_AUTO_PULL === "0" || this.confirmedModels.has(name)) return;

    // Ollama lists untagged names with an explicit ":latest".
    const tagged = name.includes(":") ? name : `${name}:latest`;
    try {
      const tags = await axios.get(`${LOCAL_LLM_BASE_URL}/api/tags`, { timeout: 5000 });
      const installed = (tags.data?.models ?? []).map((model) => model.name);
      if (!installed.includes(tagged)) {
        if (!this.pulls.has(tagged)) {
          this.pulls.set(tagged, this.pullOllamaModel(name).finally(() => this.pulls.delete(tagged)));
        }
        await this.pulls.get(tagged);
      }
    } catch (error) {
      if (error.code === "ECONNREFUSED") {
        throw new Error(this.connectionErrorMessage(), { cause: error });
      }
      throw error;
    }
    this.confirmedModels.add(name);
  }

  async pullOllamaModel(name) {
    console.error(`[local-llm-mcp] ${name} is configured but not installed; pulling it from Ollama...`);
    try {
      await axios.post(`${LOCAL_LLM_BASE_URL}/api/pull`, { model: name, stream: false }, { timeout: PULL_TIMEOUT_MS });
    } catch (error) {
      if (error.code === "ECONNREFUSED") throw error;
      const reason = error.response?.data?.error ?? error.message;
      throw new Error(`${name} is configured but not installed, and pulling it failed: ${reason}`, { cause: error });
    }
    this.modelsCache = null; // the installed list just changed
    console.error(`[local-llm-mcp] pulled ${name}.`);
  }

  async callLocalLlm(toolName, args) {
    const { model: explicitModel, ...promptArgs } = args;
    const resolved = await this.resolveModel(explicitModel, toolName);
    const messages = buildMessages(toolName, promptArgs, resolved.family);
    const enableThinking = TOOL_REASONING_OVERRIDES[toolName];

    let completion;
    try {
      completion =
        (await this.detectBackend()) === "ollama"
          ? await this.ollamaChat(resolved.id, messages, enableThinking)
          : await this.openAiChat(resolved.id, messages, enableThinking);
    } catch (error) {
      if (error.code === "ECONNREFUSED") {
        throw new Error(this.connectionErrorMessage(), { cause: error });
      }
      // Ollama and llama-server explain failures (e.g. "model 'x' not found") in the body's
      // `error` field, which axios's own message ("status code 404") leaves out.
      const detail = error.response?.data?.error;
      const reason = typeof detail === "string" ? detail : detail?.message;
      throw new Error(`local LLM error: ${error.message}${reason ? ` (${reason})` : ""}`, { cause: error });
    }

    this.recordTokenUsage(toolName, completion.usage);
    let text = completion.content;
    if (completion.promptTruncated) {
      const { contextWindow, atMaxContext } = completion;
      const window = contextWindow ? `${contextWindow}-token` : "default";
      const advice = atMaxContext
        ? "That's the largest window this model supports, so send less content per call."
        : `Send less content per call, or raise LOCAL_LLM_MIN_CTX${contextWindow ? ` above ${contextWindow}` : ""}.`;
      text += `\n\n[WARNING: the input didn't fit the model's ${window} context window and was cut down to ${completion.usage.prompt_tokens} tokens before the model saw it, so this answer is based on partial input. ${advice}]`;
    }
    if (completion.finishReason === "length") {
      text += `\n\n[WARNING: response was truncated at the ${MAX_TOKENS}-token generation limit (LOCAL_LLM_MAX_TOKENS) before finishing. Consider a narrower request, or raise the limit.]`;
    }
    return text;
  }

  async openAiChat(model, messages, enableThinking) {
    const body = {
      model,
      messages,
      stream: false,
      max_tokens: MAX_TOKENS,
    };
    if (enableThinking !== undefined) {
      body.chat_template_kwargs = { enable_thinking: enableThinking };
    }

    const response = await axios.post(`${LOCAL_LLM_BASE_URL}/v1/chat/completions`, body, {
      timeout: 900000, // 15 minute timeout (overly long, to account for slow local models)
    });
    const choice = response.data.choices[0];
    return { content: choice.message.content, finishReason: choice.finish_reason, usage: response.data.usage };
  }

  // Ollama's native chat API: num_ctx and num_predict go in `options`, and thinking is toggled
  // with its own top-level `think` flag rather than a chat-template kwarg. Like the OpenAI path,
  // only message.content is returned -- Ollama keeps reasoning separate in message.thinking.
  async ollamaChat(model, messages, enableThinking) {
    const options = { num_predict: MAX_TOKENS };
    if (MIN_CTX > 0) options.num_ctx = await this.ollamaContextWindow(model);
    const body = { model, messages, stream: false, options };
    if (enableThinking !== undefined) body.think = enableThinking;

    const response = await axios.post(`${LOCAL_LLM_BASE_URL}/api/chat`, body, {
      timeout: 900000, // 15 minute timeout (overly long, to account for slow local models)
    });
    const data = response.data;
    const promptTokens = data.prompt_eval_count ?? 0;
    const completionTokens = data.eval_count ?? 0;
    const promptChars = messages.reduce((sum, message) => sum + message.content.length, 0);
    // Ollama drops prompt tokens that don't fit num_ctx without reporting an error.
    const promptTruncated = promptTokens > 0 && promptChars / promptTokens > TRUNCATED_CHARS_PER_TOKEN;
    const completion = {
      content: data.message?.content ?? "",
      finishReason: data.done_reason,
      usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
      promptTruncated,
    };
    if (promptTruncated) {
      // For the warning: the window the model ran at (looked up when num_ctx was left to
      // Ollama), and whether it's already the most this model supports.
      completion.contextWindow = options.num_ctx ?? (await this.ollamaRunningContext(model));
      const maxContext = await this.ollamaMaxContext(model);
      completion.atMaxContext = Boolean(maxContext && completion.contextWindow >= maxContext);
    }
    return completion;
  }

  // The num_ctx to send. Ollama reloads a model whenever a request's num_ctx differs from the
  // window it's running at, and a request *without* num_ctx means "Ollama's default", which
  // reloads it just the same if it was loaded with another window. So the current window is
  // always sent back explicitly, keeping a larger configured window (OLLAMA_CONTEXT_LENGTH)
  // and avoiding reloads, unless it's below MIN_CTX. An unloaded model is first loaded with an
  // empty chat request, which loads it at Ollama's default window without generating anything.
  async ollamaContextWindow(model) {
    const maxContext = await this.ollamaMaxContext(model);
    // A model whose maximum is at or below MIN_CTX always runs at its maximum.
    if (maxContext && maxContext <= MIN_CTX) return maxContext;

    let running = await this.ollamaRunningContext(model);
    if (running === null) {
      await axios.post(`${LOCAL_LLM_BASE_URL}/api/chat`, { model, messages: [] }, { timeout: 900000 });
      running = await this.ollamaRunningContext(model);
    }
    return running !== null && running >= MIN_CTX ? running : MIN_CTX;
  }

  // The context window a loaded model is running at, from /api/ps. Best-effort: null if the
  // model isn't loaded or the lookup fails.
  async ollamaRunningContext(model) {
    try {
      const ps = await axios.get(`${LOCAL_LLM_BASE_URL}/api/ps`, { timeout: 5000 });
      return ps.data?.models?.find((entry) => entry.name === model)?.context_length ?? null;
    } catch {
      return null;
    }
  }

  // The largest context window a model supports, from /api/show's model_info, where the key is
  // prefixed with the model architecture (e.g. "llama.context_length"). Known without loading
  // the model, and cached since it never changes. Best-effort: null if the model doesn't report
  // it or the lookup fails.
  async ollamaMaxContext(model) {
    if (this.maxContextCache.has(model)) return this.maxContextCache.get(model);
    try {
      const show = await axios.post(`${LOCAL_LLM_BASE_URL}/api/show`, { model }, { timeout: 5000 });
      const modelInfo = show.data?.model_info ?? {};
      const key = Object.keys(modelInfo).find((name) => name.endsWith(".context_length"));
      const maxContext = key ? modelInfo[key] : null;
      if (maxContext) this.maxContextCache.set(model, maxContext);
      return maxContext;
    } catch {
      return null;
    }
  }

  recordTokenUsage(toolName, usage) {
    if (!usage) return;
    const promptTokens = usage.prompt_tokens ?? 0;
    const completionTokens = usage.completion_tokens ?? 0;
    const totalTokens = usage.total_tokens ?? promptTokens + completionTokens;

    this.tokenStats.calls += 1;
    this.tokenStats.promptTokens += promptTokens;
    this.tokenStats.completionTokens += completionTokens;
    this.tokenStats.totalTokens += totalTokens;

    const perTool = this.tokenStats.perTool[toolName] ?? {
      calls: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
    };
    perTool.calls += 1;
    perTool.promptTokens += promptTokens;
    perTool.completionTokens += completionTokens;
    perTool.totalTokens += totalTokens;
    this.tokenStats.perTool[toolName] = perTool;
  }

  sessionStats() {
    return { content: [{ type: "text", text: JSON.stringify(this.tokenStats, null, 2) }] };
  }

  async tokenize(args) {
    const { text, include_tokens } = args;
    if ((await this.detectBackend()) === "ollama") {
      throw new Error("Tokenizing isn't supported with Ollama, which has no tokenize endpoint. It needs llama-server's /tokenize.");
    }
    try {
      const response = await axios.post(
        `${LOCAL_LLM_BASE_URL}/tokenize`,
        { content: text },
        { timeout: 10000 }
      );
      const tokens = response.data?.tokens ?? [];
      const info = { token_count: tokens.length };
      if (include_tokens) info.tokens = tokens;
      return { content: [{ type: "text", text: JSON.stringify(info, null, 2) }] };
    } catch (error) {
      if (error.code === "ECONNREFUSED") {
        throw new Error(this.connectionErrorMessage(), { cause: error });
      }
      throw new Error(`Failed to tokenize text: ${error.message}`, { cause: error });
    }
  }

  async semanticSimilarity(args) {
    const { query, candidates, model: explicitModel } = args;
    const resolved = await this.resolveModel(explicitModel, "semantic_similarity");

    try {
      const response = await axios.post(
        `${LOCAL_LLM_BASE_URL}/v1/embeddings`,
        { model: resolved.id, input: [query, ...candidates] },
        { timeout: 60000 }
      );

      const vectors = response.data?.data?.map((entry) => entry.embedding) ?? [];
      const [queryVector, ...candidateVectors] = vectors;

      const results = candidateVectors
        .map((vector, index) => ({
          candidate: index,
          score: Math.round(cosineSimilarity(queryVector, vector) * 10000) / 10000,
        }))
        .sort((a, b) => b.score - a.score);

      return { content: [{ type: "text", text: JSON.stringify({ results }, null, 2) }] };
    } catch (error) {
      if (error.code === "ECONNREFUSED") {
        throw new Error(this.connectionErrorMessage(), { cause: error });
      }
      throw new Error(`Failed to compute semantic similarity: ${error.message}`, { cause: error });
    }
  }

  async generateCode(args) {
    const response = await this.callLocalLlm("generate_code", args);
    return { content: [{ type: "text", text: response }] };
  }

  async explainCode(args) {
    const response = await this.callLocalLlm("explain_code", args);
    return { content: [{ type: "text", text: response }] };
  }

  async reviewCode(args) {
    const response = await this.callLocalLlm("review_code", args);
    return { content: [{ type: "text", text: response }] };
  }

  async refactorCode(args) {
    const response = await this.callLocalLlm("refactor_code", args);
    return { content: [{ type: "text", text: response }] };
  }

  async fixCode(args) {
    const response = await this.callLocalLlm("fix_code", args);
    return { content: [{ type: "text", text: response }] };
  }

  async writeTests(args) {
    const response = await this.callLocalLlm("write_tests", args);
    return { content: [{ type: "text", text: response }] };
  }

  async generalTask(args) {
    const response = await this.callLocalLlm("general_task", args);
    return { content: [{ type: "text", text: response }] };
  }

  async readFile(filePath) {
    try {
      const content = await fs.readFile(filePath, "utf-8");
      return content;
    } catch (error) {
      throw new Error(`Failed to read file ${filePath}: ${error.message}`, { cause: error });
    }
  }

  async reviewFile(args) {
    const { file_path, focus, model } = args;
    const code = await this.readFile(file_path);
    const fileName = path.basename(file_path);
    const codeGraphContext = await getCodeGraphContext(fileName);

    const response = await this.callLocalLlm("review_file", {
      code: numberLines(code),
      focus,
      fileName,
      filePath: file_path,
      codeGraphContext,
      model,
    });
    return { content: [{ type: "text", text: response }] };
  }

  async explainFile(args) {
    const { file_path, context, model } = args;
    const code = await this.readFile(file_path);
    const fileName = path.basename(file_path);
    const codeGraphContext = await getCodeGraphContext(fileName);

    const response = await this.callLocalLlm("explain_file", {
      code: numberLines(code),
      context,
      fileName,
      filePath: file_path,
      codeGraphContext,
      model,
    });
    return { content: [{ type: "text", text: response }] };
  }

  async analyzeFiles(args) {
    const { file_paths, task, model } = args;
    const expandedPaths = await expandFilePaths(file_paths);

    const filesContent = await Promise.all(
      expandedPaths.map(async (filePath) => {
        const content = await this.readFile(filePath);
        const fileName = path.basename(filePath);
        return `FILE: ${fileName}\nPATH: ${filePath}\n\nCODE:\n${numberLines(content)}\n\n${"=".repeat(80)}\n`;
      })
    );
    const codeGraphContext = await getCodeGraphContext(expandedPaths.map((p) => path.basename(p)).join(" "));

    const response = await this.callLocalLlm("analyze_files", {
      task,
      filesContent: filesContent.join("\n"),
      codeGraphContext,
      model,
    });
    return { content: [{ type: "text", text: response }] };
  }

  async generateCodeWithContext(args) {
    const { prompt, language, context_files, model } = args;

    let contextSection = "";
    let codeGraphContext = null;
    if (context_files && context_files.length > 0) {
      const expandedContextFiles = await expandFilePaths(context_files);
      const contextContent = await Promise.all(
        expandedContextFiles.map(async (filePath) => {
          const content = await this.readFile(filePath);
          const fileName = path.basename(filePath);
          return `EXAMPLE FILE: ${fileName}\n${content}\n\n${"=".repeat(80)}\n`;
        })
      );
      contextSection = `\n\nREFERENCE FILES (for context and patterns):\n${contextContent.join("\n")}`;
      codeGraphContext = await getCodeGraphContext(expandedContextFiles.map((p) => path.basename(p)).join(" "));
    }

    const response = await this.callLocalLlm("generate_code_with_context", {
      prompt,
      language,
      contextSection,
      codeGraphContext,
      model,
    });
    return { content: [{ type: "text", text: response }] };
  }

  async serverInfo() {
    if ((await this.detectBackend()) === "ollama") {
      return this.ollamaServerInfo();
    }

    let modelsResponse;
    try {
      modelsResponse = await axios.get(`${LOCAL_LLM_BASE_URL}/v1/models`, { timeout: 5000 });
    } catch (error) {
      if (error.code === "ECONNREFUSED") {
        throw new Error(this.connectionErrorMessage(), { cause: error });
      }
      throw new Error(`Failed to fetch local LLM server info: ${error.message}`, { cause: error });
    }

    // /props is a llama.cpp-specific extension, not part of the OpenAI-compatible surface
    // every backend implements -- best-effort only, so this tool still works against
    // vLLM, LM Studio, etc. that don't have it.
    let props = null;
    try {
      props = (await axios.get(`${LOCAL_LLM_BASE_URL}/props`, { timeout: 5000 })).data;
    } catch {
      // no-op: props stays null, matching a backend that doesn't implement /props
    }

    const model = modelsResponse.data?.data?.[0];
    const info = {
      backend: "openai",
      base_url: LOCAL_LLM_BASE_URL,
      model_id: model?.id ?? null,
      model_family: model?.id ? resolveFamily(model.id) : null,
      context_size: props?.default_generation_settings?.n_ctx ?? props?.n_ctx ?? null,
      total_slots: props?.total_slots ?? null,
      has_chat_template: props ? Boolean(props.chat_template) : null,
    };

    return {
      content: [{ type: "text", text: JSON.stringify(info, null, 2) }],
    };
  }

  // Reports the model chat tools would use right now. max_context_size (/api/show) is always
  // known; context_size is the window it's actually running at (/api/ps), so it's null until
  // the model is loaded.
  async ollamaServerInfo() {
    let version;
    let ps;
    let model;
    try {
      [version, ps] = await Promise.all([
        axios.get(`${LOCAL_LLM_BASE_URL}/api/version`, { timeout: 5000 }),
        axios.get(`${LOCAL_LLM_BASE_URL}/api/ps`, { timeout: 5000 }),
      ]);
      model = await this.resolveModel(undefined, "server_info");
    } catch (error) {
      if (error.code === "ECONNREFUSED") {
        throw new Error(this.connectionErrorMessage(), { cause: error });
      }
      throw new Error(`Failed to fetch local LLM server info: ${error.message}`, { cause: error });
    }

    const running = ps.data?.models?.find((entry) => entry.name === model.id);
    const maxContext = await this.ollamaMaxContext(model.id);
    const info = {
      backend: "ollama",
      base_url: LOCAL_LLM_BASE_URL,
      ollama_version: version.data?.version ?? null,
      model_id: model.id,
      model_family: model.family,
      model_loaded: Boolean(running),
      context_size: running?.context_length ?? null,
      max_context_size: maxContext,
      min_context_size: MIN_CTX > 0 ? MIN_CTX : null,
    };

    return {
      content: [{ type: "text", text: JSON.stringify(info, null, 2) }],
    };
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    process.on("SIGINT", async () => {
      await this.server.close();
      process.exit(0);
    });
    console.error(`local LLM MCP server running on stdio (target: ${LOCAL_LLM_BASE_URL})`);
  }
}

export { LocalLlmServer };

if (process.argv[1] && import.meta.url === `file://${realpathSync(process.argv[1])}`) {
  const server = new LocalLlmServer();
  server.run();
}
