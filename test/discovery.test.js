import { test, describe, before, after, afterEach } from "node:test";
import assert from "node:assert/strict";
import fs from "fs/promises";
import http from "http";
import os from "os";
import path from "path";
import { startMockOllamaServer } from "./helpers/mockOllamaServer.js";
import { startMockLlamaServer } from "./helpers/mockLlamaServer.js";

// Server auto-detection, with LOCAL_LLM_BASE_URL unset. Each test points the instance's
// candidate list (`defaultServers`) at mocks instead of the real default ports, so it never
// touches whatever is actually running on localhost:11434 or :8080.
let LocalLlmServer;
let ollama;
let llama;
let unrelated;
let closedUrl;
let tmpDir;
const originalBaseUrl = process.env.LOCAL_LLM_BASE_URL;
const originalPath = process.env.PATH;

const ollamaAt = (url) => ({ name: "Ollama", url, backend: "ollama", binary: "ollama", start: "`ollama serve`" });
const llamaAt = (url) => ({ name: "llama-server", url, backend: "openai", binary: "llama-server", start: "`llama-server`" });

function serverWith(...candidates) {
  const server = new LocalLlmServer();
  server.defaultServers = candidates;
  return server;
}

before(async () => {
  delete process.env.LOCAL_LLM_BASE_URL;
  ({ LocalLlmServer } = await import("../index.js"));
  ollama = await startMockOllamaServer();
  llama = await startMockLlamaServer();
  // Something else listening on a default port: a dev server answering every path with HTML.
  unrelated = http.createServer((req, res) => {
    res.writeHead(200, { "Content-Type": "text/html" });
    res.end("<html>my app</html>");
  });
  await new Promise((resolve) => unrelated.listen(0, "127.0.0.1", resolve));
  const gone = await startMockLlamaServer();
  closedUrl = gone.url;
  await gone.close();
  tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "local-llm-mcp-discovery-test-"));
});

after(async () => {
  if (originalBaseUrl !== undefined) process.env.LOCAL_LLM_BASE_URL = originalBaseUrl;
  await ollama?.close();
  await llama?.close();
  await new Promise((resolve) => (unrelated ? unrelated.close(resolve) : resolve()));
  if (tmpDir) await fs.rm(tmpDir, { recursive: true, force: true });
});

afterEach(() => {
  delete process.env.LOCAL_LLM_BASE_URL;
  process.env.PATH = originalPath;
});

const unrelatedUrl = () => `http://127.0.0.1:${unrelated.address().port}`;

describe("server auto-detection", () => {
  test("prefers Ollama when both Ollama and llama-server are running", async () => {
    const server = serverWith(ollamaAt(ollama.url), llamaAt(llama.url));
    assert.deepEqual(await server.locateServer(), { url: ollama.url, backend: "ollama", source: "auto-detected Ollama" });
  });

  test("falls back to llama-server when Ollama isn't running", async () => {
    const server = serverWith(ollamaAt(closedUrl), llamaAt(llama.url));
    assert.deepEqual(await server.locateServer(), { url: llama.url, backend: "openai", source: "auto-detected llama-server" });
  });

  test("an unrelated server on a default port doesn't count", async () => {
    const server = serverWith(ollamaAt(unrelatedUrl()), llamaAt(unrelatedUrl()));
    await assert.rejects(() => server.locateServer(), /none answered at the default addresses/);
  });

  test("chat requests go to the detected server", async () => {
    const server = serverWith(ollamaAt(closedUrl), llamaAt(llama.url));
    llama.state.requests.length = 0;
    await server.generateCode({ prompt: "x", language: "python" });
    assert.ok(llama.state.requests.some((r) => r.url === "/v1/chat/completions"));
  });

  test("LOCAL_LLM_BASE_URL wins over auto-detection", async () => {
    process.env.LOCAL_LLM_BASE_URL = llama.url;
    const server = serverWith(ollamaAt(ollama.url));
    assert.deepEqual(await server.locateServer(), { url: llama.url, backend: "openai", source: "LOCAL_LLM_BASE_URL" });
  });

  test("serverInfo says how the server was found", async () => {
    const server = serverWith(ollamaAt(ollama.url));
    const info = JSON.parse((await server.serverInfo()).content[0].text);
    assert.equal(info.base_url, ollama.url);
    assert.equal(info.base_url_source, "auto-detected Ollama");
  });

  test("after the detected server goes away, the next call finds the other one", async () => {
    const shortLived = await startMockOllamaServer();
    const server = serverWith(ollamaAt(shortLived.url), llamaAt(llama.url));
    assert.equal((await server.locateServer()).url, shortLived.url);
    // Ollama stops; it's still first on the candidate list, but no longer answers there.
    await shortLived.close();
    await assert.rejects(() => server.generateCode({ prompt: "x", language: "python" }), /Cannot connect to local LLM server/);
    assert.equal((await server.locateServer()).url, llama.url);
  });
});

describe("when no server is found", () => {
  test("the tool error names what was tried, unwrapped", async () => {
    process.env.PATH = tmpDir; // nothing installed
    const server = serverWith(ollamaAt(closedUrl), llamaAt(closedUrl));
    await assert.rejects(
      () => server.generateCode({ prompt: "x", language: "python" }),
      (error) => {
        assert.match(error.message, /^Cannot connect to local LLM server: LOCAL_LLM_BASE_URL isn't set/);
        assert.match(error.message, new RegExp(`Ollama at ${closedUrl}, llama-server at ${closedUrl}`));
        assert.match(error.message, /Install Ollama or llama\.cpp, or set LOCAL_LLM_BASE_URL/);
        return true;
      }
    );
  });

  test("says when a server is installed but not running, and how to start it", async () => {
    const binDir = path.join(tmpDir, "bin");
    await fs.mkdir(binDir, { recursive: true });
    await fs.writeFile(path.join(binDir, "ollama"), "#!/bin/sh\n", { mode: 0o755 });
    process.env.PATH = binDir;
    const server = serverWith(ollamaAt(closedUrl), llamaAt(closedUrl));
    await assert.rejects(() => server.locateServer(), /Ollama is installed but not running; start it with `ollama serve`\./);
  });

  test("keeps looking on later calls instead of caching the failure", async () => {
    const server = serverWith(ollamaAt(closedUrl));
    await assert.rejects(() => server.locateServer(), /none answered/);
    server.defaultServers = [ollamaAt(ollama.url)];
    assert.equal((await server.locateServer()).url, ollama.url);
  });
});
