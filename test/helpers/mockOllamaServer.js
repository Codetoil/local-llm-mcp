import http from "http";

// Minimal stand-in for Ollama's native API (plus the OpenAI-compatible /v1/embeddings it also
// serves), shaped after real Ollama 0.40 responses. `state.installed` mirrors /api/tags (name +
// capabilities) and /api/show (maxContext), `state.loaded` mirrors /api/ps (name +
// context_length). Like Ollama, a chat request (re)loads its model whenever its num_ctx --
// or, without one, `state.defaultContext` -- differs from the running window, capped at
// maxContext, counting each load in `state.loads`; an empty `messages` array only loads.
// Requests are recorded in `state.requests` so tests can assert on what was sent.
export function startMockOllamaServer() {
  const state = {
    version: "0.40.2",
    installed: [
      { name: "qwen3:8b", capabilities: ["completion", "tools", "thinking"], maxContext: 40960 },
      { name: "llama3.2:latest", capabilities: ["completion", "tools"], maxContext: 131072 },
      { name: "nomic-embed-text:latest", capabilities: ["embedding"], maxContext: 2048 },
    ],
    loaded: [],
    defaultContext: 4096, // Ollama's stock default; OLLAMA_CONTEXT_LENGTH changes it
    // Models /api/pull can fetch (by tagged name); pulling one appends it to `installed`.
    pullable: {},
    loads: 0,
    requests: [],
    chatResponse: (body) => `echo:${JSON.stringify(body.messages)}`,
    // Roughly 4 characters per token, like a real tokenizer on ordinary text.
    promptEvalCount: (body) => Math.ceil(body.messages.reduce((sum, m) => sum + m.content.length, 0) / 4),
    evalCount: 20,
    doneReason: "stop",
    embeddings: (inputs) => inputs.map((text) => [text.length, 1, 0]),
  };

  const server = http.createServer((req, res) => {
    let raw = "";
    req.on("data", (chunk) => (raw += chunk));
    req.on("end", () => {
      const body = raw ? JSON.parse(raw) : undefined;
      state.requests.push({ method: req.method, url: req.url, body });
      const reply = (payload) => {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(payload));
      };

      if (req.method === "GET" && req.url === "/api/version") {
        return reply({ version: state.version });
      }

      if (req.method === "POST" && req.url === "/api/pull") {
        const tagged = body.model.includes(":") ? body.model : `${body.model}:latest`;
        const model = state.pullable[tagged];
        if (!model) {
          res.writeHead(500, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ error: "pull model manifest: file does not exist" }));
        }
        state.installed = [...state.installed, { name: tagged, ...model }];
        return reply({ status: "success" });
      }

      if (req.method === "GET" && req.url === "/api/tags") {
        return reply({ models: state.installed.map(({ name, capabilities }) => ({ name, model: name, capabilities })) });
      }

      if (req.method === "GET" && req.url === "/api/ps") {
        return reply({ models: state.loaded.map(({ name, context_length }) => ({ name, model: name, context_length })) });
      }

      if (req.method === "POST" && req.url === "/api/show") {
        const model = state.installed.find((entry) => entry.name === body.model);
        if (!model) {
          res.writeHead(404, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ error: `model '${body.model}' not found` }));
        }
        const arch = model.name.split(/[:.\d]/)[0];
        return reply({ capabilities: model.capabilities, model_info: { [`${arch}.context_length`]: model.maxContext } });
      }

      if (req.method === "POST" && req.url === "/api/chat") {
        if (!state.installed.some((entry) => entry.name === body.model)) {
          res.writeHead(404, { "Content-Type": "application/json" });
          return res.end(JSON.stringify({ error: `model '${body.model}' not found` }));
        }
        const maxContext = state.installed.find((entry) => entry.name === body.model)?.maxContext ?? Infinity;
        const contextLength = Math.min(body.options?.num_ctx ?? state.defaultContext, maxContext);
        const running = state.loaded.find((entry) => entry.name === body.model);
        if (running?.context_length !== contextLength) {
          state.loads += 1;
          state.loaded = [
            { name: body.model, context_length: contextLength },
            ...state.loaded.filter((entry) => entry.name !== body.model),
          ];
        }
        if (!body.messages?.length) {
          return reply({ model: body.model, message: { role: "assistant", content: "" }, done: true, done_reason: "load" });
        }
        return reply({
          model: body.model,
          message: { role: "assistant", content: state.chatResponse(body) },
          done: true,
          done_reason: state.doneReason,
          prompt_eval_count: state.promptEvalCount(body),
          eval_count: state.evalCount,
        });
      }

      if (req.method === "POST" && req.url === "/v1/embeddings") {
        const vectors = state.embeddings(body.input);
        return reply({ data: vectors.map((embedding, index) => ({ embedding, index })) });
      }

      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("404 page not found");
    });
  });

  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      resolve({
        url: `http://127.0.0.1:${port}`,
        state,
        close: () => new Promise((r) => server.close(r)),
      });
    });
  });
}
