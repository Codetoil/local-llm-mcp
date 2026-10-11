import { test, describe, before, after, beforeEach } from "node:test";
import assert from "node:assert/strict";
import fs from "fs/promises";
import os from "os";
import path from "path";
import { execFile } from "child_process";
import { promisify } from "util";
import { startMockLlamaServer } from "./helpers/mockLlamaServer.js";

// review_diff against real temporary git repositories, with a mock chat backend. Its own
// file/process so LOCAL_LLM_BASE_URL can point at this file's mock.
const exec = promisify(execFile);
let mock;
let LocalLlmServer;
let tmpDir;

async function makeRepo(name) {
  const repo = path.join(tmpDir, name);
  await fs.mkdir(repo);
  const git = (...args) => exec("git", args, { cwd: repo });
  await git("init", "-q", "-b", "main");
  await git("config", "user.email", "test@example.com");
  await git("config", "user.name", "Test");
  await git("config", "commit.gpgsign", "false");
  await fs.writeFile(path.join(repo, "app.js"), "const a = 1;\nconst b = 2;\nconst c = 3;\n");
  await fs.writeFile(path.join(repo, "notes.txt"), "first\n");
  await git("add", ".");
  await git("commit", "-q", "-m", "initial");
  return { repo, git };
}

const chatRequests = () => mock.state.requests.filter((r) => r.url === "/v1/chat/completions");
const lastPrompt = () => chatRequests().at(-1).body.messages[1].content;

before(async () => {
  mock = await startMockLlamaServer();
  process.env.LOCAL_LLM_BASE_URL = mock.url;
  ({ LocalLlmServer } = await import("../index.js"));
  tmpDir = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), "local-llm-mcp-diff-test-")));
});

after(async () => {
  await mock?.close();
  if (tmpDir) await fs.rm(tmpDir, { recursive: true, force: true });
});

beforeEach(() => {
  mock.state.requests.length = 0;
});

describe("review_diff", () => {
  test("reviews uncommitted changes, staged and unstaged, with new-file line numbers", async () => {
    const { repo, git } = await makeRepo("uncommitted");
    await fs.writeFile(path.join(repo, "app.js"), "const a = 1;\nconst b = 20;\nconst c = 3;\n");
    await fs.writeFile(path.join(repo, "notes.txt"), "first\nsecond\n");
    await git("add", "notes.txt");
    await new LocalLlmServer().reviewDiff({ repo_path: repo });
    const prompt = lastPrompt();
    assert.match(prompt, /uncommitted changes \(staged and unstaged\)/);
    assert.match(prompt, /CHANGED FILES:\napp\.js\nnotes\.txt/);
    assert.match(prompt, /^ {5}2 \+const b = 20;$/m); // added line, numbered in the new file
    assert.match(prompt, /^ {7}-const b = 2;$/m); // removed line, no number
    assert.match(prompt, /^ {5}1 {2}const a = 1;$/m); // context line
    assert.match(prompt, /^ {5}2 \+second$/m);
  });

  test("staged: only what's in the index", async () => {
    const { repo, git } = await makeRepo("staged");
    await fs.writeFile(path.join(repo, "app.js"), "const a = 100;\nconst b = 2;\nconst c = 3;\n");
    await fs.writeFile(path.join(repo, "notes.txt"), "first\nstaged line\n");
    await git("add", "notes.txt");
    await new LocalLlmServer().reviewDiff({ repo_path: repo, staged: true });
    const prompt = lastPrompt();
    assert.match(prompt, /staged line/);
    assert.doesNotMatch(prompt, /const a = 100/);
  });

  test("base: what the branch adds since base, like a pull request", async () => {
    const { repo, git } = await makeRepo("branch");
    await git("checkout", "-q", "-b", "feature");
    await fs.writeFile(path.join(repo, "app.js"), "const a = 1;\nconst b = 2;\nconst c = 3;\nconst d = 4;\n");
    await git("commit", "-q", "-am", "add d");
    await fs.writeFile(path.join(repo, "notes.txt"), "uncommitted\n");
    await new LocalLlmServer().reviewDiff({ repo_path: repo, base: "main" });
    const prompt = lastPrompt();
    assert.match(prompt, /changes on the current branch since main/);
    assert.match(prompt, /^ {5}4 \+const d = 4;$/m);
    assert.doesNotMatch(prompt, /uncommitted/);
  });

  test("paths limit the review", async () => {
    const { repo } = await makeRepo("paths");
    await fs.writeFile(path.join(repo, "app.js"), "changed\n");
    await fs.writeFile(path.join(repo, "notes.txt"), "changed too\n");
    await new LocalLlmServer().reviewDiff({ repo_path: repo, paths: ["notes.txt"] });
    assert.match(lastPrompt(), /CHANGED FILES:\nnotes\.txt\n\n/);
  });

  test("no changes: says so without calling the model", async () => {
    const { repo } = await makeRepo("clean");
    const result = await new LocalLlmServer().reviewDiff({ repo_path: repo });
    assert.match(result.content[0].text, /No changes to review: uncommitted changes \(staged and unstaged\) are empty\./);
    assert.equal(chatRequests().length, 0);
  });

  test("a focus is passed through", async () => {
    const { repo } = await makeRepo("focus");
    await fs.writeFile(path.join(repo, "app.js"), "eval(input);\n");
    await new LocalLlmServer().reviewDiff({ repo_path: repo, focus: "security" });
    assert.match(lastPrompt(), /with focus on security/);
  });
});

describe("review_diff safety", () => {
  test("a base that could be read as an option is refused before git runs", async () => {
    const { repo } = await makeRepo("option-injection");
    const target = path.join(tmpDir, "written-by-git.txt");
    await fs.writeFile(path.join(repo, "app.js"), "changed\n");
    for (const base of [`--output=${target}`, "-p", "main; touch x", "main $(touch x)", "a b"]) {
      await assert.rejects(() => new LocalLlmServer().reviewDiff({ repo_path: repo, base }), /Not a valid git ref/);
    }
    await assert.rejects(() => fs.access(target), "git must not have written the --output file");
    assert.equal(chatRequests().length, 0);
  });

  test("a path starting with '-' stays a path, never an option", async () => {
    const { repo } = await makeRepo("dash-path");
    const target = path.join(tmpDir, "dash-output.txt");
    await fs.writeFile(path.join(repo, "app.js"), "changed\n");
    const result = await new LocalLlmServer().reviewDiff({ repo_path: repo, paths: [`--output=${target}`] });
    assert.match(result.content[0].text, /No changes to review/);
    await assert.rejects(() => fs.access(target));
  });

  test("the repository's own config can't make git run programs", async () => {
    const { repo, git } = await makeRepo("hostile-config");
    const marker = path.join(tmpDir, "program-ran");
    const script = path.join(tmpDir, "evil.sh");
    await fs.writeFile(script, `#!/bin/sh\ntouch "${marker}"\ncat "$1" 2>/dev/null\n`, { mode: 0o755 });
    await git("config", "diff.external", script);
    await git("config", "diff.evil.textconv", script);
    await git("config", "core.fsmonitor", script);
    await fs.writeFile(path.join(repo, ".gitattributes"), "*.txt diff=evil\n");
    await fs.writeFile(path.join(repo, "notes.txt"), "changed\n");
    await fs.writeFile(path.join(repo, "app.js"), "changed\n");
    await new LocalLlmServer().reviewDiff({ repo_path: repo });
    await assert.rejects(() => fs.access(marker), "a configured program ran");
    assert.match(lastPrompt(), /\+changed/);
  });

  test("staged and base together are refused", async () => {
    const { repo } = await makeRepo("both");
    await assert.rejects(() => new LocalLlmServer().reviewDiff({ repo_path: repo, staged: true, base: "main" }), /either base or staged/);
  });

  test("repo_path must be absolute", async () => {
    await assert.rejects(() => new LocalLlmServer().reviewDiff({ repo_path: "relative/repo" }), /absolute path/);
  });

  test("a directory that isn't a repository reports git's own error", async () => {
    const plain = path.join(tmpDir, "not-a-repo");
    await fs.mkdir(plain);
    await assert.rejects(
      () => new LocalLlmServer().reviewDiff({ repo_path: plain }),
      /git diff failed in .*not-a-repo: .*not a git repository/i
    );
  });
});
