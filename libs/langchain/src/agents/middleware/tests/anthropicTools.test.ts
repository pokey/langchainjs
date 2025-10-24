/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * Tests for Anthropic text editor and memory tool middleware.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { HumanMessage, ToolMessage } from "@langchain/core/messages";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

import {
  createStateClaudeTextEditorMiddleware,
  createStateClaudeMemoryMiddleware,
  createFilesystemClaudeTextEditorMiddleware,
  createFilesystemClaudeMemoryMiddleware,
  TEXT_EDITOR_TOOL_NAME,
  MEMORY_TOOL_NAME,
  filesReducer,
  type AnthropicToolsState,
} from "../anthropicTools/index.js";
import { StateFileSystem } from "../anthropicTools/StateFileSystem.js";
import { createAgent } from "../../index.js";
import { FakeToolCallingModel } from "../../tests/utils.js";

describe("filesReducer", () => {
  it("should initialize from empty object with non-null values", () => {
    const result = filesReducer({}, {
      "/file1.txt": {
        content: "line1",
        created_at: "2024-01-01T00:00:00Z",
        modified_at: "2024-01-01T00:00:00Z",
      },
      "/file2.txt": null,
    });

    expect(result).toEqual({
      "/file1.txt": {
        content: "line1",
        created_at: "2024-01-01T00:00:00Z",
        modified_at: "2024-01-01T00:00:00Z",
      },
    });
  });

  it("should merge and delete files", () => {
    const left = {
      "/file1.txt": {
        content: "old content",
        created_at: "2024-01-01T00:00:00Z",
        modified_at: "2024-01-01T00:00:00Z",
      },
      "/file2.txt": {
        content: "keep me",
        created_at: "2024-01-01T00:00:00Z",
        modified_at: "2024-01-01T00:00:00Z",
      },
    };

    const result = filesReducer(left, {
      "/file1.txt": {
        content: "new content",
        created_at: "2024-01-01T00:00:00Z",
        modified_at: "2024-01-02T00:00:00Z",
      },
      "/file2.txt": null, // Delete this file
      "/file3.txt": {
        content: "added",
        created_at: "2024-01-02T00:00:00Z",
        modified_at: "2024-01-02T00:00:00Z",
      },
    });

    expect(result).toEqual({
      "/file1.txt": {
        content: "new content",
        created_at: "2024-01-01T00:00:00Z",
        modified_at: "2024-01-02T00:00:00Z",
      },
      "/file3.txt": {
        content: "added",
        created_at: "2024-01-02T00:00:00Z",
        modified_at: "2024-01-02T00:00:00Z",
      },
    });
  });
});

describe("validatePath", () => {
  // Helper to create a StateFileSystem instance for testing path validation
  const createFs = (allowedPrefixes?: string[]) =>
    new StateFileSystem({}, allowedPrefixes, () => {});

  describe("basic path normalization", () => {
    it("should return normalized absolute path", () => {
      const fs = createFs();
      expect(fs.validatePath("/foo/bar")).toBe("/foo/bar");
    });

    it("should add leading slash to relative paths", () => {
      const fs = createFs();
      expect(fs.validatePath("foo/bar")).toBe("/foo/bar");
    });

    it("should normalize double slashes", () => {
      const fs = createFs();
      expect(fs.validatePath("/foo//bar")).toBe("/foo/bar");
    });

    it("should normalize dot segments", () => {
      const fs = createFs();
      expect(fs.validatePath("/foo/./bar")).toBe("/foo/bar");
    });
  });

  describe("path traversal protection", () => {
    it("should block .. in absolute paths", () => {
      const fs = createFs();
      expect(() => fs.validatePath("/foo/../etc/passwd")).toThrow(
        "Path traversal not allowed"
      );
    });

    it("should block .. in relative paths", () => {
      const fs = createFs();
      expect(() => fs.validatePath("../etc/passwd")).toThrow(
        "Path traversal not allowed"
      );
    });

    it("should block tilde paths", () => {
      const fs = createFs();
      expect(() => fs.validatePath("~/.ssh/id_rsa")).toThrow(
        "Path traversal not allowed"
      );
    });
  });

  describe("allowed prefix validation", () => {
    it("should allow paths with correct prefix", () => {
      const fs = createFs(["/workspace"]);
      expect(fs.validatePath("/workspace/file.txt")).toBe(
        "/workspace/file.txt"
      );
    });

    it("should reject paths without allowed prefix", () => {
      const fs = createFs(["/workspace"]);
      expect(() => fs.validatePath("/etc/passwd")).toThrow(
        "Path must start with"
      );
    });

    it("should reject paths that only partially match prefix", () => {
      // This test catches the edge case where /workspacemalicious starts with /workspace
      const fs = createFs(["/workspace/"]);
      expect(() => fs.validatePath("/workspacemalicious/file.txt")).toThrow(
        "Path must start with"
      );
    });
  });

  describe("memories prefix validation", () => {
    it("should allow /memories paths with /memories prefix", () => {
      const fs = createFs(["/memories"]);
      expect(fs.validatePath("/memories/notes.txt")).toBe(
        "/memories/notes.txt"
      );
    });

    it("should reject non-/memories paths when /memories prefix required", () => {
      const fs = createFs(["/memories"]);
      expect(() => fs.validatePath("/other/notes.txt")).toThrow(
        "Path must start with"
      );
    });
  });
});

describe("StateClaudeTextEditorMiddleware", () => {
  describe("initialization", () => {
    it("should initialize with correct defaults", () => {
      const mw = createStateClaudeTextEditorMiddleware();
      expect(mw.name).toBe("StateClaudeTextEditorMiddleware");
      // Test that middleware has the text editor tool
      expect(mw.tools).toBeDefined();
      expect(mw.tools!).toHaveLength(1);
      expect(mw.tools![0].name).toBe(TEXT_EDITOR_TOOL_NAME);
      // Verify the tool has the provider definition
      expect((mw.tools![0] as any).providerToolDefinition).toEqual({
        type: "text_editor_20250728",
        name: TEXT_EDITOR_TOOL_NAME,
      });
    });

    it("should initialize with custom allowed path prefixes", () => {
      const mw = createStateClaudeTextEditorMiddleware({
        allowedPathPrefixes: ["/workspace"],
      });
      expect(mw.name).toBe("StateClaudeTextEditorMiddleware");
    });
  });

  describe("tool execution - view", () => {
    it("should view file with line numbers", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: TEXT_EDITOR_TOOL_NAME,
              args: {
                command: "view",
                path: "/test.txt",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [createStateClaudeTextEditorMiddleware()],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("View the file")],
        text_editor_files: {
          "/test.txt": {
            content: "line 1\nline 2\nline 3",
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      });

      const toolMessage = result.messages.find((m) =>
        ToolMessage.isInstance(m)
      ) as ToolMessage;
      expect(toolMessage).toBeDefined();
      expect(toolMessage.content).toBe("1|line 1\n2|line 2\n3|line 3");
    });

    it("should list directory when viewing non-existent file", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: TEXT_EDITOR_TOOL_NAME,
              args: {
                command: "view",
                path: "/dir",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [createStateClaudeTextEditorMiddleware()],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("View directory")],
        text_editor_files: {
          "/dir/file1.txt": {
            content: "content",
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
          "/dir/file2.txt": {
            content: "content",
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      });

      const toolMessage = result.messages.find((m) =>
        ToolMessage.isInstance(m)
      ) as ToolMessage;
      expect(toolMessage.content).toBe("/dir/file1.txt\n/dir/file2.txt");
    });

    it("should return error for non-existent file", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: TEXT_EDITOR_TOOL_NAME,
              args: {
                command: "view",
                path: "/nonexistent.txt",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [createStateClaudeTextEditorMiddleware()],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("View file")],
        text_editor_files: {},
      });

      const toolMessage = result.messages.find((m) =>
        ToolMessage.isInstance(m)
      ) as ToolMessage;
      expect(toolMessage.status).toBe("error");
      expect(toolMessage.content).toContain("File not found");
    });
  });

  describe("tool execution - create", () => {
    it("should create new file", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: TEXT_EDITOR_TOOL_NAME,
              args: {
                command: "create",
                path: "/new.txt",
                file_text: "line 1\nline 2",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [createStateClaudeTextEditorMiddleware()],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("Create file")],
        text_editor_files: {},
      });

      const state = result as unknown as AnthropicToolsState;
      expect(state.text_editor_files).toBeDefined();
      expect(state.text_editor_files!["/new.txt"]).toBeDefined();
      expect(state.text_editor_files!["/new.txt"].content).toBe(
        "line 1\nline 2"
      );
      expect(state.text_editor_files!["/new.txt"].created_at).toBeDefined();

      const toolMessage = result.messages.find((m) =>
        ToolMessage.isInstance(m)
      ) as ToolMessage;
      expect(toolMessage.content).toBe("File created: /new.txt");
    });
  });

  describe("tool execution - str_replace", () => {
    it("should replace string in file", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: TEXT_EDITOR_TOOL_NAME,
              args: {
                command: "str_replace",
                path: "/test.txt",
                old_str: "hello world",
                new_str: "goodbye world",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [createStateClaudeTextEditorMiddleware()],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("Replace text")],
        text_editor_files: {
          "/test.txt": {
            content: "hello world\nhello universe",
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      });

      const state = result as unknown as AnthropicToolsState;
      expect(state.text_editor_files!["/test.txt"].content).toBe(
        "goodbye world\nhello universe"
      );
    });

    it("should replace only first occurrence", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: TEXT_EDITOR_TOOL_NAME,
              args: {
                command: "str_replace",
                path: "/test.txt",
                old_str: "hello",
                new_str: "goodbye",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [createStateClaudeTextEditorMiddleware()],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("Replace text")],
        text_editor_files: {
          "/test.txt": {
            content: "hello\nhello again",
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      });

      const state = result as unknown as AnthropicToolsState;
      // Should replace first occurrence of "hello" with "goodbye"
      // Content is "hello\nhello again" becomes "goodbye\nhello again"
      expect(state.text_editor_files!["/test.txt"].content).toBe(
        "goodbye\nhello again"
      );
    });

    it("should return error if string not found", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: TEXT_EDITOR_TOOL_NAME,
              args: {
                command: "str_replace",
                path: "/test.txt",
                old_str: "nonexistent",
                new_str: "replacement",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [createStateClaudeTextEditorMiddleware()],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("Replace text")],
        text_editor_files: {
          "/test.txt": {
            content: "hello world",
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      });

      const toolMessage = result.messages.find((m) =>
        ToolMessage.isInstance(m)
      ) as ToolMessage;
      expect(toolMessage.status).toBe("error");
      expect(toolMessage.content).toContain("String not found");
    });
  });

  describe("tool execution - insert", () => {
    it("should insert text at specified line", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: TEXT_EDITOR_TOOL_NAME,
              args: {
                command: "insert",
                path: "/test.txt",
                insert_line: 1,
                new_str: "inserted line",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [createStateClaudeTextEditorMiddleware()],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("Insert text")],
        text_editor_files: {
          "/test.txt": {
            content: "line 1\nline 2\nline 3",
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      });

      const state = result as unknown as AnthropicToolsState;
      expect(state.text_editor_files!["/test.txt"].content).toBe(
        "line 1\ninserted line\nline 2\nline 3"
      );
    });
  });

  describe("path validation", () => {
    it("should reject path traversal with ..", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: TEXT_EDITOR_TOOL_NAME,
              args: {
                command: "view",
                path: "/../etc/passwd",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [createStateClaudeTextEditorMiddleware()],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("View file")],
        text_editor_files: {},
      });

      const toolMessage = result.messages.find((m) =>
        ToolMessage.isInstance(m)
      ) as ToolMessage;
      expect(toolMessage.status).toBe("error");
      expect(toolMessage.content).toContain("Path traversal not allowed");
    });

    it("should reject path traversal with ~", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: TEXT_EDITOR_TOOL_NAME,
              args: {
                command: "view",
                path: "~/file.txt",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [createStateClaudeTextEditorMiddleware()],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("View file")],
        text_editor_files: {},
      });

      const toolMessage = result.messages.find((m) =>
        ToolMessage.isInstance(m)
      ) as ToolMessage;
      expect(toolMessage.status).toBe("error");
      expect(toolMessage.content).toContain("Path traversal not allowed");
    });

    it("should enforce allowed prefixes", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: TEXT_EDITOR_TOOL_NAME,
              args: {
                command: "view",
                path: "/etc/passwd",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [
          createStateClaudeTextEditorMiddleware({
            allowedPathPrefixes: ["/workspace"],
          }),
        ],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("View file")],
        text_editor_files: {},
      });

      const toolMessage = result.messages.find((m) =>
        ToolMessage.isInstance(m)
      ) as ToolMessage;
      expect(toolMessage.status).toBe("error");
      expect(toolMessage.content).toContain("Path must start with");
    });
  });
});

describe("StateClaudeMemoryMiddleware", () => {
  describe("initialization", () => {
    it("should initialize with correct defaults", () => {
      const mw = createStateClaudeMemoryMiddleware();
      expect(mw.name).toBe("StateClaudeMemoryMiddleware");
      // Memory middleware has both tools and wrapModelCall (for system prompt)
      expect(mw.tools).toBeDefined();
      expect(mw.tools!).toHaveLength(1);
      expect(mw.tools![0].name).toBe(MEMORY_TOOL_NAME);
      expect(mw.wrapModelCall).toBeDefined();
    });

    it("should initialize with custom system prompt", () => {
      const customPrompt = "Custom memory instructions";
      const mw = createStateClaudeMemoryMiddleware({
        systemPrompt: customPrompt,
      });
      expect(mw.name).toBe("StateClaudeMemoryMiddleware");
    });
  });

  describe("system prompt injection", () => {
    it("should inject memory system prompt", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [[]], // No tool calls
      });

      const agent = createAgent({
        model,
        middleware: [createStateClaudeMemoryMiddleware()],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("Hello")],
      });

      // The model response should contain the memory system prompt
      // FakeToolCallingModel concatenates message content, so the system prompt
      // will be included in the AI response content
      const aiMessage = result.messages.find((m: any) =>
        m._getType() === "ai"
      );
      expect(aiMessage).toBeDefined();
      expect(aiMessage!.content).toContain("MEMORY PROTOCOL");
      expect(aiMessage!.content).toContain(
        "IMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE"
      );
    });

    it("should append to existing system prompt", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [[]], // No tool calls
      });

      const agent = createAgent({
        model,
        middleware: [createStateClaudeMemoryMiddleware()],
        systemPrompt: "Existing prompt.",
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("Hello")],
      });

      // Verify both prompts are present in the AI message content
      // FakeToolCallingModel concatenates message content, so both the existing
      // prompt and memory prompt will be in the response
      const aiMessage = result.messages.find((m: any) =>
        m._getType() === "ai"
      );
      expect(aiMessage).toBeDefined();
      expect(aiMessage!.content).toContain("Existing prompt.");
      expect(aiMessage!.content).toContain("MEMORY PROTOCOL");
      expect(aiMessage!.content).toContain(
        "IMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING ANYTHING ELSE"
      );
    });
  });

  describe("path prefix enforcement", () => {
    it("should enforce /memories prefix by default", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: MEMORY_TOOL_NAME,
              args: {
                command: "view",
                path: "/etc/passwd",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [createStateClaudeMemoryMiddleware()],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("View file")],
        memory_files: {},
      });

      const toolMessage = result.messages.find((m) =>
        ToolMessage.isInstance(m)
      ) as ToolMessage;
      expect(toolMessage.status).toBe("error");
      expect(toolMessage.content).toContain("Path must start with");
    });

    it("should allow paths with /memories prefix", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: MEMORY_TOOL_NAME,
              args: {
                command: "view",
                path: "/memories/progress.txt",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [createStateClaudeMemoryMiddleware()],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("View memory")],
        memory_files: {
          "/memories/progress.txt": {
            content: "task 1 done",
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      });

      const toolMessage = result.messages.find((m) =>
        ToolMessage.isInstance(m)
      ) as ToolMessage;
      expect(toolMessage.content).toBe("1|task 1 done");
    });
  });

  describe("tool execution - delete", () => {
    it("should delete file", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: MEMORY_TOOL_NAME,
              args: {
                command: "delete",
                path: "/memories/test.txt",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [createStateClaudeMemoryMiddleware()],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("Delete file")],
        memory_files: {
          "/memories/test.txt": {
            content: "content",
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      });

      const state = result as unknown as AnthropicToolsState;
      // After filesReducer, null values are removed
      expect(state.memory_files).not.toHaveProperty("/memories/test.txt");
    });
  });

  describe("tool execution - rename", () => {
    it("should rename file", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: MEMORY_TOOL_NAME,
              args: {
                command: "rename",
                old_path: "/memories/old.txt",
                new_path: "/memories/new.txt",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [createStateClaudeMemoryMiddleware()],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("Rename file")],
        memory_files: {
          "/memories/old.txt": {
            content: "content",
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      });

      const state = result as unknown as AnthropicToolsState;
      // After filesReducer, null values are removed
      expect(state.memory_files).not.toHaveProperty("/memories/old.txt");
      expect(state.memory_files!["/memories/new.txt"]).toBeDefined();
      expect(state.memory_files!["/memories/new.txt"].content).toBe("content");
    });
  });
});

describe("FilesystemClaudeTextEditorMiddleware", () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "test-editor-"));
  });

  afterEach(() => {
    if (fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  describe("tool execution - view", () => {
    it("should view file with line numbers", async () => {
      const filePath = path.join(tempDir, "test.txt");
      fs.writeFileSync(filePath, "line 1\nline 2\nline 3");

      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: TEXT_EDITOR_TOOL_NAME,
              args: {
                command: "view",
                path: "/test.txt",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [
          createFilesystemClaudeTextEditorMiddleware({
            rootPath: tempDir,
          }),
        ],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("View file")],
      });

      const toolMessage = result.messages.find((m) =>
        ToolMessage.isInstance(m)
      ) as ToolMessage;
      expect(toolMessage.content).toBe("1|line 1\n2|line 2\n3|line 3");
    });

    it("should return error for non-existent file", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: TEXT_EDITOR_TOOL_NAME,
              args: {
                command: "view",
                path: "/nonexistent.txt",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [
          createFilesystemClaudeTextEditorMiddleware({
            rootPath: tempDir,
          }),
        ],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("View file")],
      });

      const toolMessage = result.messages.find((m) =>
        ToolMessage.isInstance(m)
      ) as ToolMessage;
      expect(toolMessage.status).toBe("error");
      expect(toolMessage.content).toContain("File not found");
    });
  });

  describe("tool execution - create", () => {
    it("should create new file", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: TEXT_EDITOR_TOOL_NAME,
              args: {
                command: "create",
                path: "/new.txt",
                file_text: "line 1\nline 2",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [
          createFilesystemClaudeTextEditorMiddleware({
            rootPath: tempDir,
          }),
        ],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("Create file")],
      });

      const toolMessage = result.messages.find((m) =>
        ToolMessage.isInstance(m)
      ) as ToolMessage;
      expect(toolMessage.content).toBe("File created: /new.txt");

      const filePath = path.join(tempDir, "new.txt");
      expect(fs.existsSync(filePath)).toBe(true);
      expect(fs.readFileSync(filePath, "utf8")).toBe("line 1\nline 2\n");
    });

    it("should create parent directories", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: TEXT_EDITOR_TOOL_NAME,
              args: {
                command: "create",
                path: "/subdir/new.txt",
                file_text: "content",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [
          createFilesystemClaudeTextEditorMiddleware({
            rootPath: tempDir,
          }),
        ],
      });

      await agent.invoke({
        messages: [new HumanMessage("Create file")],
      });

      const filePath = path.join(tempDir, "subdir", "new.txt");
      expect(fs.existsSync(filePath)).toBe(true);
    });
  });

  describe("tool execution - str_replace", () => {
    it("should replace string in file", async () => {
      const filePath = path.join(tempDir, "test.txt");
      fs.writeFileSync(filePath, "hello world\nhello universe");

      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: TEXT_EDITOR_TOOL_NAME,
              args: {
                command: "str_replace",
                path: "/test.txt",
                old_str: "hello world",
                new_str: "goodbye world",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [
          createFilesystemClaudeTextEditorMiddleware({
            rootPath: tempDir,
          }),
        ],
      });

      await agent.invoke({
        messages: [new HumanMessage("Replace text")],
      });

      const content = fs.readFileSync(filePath, "utf8");
      expect(content).toBe("goodbye world\nhello universe\n");
    });
  });

  describe("tool execution - insert", () => {
    it("should insert text at specified line", async () => {
      const filePath = path.join(tempDir, "test.txt");
      fs.writeFileSync(filePath, "line 1\nline 2\nline 3\n");

      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: TEXT_EDITOR_TOOL_NAME,
              args: {
                command: "insert",
                path: "/test.txt",
                insert_line: 1,
                new_str: "inserted line",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [
          createFilesystemClaudeTextEditorMiddleware({
            rootPath: tempDir,
          }),
        ],
      });

      await agent.invoke({
        messages: [new HumanMessage("Insert text")],
      });

      const content = fs.readFileSync(filePath, "utf8");
      expect(content).toBe("line 1\ninserted line\nline 2\nline 3\n");
    });
  });

  describe("path security", () => {
    it("should prevent escaping root directory", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: TEXT_EDITOR_TOOL_NAME,
              args: {
                command: "view",
                path: "/../etc/passwd",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [
          createFilesystemClaudeTextEditorMiddleware({
            rootPath: tempDir,
          }),
        ],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("View file")],
      });

      const toolMessage = result.messages.find((m) =>
        ToolMessage.isInstance(m)
      ) as ToolMessage;
      expect(toolMessage.status).toBe("error");
      expect(toolMessage.content).toContain("Path traversal not allowed");
    });
  });
});

describe("FilesystemClaudeMemoryMiddleware", () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "test-memory-"));
  });

  afterEach(() => {
    if (fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  describe("path prefix enforcement", () => {
    it("should enforce /memories prefix by default", async () => {
      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: MEMORY_TOOL_NAME,
              args: {
                command: "view",
                path: "/etc/passwd",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [
          createFilesystemClaudeMemoryMiddleware({
            rootPath: tempDir,
          }),
        ],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("View file")],
      });

      const toolMessage = result.messages.find((m) =>
        ToolMessage.isInstance(m)
      ) as ToolMessage;
      expect(toolMessage.status).toBe("error");
      expect(toolMessage.content).toContain("Path must start with");
    });

    it("should allow paths with /memories prefix", async () => {
      const memoriesDir = path.join(tempDir, "memories");
      fs.mkdirSync(memoriesDir, { recursive: true });
      const filePath = path.join(memoriesDir, "progress.txt");
      fs.writeFileSync(filePath, "task 1 done");

      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: MEMORY_TOOL_NAME,
              args: {
                command: "view",
                path: "/memories/progress.txt",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [
          createFilesystemClaudeMemoryMiddleware({
            rootPath: tempDir,
          }),
        ],
      });

      const result = await agent.invoke({
        messages: [new HumanMessage("View file")],
      });

      const toolMessage = result.messages.find((m) =>
        ToolMessage.isInstance(m)
      ) as ToolMessage;
      expect(toolMessage.content).toBe("1|task 1 done");
    });
  });

  describe("tool execution - delete", () => {
    it("should delete file", async () => {
      const memoriesDir = path.join(tempDir, "memories");
      fs.mkdirSync(memoriesDir, { recursive: true });
      const filePath = path.join(memoriesDir, "test.txt");
      fs.writeFileSync(filePath, "content");

      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: MEMORY_TOOL_NAME,
              args: {
                command: "delete",
                path: "/memories/test.txt",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [
          createFilesystemClaudeMemoryMiddleware({
            rootPath: tempDir,
          }),
        ],
      });

      await agent.invoke({
        messages: [new HumanMessage("Delete file")],
      });

      expect(fs.existsSync(filePath)).toBe(false);
    });

    it("should delete directory recursively", async () => {
      const memoriesDir = path.join(tempDir, "memories");
      fs.mkdirSync(memoriesDir, { recursive: true });
      const subdirPath = path.join(memoriesDir, "subdir");
      fs.mkdirSync(subdirPath);
      fs.writeFileSync(path.join(subdirPath, "file.txt"), "content");

      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: MEMORY_TOOL_NAME,
              args: {
                command: "delete",
                path: "/memories/subdir",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [
          createFilesystemClaudeMemoryMiddleware({
            rootPath: tempDir,
          }),
        ],
      });

      await agent.invoke({
        messages: [new HumanMessage("Delete directory")],
      });

      expect(fs.existsSync(subdirPath)).toBe(false);
    });
  });

  describe("tool execution - rename", () => {
    it("should rename file", async () => {
      const memoriesDir = path.join(tempDir, "memories");
      fs.mkdirSync(memoriesDir, { recursive: true });
      const oldPath = path.join(memoriesDir, "old.txt");
      fs.writeFileSync(oldPath, "content");

      const model = new FakeToolCallingModel({
        toolCalls: [
          [
            {
              id: "call_1",
              name: MEMORY_TOOL_NAME,
              args: {
                command: "rename",
                old_path: "/memories/old.txt",
                new_path: "/memories/new.txt",
              },
            },
          ],
        ],
      });

      const agent = createAgent({
        model,
        middleware: [
          createFilesystemClaudeMemoryMiddleware({
            rootPath: tempDir,
          }),
        ],
      });

      await agent.invoke({
        messages: [new HumanMessage("Rename file")],
      });

      expect(fs.existsSync(oldPath)).toBe(false);
      const newPath = path.join(memoriesDir, "new.txt");
      expect(fs.existsSync(newPath)).toBe(true);
      expect(fs.readFileSync(newPath, "utf8")).toBe("content");
    });
  });
});
