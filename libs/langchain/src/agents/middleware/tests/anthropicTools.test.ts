/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * Tests for Anthropic text editor and memory tool middleware.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { Command } from "@langchain/langgraph";
import { AIMessage, ToolMessage } from "@langchain/core/messages";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

import {
  StateClaudeTextEditorMiddleware,
  StateClaudeMemoryMiddleware,
  FilesystemClaudeTextEditorMiddleware,
  FilesystemClaudeMemoryMiddleware,
  TEXT_EDITOR_TOOL_NAME,
  MEMORY_TOOL_NAME,
  MEMORY_SYSTEM_PROMPT,
  filesReducer,
  validatePath,
  type AnthropicToolsState,
} from "../anthropicTools.js";

describe("filesReducer", () => {
  it("should initialize from undefined with non-null values", () => {
    const result = filesReducer(undefined, {
      "/file1.txt": {
        content: ["line1"],
        created_at: "2024-01-01T00:00:00Z",
        modified_at: "2024-01-01T00:00:00Z",
      },
      "/file2.txt": null,
    });

    expect(result).toEqual({
      "/file1.txt": {
        content: ["line1"],
        created_at: "2024-01-01T00:00:00Z",
        modified_at: "2024-01-01T00:00:00Z",
      },
    });
  });

  it("should merge and delete files", () => {
    const left = {
      "/file1.txt": {
        content: ["old content"],
        created_at: "2024-01-01T00:00:00Z",
        modified_at: "2024-01-01T00:00:00Z",
      },
      "/file2.txt": {
        content: ["keep me"],
        created_at: "2024-01-01T00:00:00Z",
        modified_at: "2024-01-01T00:00:00Z",
      },
    };

    const result = filesReducer(left, {
      "/file1.txt": {
        content: ["new content"],
        created_at: "2024-01-01T00:00:00Z",
        modified_at: "2024-01-02T00:00:00Z",
      },
      "/file2.txt": null, // Delete this file
      "/file3.txt": {
        content: ["added"],
        created_at: "2024-01-02T00:00:00Z",
        modified_at: "2024-01-02T00:00:00Z",
      },
    });

    expect(result).toEqual({
      "/file1.txt": {
        content: ["new content"],
        created_at: "2024-01-01T00:00:00Z",
        modified_at: "2024-01-02T00:00:00Z",
      },
      "/file3.txt": {
        content: ["added"],
        created_at: "2024-01-02T00:00:00Z",
        modified_at: "2024-01-02T00:00:00Z",
      },
    });
  });
});

describe("validatePath", () => {
  describe("basic path normalization", () => {
    it("should return normalized absolute path", () => {
      expect(validatePath("/foo/bar")).toBe("/foo/bar");
    });

    it("should add leading slash to relative paths", () => {
      expect(validatePath("foo/bar")).toBe("/foo/bar");
    });

    it("should normalize double slashes", () => {
      expect(validatePath("/foo//bar")).toBe("/foo/bar");
    });

    it("should normalize dot segments", () => {
      expect(validatePath("/foo/./bar")).toBe("/foo/bar");
    });
  });

  describe("path traversal protection", () => {
    it("should block .. in absolute paths", () => {
      expect(() => validatePath("/foo/../etc/passwd")).toThrow(
        "Path traversal not allowed"
      );
    });

    it("should block .. in relative paths", () => {
      expect(() => validatePath("../etc/passwd")).toThrow(
        "Path traversal not allowed"
      );
    });

    it("should block tilde paths", () => {
      expect(() => validatePath("~/.ssh/id_rsa")).toThrow(
        "Path traversal not allowed"
      );
    });
  });

  describe("allowed prefix validation", () => {
    it("should allow paths with correct prefix", () => {
      expect(validatePath("/workspace/file.txt", ["/workspace"])).toBe(
        "/workspace/file.txt"
      );
    });

    it("should reject paths without allowed prefix", () => {
      expect(() => validatePath("/etc/passwd", ["/workspace"])).toThrow(
        "Path must start with"
      );
    });

    it("should reject paths that only partially match prefix", () => {
      // This test catches the edge case where /workspacemalicious starts with /workspace
      expect(() =>
        validatePath("/workspacemalicious/file.txt", ["/workspace/"])
      ).toThrow("Path must start with");
    });
  });

  describe("memories prefix validation", () => {
    it("should allow /memories paths with /memories prefix", () => {
      expect(validatePath("/memories/notes.txt", ["/memories"])).toBe(
        "/memories/notes.txt"
      );
    });

    it("should reject non-/memories paths when /memories prefix required", () => {
      expect(() => validatePath("/other/notes.txt", ["/memories"])).toThrow(
        "Path must start with"
      );
    });
  });
});

describe("StateClaudeTextEditorMiddleware", () => {
  let middleware: StateClaudeTextEditorMiddleware;

  beforeEach(() => {
    middleware = new StateClaudeTextEditorMiddleware();
  });

  describe("initialization", () => {
    it("should initialize with correct defaults", () => {
      const mw = new StateClaudeTextEditorMiddleware();
      expect(mw.name).toBe("StateClaudeTextEditorMiddleware");
      // Test that middleware has wrapModelCall and wrapToolCall
      expect(mw.wrapModelCall).toBeDefined();
      expect(mw.wrapToolCall).toBeDefined();
    });

    it("should initialize with custom allowed path prefixes", () => {
      const mw = new StateClaudeTextEditorMiddleware({
        allowedPathPrefixes: ["/workspace"],
      });
      expect(mw.name).toBe("StateClaudeTextEditorMiddleware");
    });
  });

  describe("wrapModelCall", () => {
    it("should inject text editor tool into model request", async () => {
      const request = {
        model: "test-model",
        messages: [],
        tools: [],
      } as any;

      const handler = async (req: any) => {
        expect(req.tools).toHaveLength(1);
        expect(req.tools[0]).toEqual({
          type: "text_editor_20250728",
          name: TEXT_EDITOR_TOOL_NAME,
        });
        return new AIMessage({ content: "test" });
      };

      await middleware.wrapModelCall!(request, handler);
    });
  });

  describe("wrapToolCall - view", () => {
    it("should view file with line numbers", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/test.txt": {
            content: ["line 1", "line 2", "line 3"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "view",
            path: "/test.txt",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const handler = async () => {
        throw new Error("Should not call handler");
      };

      const result = await middleware.wrapToolCall!(request, handler);
      expect(result).toBeInstanceOf(Command);

      const cmd = result as Command;
      expect(cmd.update).toBeDefined();
      expect((cmd.update as any).messages).toHaveLength(1);
      expect((cmd.update as any).messages[0].content).toBe(
        "1|line 1\n2|line 2\n3|line 3"
      );
    });

    it("should list directory when viewing non-existent file", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/dir/file1.txt": {
            content: ["content"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
          "/dir/file2.txt": {
            content: ["content"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "view",
            path: "/dir",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      const cmd = result as Command;
      expect((cmd.update as any).messages[0].content).toBe(
        "/dir/file1.txt\n/dir/file2.txt"
      );
    });

    it("should return error for non-existent file", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {},
      };

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "view",
            path: "/nonexistent.txt",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect(result).toHaveProperty("status", "error");
      expect((result as ToolMessage).content).toContain("File not found");
    });
  });

  describe("wrapToolCall - create", () => {
    it("should create new file", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {},
      };

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "create",
            path: "/new.txt",
            file_text: "line 1\nline 2",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      const cmd = result as Command;
      expect(cmd.update).toBeDefined();
      expect((cmd.update as any).text_editor_files).toBeDefined();
      expect((cmd.update as any).text_editor_files["/new.txt"]).toBeDefined();
      expect((cmd.update as any).text_editor_files["/new.txt"].content).toEqual(
        ["line 1", "line 2"]
      );
      expect(
        (cmd.update as any).text_editor_files["/new.txt"].created_at
      ).toBeDefined();
      expect((cmd.update as any).messages[0].content).toBe(
        "File created: /new.txt"
      );
    });
  });

  describe("wrapToolCall - str_replace", () => {
    it("should replace string in file", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/test.txt": {
            content: ["hello world", "hello universe"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "str_replace",
            path: "/test.txt",
            old_str: "hello world",
            new_str: "goodbye world",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      const cmd = result as Command;
      expect(
        (cmd.update as any).text_editor_files["/test.txt"].content
      ).toEqual(["goodbye world", "hello universe"]);
    });

    it("should replace only first occurrence", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/test.txt": {
            content: ["hello", "hello again"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "str_replace",
            path: "/test.txt",
            old_str: "hello",
            new_str: "goodbye",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      const cmd = result as Command;
      // Should replace first occurrence of "hello" with "goodbye"
      // Content is joined with \n, so "hello\nhello again" becomes "goodbye\nhello again"
      expect(
        (cmd.update as any).text_editor_files["/test.txt"].content
      ).toEqual(["goodbye", "hello again"]);
    });

    it("should return error if string not found", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/test.txt": {
            content: ["hello world"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "str_replace",
            path: "/test.txt",
            old_str: "nonexistent",
            new_str: "replacement",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect(result).toHaveProperty("status", "error");
      expect((result as ToolMessage).content).toContain("String not found");
    });
  });

  describe("wrapToolCall - insert", () => {
    it("should insert text at specified line", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/test.txt": {
            content: ["line 1", "line 2", "line 3"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "insert",
            path: "/test.txt",
            insert_line: 1,
            new_str: "inserted line",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      const cmd = result as Command;
      expect(
        (cmd.update as any).text_editor_files["/test.txt"].content
      ).toEqual(["line 1", "inserted line", "line 2", "line 3"]);
    });
  });

  describe("wrapToolCall - delete", () => {
    it("should delete file", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/test.txt": {
            content: ["content"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "delete",
            path: "/test.txt",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      const cmd = result as Command;
      expect((cmd.update as any).text_editor_files["/test.txt"]).toBeNull();
    });
  });

  describe("wrapToolCall - rename", () => {
    it("should rename file", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/old.txt": {
            content: ["content"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "rename",
            old_path: "/old.txt",
            new_path: "/new.txt",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      const cmd = result as Command;
      expect((cmd.update as any).text_editor_files["/old.txt"]).toBeNull();
      expect((cmd.update as any).text_editor_files["/new.txt"]).toBeDefined();
      expect((cmd.update as any).text_editor_files["/new.txt"].content).toEqual(
        ["content"]
      );
    });
  });

  describe("path validation", () => {
    it("should reject path traversal with ..", async () => {
      const state: AnthropicToolsState = { text_editor_files: {} };

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "view",
            path: "/../etc/passwd",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect(result).toHaveProperty("status", "error");
      expect((result as ToolMessage).content).toContain(
        "Path traversal not allowed"
      );
    });

    it("should reject path traversal with ~", async () => {
      const state: AnthropicToolsState = { text_editor_files: {} };

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "view",
            path: "~/file.txt",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect(result).toHaveProperty("status", "error");
      expect((result as ToolMessage).content).toContain(
        "Path traversal not allowed"
      );
    });

    it("should enforce allowed prefixes", async () => {
      const middleware = new StateClaudeTextEditorMiddleware({
        allowedPathPrefixes: ["/workspace"],
      });

      const state: AnthropicToolsState = { text_editor_files: {} };

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "view",
            path: "/etc/passwd",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect(result).toHaveProperty("status", "error");
      expect((result as ToolMessage).content).toContain("Path must start with");
    });
  });
});

describe("StateClaudeMemoryMiddleware", () => {
  let middleware: StateClaudeMemoryMiddleware;

  beforeEach(() => {
    middleware = new StateClaudeMemoryMiddleware();
  });

  describe("initialization", () => {
    it("should initialize with correct defaults", () => {
      const mw = new StateClaudeMemoryMiddleware();
      expect(mw.name).toBe("StateClaudeMemoryMiddleware");
      expect(mw.wrapModelCall).toBeDefined();
      expect(mw.wrapToolCall).toBeDefined();
    });

    it("should initialize with custom system prompt", () => {
      const customPrompt = "Custom memory instructions";
      const mw = new StateClaudeMemoryMiddleware({
        systemPrompt: customPrompt,
      });
      expect(mw.name).toBe("StateClaudeMemoryMiddleware");
    });
  });

  describe("wrapModelCall", () => {
    it("should inject memory tool and system prompt", async () => {
      const request = {
        model: "test-model",
        messages: [],
        tools: [],
      } as any;

      const handler = async (req: any) => {
        expect(req.tools).toHaveLength(1);
        expect(req.tools[0]).toEqual({
          type: "memory_20250818",
          name: MEMORY_TOOL_NAME,
        });
        expect(req.systemPrompt).toContain("MEMORY PROTOCOL");
        expect(req.systemPrompt).toBe(MEMORY_SYSTEM_PROMPT);
        return new AIMessage({ content: "test" });
      };

      await middleware.wrapModelCall!(request, handler);
    });

    it("should append to existing system prompt", async () => {
      const request = {
        model: "test-model",
        messages: [],
        tools: [],
        systemPrompt: "Existing prompt.",
      } as any;

      const handler = async (req: any) => {
        expect(req.systemPrompt).toBe(
          `Existing prompt.\n\n${MEMORY_SYSTEM_PROMPT}`
        );
        return new AIMessage({ content: "test" });
      };

      await middleware.wrapModelCall!(request, handler);
    });
  });

  describe("path prefix enforcement", () => {
    it("should enforce /memories prefix by default", async () => {
      const state: AnthropicToolsState = { memory_files: {} };

      const request = {
        toolCall: {
          name: MEMORY_TOOL_NAME,
          args: {
            command: "view",
            path: "/etc/passwd",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect(result).toHaveProperty("status", "error");
      expect((result as ToolMessage).content).toContain("Path must start with");
    });

    it("should allow paths with /memories prefix", async () => {
      const state: AnthropicToolsState = {
        memory_files: {
          "/memories/progress.txt": {
            content: ["task 1 done"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: MEMORY_TOOL_NAME,
          args: {
            command: "view",
            path: "/memories/progress.txt",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect(result).toBeInstanceOf(Command);
      const cmd = result as Command;
      expect((cmd.update as any).messages[0].content).toBe("1|task 1 done");
    });
  });
});

describe("FilesystemClaudeTextEditorMiddleware", () => {
  let middleware: FilesystemClaudeTextEditorMiddleware;
  let tempDir: string;

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "test-editor-"));
    middleware = new FilesystemClaudeTextEditorMiddleware({
      rootPath: tempDir,
    });
  });

  afterEach(() => {
    if (fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  describe("wrapToolCall - view", () => {
    it("should view file with line numbers", async () => {
      const filePath = path.join(tempDir, "test.txt");
      fs.writeFileSync(filePath, "line 1\nline 2\nline 3");

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "view",
            path: "/test.txt",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      const cmd = result as Command;
      expect((cmd.update as any).messages[0].content).toBe(
        "1|line 1\n2|line 2\n3|line 3"
      );
    });

    it("should return error for non-existent file", async () => {
      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "view",
            path: "/nonexistent.txt",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect(result).toHaveProperty("status", "error");
      expect((result as ToolMessage).content).toContain("File not found");
    });
  });

  describe("wrapToolCall - create", () => {
    it("should create new file", async () => {
      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "create",
            path: "/new.txt",
            file_text: "line 1\nline 2",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      const cmd = result as Command;
      expect((cmd.update as any).messages[0].content).toBe(
        "File created: /new.txt"
      );

      const filePath = path.join(tempDir, "new.txt");
      expect(fs.existsSync(filePath)).toBe(true);
      expect(fs.readFileSync(filePath, "utf8")).toBe("line 1\nline 2\n");
    });

    it("should create parent directories", async () => {
      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "create",
            path: "/subdir/new.txt",
            file_text: "content",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      const filePath = path.join(tempDir, "subdir", "new.txt");
      expect(fs.existsSync(filePath)).toBe(true);
    });
  });

  describe("wrapToolCall - str_replace", () => {
    it("should replace string in file", async () => {
      const filePath = path.join(tempDir, "test.txt");
      fs.writeFileSync(filePath, "hello world\nhello universe");

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "str_replace",
            path: "/test.txt",
            old_str: "hello world",
            new_str: "goodbye world",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      const content = fs.readFileSync(filePath, "utf8");
      expect(content).toBe("goodbye world\nhello universe");
    });
  });

  describe("wrapToolCall - insert", () => {
    it("should insert text at specified line", async () => {
      const filePath = path.join(tempDir, "test.txt");
      fs.writeFileSync(filePath, "line 1\nline 2\nline 3\n");

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "insert",
            path: "/test.txt",
            insert_line: 1,
            new_str: "inserted line",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      const content = fs.readFileSync(filePath, "utf8");
      expect(content).toBe("line 1\ninserted line\nline 2\nline 3\n");
    });
  });

  describe("wrapToolCall - delete", () => {
    it("should delete file", async () => {
      const filePath = path.join(tempDir, "test.txt");
      fs.writeFileSync(filePath, "content");

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "delete",
            path: "/test.txt",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect(fs.existsSync(filePath)).toBe(false);
    });

    it("should delete directory recursively", async () => {
      const dirPath = path.join(tempDir, "subdir");
      fs.mkdirSync(dirPath);
      fs.writeFileSync(path.join(dirPath, "file.txt"), "content");

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "delete",
            path: "/subdir",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect(fs.existsSync(dirPath)).toBe(false);
    });
  });

  describe("wrapToolCall - rename", () => {
    it("should rename file", async () => {
      const oldPath = path.join(tempDir, "old.txt");
      fs.writeFileSync(oldPath, "content");

      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "rename",
            old_path: "/old.txt",
            new_path: "/new.txt",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect(fs.existsSync(oldPath)).toBe(false);
      const newPath = path.join(tempDir, "new.txt");
      expect(fs.existsSync(newPath)).toBe(true);
      expect(fs.readFileSync(newPath, "utf8")).toBe("content");
    });
  });

  describe("path security", () => {
    it("should prevent escaping root directory", async () => {
      const request = {
        toolCall: {
          name: TEXT_EDITOR_TOOL_NAME,
          args: {
            command: "view",
            path: "/../etc/passwd",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect(result).toHaveProperty("status", "error");
      expect((result as ToolMessage).content).toContain(
        "Path traversal not allowed"
      );
    });
  });
});

describe("FilesystemClaudeMemoryMiddleware", () => {
  let middleware: FilesystemClaudeMemoryMiddleware;
  let tempDir: string;

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "test-memory-"));
    middleware = new FilesystemClaudeMemoryMiddleware({
      rootPath: tempDir,
    });
  });

  afterEach(() => {
    if (fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  describe("path prefix enforcement", () => {
    it("should enforce /memories prefix by default", async () => {
      const request = {
        toolCall: {
          name: MEMORY_TOOL_NAME,
          args: {
            command: "view",
            path: "/etc/passwd",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect(result).toHaveProperty("status", "error");
      expect((result as ToolMessage).content).toContain("Path must start with");
    });

    it("should allow paths with /memories prefix", async () => {
      const memoriesDir = path.join(tempDir, "memories");
      fs.mkdirSync(memoriesDir, { recursive: true });
      const filePath = path.join(memoriesDir, "progress.txt");
      fs.writeFileSync(filePath, "task 1 done");

      const request = {
        toolCall: {
          name: MEMORY_TOOL_NAME,
          args: {
            command: "view",
            path: "/memories/progress.txt",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect(result).toBeInstanceOf(Command);
      const cmd = result as Command;
      expect((cmd.update as any).messages[0].content).toBe("1|task 1 done");
    });
  });
});
