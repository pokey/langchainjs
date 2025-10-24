/* eslint-disable @typescript-eslint/no-explicit-any */
/**
 * Tests for file search middleware (Glob and Grep tools).
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { ToolMessage } from "@langchain/core/messages";
import * as fs from "node:fs";
import * as path from "node:path";
import * as os from "node:os";

import {
  StateFileSearchMiddleware,
  FilesystemFileSearchMiddleware,
} from "../anthropicTools/fileSearch.js";
import type { AnthropicToolsState } from "../anthropicTools/index.js";

describe("StateFileSearchMiddleware", () => {
  let middleware: StateFileSearchMiddleware;

  beforeEach(() => {
    middleware = new StateFileSearchMiddleware();
  });

  describe("initialization", () => {
    it("should initialize with default state key", () => {
      const mw = new StateFileSearchMiddleware();
      expect(mw.name).toBe("StateFileSearchMiddleware");
      expect(mw.tools).toBeDefined();
      expect(mw.tools).toHaveLength(2); // glob_search and grep_search
      expect(mw.tools[0].name).toBe("glob_search");
      expect(mw.tools[1].name).toBe("grep_search");
    });

    it("should initialize with custom state key", () => {
      const mw = new StateFileSearchMiddleware({ stateKey: "memory_files" });
      expect(mw.name).toBe("StateFileSearchMiddleware");
    });
  });

  describe("glob_search", () => {
    it("should find files matching pattern", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/src/index.ts": {
            content: ["content"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-02T00:00:00Z",
          },
          "/src/utils.ts": {
            content: ["content"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
          "/README.md": {
            content: ["content"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: "glob_search",
          args: {
            pattern: "*.ts",
            path: "/src",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toContain("/src/index.ts");
      expect((result as ToolMessage).content).toContain("/src/utils.ts");
      expect((result as ToolMessage).content).not.toContain("/README.md");
    });

    it("should support **/ glob pattern", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/src/components/Button.tsx": {
            content: ["content"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
          "/src/utils/helpers.ts": {
            content: ["content"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
          "/src/index.ts": {
            content: ["content"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: "glob_search",
          args: {
            pattern: "**/*.ts",
            path: "/src",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toContain(
        "/src/utils/helpers.ts"
      );
      expect((result as ToolMessage).content).toContain("/src/index.ts");
      expect((result as ToolMessage).content).not.toContain(
        "/src/components/Button.tsx"
      );
    });

    it("should sort by modification time (most recent first)", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/old.txt": {
            content: ["content"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
          "/new.txt": {
            content: ["content"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-03T00:00:00Z",
          },
          "/mid.txt": {
            content: ["content"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-02T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: "glob_search",
          args: {
            pattern: "*.txt",
            path: "/",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      const content = (result as ToolMessage).content as string;
      const lines = content.split("\n");
      expect(lines[0]).toBe("/new.txt");
      expect(lines[1]).toBe("/mid.txt");
      expect(lines[2]).toBe("/old.txt");
    });

    it("should return 'No files found' when no matches", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/test.ts": {
            content: ["content"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: "glob_search",
          args: {
            pattern: "*.py",
            path: "/",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toBe("No files found");
    });
  });

  describe("grep_search", () => {
    it("should find files containing pattern (files_with_matches mode)", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/file1.txt": {
            content: ["hello world", "goodbye world"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
          "/file2.txt": {
            content: ["hello universe"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
          "/file3.txt": {
            content: ["no match here"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: "grep_search",
          args: {
            pattern: "hello",
            path: "/",
            output_mode: "files_with_matches",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toContain("/file1.txt");
      expect((result as ToolMessage).content).toContain("/file2.txt");
      expect((result as ToolMessage).content).not.toContain("/file3.txt");
    });

    it("should return matching lines (content mode)", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/file1.txt": {
            content: [
              "line 1 with hello",
              "line 2 without",
              "line 3 with hello",
            ],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: "grep_search",
          args: {
            pattern: "hello",
            path: "/",
            output_mode: "content",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toContain(
        "/file1.txt:1:line 1 with hello"
      );
      expect((result as ToolMessage).content).toContain(
        "/file1.txt:3:line 3 with hello"
      );
      expect((result as ToolMessage).content).not.toContain("line 2 without");
    });

    it("should return match counts (count mode)", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/file1.txt": {
            content: ["hello", "hello", "hello"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
          "/file2.txt": {
            content: ["hello"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: "grep_search",
          args: {
            pattern: "hello",
            path: "/",
            output_mode: "count",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toContain("/file1.txt:3");
      expect((result as ToolMessage).content).toContain("/file2.txt:1");
    });

    it("should filter by include pattern", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/test.ts": {
            content: ["hello"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
          "/test.js": {
            content: ["hello"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
          "/test.py": {
            content: ["hello"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: "grep_search",
          args: {
            pattern: "hello",
            path: "/",
            include: "*.ts",
            output_mode: "files_with_matches",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toBe("/test.ts");
    });

    it("should support brace expansion in include patterns", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/test.ts": {
            content: ["hello"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
          "/test.tsx": {
            content: ["hello"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
          "/test.js": {
            content: ["hello"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: "grep_search",
          args: {
            pattern: "hello",
            path: "/",
            include: "*.{ts,tsx}",
            output_mode: "files_with_matches",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toContain("/test.ts");
      expect((result as ToolMessage).content).toContain("/test.tsx");
      expect((result as ToolMessage).content).not.toContain("/test.js");
    });

    it("should support regex patterns", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/test.txt": {
            content: ["test123", "test456", "testABC"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: "grep_search",
          args: {
            pattern: "test\\d+",
            path: "/",
            output_mode: "content",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toContain("test123");
      expect((result as ToolMessage).content).toContain("test456");
      expect((result as ToolMessage).content).not.toContain("testABC");
    });

    it("should return error for invalid regex", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {},
      };

      const request = {
        toolCall: {
          name: "grep_search",
          args: {
            pattern: "[invalid",
            path: "/",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toContain(
        "Invalid regex pattern"
      );
    });

    it("should return error for invalid include pattern", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {},
      };

      const request = {
        toolCall: {
          name: "grep_search",
          args: {
            pattern: "test",
            path: "/",
            include: "*.{ts",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toBe("Invalid include pattern");
    });

    it("should return 'No matches found' when no results", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/test.txt": {
            content: ["no match here"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: "grep_search",
          args: {
            pattern: "missing",
            path: "/",
          },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toBe("No matches found");
    });
  });

  describe("backend configuration", () => {
    it("should search only text_editor_files by default for glob", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/src/main.ts": {
            content: [],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
        memory_files: {
          "/memories/notes.txt": {
            content: [],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: "glob_search",
          args: { pattern: "**/*", path: "/" },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      const content = (result as ToolMessage).content as string;
      expect(content).toContain("/src/main.ts");
      expect(content).not.toContain("/memories/notes.txt");
    });

    it("should search only text_editor_files by default for grep", async () => {
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/src/main.ts": {
            content: ["hello world"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
        memory_files: {
          "/memories/notes.txt": {
            content: ["hello memories"],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: "grep_search",
          args: { pattern: "hello", path: "/" },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      const content = (result as ToolMessage).content as string;
      expect(content).toContain("/src/main.ts");
      expect(content).not.toContain("/memories/notes.txt");
    });

    it("should search custom state key when specified", async () => {
      const mw = new StateFileSearchMiddleware({ stateKey: "memory_files" });
      const state: AnthropicToolsState = {
        text_editor_files: {
          "/src/main.ts": {
            content: [],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
        memory_files: {
          "/memories/notes.txt": {
            content: [],
            created_at: "2024-01-01T00:00:00Z",
            modified_at: "2024-01-01T00:00:00Z",
          },
        },
      };

      const request = {
        toolCall: {
          name: "glob_search",
          args: { pattern: "**/*", path: "/" },
          id: "call_1",
          type: "tool_call",
        },
        state,
      } as any;

      const result = await mw.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      const content = (result as ToolMessage).content as string;
      expect(content).toContain("/memories/notes.txt");
      expect(content).not.toContain("/src/main.ts");
    });
  });
});

describe("FilesystemFileSearchMiddleware", () => {
  let middleware: FilesystemFileSearchMiddleware;
  let tempDir: string;

  beforeEach(() => {
    tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "test-search-"));
    middleware = new FilesystemFileSearchMiddleware({
      rootPath: tempDir,
      useRipgrep: false, // Disable ripgrep for consistent tests
    });
  });

  afterEach(() => {
    if (fs.existsSync(tempDir)) {
      fs.rmSync(tempDir, { recursive: true, force: true });
    }
  });

  describe("glob_search", () => {
    it("should find files matching pattern", async () => {
      fs.writeFileSync(path.join(tempDir, "file1.ts"), "content");
      fs.writeFileSync(path.join(tempDir, "file2.ts"), "content");
      fs.writeFileSync(path.join(tempDir, "file3.js"), "content");

      const request = {
        toolCall: {
          name: "glob_search",
          args: {
            pattern: "*.ts",
            path: "/",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toContain("/file1.ts");
      expect((result as ToolMessage).content).toContain("/file2.ts");
      expect((result as ToolMessage).content).not.toContain("/file3.js");
    });

    it("should support recursive glob patterns", async () => {
      fs.mkdirSync(path.join(tempDir, "src"), { recursive: true });
      fs.mkdirSync(path.join(tempDir, "src", "utils"), { recursive: true });
      fs.writeFileSync(path.join(tempDir, "src", "index.ts"), "content");
      fs.writeFileSync(
        path.join(tempDir, "src", "utils", "helper.ts"),
        "content"
      );
      fs.writeFileSync(path.join(tempDir, "README.md"), "content");

      const request = {
        toolCall: {
          name: "glob_search",
          args: {
            pattern: "**/*.ts",
            path: "/",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toContain("/src/index.ts");
      expect((result as ToolMessage).content).toContain("/src/utils/helper.ts");
      expect((result as ToolMessage).content).not.toContain("/README.md");
    });

    it("should return 'No files found' when no matches", async () => {
      fs.writeFileSync(path.join(tempDir, "test.ts"), "content");

      const request = {
        toolCall: {
          name: "glob_search",
          args: {
            pattern: "*.py",
            path: "/",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toBe("No files found");
    });
  });

  describe("grep_search", () => {
    it("should find files containing pattern", async () => {
      fs.writeFileSync(path.join(tempDir, "file1.txt"), "hello world");
      fs.writeFileSync(path.join(tempDir, "file2.txt"), "goodbye world");
      fs.writeFileSync(path.join(tempDir, "file3.txt"), "no match");

      const request = {
        toolCall: {
          name: "grep_search",
          args: {
            pattern: "hello",
            path: "/",
            output_mode: "files_with_matches",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toBe("/file1.txt");
    });

    it("should return matching lines (content mode)", async () => {
      fs.writeFileSync(
        path.join(tempDir, "test.txt"),
        "line 1 with hello\nline 2 without\nline 3 with hello"
      );

      const request = {
        toolCall: {
          name: "grep_search",
          args: {
            pattern: "hello",
            path: "/",
            output_mode: "content",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toContain(
        "/test.txt:1:line 1 with hello"
      );
      expect((result as ToolMessage).content).toContain(
        "/test.txt:3:line 3 with hello"
      );
      expect((result as ToolMessage).content).not.toContain("line 2 without");
    });

    it("should return match counts (count mode)", async () => {
      fs.writeFileSync(path.join(tempDir, "file1.txt"), "hello\nhello\nhello");
      fs.writeFileSync(path.join(tempDir, "file2.txt"), "hello");

      const request = {
        toolCall: {
          name: "grep_search",
          args: {
            pattern: "hello",
            path: "/",
            output_mode: "count",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toContain("/file1.txt:3");
      expect((result as ToolMessage).content).toContain("/file2.txt:1");
    });

    it("should filter by include pattern", async () => {
      fs.writeFileSync(path.join(tempDir, "test.ts"), "hello");
      fs.writeFileSync(path.join(tempDir, "test.js"), "hello");
      fs.writeFileSync(path.join(tempDir, "test.py"), "hello");

      const request = {
        toolCall: {
          name: "grep_search",
          args: {
            pattern: "hello",
            path: "/",
            include: "*.ts",
            output_mode: "files_with_matches",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toBe("/test.ts");
    });

    it("should support brace expansion", async () => {
      fs.writeFileSync(path.join(tempDir, "test.ts"), "hello");
      fs.writeFileSync(path.join(tempDir, "test.tsx"), "hello");
      fs.writeFileSync(path.join(tempDir, "test.js"), "hello");

      const request = {
        toolCall: {
          name: "grep_search",
          args: {
            pattern: "hello",
            path: "/",
            include: "*.{ts,tsx}",
            output_mode: "files_with_matches",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toContain("/test.ts");
      expect((result as ToolMessage).content).toContain("/test.tsx");
      expect((result as ToolMessage).content).not.toContain("/test.js");
    });

    it("should search in subdirectories", async () => {
      fs.mkdirSync(path.join(tempDir, "subdir"), { recursive: true });
      fs.writeFileSync(path.join(tempDir, "subdir", "test.txt"), "hello");

      const request = {
        toolCall: {
          name: "grep_search",
          args: {
            pattern: "hello",
            path: "/",
            output_mode: "files_with_matches",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toBe("/subdir/test.txt");
    });

    it("should skip files that are too large", async () => {
      // Create a file larger than default 10MB
      const largeContent = "a".repeat(11 * 1024 * 1024);
      fs.writeFileSync(path.join(tempDir, "large.txt"), largeContent);
      fs.writeFileSync(path.join(tempDir, "small.txt"), "hello");

      const request = {
        toolCall: {
          name: "grep_search",
          args: {
            pattern: "a",
            path: "/",
            output_mode: "files_with_matches",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      // Should not find the large file
      expect((result as ToolMessage).content).not.toContain("/large.txt");
    });

    it("should return 'No matches found' when no results", async () => {
      fs.writeFileSync(path.join(tempDir, "test.txt"), "no match here");

      const request = {
        toolCall: {
          name: "grep_search",
          args: {
            pattern: "missing",
            path: "/",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toBe("No matches found");
    });
  });

  describe("path security", () => {
    it("should prevent path traversal in glob", async () => {
      const request = {
        toolCall: {
          name: "glob_search",
          args: {
            pattern: "*",
            path: "/../etc",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toBe("No files found");
    });

    it("should prevent path traversal in grep", async () => {
      const request = {
        toolCall: {
          name: "grep_search",
          args: {
            pattern: "test",
            path: "/../etc",
          },
          id: "call_1",
          type: "tool_call",
        },
      } as any;

      const result = await middleware.wrapToolCall!(request, async () => {
        throw new Error("Should not call handler");
      });

      expect((result as ToolMessage).content).toBe("No matches found");
    });
  });
});
