/**
 * Anthropic text editor and memory tool middleware.
 *
 * This module provides client-side implementations of Anthropic's text editor and
 * memory tools using schema-less tool definitions and tool call interception.
 */

import { ToolMessage } from "@langchain/core/messages";
import { Command } from "@langchain/langgraph";
import * as fs from "node:fs";
import * as path from "node:path";
import { z } from "zod";
import type {
  AgentMiddleware,
  WrapModelCallHook,
  WrapToolCallHook,
} from "./types.js";

// Tool type constants
export const TEXT_EDITOR_TOOL_TYPE = "text_editor_20250728";
export const TEXT_EDITOR_TOOL_NAME = "str_replace_based_edit_tool";
export const MEMORY_TOOL_TYPE = "memory_20250818";
export const MEMORY_TOOL_NAME = "memory";

export const MEMORY_SYSTEM_PROMPT = `IMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE \
DOING ANYTHING ELSE.
MEMORY PROTOCOL:
1. Use the \`view\` command of your \`memory\` tool to check for earlier progress.
2. ... (work on the task) ...
   - As you make progress, record status / progress / thoughts etc in your memory.
ASSUME INTERRUPTION: Your context window might be reset at any moment, so you risk \
losing any progress that is not recorded in your memory directory.`;

/**
 * Data structure for storing file contents.
 */
export interface FileData {
  /** Lines of the file */
  content: string[];
  /** ISO 8601 timestamp of file creation */
  created_at: string;
  /** ISO 8601 timestamp of last modification */
  modified_at: string;
}

/**
 * Custom reducer that merges file updates.
 * @param left - Existing files dict
 * @param right - New files dict to merge (null values delete files)
 * @returns Merged dict where right overwrites left for matching keys
 */
export function filesReducer(
  left: Record<string, FileData> | undefined,
  right: Record<string, FileData | null>
): Record<string, FileData> {
  if (left === undefined) {
    // Filter out null values when initializing
    const result: Record<string, FileData> = {};
    for (const [k, v] of Object.entries(right)) {
      if (v !== null) {
        result[k] = v;
      }
    }
    return result;
  }

  // Merge, filtering out null values (deletions)
  const result = { ...left };
  for (const [k, v] of Object.entries(right)) {
    if (v === null) {
      delete result[k];
    } else {
      result[k] = v;
    }
  }
  return result;
}

/**
 * State schema for Anthropic text editor and memory tools.
 */
export interface AnthropicToolsState {
  /** Virtual file system for text editor tools */
  text_editor_files?: Record<string, FileData>;
  /** Virtual file system for memory tools */
  memory_files?: Record<string, FileData>;
}

/**
 * Zod schema for file data.
 */
const FileDataSchema = z.object({
  content: z.array(z.string()),
  created_at: z.string(),
  modified_at: z.string(),
});

/**
 * Zod state schemas for middleware.
 */
const TextEditorStateSchema = z.object({
  text_editor_files: z
    .record(z.string(), FileDataSchema)
    .optional()
    .default({}),
});

const MemoryStateSchema = z.object({
  memory_files: z.record(z.string(), FileDataSchema).optional().default({}),
});

/**
 * Validate and normalize file path for security.
 * @param filePath - The path to validate
 * @param allowedPrefixes - Optional list of allowed path prefixes
 * @returns Normalized canonical path
 * @throws Error if path contains traversal sequences or violates prefix rules
 */
export function validatePath(
  filePath: string,
  allowedPrefixes?: string[]
): string {
  // Reject paths with traversal attempts
  if (filePath.includes("..") || filePath.startsWith("~")) {
    throw new Error(`Path traversal not allowed: ${filePath}`);
  }

  // Normalize path (resolve ., //, etc.)
  let normalized = path.normalize(filePath);

  // Convert to forward slashes for consistency
  normalized = normalized.replace(/\\/g, "/");

  // Ensure path starts with /
  if (!normalized.startsWith("/")) {
    normalized = `/${normalized}`;
  }

  // Check allowed prefixes if specified
  if (allowedPrefixes !== undefined && allowedPrefixes.length > 0) {
    const allowed = allowedPrefixes.some((prefix) =>
      normalized.startsWith(prefix)
    );
    if (!allowed) {
      throw new Error(
        `Path must start with one of ${JSON.stringify(
          allowedPrefixes
        )}: ${filePath}`
      );
    }
  }

  return normalized;
}

/**
 * List files in a directory.
 * @param files - Files dict
 * @param dirPath - Normalized directory path
 * @returns Sorted list of file paths in the directory
 */
function listDirectory(
  files: Record<string, FileData>,
  dirPath: string
): string[] {
  // Ensure path ends with / for directory matching
  const dir = dirPath.endsWith("/") ? dirPath : `${dirPath}/`;

  const matchingFiles: string[] = [];
  for (const filePath of Object.keys(files)) {
    if (filePath.startsWith(dir)) {
      // Get relative path from directory
      const relative = filePath.slice(dir.length);
      // Only include direct children (no subdirectories)
      if (!relative.includes("/")) {
        matchingFiles.push(filePath);
      }
    }
  }

  return matchingFiles.sort();
}

/**
 * Base class for state-based file tool middleware (internal).
 */
class StateClaudeFileToolMiddleware implements AgentMiddleware {
  name: string;

  protected toolType: string;

  protected toolName: string;

  protected stateKey: string;

  protected allowedPrefixes?: string[];

  protected systemPrompt?: string;

  constructor(options: {
    toolType: string;
    toolName: string;
    stateKey: string;
    allowedPathPrefixes?: string[];
    systemPrompt?: string;
  }) {
    this.toolType = options.toolType;
    this.toolName = options.toolName;
    this.stateKey = options.stateKey;
    this.allowedPrefixes = options.allowedPathPrefixes;
    this.systemPrompt = options.systemPrompt;
    this.name = `StateClaudeFileToolMiddleware(${this.toolName})`;
  }

  wrapModelCall: WrapModelCallHook = async (request, handler) => {
    // Add tool
    const tools = [...(request.tools || [])];
    tools.push({
      type: this.toolType,
      name: this.toolName,
    });
    request.tools = tools;

    // Inject system prompt if provided
    if (this.systemPrompt) {
      request.systemPrompt = request.systemPrompt
        ? `${request.systemPrompt}\n\n${this.systemPrompt}`
        : this.systemPrompt;
    }

    return handler(request);
  };

  wrapToolCall: WrapToolCallHook = async (request, handler) => {
    const toolCall = request.toolCall;
    const toolName = toolCall.name;

    if (toolName !== this.toolName) {
      return handler(request);
    }

    // Handle tool call
    try {
      const args = (toolCall.args || {}) as Record<string, unknown>;
      const command = args.command as string;
      const state = request.state as AnthropicToolsState;

      if (command === "view") {
        return this.handleView(args, state, toolCall.id);
      }
      if (command === "create") {
        return this.handleCreate(args, state, toolCall.id);
      }
      if (command === "str_replace") {
        return this.handleStrReplace(args, state, toolCall.id);
      }
      if (command === "insert") {
        return this.handleInsert(args, state, toolCall.id);
      }
      if (command === "delete") {
        return this.handleDelete(args, state, toolCall.id);
      }
      if (command === "rename") {
        return this.handleRename(args, state, toolCall.id);
      }

      return new ToolMessage({
        content: `Unknown command: ${command}`,
        tool_call_id: toolCall.id!,
        name: toolName,
        status: "error",
      });
    } catch (error) {
      return new ToolMessage({
        content: String(error),
        tool_call_id: toolCall.id!,
        name: toolName,
        status: "error",
      });
    }
  };

  protected handleView(
    args: Record<string, unknown>,
    state: AnthropicToolsState,
    toolCallId: string | undefined
  ): Command {
    const filePath = args.path as string;
    const normalizedPath = validatePath(filePath, this.allowedPrefixes);

    const files =
      (state[this.stateKey as keyof AnthropicToolsState] as Record<
        string,
        FileData
      >) || {};
    const fileData = files[normalizedPath];

    if (!fileData) {
      // Try directory listing
      const matching = listDirectory(files, normalizedPath);

      if (matching.length > 0) {
        const content = matching.join("\n");
        return new Command({
          update: {
            messages: [
              new ToolMessage({
                content,
                tool_call_id: toolCallId!,
                name: this.toolName,
              }),
            ],
          },
        });
      }

      throw new Error(`File not found: ${filePath}`);
    }

    // Format file content with line numbers
    const linesContent = fileData.content;
    const formattedLines = linesContent.map((line, i) => `${i + 1}|${line}`);
    const content = formattedLines.join("\n");

    return new Command({
      update: {
        messages: [
          new ToolMessage({
            content,
            tool_call_id: toolCallId!,
            name: this.toolName,
          }),
        ],
      },
    });
  }

  protected handleCreate(
    args: Record<string, unknown>,
    state: AnthropicToolsState,
    toolCallId: string | undefined
  ): Command {
    const filePath = args.path as string;
    const fileText = args.file_text as string;

    const normalizedPath = validatePath(filePath, this.allowedPrefixes);

    // Get existing files
    const files =
      (state[this.stateKey as keyof AnthropicToolsState] as Record<
        string,
        FileData
      >) || {};
    const existing = files[normalizedPath];

    // Create file data
    const now = new Date().toISOString();
    const createdAt = existing ? existing.created_at : now;

    const contentLines = fileText.split("\n");

    return new Command({
      update: {
        [this.stateKey]: {
          [normalizedPath]: {
            content: contentLines,
            created_at: createdAt,
            modified_at: now,
          },
        },
        messages: [
          new ToolMessage({
            content: `File created: ${filePath}`,
            tool_call_id: toolCallId!,
            name: this.toolName,
          }),
        ],
      },
    });
  }

  protected handleStrReplace(
    args: Record<string, unknown>,
    state: AnthropicToolsState,
    toolCallId: string | undefined
  ): Command {
    const filePath = args.path as string;
    const oldStr = args.old_str as string;
    const newStr = (args.new_str as string) || "";

    const normalizedPath = validatePath(filePath, this.allowedPrefixes);

    // Read file
    const files =
      (state[this.stateKey as keyof AnthropicToolsState] as Record<
        string,
        FileData
      >) || {};
    const fileData = files[normalizedPath];
    if (!fileData) {
      throw new Error(`File not found: ${filePath}`);
    }

    const linesContent = fileData.content;
    const content = linesContent.join("\n");

    // Replace string
    if (!content.includes(oldStr)) {
      throw new Error(`String not found in file: ${oldStr}`);
    }

    const newContent = content.replace(oldStr, newStr);
    const newLines = newContent.split("\n");

    // Update file
    const now = new Date().toISOString();

    return new Command({
      update: {
        [this.stateKey]: {
          [normalizedPath]: {
            content: newLines,
            created_at: fileData.created_at,
            modified_at: now,
          },
        },
        messages: [
          new ToolMessage({
            content: `String replaced in ${filePath}`,
            tool_call_id: toolCallId!,
            name: this.toolName,
          }),
        ],
      },
    });
  }

  protected handleInsert(
    args: Record<string, unknown>,
    state: AnthropicToolsState,
    toolCallId: string | undefined
  ): Command {
    const filePath = args.path as string;
    const insertLine = args.insert_line as number;
    const textToInsert = args.new_str as string;

    const normalizedPath = validatePath(filePath, this.allowedPrefixes);

    // Read file
    const files =
      (state[this.stateKey as keyof AnthropicToolsState] as Record<
        string,
        FileData
      >) || {};
    const fileData = files[normalizedPath];
    if (!fileData) {
      throw new Error(`File not found: ${filePath}`);
    }

    const linesContent = fileData.content;
    const newLines = textToInsert.split("\n");

    // Insert after insert_line (0-indexed)
    const updatedLines = [
      ...linesContent.slice(0, insertLine),
      ...newLines,
      ...linesContent.slice(insertLine),
    ];

    // Update file
    const now = new Date().toISOString();

    return new Command({
      update: {
        [this.stateKey]: {
          [normalizedPath]: {
            content: updatedLines,
            created_at: fileData.created_at,
            modified_at: now,
          },
        },
        messages: [
          new ToolMessage({
            content: `Text inserted in ${filePath}`,
            tool_call_id: toolCallId!,
            name: this.toolName,
          }),
        ],
      },
    });
  }

  protected handleDelete(
    args: Record<string, unknown>,
    _state: AnthropicToolsState,
    toolCallId: string | undefined
  ): Command {
    const filePath = args.path as string;

    const normalizedPath = validatePath(filePath, this.allowedPrefixes);

    return new Command({
      update: {
        [this.stateKey]: { [normalizedPath]: null },
        messages: [
          new ToolMessage({
            content: `File deleted: ${filePath}`,
            tool_call_id: toolCallId!,
            name: this.toolName,
          }),
        ],
      },
    });
  }

  protected handleRename(
    args: Record<string, unknown>,
    state: AnthropicToolsState,
    toolCallId: string | undefined
  ): Command {
    const oldPath = args.old_path as string;
    const newPath = args.new_path as string;

    const normalizedOld = validatePath(oldPath, this.allowedPrefixes);
    const normalizedNew = validatePath(newPath, this.allowedPrefixes);

    // Read file
    const files =
      (state[this.stateKey as keyof AnthropicToolsState] as Record<
        string,
        FileData
      >) || {};
    const fileData = files[normalizedOld];
    if (!fileData) {
      throw new Error(`File not found: ${oldPath}`);
    }

    // Update timestamp
    const now = new Date().toISOString();
    const fileDataCopy = { ...fileData, modified_at: now };

    return new Command({
      update: {
        [this.stateKey]: {
          [normalizedOld]: null,
          [normalizedNew]: fileDataCopy,
        },
        messages: [
          new ToolMessage({
            content: `File renamed: ${oldPath} -> ${newPath}`,
            tool_call_id: toolCallId!,
            name: this.toolName,
          }),
        ],
      },
    });
  }
}

/**
 * State-based text editor tool middleware.
 *
 * Provides Anthropic's text_editor tool using LangGraph state for storage.
 * Files persist for the conversation thread.
 *
 * @example
 * ```ts
 * import { createAgent } from "langchain/agents";
 * import { StateClaudeTextEditorMiddleware } from "langchain/agents/middleware";
 *
 * const agent = createAgent({
 *   model,
 *   tools: [],
 *   middleware: [new StateClaudeTextEditorMiddleware()],
 * });
 * ```
 */
export class StateClaudeTextEditorMiddleware extends StateClaudeFileToolMiddleware {
  stateSchema = TextEditorStateSchema;

  constructor(options?: { allowedPathPrefixes?: string[] }) {
    super({
      toolType: TEXT_EDITOR_TOOL_TYPE,
      toolName: TEXT_EDITOR_TOOL_NAME,
      stateKey: "text_editor_files",
      allowedPathPrefixes: options?.allowedPathPrefixes,
    });
    this.name = "StateClaudeTextEditorMiddleware";
  }
}

/**
 * State-based memory tool middleware.
 *
 * Provides Anthropic's memory tool using LangGraph state for storage.
 * Files persist for the conversation thread. Enforces /memories prefix
 * and injects Anthropic's recommended system prompt.
 *
 * @example
 * ```ts
 * import { createAgent } from "langchain/agents";
 * import { StateClaudeMemoryMiddleware } from "langchain/agents/middleware";
 *
 * const agent = createAgent({
 *   model,
 *   tools: [],
 *   middleware: [new StateClaudeMemoryMiddleware()],
 * });
 * ```
 */
export class StateClaudeMemoryMiddleware extends StateClaudeFileToolMiddleware {
  stateSchema = MemoryStateSchema;

  constructor(options?: {
    allowedPathPrefixes?: string[];
    systemPrompt?: string;
  }) {
    super({
      toolType: MEMORY_TOOL_TYPE,
      toolName: MEMORY_TOOL_NAME,
      stateKey: "memory_files",
      allowedPathPrefixes: options?.allowedPathPrefixes || ["/memories"],
      systemPrompt:
        options?.systemPrompt !== undefined
          ? options.systemPrompt
          : MEMORY_SYSTEM_PROMPT,
    });
    this.name = "StateClaudeMemoryMiddleware";
  }
}

/**
 * Base class for filesystem-based file tool middleware (internal).
 */
class FilesystemClaudeFileToolMiddleware implements AgentMiddleware {
  name: string;

  protected toolType: string;

  protected toolName: string;

  protected rootPath: string;

  protected allowedPrefixes: string[];

  protected maxFileSizeBytes: number;

  protected systemPrompt?: string;

  constructor(options: {
    toolType: string;
    toolName: string;
    rootPath: string;
    allowedPrefixes?: string[];
    maxFileSizeMb?: number;
    systemPrompt?: string;
  }) {
    this.toolType = options.toolType;
    this.toolName = options.toolName;
    this.rootPath = path.resolve(options.rootPath);
    this.allowedPrefixes = options.allowedPrefixes || ["/"];
    this.maxFileSizeBytes = (options.maxFileSizeMb || 10) * 1024 * 1024;
    this.systemPrompt = options.systemPrompt;
    this.name = `FilesystemClaudeFileToolMiddleware(${this.toolName})`;

    // Create root directory if it doesn't exist
    if (!fs.existsSync(this.rootPath)) {
      fs.mkdirSync(this.rootPath, { recursive: true });
    }
  }

  wrapModelCall: WrapModelCallHook = async (request, handler) => {
    // Inject system prompt if provided
    if (this.systemPrompt) {
      request.systemPrompt = request.systemPrompt
        ? `${request.systemPrompt}\n\n${this.systemPrompt}`
        : this.systemPrompt;
    }

    return handler(request);
  };

  wrapToolCall: WrapToolCallHook = async (request, handler) => {
    const toolCall = request.toolCall;
    const toolName = toolCall.name;

    if (toolName !== this.toolName) {
      return handler(request);
    }

    // Handle tool call
    try {
      const args = (toolCall.args || {}) as Record<string, unknown>;
      const command = args.command as string;

      if (command === "view") {
        return this.handleView(args, toolCall.id);
      }
      if (command === "create") {
        return this.handleCreate(args, toolCall.id);
      }
      if (command === "str_replace") {
        return this.handleStrReplace(args, toolCall.id);
      }
      if (command === "insert") {
        return this.handleInsert(args, toolCall.id);
      }
      if (command === "delete") {
        return this.handleDelete(args, toolCall.id);
      }
      if (command === "rename") {
        return this.handleRename(args, toolCall.id);
      }

      return new ToolMessage({
        content: `Unknown command: ${command}`,
        tool_call_id: toolCall.id!,
        name: toolName,
        status: "error",
      });
    } catch (error) {
      return new ToolMessage({
        content: String(error),
        tool_call_id: toolCall.id!,
        name: toolName,
        status: "error",
      });
    }
  };

  protected validateAndResolvePath(virtualPath: string): string {
    // Normalize path
    let normalizedVirtual = virtualPath;
    if (!normalizedVirtual.startsWith("/")) {
      normalizedVirtual = `/${normalizedVirtual}`;
    }

    // Check for path traversal
    if (normalizedVirtual.includes("..") || normalizedVirtual.includes("~")) {
      throw new Error("Path traversal not allowed");
    }

    // Convert virtual path to filesystem path
    const relative = normalizedVirtual.slice(1); // Remove leading /
    const fullPath = path.resolve(this.rootPath, relative);

    // Ensure path is within root
    if (!fullPath.startsWith(this.rootPath)) {
      throw new Error(`Path outside root directory: ${virtualPath}`);
    }

    // Check allowed prefixes
    const virtualForCheck = `/${path
      .relative(this.rootPath, fullPath)
      .replace(/\\/g, "/")}`;
    const allowed = this.allowedPrefixes.some(
      (prefix) =>
        virtualForCheck.startsWith(prefix) ||
        virtualForCheck === prefix.replace(/\/$/, "")
    );
    if (!allowed) {
      throw new Error(
        `Path must start with one of: ${JSON.stringify(this.allowedPrefixes)}`
      );
    }

    return fullPath;
  }

  protected handleView(
    args: Record<string, unknown>,
    toolCallId: string | undefined
  ): Command {
    const virtualPath = args.path as string;
    const fullPath = this.validateAndResolvePath(virtualPath);

    if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isFile()) {
      throw new Error(`File not found: ${virtualPath}`);
    }

    // Check file size
    const stats = fs.statSync(fullPath);
    if (stats.size > this.maxFileSizeBytes) {
      const maxMb = this.maxFileSizeBytes / 1024 / 1024;
      throw new Error(`File too large: ${virtualPath} exceeds ${maxMb}MB`);
    }

    // Read file
    let content: string;
    try {
      content = fs.readFileSync(fullPath, "utf8");
    } catch (error) {
      throw new Error(`Cannot decode file ${virtualPath}: ${error}`);
    }

    // Format with line numbers
    let lines = content.split("\n");
    // Remove trailing newline's empty string if present
    if (lines.length > 0 && lines[lines.length - 1] === "") {
      lines = lines.slice(0, -1);
    }
    const formattedLines = lines.map((line, i) => `${i + 1}|${line}`);
    const formattedContent = formattedLines.join("\n");

    return new Command({
      update: {
        messages: [
          new ToolMessage({
            content: formattedContent,
            tool_call_id: toolCallId!,
            name: this.toolName,
          }),
        ],
      },
    });
  }

  protected handleCreate(
    args: Record<string, unknown>,
    toolCallId: string | undefined
  ): Command {
    const virtualPath = args.path as string;
    const fileText = args.file_text as string;

    const fullPath = this.validateAndResolvePath(virtualPath);

    // Create parent directories
    const dir = path.dirname(fullPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    // Write file
    fs.writeFileSync(fullPath, `${fileText}\n`, "utf8");

    return new Command({
      update: {
        messages: [
          new ToolMessage({
            content: `File created: ${virtualPath}`,
            tool_call_id: toolCallId!,
            name: this.toolName,
          }),
        ],
      },
    });
  }

  protected handleStrReplace(
    args: Record<string, unknown>,
    toolCallId: string | undefined
  ): Command {
    const virtualPath = args.path as string;
    const oldStr = args.old_str as string;
    const newStr = (args.new_str as string) || "";

    const fullPath = this.validateAndResolvePath(virtualPath);

    if (!fs.existsSync(fullPath)) {
      throw new Error(`File not found: ${virtualPath}`);
    }

    // Read file
    const content = fs.readFileSync(fullPath, "utf8");

    // Replace string
    if (!content.includes(oldStr)) {
      throw new Error(`String not found in file: ${oldStr}`);
    }

    const newContent = content.replace(oldStr, newStr);

    // Write back
    fs.writeFileSync(fullPath, newContent, "utf8");

    return new Command({
      update: {
        messages: [
          new ToolMessage({
            content: `String replaced in ${virtualPath}`,
            tool_call_id: toolCallId!,
            name: this.toolName,
          }),
        ],
      },
    });
  }

  protected handleInsert(
    args: Record<string, unknown>,
    toolCallId: string | undefined
  ): Command {
    const virtualPath = args.path as string;
    const insertLine = args.insert_line as number;
    const textToInsert = args.new_str as string;

    const fullPath = this.validateAndResolvePath(virtualPath);

    if (!fs.existsSync(fullPath)) {
      throw new Error(`File not found: ${virtualPath}`);
    }

    // Read file
    const content = fs.readFileSync(fullPath, "utf8");
    let lines = content.split("\n");
    // Handle trailing newline
    let hadTrailingNewline = false;
    if (lines.length > 0 && lines[lines.length - 1] === "") {
      lines = lines.slice(0, -1);
      hadTrailingNewline = true;
    }

    const newLines = textToInsert.split("\n");

    // Insert after insert_line (0-indexed)
    const updatedLines = [
      ...lines.slice(0, insertLine),
      ...newLines,
      ...lines.slice(insertLine),
    ];

    // Write back
    let newContent = updatedLines.join("\n");
    if (hadTrailingNewline) {
      newContent += "\n";
    }
    fs.writeFileSync(fullPath, newContent, "utf8");

    return new Command({
      update: {
        messages: [
          new ToolMessage({
            content: `Text inserted in ${virtualPath}`,
            tool_call_id: toolCallId!,
            name: this.toolName,
          }),
        ],
      },
    });
  }

  protected handleDelete(
    args: Record<string, unknown>,
    toolCallId: string | undefined
  ): Command {
    const virtualPath = args.path as string;
    const fullPath = this.validateAndResolvePath(virtualPath);

    if (fs.existsSync(fullPath)) {
      const stats = fs.statSync(fullPath);
      if (stats.isFile()) {
        fs.unlinkSync(fullPath);
      } else if (stats.isDirectory()) {
        fs.rmSync(fullPath, { recursive: true });
      }
    }
    // If doesn't exist, silently succeed

    return new Command({
      update: {
        messages: [
          new ToolMessage({
            content: `File deleted: ${virtualPath}`,
            tool_call_id: toolCallId!,
            name: this.toolName,
          }),
        ],
      },
    });
  }

  protected handleRename(
    args: Record<string, unknown>,
    toolCallId: string | undefined
  ): Command {
    const oldPath = args.old_path as string;
    const newPath = args.new_path as string;

    const oldFull = this.validateAndResolvePath(oldPath);
    const newFull = this.validateAndResolvePath(newPath);

    if (!fs.existsSync(oldFull)) {
      throw new Error(`File not found: ${oldPath}`);
    }

    // Create parent directory for new path
    const dir = path.dirname(newFull);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    // Rename
    fs.renameSync(oldFull, newFull);

    return new Command({
      update: {
        messages: [
          new ToolMessage({
            content: `File renamed: ${oldPath} -> ${newPath}`,
            tool_call_id: toolCallId!,
            name: this.toolName,
          }),
        ],
      },
    });
  }
}

/**
 * Filesystem-based text editor tool middleware.
 *
 * Provides Anthropic's text_editor tool using local filesystem for storage.
 * User handles persistence via volumes, git, or other mechanisms.
 *
 * @example
 * ```ts
 * import { createAgent } from "langchain/agents";
 * import { FilesystemClaudeTextEditorMiddleware } from "langchain/agents/middleware";
 *
 * const agent = createAgent({
 *   model,
 *   tools: [],
 *   middleware: [
 *     new FilesystemClaudeTextEditorMiddleware({ rootPath: "/workspace" })
 *   ],
 * });
 * ```
 */
export class FilesystemClaudeTextEditorMiddleware extends FilesystemClaudeFileToolMiddleware {
  constructor(options: {
    rootPath: string;
    allowedPrefixes?: string[];
    maxFileSizeMb?: number;
  }) {
    super({
      toolType: TEXT_EDITOR_TOOL_TYPE,
      toolName: TEXT_EDITOR_TOOL_NAME,
      rootPath: options.rootPath,
      allowedPrefixes: options.allowedPrefixes,
      maxFileSizeMb: options.maxFileSizeMb,
    });
    this.name = "FilesystemClaudeTextEditorMiddleware";
  }
}

/**
 * Filesystem-based memory tool middleware.
 *
 * Provides Anthropic's memory tool using local filesystem for storage.
 * User handles persistence via volumes, git, or other mechanisms.
 * Enforces /memories prefix and injects Anthropic's recommended system prompt.
 *
 * @example
 * ```ts
 * import { createAgent } from "langchain/agents";
 * import { FilesystemClaudeMemoryMiddleware } from "langchain/agents/middleware";
 *
 * const agent = createAgent({
 *   model,
 *   tools: [],
 *   middleware: [
 *     new FilesystemClaudeMemoryMiddleware({ rootPath: "/workspace" })
 *   ],
 * });
 * ```
 */
export class FilesystemClaudeMemoryMiddleware extends FilesystemClaudeFileToolMiddleware {
  constructor(options: {
    rootPath: string;
    allowedPrefixes?: string[];
    maxFileSizeMb?: number;
    systemPrompt?: string;
  }) {
    super({
      toolType: MEMORY_TOOL_TYPE,
      toolName: MEMORY_TOOL_NAME,
      rootPath: options.rootPath,
      allowedPrefixes: options.allowedPrefixes || ["/memories"],
      maxFileSizeMb: options.maxFileSizeMb,
      systemPrompt:
        options.systemPrompt !== undefined
          ? options.systemPrompt
          : MEMORY_SYSTEM_PROMPT,
    });
    this.name = "FilesystemClaudeMemoryMiddleware";
  }
}
