/**
 * Anthropic text editor and memory tool middleware.
 *
 * This module provides client-side implementations of Anthropic's text editor and
 * memory tools using proper tool definitions with providerToolDefinition.
 */

import { ToolMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { Command, getCurrentTaskInput } from "@langchain/langgraph";
import * as fs from "node:fs";
import * as path from "node:path";
import { z } from "zod";
import { createMiddleware } from "../index.js";

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
 * Command schemas for file tools (text editor and memory).
 * These match Anthropic's built-in tool command structure.
 */
const ViewCommandSchema = z.object({
  command: z.literal("view"),
  path: z.string().describe("Path to the file or directory to view"),
  view_range: z
    .tuple([z.number(), z.number()])
    .optional()
    .describe(
      "Optional line range to view [start, end]. Only applies to files, not directories."
    ),
});

const CreateCommandSchema = z.object({
  command: z.literal("create"),
  path: z.string().describe("Path where the new file should be created"),
  file_text: z.string().describe("Content to write to the new file"),
});

const StrReplaceCommandSchema = z.object({
  command: z.literal("str_replace"),
  path: z.string().describe("Path to the file to modify"),
  old_str: z
    .string()
    .describe("Text to replace (must match exactly, including whitespace)"),
  new_str: z.string().describe("New text to insert in place of old text"),
});

const InsertCommandSchema = z.object({
  command: z.literal("insert"),
  path: z.string().describe("Path to the file to modify"),
  insert_line: z
    .number()
    .describe("Line number after which to insert text (0 for beginning)"),
  new_str: z.string().describe("Text to insert"),
});

const DeleteCommandSchema = z.object({
  command: z.literal("delete"),
  path: z.string().describe("Path to the file or directory to delete"),
});

const RenameCommandSchema = z.object({
  command: z.literal("rename"),
  old_path: z.string().describe("Current path of the file/directory"),
  new_path: z.string().describe("New path for the file/directory"),
});

/**
 * Discriminated union of all file tool commands.
 */
const FileToolCommandSchema = z.discriminatedUnion("command", [
  ViewCommandSchema,
  CreateCommandSchema,
  StrReplaceCommandSchema,
  InsertCommandSchema,
  DeleteCommandSchema,
  RenameCommandSchema,
]);

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
 * Helper function to create a state-based file tool handler.
 * Handles command execution and state updates for text editor and memory tools.
 */
function createStateFileToolHandler(options: {
  toolName: string;
  stateKey: "text_editor_files" | "memory_files";
  allowedPrefixes?: string[];
}) {
  return async (
    args: z.infer<typeof FileToolCommandSchema>,
    config: any
  ): Promise<Command> => {
    const state = await getCurrentTaskInput<AnthropicToolsState>(config);
    const files = (state[options.stateKey] || {}) as Record<string, FileData>;
    const toolCallId = config.toolCall?.id as string;

    try {
      switch (args.command) {
        case "view": {
          const normalizedPath = validatePath(args.path, options.allowedPrefixes);
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
                      tool_call_id: toolCallId,
                      name: options.toolName,
                    }),
                  ],
                },
              });
            }

            throw new Error(`File not found: ${args.path}`);
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
                  tool_call_id: toolCallId,
                  name: options.toolName,
                }),
              ],
            },
          });
        }

        case "create": {
          const normalizedPath = validatePath(args.path, options.allowedPrefixes);
          const existing = files[normalizedPath];

          // Create file data
          const now = new Date().toISOString();
          const createdAt = existing ? existing.created_at : now;
          const contentLines = args.file_text.split("\n");

          return new Command({
            update: {
              [options.stateKey]: {
                [normalizedPath]: {
                  content: contentLines,
                  created_at: createdAt,
                  modified_at: now,
                },
              },
              messages: [
                new ToolMessage({
                  content: `File created: ${args.path}`,
                  tool_call_id: toolCallId,
                  name: options.toolName,
                }),
              ],
            },
          });
        }

        case "str_replace": {
          const normalizedPath = validatePath(args.path, options.allowedPrefixes);
          const fileData = files[normalizedPath];
          if (!fileData) {
            throw new Error(`File not found: ${args.path}`);
          }

          const content = fileData.content.join("\n");

          // Replace string
          if (!content.includes(args.old_str)) {
            throw new Error(`String not found in file: ${args.old_str}`);
          }

          const newContent = content.replace(args.old_str, args.new_str);
          const newLines = newContent.split("\n");

          // Update file
          const now = new Date().toISOString();

          return new Command({
            update: {
              [options.stateKey]: {
                [normalizedPath]: {
                  content: newLines,
                  created_at: fileData.created_at,
                  modified_at: now,
                },
              },
              messages: [
                new ToolMessage({
                  content: `String replaced in ${args.path}`,
                  tool_call_id: toolCallId,
                  name: options.toolName,
                }),
              ],
            },
          });
        }

        case "insert": {
          const normalizedPath = validatePath(args.path, options.allowedPrefixes);
          const fileData = files[normalizedPath];
          if (!fileData) {
            throw new Error(`File not found: ${args.path}`);
          }

          const newLines = args.new_str.split("\n");

          // Insert after insert_line (0-indexed)
          const updatedLines = [
            ...fileData.content.slice(0, args.insert_line),
            ...newLines,
            ...fileData.content.slice(args.insert_line),
          ];

          // Update file
          const now = new Date().toISOString();

          return new Command({
            update: {
              [options.stateKey]: {
                [normalizedPath]: {
                  content: updatedLines,
                  created_at: fileData.created_at,
                  modified_at: now,
                },
              },
              messages: [
                new ToolMessage({
                  content: `Text inserted in ${args.path}`,
                  tool_call_id: toolCallId,
                  name: options.toolName,
                }),
              ],
            },
          });
        }

        case "delete": {
          const normalizedPath = validatePath(args.path, options.allowedPrefixes);

          return new Command({
            update: {
              [options.stateKey]: { [normalizedPath]: null },
              messages: [
                new ToolMessage({
                  content: `File deleted: ${args.path}`,
                  tool_call_id: toolCallId,
                  name: options.toolName,
                }),
              ],
            },
          });
        }

        case "rename": {
          const normalizedOld = validatePath(args.old_path, options.allowedPrefixes);
          const normalizedNew = validatePath(args.new_path, options.allowedPrefixes);

          const fileData = files[normalizedOld];
          if (!fileData) {
            throw new Error(`File not found: ${args.old_path}`);
          }

          // Update timestamp
          const now = new Date().toISOString();
          const fileDataCopy = { ...fileData, modified_at: now };

          return new Command({
            update: {
              [options.stateKey]: {
                [normalizedOld]: null,
                [normalizedNew]: fileDataCopy,
              },
              messages: [
                new ToolMessage({
                  content: `File renamed: ${args.old_path} -> ${args.new_path}`,
                  tool_call_id: toolCallId,
                  name: options.toolName,
                }),
              ],
            },
          });
        }
      }
    } catch (error) {
      return new Command({
        update: {
          messages: [
            new ToolMessage({
              content: String(error),
              tool_call_id: toolCallId,
              name: options.toolName,
              status: "error",
            }),
          ],
        },
      });
    }
  };
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
 *   middleware: [StateClaudeTextEditorMiddleware()],
 * });
 * ```
 */
export function StateClaudeTextEditorMiddleware(options?: {
  allowedPathPrefixes?: string[];
}) {
  const textEditorTool = tool(
    createStateFileToolHandler({
      toolName: TEXT_EDITOR_TOOL_NAME,
      stateKey: "text_editor_files",
      allowedPrefixes: options?.allowedPathPrefixes,
    }),
    {
      name: TEXT_EDITOR_TOOL_NAME,
      description:
        "Edit files using Anthropic's text editor tool with state-based storage",
      schema: FileToolCommandSchema,
      providerToolDefinition: {
        type: TEXT_EDITOR_TOOL_TYPE,
        name: TEXT_EDITOR_TOOL_NAME,
      },
    }
  );

  return createMiddleware({
    name: "StateClaudeTextEditorMiddleware",
    stateSchema: TextEditorStateSchema,
    tools: [textEditorTool],
  });
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
 *   middleware: [StateClaudeMemoryMiddleware()],
 * });
 * ```
 */
export function StateClaudeMemoryMiddleware(options?: {
  allowedPathPrefixes?: string[];
  systemPrompt?: string;
}) {
  const memoryTool = tool(
    createStateFileToolHandler({
      toolName: MEMORY_TOOL_NAME,
      stateKey: "memory_files",
      allowedPrefixes: options?.allowedPathPrefixes || ["/memories"],
    }),
    {
      name: MEMORY_TOOL_NAME,
      description:
        "Store and retrieve information across conversations using Anthropic's memory tool",
      schema: FileToolCommandSchema,
      providerToolDefinition: {
        type: MEMORY_TOOL_TYPE,
        name: MEMORY_TOOL_NAME,
      },
    }
  );

  const systemPrompt =
    options?.systemPrompt !== undefined
      ? options.systemPrompt
      : MEMORY_SYSTEM_PROMPT;

  return createMiddleware({
    name: "StateClaudeMemoryMiddleware",
    stateSchema: MemoryStateSchema,
    tools: [memoryTool],
    wrapModelCall: systemPrompt
      ? (request, handler) =>
          handler({
            ...request,
            systemPrompt:
              (request.systemPrompt ? `${request.systemPrompt}\n\n` : "") +
              systemPrompt,
          })
      : undefined,
  });
}

/**
 * Helper function to validate and resolve a virtual path to a filesystem path.
 */
function validateAndResolvePath(
  virtualPath: string,
  rootPath: string,
  allowedPrefixes: string[]
): string {
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
  const fullPath = path.resolve(rootPath, relative);

  // Ensure path is within root
  if (!fullPath.startsWith(rootPath)) {
    throw new Error(`Path outside root directory: ${virtualPath}`);
  }

  // Check allowed prefixes
  const virtualForCheck = `/${path
    .relative(rootPath, fullPath)
    .replace(/\\/g, "/")}`;
  const allowed = allowedPrefixes.some(
    (prefix) =>
      virtualForCheck.startsWith(prefix) ||
      virtualForCheck === prefix.replace(/\/$/, "")
  );
  if (!allowed) {
    throw new Error(
      `Path must start with one of: ${JSON.stringify(allowedPrefixes)}`
    );
  }

  return fullPath;
}

/**
 * Helper function to create a filesystem-based file tool handler.
 * Handles command execution with actual filesystem operations.
 */
function createFilesystemFileToolHandler(options: {
  toolName: string;
  rootPath: string;
  allowedPrefixes: string[];
  maxFileSizeBytes: number;
}) {
  // Create root directory if it doesn't exist
  if (!fs.existsSync(options.rootPath)) {
    fs.mkdirSync(options.rootPath, { recursive: true });
  }

  return async (
    args: z.infer<typeof FileToolCommandSchema>,
    config: any
  ): Promise<Command> => {
    const toolCallId = config.toolCall?.id as string;

    try {
      switch (args.command) {
        case "view": {
          const fullPath = validateAndResolvePath(
            args.path,
            options.rootPath,
            options.allowedPrefixes
          );

          if (!fs.existsSync(fullPath) || !fs.statSync(fullPath).isFile()) {
            throw new Error(`File not found: ${args.path}`);
          }

          // Check file size
          const stats = fs.statSync(fullPath);
          if (stats.size > options.maxFileSizeBytes) {
            const maxMb = options.maxFileSizeBytes / 1024 / 1024;
            throw new Error(`File too large: ${args.path} exceeds ${maxMb}MB`);
          }

          // Read file
          let content: string;
          try {
            content = fs.readFileSync(fullPath, "utf8");
          } catch (error) {
            throw new Error(`Cannot decode file ${args.path}: ${error}`);
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
                  tool_call_id: toolCallId,
                  name: options.toolName,
                }),
              ],
            },
          });
        }

        case "create": {
          const fullPath = validateAndResolvePath(
            args.path,
            options.rootPath,
            options.allowedPrefixes
          );

          // Create parent directories
          const dir = path.dirname(fullPath);
          if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
          }

          // Write file
          fs.writeFileSync(fullPath, `${args.file_text}\n`, "utf8");

          return new Command({
            update: {
              messages: [
                new ToolMessage({
                  content: `File created: ${args.path}`,
                  tool_call_id: toolCallId,
                  name: options.toolName,
                }),
              ],
            },
          });
        }

        case "str_replace": {
          const fullPath = validateAndResolvePath(
            args.path,
            options.rootPath,
            options.allowedPrefixes
          );

          if (!fs.existsSync(fullPath)) {
            throw new Error(`File not found: ${args.path}`);
          }

          // Read file
          const content = fs.readFileSync(fullPath, "utf8");

          // Replace string
          if (!content.includes(args.old_str)) {
            throw new Error(`String not found in file: ${args.old_str}`);
          }

          const newContent = content.replace(args.old_str, args.new_str);

          // Write back
          fs.writeFileSync(fullPath, newContent, "utf8");

          return new Command({
            update: {
              messages: [
                new ToolMessage({
                  content: `String replaced in ${args.path}`,
                  tool_call_id: toolCallId,
                  name: options.toolName,
                }),
              ],
            },
          });
        }

        case "insert": {
          const fullPath = validateAndResolvePath(
            args.path,
            options.rootPath,
            options.allowedPrefixes
          );

          if (!fs.existsSync(fullPath)) {
            throw new Error(`File not found: ${args.path}`);
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

          const newLines = args.new_str.split("\n");

          // Insert after insert_line (0-indexed)
          const updatedLines = [
            ...lines.slice(0, args.insert_line),
            ...newLines,
            ...lines.slice(args.insert_line),
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
                  content: `Text inserted in ${args.path}`,
                  tool_call_id: toolCallId,
                  name: options.toolName,
                }),
              ],
            },
          });
        }

        case "delete": {
          const fullPath = validateAndResolvePath(
            args.path,
            options.rootPath,
            options.allowedPrefixes
          );

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
                  content: `File deleted: ${args.path}`,
                  tool_call_id: toolCallId,
                  name: options.toolName,
                }),
              ],
            },
          });
        }

        case "rename": {
          const oldFull = validateAndResolvePath(
            args.old_path,
            options.rootPath,
            options.allowedPrefixes
          );
          const newFull = validateAndResolvePath(
            args.new_path,
            options.rootPath,
            options.allowedPrefixes
          );

          if (!fs.existsSync(oldFull)) {
            throw new Error(`File not found: ${args.old_path}`);
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
                  content: `File renamed: ${args.old_path} -> ${args.new_path}`,
                  tool_call_id: toolCallId,
                  name: options.toolName,
                }),
              ],
            },
          });
        }
      }
    } catch (error) {
      return new Command({
        update: {
          messages: [
            new ToolMessage({
              content: String(error),
              tool_call_id: toolCallId,
              name: options.toolName,
              status: "error",
            }),
          ],
        },
      });
    }
  };
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
 *     FilesystemClaudeTextEditorMiddleware({ rootPath: "/workspace" })
 *   ],
 * });
 * ```
 */
export function FilesystemClaudeTextEditorMiddleware(options: {
  rootPath: string;
  allowedPrefixes?: string[];
  maxFileSizeMb?: number;
}) {
  const resolvedRootPath = path.resolve(options.rootPath);
  const maxFileSizeBytes = (options.maxFileSizeMb || 10) * 1024 * 1024;
  const allowedPrefixes = options.allowedPrefixes || ["/"];

  const textEditorTool = tool(
    createFilesystemFileToolHandler({
      toolName: TEXT_EDITOR_TOOL_NAME,
      rootPath: resolvedRootPath,
      allowedPrefixes,
      maxFileSizeBytes,
    }),
    {
      name: TEXT_EDITOR_TOOL_NAME,
      description:
        "Edit files using Anthropic's text editor tool with filesystem-based storage",
      schema: FileToolCommandSchema,
      providerToolDefinition: {
        type: TEXT_EDITOR_TOOL_TYPE,
        name: TEXT_EDITOR_TOOL_NAME,
      },
    }
  );

  return createMiddleware({
    name: "FilesystemClaudeTextEditorMiddleware",
    tools: [textEditorTool],
  });
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
 *     FilesystemClaudeMemoryMiddleware({ rootPath: "/workspace" })
 *   ],
 * });
 * ```
 */
export function FilesystemClaudeMemoryMiddleware(options: {
  rootPath: string;
  allowedPrefixes?: string[];
  maxFileSizeMb?: number;
  systemPrompt?: string;
}) {
  const resolvedRootPath = path.resolve(options.rootPath);
  const maxFileSizeBytes = (options.maxFileSizeMb || 10) * 1024 * 1024;
  const allowedPrefixes = options.allowedPrefixes || ["/memories"];

  const memoryTool = tool(
    createFilesystemFileToolHandler({
      toolName: MEMORY_TOOL_NAME,
      rootPath: resolvedRootPath,
      allowedPrefixes,
      maxFileSizeBytes,
    }),
    {
      name: MEMORY_TOOL_NAME,
      description:
        "Store and retrieve information across conversations using Anthropic's memory tool with filesystem-based storage",
      schema: FileToolCommandSchema,
      providerToolDefinition: {
        type: MEMORY_TOOL_TYPE,
        name: MEMORY_TOOL_NAME,
      },
    }
  );

  const systemPrompt =
    options.systemPrompt !== undefined
      ? options.systemPrompt
      : MEMORY_SYSTEM_PROMPT;

  return createMiddleware({
    name: "FilesystemClaudeMemoryMiddleware",
    tools: [memoryTool],
    wrapModelCall: systemPrompt
      ? (request, handler) =>
          handler({
            ...request,
            systemPrompt:
              (request.systemPrompt ? `${request.systemPrompt}\n\n` : "") +
              systemPrompt,
          })
      : undefined,
  });
}
