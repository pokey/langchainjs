/**
 * Anthropic text editor and memory tool middleware.
 *
 * This module provides client-side implementations of Anthropic's text editor and
 * memory tools using proper tool definitions with providerToolDefinition.
 */

import { ToolMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { Command, getCurrentTaskInput } from "@langchain/langgraph";
import * as path from "node:path";
import { z } from "zod";
import { createMiddleware } from "../../index.js";
import { CommandHandler } from "./CommandHandler.js";
import { FileData, FileDataSchema } from "./FileData.js";
import { PhysicalFileSystem } from "./PhysicalFileSystem.js";
import { StateFileSystem } from "./StateFileSystem.js";
import {
  TextEditorCommandSchema,
  MemoryCommandSchema,
} from "./anthropicCommandSchemas.js";
import { ModelRequest } from "../../nodes/types.js";
import { AgentBuiltInState } from "../../runtime.js";

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
 * Custom reducer that merges file updates.
 * @param left - Existing files dict
 * @param right - New files dict to merge (null values delete files)
 * @returns Merged dict where right overwrites left for matching keys
 */
export function filesReducer(
  left: Record<string, FileData>,
  right: Record<string, FileData | null>
): Record<string, FileData> {
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
 * Re-export FileData type for external use
 */
export type { FileData } from "./FileData.js";

/**
 * Zod state schemas for middleware with registered reducers.
 * Uses withLangGraph to attach filesReducer (JavaScript equivalent of Python's Annotated).
 * The reducer handles file updates and deletions (null values remove files).
 */
const TextEditorStateSchema = z.object({
  text_editor_files: z.record(z.string(), FileDataSchema).default(() => ({})),
});

const MemoryStateSchema = z.object({
  memory_files: z.record(z.string(), FileDataSchema).default(() => ({})),
});

/**
 * Handle text editor tool commands.
 * Supports: view, create, str_replace, insert
 */
async function handleTextEditorCommand(
  commandHandler: CommandHandler,
  args: z.infer<typeof TextEditorCommandSchema>
): Promise<string> {
  switch (args.command) {
    case "view":
      return commandHandler.handleViewCommand(args.path);

    case "create":
      return commandHandler.handleCreateCommand(args.path, args.file_text);

    case "str_replace":
      return commandHandler.handleStrReplaceCommand(
        args.path,
        args.old_str,
        args.new_str
      );

    case "insert":
      return commandHandler.handleInsertCommand(
        args.path,
        args.insert_line,
        args.new_str
      );

    default:
      throw new Error(
        `Unknown command: ${(args as { command?: string }).command}`
      );
  }
}

/**
 * Handle memory tool commands.
 * Supports: view, create, str_replace, insert, delete, rename
 */
async function handleMemoryCommand(
  commandHandler: CommandHandler,
  args: z.infer<typeof MemoryCommandSchema>
): Promise<string> {
  switch (args.command) {
    case "view":
      return commandHandler.handleViewCommand(args.path);

    case "create":
      return commandHandler.handleCreateCommand(args.path, args.file_text);

    case "str_replace":
      return commandHandler.handleStrReplaceCommand(
        args.path,
        args.old_str,
        args.new_str
      );

    case "insert":
      return commandHandler.handleInsertCommand(
        args.path,
        args.insert_line,
        args.insert_text
      );

    case "delete":
      return commandHandler.handleDeleteCommand(args.path);

    case "rename":
      return commandHandler.handleRenameCommand(args.old_path, args.new_path);

    default:
      throw new Error(
        `Unknown command: ${(args as { command?: string }).command}`
      );
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
 * import { createStateClaudeTextEditorMiddleware } from "langchain/agents/middleware";
 *
 * const agent = createAgent({
 *   model,
 *   tools: [],
 *   middleware: [createStateClaudeTextEditorMiddleware()],
 * });
 * ```
 */
export function createStateClaudeTextEditorMiddleware(options?: {
  allowedPathPrefixes?: string[];
}) {
  const allowedPrefixes = options?.allowedPathPrefixes || ["/"];

  return createMiddleware({
    name: "StateClaudeTextEditorMiddleware",
    stateSchema: TextEditorStateSchema,
    tools: [
      tool(
        async (args, c) => {
          const state =
            getCurrentTaskInput<z.infer<typeof TextEditorStateSchema>>(c);
          try {
            let files = state.text_editor_files;
            const fileSystem = new StateFileSystem(
              state.text_editor_files,
              allowedPrefixes,
              (update) => {
                files = filesReducer(files, update);
              }
            );
            const commandHandler = new CommandHandler(fileSystem);
            const message = await handleTextEditorCommand(commandHandler, args);

            return new Command({
              update: {
                messages: [
                  new ToolMessage({
                    content: message,
                    tool_call_id: c.toolCall?.id,
                    name: TEXT_EDITOR_TOOL_NAME,
                  }),
                ],
                ...(files === state.text_editor_files
                  ? {}
                  : { text_editor_files: files }),
              },
            });
          } catch (error) {
            return new ToolMessage({
              content: String(error),
              tool_call_id: c.toolCall?.id,
              name: TEXT_EDITOR_TOOL_NAME,
              status: "error",
            });
          }
        },
        {
          name: TEXT_EDITOR_TOOL_NAME,
          description:
            "Anthropic text editor tool (client-side implementation)",
          schema: TextEditorCommandSchema,
          providerToolDefinition: {
            type: TEXT_EDITOR_TOOL_TYPE,
            name: TEXT_EDITOR_TOOL_NAME,
          },
        }
      ),
    ],
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
 * import { createStateClaudeMemoryMiddleware } from "langchain/agents/middleware";
 *
 * const agent = createAgent({
 *   model,
 *   tools: [],
 *   middleware: [createStateClaudeMemoryMiddleware()],
 * });
 * ```
 */
export function createStateClaudeMemoryMiddleware(options?: {
  allowedPathPrefixes?: string[];
  systemPrompt?: string;
}) {
  const allowedPrefixes = options?.allowedPathPrefixes || ["/memories"];
  const systemPrompt =
    options?.systemPrompt !== undefined
      ? options.systemPrompt
      : MEMORY_SYSTEM_PROMPT;

  return createMiddleware({
    name: "StateClaudeMemoryMiddleware",
    stateSchema: MemoryStateSchema,
    tools: [
      tool(
        async (args, c) => {
          const state =
            getCurrentTaskInput<z.infer<typeof MemoryStateSchema>>(c);
          try {
            const updates: Record<string, FileData | null> = {};
            const fileSystem = new StateFileSystem(
              state.memory_files,
              allowedPrefixes,
              (files) => {
                Object.assign(updates, files);
              }
            );
            const commandHandler = new CommandHandler(fileSystem);
            const message = await handleMemoryCommand(commandHandler, args);

            return new Command({
              update: {
                messages: [
                  new ToolMessage({
                    content: message,
                    tool_call_id: c.toolCall?.id,
                    name: MEMORY_TOOL_NAME,
                  }),
                ],
                memory_files: filesReducer(state.memory_files, updates),
              },
            });
          } catch (error) {
            return new ToolMessage({
              content: String(error),
              tool_call_id: c.toolCall?.id,
              name: MEMORY_TOOL_NAME,
              status: "error",
            });
          }
        },
        {
          name: MEMORY_TOOL_NAME,
          description: "Anthropic memory tool (client-side implementation)",
          schema: MemoryCommandSchema,
          providerToolDefinition: {
            type: MEMORY_TOOL_TYPE,
            name: MEMORY_TOOL_NAME,
          },
        }
      ),
    ],
    wrapModelCall: async (request, handler) => {
      return handler(updateMemoryRequest(systemPrompt, request));
    },
  });
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
 * import { createFilesystemClaudeTextEditorMiddleware } from "langchain/agents/middleware";
 *
 * const agent = createAgent({
 *   model,
 *   tools: [],
 *   middleware: [
 *     createFilesystemClaudeTextEditorMiddleware({ rootPath: "/workspace" })
 *   ],
 * });
 * ```
 */
export function createFilesystemClaudeTextEditorMiddleware(options: {
  rootPath: string;
  allowedPrefixes?: string[];
  maxFileSizeMb?: number;
}) {
  const resolvedRootPath = path.resolve(options.rootPath);
  const maxFileSizeMb = options.maxFileSizeMb || 10;
  const allowedPrefixes = options.allowedPrefixes || ["/"];

  const fileSystem = new PhysicalFileSystem(
    resolvedRootPath,
    allowedPrefixes,
    maxFileSizeMb
  );

  return createMiddleware({
    name: "FilesystemClaudeTextEditorMiddleware",
    stateSchema: undefined,
    tools: [
      tool(
        async (args, c) => {
          try {
            const commandHandler = new CommandHandler(fileSystem);
            const message = await handleTextEditorCommand(commandHandler, args);

            return new Command({
              update: {
                messages: [
                  new ToolMessage({
                    content: message,
                    tool_call_id: c.toolCall?.id,
                    name: TEXT_EDITOR_TOOL_NAME,
                  }),
                ],
              },
            });
          } catch (error) {
            return new ToolMessage({
              content: String(error),
              tool_call_id: c.toolCall?.id,
              name: TEXT_EDITOR_TOOL_NAME,
              status: "error",
            });
          }
        },
        {
          name: TEXT_EDITOR_TOOL_NAME,
          description: "Anthropic text editor tool (filesystem-based)",
          schema: TextEditorCommandSchema,
          providerToolDefinition: {
            type: TEXT_EDITOR_TOOL_TYPE,
            name: TEXT_EDITOR_TOOL_NAME,
          },
        }
      ),
    ],
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
 * import { createFilesystemClaudeMemoryMiddleware } from "langchain/agents/middleware";
 *
 * const agent = createAgent({
 *   model,
 *   tools: [],
 *   middleware: [
 *     createFilesystemClaudeMemoryMiddleware({ rootPath: "/workspace" })
 *   ],
 * });
 * ```
 */
export function createFilesystemClaudeMemoryMiddleware(options: {
  rootPath: string;
  allowedPrefixes?: string[];
  maxFileSizeMb?: number;
  systemPrompt?: string;
}) {
  const resolvedRootPath = path.resolve(options.rootPath);
  const maxFileSizeMb = options.maxFileSizeMb || 10;
  const allowedPrefixes = options.allowedPrefixes || ["/memories"];
  const systemPrompt =
    options.systemPrompt !== undefined
      ? options.systemPrompt
      : MEMORY_SYSTEM_PROMPT;

  const fileSystem = new PhysicalFileSystem(
    resolvedRootPath,
    allowedPrefixes,
    maxFileSizeMb
  );

  return createMiddleware({
    name: "FilesystemClaudeMemoryMiddleware",
    stateSchema: undefined,
    tools: [
      tool(
        async (args, c) => {
          try {
            const commandHandler = new CommandHandler(fileSystem);
            const message = await handleMemoryCommand(commandHandler, args);

            return new Command({
              update: {
                messages: [
                  new ToolMessage({
                    content: message,
                    tool_call_id: c.toolCall?.id,
                    name: MEMORY_TOOL_NAME,
                  }),
                ],
              },
            });
          } catch (error) {
            return new ToolMessage({
              content: String(error),
              tool_call_id: c.toolCall?.id,
              name: MEMORY_TOOL_NAME,
              status: "error",
            });
          }
        },
        {
          name: MEMORY_TOOL_NAME,
          description: "Anthropic memory tool (filesystem-based)",
          schema: MemoryCommandSchema,
          providerToolDefinition: {
            type: MEMORY_TOOL_TYPE,
            name: MEMORY_TOOL_NAME,
          },
        }
      ),
    ],
    wrapModelCall: async (request, handler) => {
      return handler(updateMemoryRequest(systemPrompt, request));
    },
  });
}

/**
 * Update memory request with the system prompt and headers.
 *
 * @param systemPrompt The system prompt to inject
 * @param request The request to modify
 * @returns Modified request
 */
function updateMemoryRequest(
  systemPrompt: string,
  request: ModelRequest<AgentBuiltInState, never>
): ModelRequest<AgentBuiltInState, never> {
  return {
    ...request,

    // Inject system prompt if provided
    systemPrompt: systemPrompt
      ? request.systemPrompt
        ? `${request.systemPrompt}\n\n${systemPrompt}`
        : systemPrompt
      : request.systemPrompt,

    modelSettings: {
      ...request.modelSettings,
      headers: {
        ...(request.modelSettings?.headers || {}),
        "anthropic-beta": "context-management-2025-06-27",
      },
    },
  };
}
