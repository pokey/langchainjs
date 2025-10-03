import { AIMessage } from "@langchain/core/messages";
import { z } from "zod";
import { createMiddleware } from "../middleware.js";
import { ModelRequest } from "../types.js";

/**
 * State schema for Anthropic code execution middleware.
 * Tracks the container ID to enable file persistence across conversation turns.
 */
const stateSchema = z.object({
  container: z
    .object({
      id: z.string().describe("Anthropic container ID"),
      expiresAt: z.date().describe("Container expiration"),
    })
    .optional(),
});

export type AnthropicCodeExecutionMiddlewareState = z.infer<typeof stateSchema>;

/**
 * Creates middleware for managing Anthropic's code execution tool.
 *
 * This middleware streamlines data analysis workflows by:
 * - Enabling Anthropic's code execution tool
 * - Automatically managing container IDs for Anthropic (enables file persistence across turns)
 * - Injecting required beta headers for code execution and files API
 * - Working around Anthropic API bugs with file references in conversation history
 *
 * Works with:
 * - Anthropic's `code_execution_20250825` tool with Files API
 *
 * @example
 * ```typescript
 * import { ChatAnthropic } from "@langchain/anthropic";
 * import { HumanMessage } from "@langchain/core/messages";
 * import { createAgent } from "langchain/agents";
 * import { anthropicCodeExecutionMiddleware } from "langchain/agents/middleware";
 * import {
 *   uploadFileAnthropic,
 *   extractGeneratedFilesAnthropic,
 *   getFileMetadataAnthropic,
 *   downloadFileAnthropic,
 * } from "langchain/agents/middleware";
 *
 * const model = new ChatAnthropic({
 *   model: "claude-sonnet-4-20250514",
 * });
 *
 * const agent = createAgent({
 *   model,
 *   middleware: [anthropicCodeExecutionMiddleware()],
 * });
 *
 * // Upload file using helper
 * const client = model.createClient({});
 * const uploadedFile = await uploadFileAnthropic(client, "data.csv");
 *
 * // Analyze data
 * const result = await agent.invoke({
 *   messages: new HumanMessage({
 *     content: [
 *       { type: "text", text: "Analyze this data" },
 *       { type: "container_upload", file_id: uploadedFile.fileId },
 *     ],
 *   }),
 * });
 *
 * // Extract and download generated files
 * const fileIds = extractGeneratedFilesAnthropic(result);
 * for (const fileId of fileIds) {
 *   const metadata = await getFileMetadataAnthropic(client, fileId);
 *   await downloadFileAnthropic(client, fileId, metadata.filename);
 * }
 * ```
 *
 * @returns A configured middleware instance
 *
 * @see {@link AnthropicCodeExecutionMiddlewareState} for the state schema
 * @see {@link uploadFileAnthropic} for uploading files to Anthropic
 * @see {@link extractGeneratedFilesAnthropic} for extracting generated file IDs from responses
 * @see {@link getFileMetadataAnthropic} for retrieving file metadata including filenames
 * @see {@link downloadFileAnthropic} for downloading generated files from Anthropic
 */
export function anthropicCodeExecutionMiddleware() {
  return createMiddleware({
    name: "anthropicCodeExecutionMiddleware",
    stateSchema,
    tools: [{ type: "code_execution_20250825", name: "code_execution" }],
    modifyModelRequest: (request, state) => ({
      ...request,
      messages: workAroundAnthropicCodeExecutionBug(request.messages),
      callOptions: {
        // Pass container ID to reuse files across turns
        // FIXME: What to do about container expiration?
        container: state.container?.id,

        // Automatically inject required beta headers for Anthropic code execution
        headers: {
          "anthropic-beta": "code-execution-2025-08-25,files-api-2025-04-14",
        },
      },
    }),
    afterModel: (state) => {
      const stateUpdate: Partial<AnthropicCodeExecutionMiddlewareState> = {};

      const newContainer = state.messages.find(
        (message) =>
          message.type === "ai" && message.additional_kwargs?.container
      )?.additional_kwargs?.container as Container | undefined;

      if (newContainer && newContainer.id !== state.container?.id) {
        // Persist the container ID and expiration in state
        stateUpdate.container = {
          id: newContainer.id,
          expiresAt: new Date(newContainer.expires_at),
        };
      }

      return stateUpdate;
    },
  });
}

interface Container {
  id: string;
  expires_at: string;
}

/**
 * Work around a bug in the Anthropic API where we get a 500 when we send code execution
 * results that include bash_code_execution_output blocks.
 *
 * @param messages The messages to filter
 * @returns The filtered messages
 */
function workAroundAnthropicCodeExecutionBug(
  messages: ModelRequest["messages"]
): ModelRequest["messages"] {
  return messages.map((message) => {
    if (message.type === "ai" && Array.isArray(message.content)) {
      const filteredContent = message.content.map((block) => {
        if (isBashCodeExecutionToolResult(block)) {
          // Only filter if content is a result block (not an error)
          if (block.content.type === "bash_code_execution_result") {
            return {
              ...block,
              content: {
                ...block.content,
                content: block.content.content.filter(
                  (c) => c.type !== "bash_code_execution_output"
                ),
              },
            };
          }
        }
        return block;
      });

      return new AIMessage({
        ...message,
        content: filteredContent,
      });
    }
    return message;
  });
}

/**
 * Type guard to check if a block is a bash code execution tool result.
 */
function isBashCodeExecutionToolResult(
  block: unknown
): block is BashCodeExecutionToolResultBlock {
  return (
    typeof block === "object" &&
    block !== null &&
    "type" in block &&
    block.type === "bash_code_execution_tool_result" &&
    "content" in block
  );
}

/**
 * Anthropic code execution content block types.
 * These match the types from @anthropic-ai/sdk BetaBashCodeExecution* interfaces.
 */
interface BashCodeExecutionOutputBlock {
  file_id: string;
  type: "bash_code_execution_output";
}

interface BashCodeExecutionResultBlock {
  content: Array<BashCodeExecutionOutputBlock>;
  return_code: number;
  stderr: string;
  stdout: string;
  type: "bash_code_execution_result";
}

interface BashCodeExecutionToolResultError {
  error_code:
    | "invalid_tool_input"
    | "unavailable"
    | "too_many_requests"
    | "execution_time_exceeded"
    | "output_file_too_large";
  type: "bash_code_execution_tool_result_error";
}

interface BashCodeExecutionToolResultBlock {
  content: BashCodeExecutionToolResultError | BashCodeExecutionResultBlock;
  tool_use_id: string;
  type: "bash_code_execution_tool_result";
}
