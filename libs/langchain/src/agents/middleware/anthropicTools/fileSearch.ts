/**
 * File search middleware for Anthropic text editor and memory tools.
 *
 * This module provides Glob and Grep search tools that operate on files stored
 * in state or filesystem.
 */

import * as path from "node:path";
import * as fs from "node:fs";
import { exec } from "node:child_process";
import { promisify } from "node:util";
import { tool } from "@langchain/core/tools";
import { ToolMessage } from "@langchain/core/messages";
import { z } from "zod/v3";

import type { AgentMiddleware, WrapToolCallHook } from "../types.js";
import type { AnthropicToolsState } from "./index.js";
import type { FileData } from "./FileData.js";
import type { ClientTool } from "../../tools.js";

const execAsync = promisify(exec);

/**
 * Expand brace patterns like `*.{py,pyi}` into a list of globs.
 */
function expandIncludePatterns(pattern: string): string[] | null {
  if (pattern.includes("}") && !pattern.includes("{")) {
    return null;
  }

  const expanded: string[] = [];

  function expand(current: string): void {
    const start = current.indexOf("{");
    if (start === -1) {
      expanded.push(current);
      return;
    }

    const end = current.indexOf("}", start);
    if (end === -1) {
      throw new Error("Invalid pattern");
    }

    const prefix = current.slice(0, start);
    const suffix = current.slice(end + 1);
    const inner = current.slice(start + 1, end);
    if (!inner) {
      throw new Error("Invalid pattern");
    }

    for (const option of inner.split(",")) {
      expand(prefix + option + suffix);
    }
  }

  try {
    expand(pattern);
  } catch {
    return null;
  }

  return expanded;
}

/**
 * Validate glob pattern used for include filters.
 */
function isValidIncludePattern(pattern: string): boolean {
  if (!pattern) {
    return false;
  }

  if (
    pattern.includes("\x00") ||
    pattern.includes("\n") ||
    pattern.includes("\r")
  ) {
    return false;
  }

  const expanded = expandIncludePatterns(pattern);
  if (expanded === null) {
    return false;
  }

  // Convert glob to regex to validate
  try {
    for (const candidate of expanded) {
      // Simple glob to regex conversion for validation
      const regexPattern = candidate
        .replace(/\./g, "\\.")
        .replace(/\*/g, ".*")
        .replace(/\?/g, ".");
      new RegExp(`^${regexPattern}$`);
    }
  } catch {
    return false;
  }

  return true;
}

/**
 * Return true if the basename matches the include pattern.
 */
function matchIncludePattern(basename: string, pattern: string): boolean {
  const expanded = expandIncludePatterns(pattern);
  if (!expanded) {
    return false;
  }

  return expanded.some((candidate) => {
    // Convert glob to regex
    const regexPattern = candidate
      .replace(/\./g, "\\.")
      .replace(/\*/g, ".*")
      .replace(/\?/g, ".");
    const regex = new RegExp(`^${regexPattern}$`);
    return regex.test(basename);
  });
}

/**
 * Match a path against a glob pattern (supports ** for recursive matching).
 */
function matchGlobPattern(filePath: string, pattern: string): boolean {
  // Convert glob pattern to regex
  let regexPattern = pattern
    .replace(/\./g, "\\.") // Escape dots
    .replace(/\*\*/g, "§§") // Temporarily replace ** with placeholder
    .replace(/\*/g, "[^/]*") // Replace * with non-slash characters
    .replace(/§§/g, ".*") // Replace ** with any characters including /
    .replace(/\?/g, "."); // Replace ? with single character

  // Make it match the full path
  regexPattern = `^${regexPattern}$`;

  const regex = new RegExp(regexPattern);
  return regex.test(filePath);
}

// Define tool schemas
const globSearchSchema = z.object({
  pattern: z.string().describe("The glob pattern to match files against"),
  path: z.string().default("/").describe("The directory to search in"),
});

const grepSearchSchema = z.object({
  pattern: z
    .string()
    .describe("The regular expression pattern to search for in file contents"),
  path: z.string().default("/").describe("The directory to search in"),
  include: z
    .string()
    .optional()
    .describe('File pattern to filter (e.g., "*.js", "*.{ts,tsx}")'),
  output_mode: z
    .enum(["files_with_matches", "content", "count"])
    .default("files_with_matches")
    .describe(
      'Output format: "files_with_matches" (default), "content", or "count"'
    ),
});

/**
 * Provides Glob and Grep search over state-based files.
 *
 * This middleware adds two tools that search through virtual files in state:
 * - Glob: Fast file pattern matching by file path
 * - Grep: Fast content search using regular expressions
 *
 * @example
 * ```ts
 * import { createAgent } from "langchain/agents";
 * import {
 *   StateClaudeTextEditorMiddleware,
 *   StateFileSearchMiddleware,
 * } from "langchain/agents/middleware";
 *
 * const agent = createAgent({
 *   model,
 *   tools: [],
 *   middleware: [
 *     new StateClaudeTextEditorMiddleware(),
 *     new StateFileSearchMiddleware(),
 *   ],
 * });
 * ```
 */
export class StateFileSearchMiddleware implements AgentMiddleware {
  name = "StateFileSearchMiddleware";

  tools: ClientTool[];

  protected stateKey: string;

  constructor(options?: { stateKey?: string }) {
    this.stateKey = options?.stateKey || "text_editor_files";

    // Create tool definitions
    const globSearchTool = tool(
      // This is a placeholder - actual execution happens in wrapToolCall
      () => "placeholder",
      {
        name: "glob_search",
        description: `Fast file pattern matching tool that works with any codebase size.

Supports glob patterns like **/*.js or src/**/*.ts.
Returns matching file paths sorted by modification time.
Use this tool when you need to find files by name patterns.`,
        schema: globSearchSchema,
      }
    );

    const grepSearchTool = tool(
      // This is a placeholder - actual execution happens in wrapToolCall
      () => "placeholder",
      {
        name: "grep_search",
        description: `Fast content search tool that works with any codebase size.

Searches file contents using regular expressions. Supports full regex
syntax and filters files by pattern with the include parameter.`,
        schema: grepSearchSchema,
      }
    );

    this.tools = [globSearchTool, grepSearchTool];
  }

  wrapToolCall: WrapToolCallHook = async (request, handler) => {
    const toolCall = request.toolCall;
    const toolName = toolCall.name;

    if (toolName === "glob_search") {
      return this.handleGlobSearch(request);
    }

    if (toolName === "grep_search") {
      return this.handleGrepSearch(request);
    }

    return handler(request);
  };

  protected handleGlobSearch(
    request: import("../types.js").ToolCallRequest
  ): ToolMessage {
    const args = (request.toolCall.args || {}) as z.infer<
      typeof globSearchSchema
    >;
    const pattern = args.pattern;
    const basePath = args.path.startsWith("/") ? args.path : `/${args.path}`;

    // Get files from state
    const state = request.state as AnthropicToolsState;
    const files =
      (state[this.stateKey as keyof AnthropicToolsState] as Record<
        string,
        FileData
      >) || {};

    // Match files
    const matches: Array<[string, string]> = [];
    for (const [filePath, fileData] of Object.entries(files)) {
      if (filePath.startsWith(basePath)) {
        // Get relative path from base
        let relative: string;
        if (basePath === "/") {
          relative = filePath.slice(1); // Remove leading /
        } else if (filePath === basePath) {
          relative = path.basename(filePath);
        } else if (filePath.startsWith(`${basePath}/`)) {
          relative = filePath.slice(basePath.length + 1);
        } else {
          continue;
        }

        // Match against pattern
        let isMatch = matchGlobPattern(relative, pattern);
        // Handle ** pattern which requires special care
        if (!isMatch && pattern.startsWith("**/")) {
          // Also try matching without the **/ prefix for files in base dir
          isMatch = matchGlobPattern(relative, pattern.slice(3));
        }

        if (isMatch) {
          matches.push([filePath, fileData.modified_at]);
        }
      }
    }

    if (matches.length === 0) {
      return new ToolMessage({
        content: "No files found",
        tool_call_id: request.toolCall.id!,
        name: "glob_search",
      });
    }

    // Sort by modification time (most recent first)
    matches.sort((a, b) => b[1].localeCompare(a[1]));
    const filePaths = matches.map(([p]) => p);

    return new ToolMessage({
      content: filePaths.join("\n"),
      tool_call_id: request.toolCall.id!,
      name: "glob_search",
    });
  }

  protected handleGrepSearch(
    request: import("../types.js").ToolCallRequest
  ): ToolMessage {
    const args = (request.toolCall.args || {}) as z.infer<
      typeof grepSearchSchema
    >;
    const pattern = args.pattern;
    const basePath = args.path.startsWith("/") ? args.path : `/${args.path}`;
    const include = args.include;
    const outputMode = args.output_mode || "files_with_matches";

    // Compile regex pattern (for validation)
    let regex: RegExp;
    try {
      regex = new RegExp(pattern);
    } catch (error) {
      return new ToolMessage({
        content: `Invalid regex pattern: ${error}`,
        tool_call_id: request.toolCall.id!,
        name: "grep_search",
      });
    }

    if (include && !isValidIncludePattern(include)) {
      return new ToolMessage({
        content: "Invalid include pattern",
        tool_call_id: request.toolCall.id!,
        name: "grep_search",
      });
    }

    // Search files
    const state = request.state as AnthropicToolsState;
    const files =
      (state[this.stateKey as keyof AnthropicToolsState] as Record<
        string,
        FileData
      >) || {};
    const results: Record<string, Array<[number, string]>> = {};

    for (const [filePath, fileData] of Object.entries(files)) {
      if (!filePath.startsWith(basePath)) {
        continue;
      }

      // Check include filter
      if (include) {
        const basename = path.basename(filePath);
        if (!matchIncludePattern(basename, include)) {
          continue;
        }
      }

      // Search file content
      for (let lineNum = 0; lineNum < fileData.content.length; lineNum++) {
        const line = fileData.content[lineNum];
        if (regex.test(line)) {
          if (!results[filePath]) {
            results[filePath] = [];
          }
          results[filePath].push([lineNum + 1, line]);
        }
      }
    }

    if (Object.keys(results).length === 0) {
      return new ToolMessage({
        content: "No matches found",
        tool_call_id: request.toolCall.id!,
        name: "grep_search",
      });
    }

    // Format output based on mode
    const formatted = this.formatGrepResults(results, outputMode);
    return new ToolMessage({
      content: formatted,
      tool_call_id: request.toolCall.id!,
      name: "grep_search",
    });
  }

  protected formatGrepResults(
    results: Record<string, Array<[number, string]>>,
    outputMode: string
  ): string {
    const sortedPaths = Object.keys(results).sort();

    if (outputMode === "files_with_matches") {
      // Just return file paths
      return sortedPaths.join("\n");
    }

    if (outputMode === "content") {
      // Return file:line:content format
      const lines: string[] = [];
      for (const filePath of sortedPaths) {
        for (const [lineNum, line] of results[filePath]) {
          lines.push(`${filePath}:${lineNum}:${line}`);
        }
      }
      return lines.join("\n");
    }

    if (outputMode === "count") {
      // Return file:count format
      const lines: string[] = [];
      for (const filePath of sortedPaths) {
        const count = results[filePath].length;
        lines.push(`${filePath}:${count}`);
      }
      return lines.join("\n");
    }

    // Default to files_with_matches
    return sortedPaths.join("\n");
  }
}

/**
 * Provides Glob and Grep search over filesystem files.
 *
 * This middleware adds two tools that search through local filesystem:
 * - Glob: Fast file pattern matching by file path
 * - Grep: Fast content search using ripgrep or JavaScript fallback
 *
 * @example
 * ```ts
 * import { createAgent } from "langchain/agents";
 * import {
 *   FilesystemClaudeTextEditorMiddleware,
 *   FilesystemFileSearchMiddleware,
 * } from "langchain/agents/middleware";
 *
 * const agent = createAgent({
 *   model,
 *   tools: [],
 *   middleware: [
 *     new FilesystemClaudeTextEditorMiddleware({ rootPath: "/workspace" }),
 *     new FilesystemFileSearchMiddleware({ rootPath: "/workspace" }),
 *   ],
 * });
 * ```
 */
export class FilesystemFileSearchMiddleware implements AgentMiddleware {
  name = "FilesystemFileSearchMiddleware";

  tools: ClientTool[];

  protected rootPath: string;

  protected useRipgrep: boolean;

  protected maxFileSizeBytes: number;

  constructor(options: {
    rootPath: string;
    useRipgrep?: boolean;
    maxFileSizeMb?: number;
  }) {
    this.rootPath = path.resolve(options.rootPath);
    this.useRipgrep =
      options.useRipgrep !== undefined ? options.useRipgrep : true;
    this.maxFileSizeBytes = (options.maxFileSizeMb || 10) * 1024 * 1024;

    // Create tool definitions
    const globSearchTool = tool(
      // This is a placeholder - actual execution happens in wrapToolCall
      () => "placeholder",
      {
        name: "glob_search",
        description: `Fast file pattern matching tool that works with any codebase size.

Supports glob patterns like **/*.js or src/**/*.ts.
Returns matching file paths sorted by modification time.
Use this tool when you need to find files by name patterns.`,
        schema: globSearchSchema,
      }
    );

    const grepSearchTool = tool(
      // This is a placeholder - actual execution happens in wrapToolCall
      () => "placeholder",
      {
        name: "grep_search",
        description: `Fast content search tool that works with any codebase size.

Searches file contents using regular expressions. Supports full regex
syntax and filters files by pattern with the include parameter.`,
        schema: grepSearchSchema,
      }
    );

    this.tools = [globSearchTool, grepSearchTool];
  }

  wrapToolCall: WrapToolCallHook = async (request, handler) => {
    const toolCall = request.toolCall;
    const toolName = toolCall.name;

    if (toolName === "glob_search") {
      return this.handleGlobSearch(request);
    }

    if (toolName === "grep_search") {
      return await this.handleGrepSearch(request);
    }

    return handler(request);
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

    return fullPath;
  }

  protected handleGlobSearch(
    request: import("../types.js").ToolCallRequest
  ): ToolMessage {
    const args = (request.toolCall.args || {}) as z.infer<
      typeof globSearchSchema
    >;
    const pattern = args.pattern;
    const basePath = args.path;

    try {
      const baseFull = this.validateAndResolvePath(basePath);

      if (!fs.existsSync(baseFull) || !fs.statSync(baseFull).isDirectory()) {
        return new ToolMessage({
          content: "No files found",
          tool_call_id: request.toolCall.id!,
          name: "glob_search",
        });
      }

      // Use recursive directory search with pattern matching
      const matches: Array<[string, string]> = [];

      const searchDir = (dir: string): void => {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          if (entry.isDirectory()) {
            searchDir(fullPath);
          } else if (entry.isFile()) {
            // Get relative path from baseFull
            const relativePath = path.relative(baseFull, fullPath);
            if (matchGlobPattern(relativePath, pattern)) {
              const virtualPath = `/${path
                .relative(this.rootPath, fullPath)
                .replace(/\\/g, "/")}`;
              const stats = fs.statSync(fullPath);
              const modifiedAt = stats.mtime.toISOString();
              matches.push([virtualPath, modifiedAt]);
            }
          }
        }
      };

      searchDir(baseFull);

      if (matches.length === 0) {
        return new ToolMessage({
          content: "No files found",
          tool_call_id: request.toolCall.id!,
          name: "glob_search",
        });
      }

      // Sort by modification time (most recent first)
      matches.sort((a, b) => b[1].localeCompare(a[1]));
      const filePaths = matches.map(([p]) => p);

      return new ToolMessage({
        content: filePaths.join("\n"),
        tool_call_id: request.toolCall.id!,
        name: "glob_search",
      });
    } catch (_error) {
      return new ToolMessage({
        content: "No files found",
        tool_call_id: request.toolCall.id!,
        name: "glob_search",
      });
    }
  }

  protected async handleGrepSearch(
    request: import("../types.js").ToolCallRequest
  ): Promise<ToolMessage> {
    const args = (request.toolCall.args || {}) as z.infer<
      typeof grepSearchSchema
    >;
    const pattern = args.pattern;
    const basePath = args.path;
    const include = args.include;
    const outputMode = args.output_mode || "files_with_matches";

    // Compile regex pattern (for validation)
    try {
      new RegExp(pattern);
    } catch (error) {
      return new ToolMessage({
        content: `Invalid regex pattern: ${error}`,
        tool_call_id: request.toolCall.id!,
        name: "grep_search",
      });
    }

    if (include && !isValidIncludePattern(include)) {
      return new ToolMessage({
        content: "Invalid include pattern",
        tool_call_id: request.toolCall.id!,
        name: "grep_search",
      });
    }

    // Try ripgrep first if enabled
    let results: Record<string, Array<[number, string]>> | null = null;
    if (this.useRipgrep) {
      try {
        results = await this.ripgrepSearch(pattern, basePath, include);
      } catch {
        // Fallback to JavaScript search
        results = null;
      }
    }

    // JavaScript fallback if ripgrep failed or is disabled
    if (results === null) {
      results = this.javascriptSearch(pattern, basePath, include);
    }

    if (Object.keys(results).length === 0) {
      return new ToolMessage({
        content: "No matches found",
        tool_call_id: request.toolCall.id!,
        name: "grep_search",
      });
    }

    // Format output based on mode
    const formatted = this.formatGrepResults(results, outputMode);
    return new ToolMessage({
      content: formatted,
      tool_call_id: request.toolCall.id!,
      name: "grep_search",
    });
  }

  protected async ripgrepSearch(
    pattern: string,
    basePath: string,
    include: string | undefined
  ): Promise<Record<string, Array<[number, string]>>> {
    const baseFull = this.validateAndResolvePath(basePath);

    if (!fs.existsSync(baseFull)) {
      return {};
    }

    // Build ripgrep command
    const cmd: string[] = ["rg", "--json"];

    if (include) {
      cmd.push("--glob", include);
    }

    cmd.push("--", pattern, baseFull);

    const { stdout } = await execAsync(cmd.join(" "), {
      timeout: 30000,
    });

    // Parse ripgrep JSON output
    const results: Record<string, Array<[number, string]>> = {};
    for (const line of stdout.split("\n")) {
      if (!line.trim()) continue;
      try {
        const data = JSON.parse(line);
        if (data.type === "match") {
          const filePath = data.data.path.text;
          // Convert to virtual path
          const virtualPath = `/${path
            .relative(this.rootPath, filePath)
            .replace(/\\/g, "/")}`;
          const lineNum = data.data.line_number;
          const lineText = data.data.lines.text.replace(/\n$/, "");

          if (!results[virtualPath]) {
            results[virtualPath] = [];
          }
          results[virtualPath].push([lineNum, lineText]);
        }
      } catch {
        // Skip invalid JSON lines
        continue;
      }
    }

    return results;
  }

  protected javascriptSearch(
    pattern: string,
    basePath: string,
    include: string | undefined
  ): Record<string, Array<[number, string]>> {
    try {
      const baseFull = this.validateAndResolvePath(basePath);

      if (!fs.existsSync(baseFull)) {
        return {};
      }

      const regex = new RegExp(pattern);
      const results: Record<string, Array<[number, string]>> = {};

      // Walk directory tree
      const searchDir = (dir: string): void => {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          if (entry.isDirectory()) {
            searchDir(fullPath);
          } else if (entry.isFile()) {
            // Check include filter
            if (include && !matchIncludePattern(entry.name, include)) {
              continue;
            }

            // Skip files that are too large
            const stats = fs.statSync(fullPath);
            if (stats.size > this.maxFileSizeBytes) {
              continue;
            }

            try {
              const content = fs.readFileSync(fullPath, "utf8");
              const lines = content.split("\n");

              for (let lineNum = 0; lineNum < lines.length; lineNum++) {
                const line = lines[lineNum];
                if (regex.test(line)) {
                  const virtualPath = `/${path
                    .relative(this.rootPath, fullPath)
                    .replace(/\\/g, "/")}`;
                  if (!results[virtualPath]) {
                    results[virtualPath] = [];
                  }
                  results[virtualPath].push([lineNum + 1, line]);
                }
              }
            } catch {
              // Skip files that can't be read
              continue;
            }
          }
        }
      };

      searchDir(baseFull);
      return results;
    } catch {
      return {};
    }
  }

  protected formatGrepResults(
    results: Record<string, Array<[number, string]>>,
    outputMode: string
  ): string {
    const sortedPaths = Object.keys(results).sort();

    if (outputMode === "files_with_matches") {
      // Just return file paths
      return sortedPaths.join("\n");
    }

    if (outputMode === "content") {
      // Return file:line:content format
      const lines: string[] = [];
      for (const filePath of sortedPaths) {
        for (const [lineNum, line] of results[filePath]) {
          lines.push(`${filePath}:${lineNum}:${line}`);
        }
      }
      return lines.join("\n");
    }

    if (outputMode === "count") {
      // Return file:count format
      const lines: string[] = [];
      for (const filePath of sortedPaths) {
        const count = results[filePath].length;
        lines.push(`${filePath}:${count}`);
      }
      return lines.join("\n");
    }

    // Default to files_with_matches
    return sortedPaths.join("\n");
  }
}
