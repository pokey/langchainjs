import z from "zod";

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

const TextEditorInsertCommandSchema = z.object({
  command: z.literal("insert"),
  path: z.string().describe("Path to the file to modify"),
  insert_line: z
    .number()
    .describe("Line number after which to insert text (0 for beginning)"),
  new_str: z.string().describe("Text to insert"),
});

const MemoryInsertCommandSchema = z.object({
  command: z.literal("insert"),
  path: z.string().describe("Path to the file to modify"),
  insert_line: z
    .number()
    .describe("Line number after which to insert text (0 for beginning)"),
  insert_text: z.string().describe("Text to insert"),
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
 * Text editor tool commands (text_editor_20250728).
 * Supports: view, create, str_replace, insert
 */
export const TextEditorCommandSchema = z.discriminatedUnion("command", [
  ViewCommandSchema,
  CreateCommandSchema,
  StrReplaceCommandSchema,
  TextEditorInsertCommandSchema,
]);

/**
 * Memory tool commands (memory_20250818).
 * Supports: view, create, str_replace, insert, delete, rename
 */
export const MemoryCommandSchema = z.discriminatedUnion("command", [
  ViewCommandSchema,
  CreateCommandSchema,
  StrReplaceCommandSchema,
  MemoryInsertCommandSchema,
  DeleteCommandSchema,
  RenameCommandSchema,
]);
