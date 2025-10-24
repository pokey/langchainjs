/**
 * Integration tests for Anthropic text editor and memory tool middleware.
 * These tests use real API calls to Claude Sonnet 4.5.
 */
import { describe, it, expect } from "vitest";
import { ChatAnthropic } from "@langchain/anthropic";
import { HumanMessage } from "@langchain/core/messages";

import { createAgent } from "../../index.js";
import {
  type AnthropicToolsState,
  createStateClaudeMemoryMiddleware,
  createStateClaudeTextEditorMiddleware,
} from "../anthropicTools/index.js";

describe("StateClaudeTextEditorMiddleware integration", () => {
  it("should create and view a file", async () => {
    const model = new ChatAnthropic({
      model: "claude-sonnet-4-5-20250929",
      temperature: 0,
    });

    const agent = createAgent({
      model,
      middleware: [createStateClaudeTextEditorMiddleware()],
    });

    // Create a file
    const result = await agent.invoke({
      messages: [
        new HumanMessage("Create a file /hello.py that prints 'Hello, World!'"),
      ],
    });

    // Verify tool was called and file was created
    const state = result as unknown as AnthropicToolsState;
    const toolMessages = result.messages.filter((msg) => msg.type === "tool");

    expect(toolMessages.length).toBeGreaterThan(0);
    expect(state.text_editor_files).toBeDefined();

    const fileKeys = Object.keys(state.text_editor_files || {});
    expect(fileKeys.length).toBeGreaterThan(0);

    // Verify file contains expected content
    const createdFile = fileKeys.find((k) => k.includes("hello"));
    expect(createdFile).toBeDefined();
    const content = state.text_editor_files![createdFile!].content;
    expect(content.toLowerCase()).toContain("print");
    expect(content.toLowerCase()).toContain("hello");
  }, 60000);

  it("should modify existing file with str_replace", async () => {
    const model = new ChatAnthropic({
      model: "claude-sonnet-4-5-20250929",
      temperature: 0,
    });

    // Start with an existing file
    const initialState: AnthropicToolsState = {
      text_editor_files: {
        "/config.txt": {
          content: "api_key=old_value\ntimeout=30",
          created_at: new Date().toISOString(),
          modified_at: new Date().toISOString(),
        },
      },
    };

    const agent = createAgent({
      model,
      middleware: [createStateClaudeTextEditorMiddleware()],
    });

    const result = await agent.invoke({
      messages: [
        new HumanMessage(
          "In the /config.txt file, change 'old_value' to 'new_secret_key'"
        ),
      ],
      ...initialState,
    });

    // Verify file was modified
    const state = result as unknown as AnthropicToolsState;
    const toolMessages = result.messages.filter((msg) => msg.type === "tool");

    expect(toolMessages.length).toBeGreaterThan(0);
    expect(state.text_editor_files!["/config.txt"]).toBeDefined();

    const content = state.text_editor_files!["/config.txt"].content;
    expect(content).toContain("new_secret_key");
    expect(content).not.toContain("old_value");
  }, 60000);

  it("should delete files and apply reducer correctly", async () => {
    const model = new ChatAnthropic({
      model: "claude-sonnet-4-5-20250929",
      temperature: 0,
      clientOptions: {
        defaultHeaders: {
          "anthropic-beta": "context-management-2025-06-27",
        },
      },
    });

    // Start with two memory files
    const initialState: AnthropicToolsState = {
      memory_files: {
        "/memories/keep.txt": {
          content: "This file should remain",
          created_at: new Date().toISOString(),
          modified_at: new Date().toISOString(),
        },
        "/memories/delete.txt": {
          content: "This file should be deleted",
          created_at: new Date().toISOString(),
          modified_at: new Date().toISOString(),
        },
      },
    };

    const agent = createAgent({
      model,
      middleware: [createStateClaudeMemoryMiddleware()],
    });

    const result = await agent.invoke({
      messages: [
        new HumanMessage(
          "Delete the file /memories/delete.txt using the memory tool"
        ),
      ],
      ...initialState,
    });

    // Verify the deleted file is removed from state (not present with null value)
    const state = result as unknown as AnthropicToolsState;
    expect(state.memory_files).toBeDefined();
    expect(state.memory_files!["/memories/keep.txt"]).toBeDefined();

    // This is the key test: the file should be REMOVED from state, not set to null
    expect(state.memory_files).not.toHaveProperty("/memories/delete.txt");

    // Only one file should remain
    expect(Object.keys(state.memory_files!)).toEqual(["/memories/keep.txt"]);
  }, 60000);
});

describe("StateClaudeMemoryMiddleware integration", () => {
  it("should store and retrieve information using memory tool", async () => {
    const model = new ChatAnthropic({
      model: "claude-sonnet-4-5",
      temperature: 0,
    });

    const agent = createAgent({
      model,
      middleware: [createStateClaudeMemoryMiddleware()],
    });

    // First invocation: store information in memory
    const result1 = await agent.invoke({
      messages: [
        new HumanMessage(
          "Please save to your memory that my favorite programming language is TypeScript and I work at a company called Acme Corp."
        ),
      ],
    });

    const state1 = result1 as unknown as AnthropicToolsState;
    const toolMessages1 = result1.messages.filter((msg) => msg.type === "tool");

    // Verify memory tool was called
    expect(toolMessages1.length).toBeGreaterThan(0);
    expect(state1.memory_files).toBeDefined();

    // Verify at least one memory file was created
    const memoryFileKeys = Object.keys(state1.memory_files || {});
    expect(memoryFileKeys.length).toBeGreaterThan(0);

    // Verify memory files are stored under /memories/ prefix
    memoryFileKeys.forEach((key) => {
      expect(key).toMatch(/^\/memories\//);
    });

    // Check that the content includes the stored information
    const memoryContents = memoryFileKeys.map(
      (key) => state1.memory_files![key].content
    );
    const combinedMemoryText = memoryContents.join(" ").toLowerCase();
    expect(combinedMemoryText).toContain("typescript");
    expect(combinedMemoryText).toContain("acme corp");

    // Second invocation: retrieve stored information
    const result2 = await agent.invoke({
      messages: [
        new HumanMessage(
          "What's my favorite programming language and where do I work?"
        ),
      ],
      ...state1, // Pass state from first invocation
    });

    // Find the AI response
    const aiMessages = result2.messages.filter((msg) => msg.type === "ai");
    expect(aiMessages.length).toBeGreaterThan(0);

    // Extract text from all AI messages (handle both string and array content)
    const responseText = aiMessages
      .map((msg) => {
        if (typeof msg.content === "string") {
          return msg.content;
        } else if (Array.isArray(msg.content)) {
          return msg.content
            .filter(
              (block: {
                type?: string;
              }): block is { type: "text"; text: string } =>
                block.type === "text"
            )
            .map((block) => block.text)
            .join(" ");
        }
        return "";
      })
      .join(" ")
      .toLowerCase();

    expect(responseText).toBeTruthy();

    // Verify Claude can recall the stored information
    expect(responseText).toContain("typescript");
    expect(responseText).toContain("acme");
  }, 60000);
});

describe("Combined middleware integration", () => {
  it("should work with both text editor and memory middleware", async () => {
    const model = new ChatAnthropic({
      model: "claude-sonnet-4-5",
      temperature: 0,
    });

    const agent = createAgent({
      model,
      middleware: [
        createStateClaudeTextEditorMiddleware(),
        createStateClaudeMemoryMiddleware(),
      ],
    });

    const result = await agent.invoke({
      messages: [
        new HumanMessage(
          "Create a file /note.txt with 'Meeting at 3pm' and remember this in your memory"
        ),
      ],
    });

    const state = result as unknown as AnthropicToolsState;
    const toolMessages = result.messages.filter((msg) => msg.type === "tool");

    // At least one tool should have been called
    expect(toolMessages.length).toBeGreaterThan(0);

    // At least one of the file systems should have files
    const hasTextEditorFiles =
      state.text_editor_files &&
      Object.keys(state.text_editor_files).length > 0;
    const hasMemoryFiles =
      state.memory_files && Object.keys(state.memory_files).length > 0;

    expect(hasTextEditorFiles || hasMemoryFiles).toBe(true);
  }, 60000);
});
