/* eslint-disable @typescript-eslint/no-explicit-any */
import { ChatAnthropic } from "@langchain/anthropic";
import { HumanMessage } from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph-checkpoint";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createAgent } from "../../index.js";
import { anthropicCodeExecutionMiddleware } from "../anthropicCodeExecution.js";

describe("anthropicCodeExecution middleware unit tests", () => {
  let mockCreate: any;
  let model: ChatAnthropic;

  beforeEach(() => {
    mockCreate = vi.fn();

    const mockClient = {
      messages: {
        create: mockCreate,
      },
    } as any;

    model = new ChatAnthropic({
      model: "claude-sonnet-4-20250514",
      temperature: 0,
      createClient: () => mockClient,
    });
  });

  it("should manage container across multiple turns", async () => {
    const checkpointer = new MemorySaver();
    const agent = createAgent({
      model,
      middleware: [anthropicCodeExecutionMiddleware()],
      checkpointer,
    });

    const containerId = "container_test_123";
    const thread = { configurable: { thread_id: "test-thread" } };

    // Mock first response with container
    mockCreate.mockResolvedValueOnce({
      id: "msg_123",
      type: "message",
      role: "assistant",
      content: [
        { type: "text", text: "Filtered to Widget A" },
        {
          type: "bash_code_execution_tool_result",
          tool_use_id: "tool_1",
          content: {
            type: "bash_code_execution_result",
            stdout: "",
            stderr: "",
            return_code: 0,
            content: [
              {
                type: "bash_code_execution_output",
                file_id: "file_filtered_123",
              },
            ],
          },
        },
      ],
      model: "claude-sonnet-4-20250514",
      stop_reason: "end_turn",
      usage: { input_tokens: 100, output_tokens: 200 },
      container: {
        id: containerId,
        expires_at: new Date(Date.now() + 3600000).toISOString(),
      },
    });

    // First invocation
    const result1 = await agent.invoke(
      {
        messages: new HumanMessage({
          content: [
            { type: "text", text: "Filter to just widget A" },
            { type: "container_upload", file_id: "file_upload_123" },
          ],
        }),
      },
      thread
    );

    expect(mockCreate).toHaveBeenCalledTimes(1);
    expect(result1.messages).toBeTruthy();

    // Mock second response with same container
    mockCreate.mockResolvedValueOnce({
      id: "msg_124",
      type: "message",
      role: "assistant",
      content: [
        { type: "text", text: "Graph created" },
        {
          type: "bash_code_execution_tool_result",
          tool_use_id: "tool_2",
          content: {
            type: "bash_code_execution_result",
            stdout: "",
            stderr: "",
            return_code: 0,
            content: [
              {
                type: "bash_code_execution_output",
                file_id: "file_graph_456",
              },
            ],
          },
        },
      ],
      model: "claude-sonnet-4-20250514",
      stop_reason: "end_turn",
      usage: { input_tokens: 150, output_tokens: 300 },
      container: {
        id: containerId,
        expires_at: new Date(Date.now() + 3600000).toISOString(),
      },
    });

    // Second invocation - should reuse container
    await agent.invoke(
      {
        messages: new HumanMessage(
          "Turn that into a graph of sales and units over time."
        ),
      },
      thread
    );

    expect(mockCreate).toHaveBeenCalledTimes(2);
    const secondCall = mockCreate.mock.calls[1][0];

    // Verify container was passed in second call
    expect(secondCall.container).toBe(containerId);

    // Verify container state was persisted
    const state = await checkpointer.get(thread);
    const middlewareContainer = (state?.channel_values as any)?.container;
    expect(middlewareContainer?.id).toBe(containerId);
  });

  it("should inject required beta headers", async () => {
    const checkpointer = new MemorySaver();
    const agent = createAgent({
      model,
      middleware: [anthropicCodeExecutionMiddleware()],
      checkpointer,
    });

    const thread = { configurable: { thread_id: "test-thread" } };

    mockCreate.mockResolvedValueOnce({
      id: "msg_123",
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "Done" }],
      model: "claude-sonnet-4-20250514",
      stop_reason: "end_turn",
      usage: { input_tokens: 10, output_tokens: 20 },
    });

    await agent.invoke(
      {
        messages: new HumanMessage("Test"),
      },
      thread
    );

    expect(mockCreate).toHaveBeenCalledTimes(1);
    const options = mockCreate.mock.calls[0][1];

    // The middleware should have injected the headers through callOptions
    // Note: we can't easily verify headers in the mock since they're passed through
    // the SDK, but we can verify the call was made
    expect(options).toHaveProperty("headers");
    expect(options.headers).toHaveProperty("anthropic-beta");
  });
});
