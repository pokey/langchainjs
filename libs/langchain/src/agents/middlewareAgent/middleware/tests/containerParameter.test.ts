/* eslint-disable @typescript-eslint/no-explicit-any */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { ChatAnthropic } from "@langchain/anthropic";
import { HumanMessage } from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph-checkpoint";
import { createAgent } from "../../index.js";
import { anthropicCodeExecutionMiddleware } from "../anthropicCodeExecution.js";

describe("container parameter propagation", () => {
  let mockCreate: any;
  let model: ChatAnthropic;

  beforeEach(() => {
    // Mock the Anthropic client's messages.create method
    mockCreate = vi.fn();

    // Create a mock client
    const mockClient = {
      messages: {
        create: mockCreate,
      },
    } as any;

    // Create ChatAnthropic with mocked client
    model = new ChatAnthropic({
      model: "claude-sonnet-4-20250514",
      temperature: 0,
      createClient: () => mockClient,
    });
  });

  it("should pass container parameter from middleware to Anthropic API", async () => {
    const checkpointer = new MemorySaver();
    const agent = createAgent({
      model,
      middleware: [anthropicCodeExecutionMiddleware()],
      checkpointer,
    });

    const containerId = "test-container-123";
    const thread = { configurable: { thread_id: "test-thread" } };

    // Mock first response with a container - this will be extracted by middleware
    mockCreate.mockResolvedValueOnce({
      id: "msg_123",
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "Analysis complete" }],
      model: "claude-sonnet-4-20250514",
      stop_reason: "end_turn",
      usage: { input_tokens: 10, output_tokens: 20 },
      container: {
        id: containerId,
        expires_at: new Date(Date.now() + 3600000).toISOString(),
      },
    });

    // First invocation - no container sent yet, but response will include one
    await agent.invoke(
      {
        messages: new HumanMessage("Analyze this data"),
      },
      thread
    );

    // Check the first call - container should be undefined (no existing container)
    expect(mockCreate).toHaveBeenCalledTimes(1);
    const firstCall = mockCreate.mock.calls[0][0];
    console.log("First API call params:", JSON.stringify(firstCall, null, 2));
    expect(firstCall.container).toBeUndefined();

    // Mock second response
    mockCreate.mockResolvedValueOnce({
      id: "msg_124",
      type: "message",
      role: "assistant",
      content: [{ type: "text", text: "Second analysis" }],
      model: "claude-sonnet-4-20250514",
      stop_reason: "end_turn",
      usage: { input_tokens: 10, output_tokens: 20 },
      container: {
        id: containerId,
        expires_at: new Date(Date.now() + 3600000).toISOString(),
      },
    });

    // Second invocation - container should be extracted from first response and reused
    await agent.invoke(
      {
        messages: new HumanMessage("Do more analysis"),
      },
      thread
    );

    // Check the second call - container should be present
    expect(mockCreate).toHaveBeenCalledTimes(2);
    const secondCall = mockCreate.mock.calls[1][0];
    console.log("Second API call params:", JSON.stringify(secondCall, null, 2));

    // Verify the container was extracted from the first response's additional_kwargs
    // and stored in middleware state
    const state = await checkpointer.get(thread);
    const middlewareContainer = (state?.channel_values as any)?.container;
    console.log(
      "Middleware container:",
      JSON.stringify(middlewareContainer, null, 2)
    );
    expect(middlewareContainer?.id).toBe(containerId);

    // Container should now be passed from middleware state to the API
    expect(secondCall.container).toBe(containerId);
  });
});
