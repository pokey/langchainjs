/**
 * Example demonstrating the Code Execution Middleware with Daytona.
 *
 * This middleware enables Claude to execute bash commands in a secure, isolated
 * Daytona sandbox, analyze uploaded files, and generate output files. The example
 * demonstrates multi-turn conversation with container and file state persistence.
 *
 * Requirements:
 * - Set DAYTONA_API_KEY environment variable (get from https://app.daytona.io)
 * - Install dependencies: @langchain/daytona, @langchain/anthropic, langchain
 */

import { ChatAnthropic } from "@langchain/anthropic";
import { DaytonaContainerProvider } from "@langchain/daytona/middleware";
import { HumanMessage } from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph";
import {
  codeExecutionMiddleware,
  createAgent,
  MemoryFileProvider,
} from "langchain";
import fs from "node:fs/promises";
import { join } from "node:path";

// Initial setup
const model = new ChatAnthropic({
  model: "claude-sonnet-4-5-20250929",
});

const middleware = codeExecutionMiddleware(
  new DaytonaContainerProvider(),
  new MemoryFileProvider()
);

const agent = createAgent({
  model,
  middleware: [middleware],
  checkpointer: new MemorySaver(),
});

const thread = {
  configurable: {
    thread_id: "test-123",
  },
};

// Read and add the test data file
const testDataPath = "test_data.csv";
const fileContent = await fs.readFile(testDataPath);

// First invocation - should create container and analyze uploaded data
const response1 = await agent.invoke(
  {
    messages: new HumanMessage("Filter to just widget A and save to a new csv"),
    files: [await middleware.addFile(testDataPath, fileContent)],
  },
  thread
);
console.log("Response 1:", response1.messages);

// Second invocation - should reuse container and previous analysis
const response2 = await agent.invoke(
  {
    messages: new HumanMessage(
      "Turn that into a graph of sales and units over time."
    ),
  },
  thread
);
console.log("Response 2:", response2.messages);

// Extract and download generated files
const generatedFiles = middleware
  .files(response2)
  .filter(({ type, path }) => type === "tool" && path.endsWith(".png"));

for (const file of generatedFiles) {
  const content = await file.getContent();
  // Extract just the filename from the absolute path
  const filename = file.path.split("/").pop() || file.path;
  const outputPath = join(".", filename);
  await fs.writeFile(outputPath, content);
  console.log(`Downloaded generated file: ${outputPath}`);
}
