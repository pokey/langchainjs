import { ChatAnthropic } from "@langchain/anthropic";
import { HumanMessage } from "@langchain/core/messages";

import { MemorySaver } from "@langchain/langgraph-checkpoint";
import { createAgent } from "./agents/middlewareAgent/index.js";
import { anthropicCodeExecutionMiddleware } from "./agents/middlewareAgent/middleware/anthropicCodeExecution.js";
import {
  downloadFileAnthropic,
  extractGeneratedFilesAnthropic,
  getFileMetadataAnthropic,
  uploadFileAnthropic,
} from "./agents/middlewareAgent/middleware/anthropicHelpers.js";

// Initial setup
const model = new ChatAnthropic({
  model: "claude-sonnet-4-20250514",
});
const client = model.createClient({});
const thread = {
  configurable: {
    thread_id: "test-123",
  },
};
const agent = createAgent({
  model,
  middleware: [anthropicCodeExecutionMiddleware()],
  checkpointSaver: new MemorySaver(),
});

// Upload the file to Anthropic
const uploadedFile = await uploadFileAnthropic(client, "test_data.csv");

// First invocation - should create container and analyze uploaded data
const response1 = await agent.invoke(
  {
    messages: new HumanMessage({
      content: [
        {
          type: "text",
          text: "Filter to just widget A",
        },
        { type: "container_upload", file_id: uploadedFile.fileId },
      ],
    }),
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
for (const fileId of extractGeneratedFilesAnthropic(response2)) {
  const metadata = await getFileMetadataAnthropic(client, fileId);
  await downloadFileAnthropic(client, fileId, metadata.filename);
  console.log(`Downloaded generated file: ${metadata.filename}`);
}
