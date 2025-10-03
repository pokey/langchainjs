import { ChatAnthropic } from "@langchain/anthropic";
import { HumanMessage } from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph-checkpoint";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { createAgent } from "../../index.js";
import { anthropicCodeExecutionMiddleware } from "../anthropicCodeExecution.js";
import {
  downloadFileAnthropic,
  extractGeneratedFilesAnthropic,
  getFileMetadataAnthropic,
  uploadFileAnthropic,
} from "../anthropicHelpers.js";

const thread = {
  configurable: {
    thread_id: "test-123",
  },
};

describe("dataAnalysisMiddleware integration tests", () => {
  let model: ChatAnthropic;
  let outputDir: string;
  const testDataPath = join(__dirname, "fixtures", "test_data.csv");

  beforeAll(() => {
    model = new ChatAnthropic({
      model: "claude-sonnet-4-20250514", // Haiku is a bit too dumb
    });

    // Create temporary directory for test outputs
    outputDir = mkdtempSync(join(tmpdir(), "langchain-test-"));
  });

  afterAll(() => {
    // Clean up temporary directory
    if (existsSync(outputDir)) {
      rmSync(outputDir, { recursive: true, force: true });
    }
  });

  it(
    "should upload file, analyze data, and download generated files",
    {
      timeout: 120000, // 2 minute timeout for API calls
    },
    async () => {
      const client = model.createClient({});

      // Upload test data
      const uploadedFile = await uploadFileAnthropic(client, testDataPath);

      expect(uploadedFile.fileId).toBeTruthy();
      expect(uploadedFile.filename).toBe("test_data.csv");
      expect(uploadedFile.provider).toBe("anthropic");

      // Verify file metadata
      const metadata = await getFileMetadataAnthropic(
        client,
        uploadedFile.fileId
      );
      expect(metadata.filename).toBe("test_data.csv");

      // Create agent with data analysis middleware
      const agent = createAgent({
        model,
        middleware: [anthropicCodeExecutionMiddleware()],
        checkpointSaver: new MemorySaver(),
      });

      // Invoke agent with analysis task
      const result1 = await agent.invoke(
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

      // Verify first response
      expect(result1.messages).toBeTruthy();
      expect(result1.messages.length).toBeGreaterThan(0);

      const result2 = await agent.invoke(
        {
          messages: new HumanMessage(
            "Turn that into a graph of sales and units over time."
          ),
        },
        thread
      );

      // Verify second response and extract generated files
      expect(result2.messages).toBeTruthy();
      expect(result2.messages.length).toBeGreaterThan(0);

      const fileIds = extractGeneratedFilesAnthropic(result2);
      expect(fileIds.length).toBeGreaterThan(0);

      // Download and verify each generated file
      for (const fileId of fileIds) {
        const metadata = await getFileMetadataAnthropic(client, fileId);
        expect(metadata.filename).toBeTruthy();

        const outputPath = join(outputDir, metadata.filename);
        await downloadFileAnthropic(client, fileId, outputPath);

        // Verify file was downloaded
        expect(existsSync(outputPath)).toBe(true);
      }
    }
  );
});
