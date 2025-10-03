/* eslint-disable @typescript-eslint/no-explicit-any */
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  downloadFileAnthropic,
  extractGeneratedFilesAnthropic,
  getFileMetadataAnthropic,
  uploadFileAnthropic,
} from "../anthropicHelpers.js";

describe("extractGeneratedFilesAnthropic", () => {
  it("should extract file IDs from response with messages array", () => {
    const response = {
      messages: [
        new HumanMessage("test"),
        new AIMessage({
          content: [
            { type: "text", text: "Analysis complete" },
            {
              type: "bash_code_execution_tool_result",
              tool_use_id: "tool_123",
              content: {
                type: "bash_code_execution_result",
                stdout: "",
                stderr: "",
                return_code: 0,
                content: [
                  {
                    type: "bash_code_execution_output",
                    file_id: "file_abc123",
                  },
                  {
                    type: "bash_code_execution_output",
                    file_id: "file_def456",
                  },
                ],
              },
            },
          ],
        }),
      ],
    };

    const fileIds = extractGeneratedFilesAnthropic(response);
    expect(fileIds).toEqual(["file_abc123", "file_def456"]);
  });

  it("should extract file IDs from single message response", () => {
    const response = new AIMessage({
      content: [
        { type: "text", text: "Graph created" },
        {
          type: "bash_code_execution_tool_result",
          tool_use_id: "tool_456",
          content: {
            type: "bash_code_execution_result",
            stdout: "",
            stderr: "",
            return_code: 0,
            content: [
              {
                type: "bash_code_execution_output",
                file_id: "file_xyz789",
              },
            ],
          },
        },
      ],
    });

    const fileIds = extractGeneratedFilesAnthropic(response);
    expect(fileIds).toEqual(["file_xyz789"]);
  });

  it("should return empty array when no files present", () => {
    const response = {
      messages: [
        new AIMessage({
          content: [{ type: "text", text: "No files generated" }],
        }),
      ],
    };

    const fileIds = extractGeneratedFilesAnthropic(response);
    expect(fileIds).toEqual([]);
  });

  it("should handle response with no content", () => {
    const response = {
      messages: [new HumanMessage("test")],
    };

    const fileIds = extractGeneratedFilesAnthropic(response);
    expect(fileIds).toEqual([]);
  });
});

describe("helper functions with mocked API", () => {
  let mockClient: any;
  let outputDir: string;

  beforeEach(() => {
    mockClient = {
      beta: {
        files: {
          upload: vi.fn(),
          download: vi.fn(),
          retrieveMetadata: vi.fn(),
        },
      },
    };

    outputDir = mkdtempSync(join(tmpdir(), "langchain-test-"));
  });

  afterEach(() => {
    if (existsSync(outputDir)) {
      rmSync(outputDir, { recursive: true, force: true });
    }
  });

  it("should upload file and return file info", async () => {
    mockClient.beta.files.upload.mockResolvedValue({
      id: "file_test123",
      type: "file",
    });

    const testFilePath = join(__dirname, "fixtures", "test_data.csv");
    const result = await uploadFileAnthropic(mockClient, testFilePath);

    expect(result).toEqual({
      fileId: "file_test123",
      filename: "test_data.csv",
      provider: "anthropic",
    });

    expect(mockClient.beta.files.upload).toHaveBeenCalledWith({
      file: expect.any(Object),
    });
  });

  it("should get file metadata", async () => {
    mockClient.beta.files.retrieveMetadata.mockResolvedValue({
      id: "file_test123",
      filename: "widget_a_filtered.csv",
      type: "file",
    });

    const metadata = await getFileMetadataAnthropic(mockClient, "file_test123");

    expect(metadata).toEqual({
      id: "file_test123",
      filename: "widget_a_filtered.csv",
    });

    expect(mockClient.beta.files.retrieveMetadata).toHaveBeenCalledWith(
      "file_test123"
    );
  });

  it("should download file", async () => {
    const fileContent = "date,region,product\n2024-01-15,North,Widget A\n";

    // Create a web ReadableStream instead of Node Readable
    const webStream = new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(fileContent));
        controller.close();
      },
    });

    mockClient.beta.files.download.mockResolvedValue({
      body: webStream,
    });

    const outputPath = join(outputDir, "downloaded.csv");
    await downloadFileAnthropic(mockClient, "file_test123", outputPath);

    expect(existsSync(outputPath)).toBe(true);
    const content = readFileSync(outputPath, "utf-8");
    expect(content).toBe(fileContent);

    expect(mockClient.beta.files.download).toHaveBeenCalledWith("file_test123");
  });
});
