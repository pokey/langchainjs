import type { Anthropic } from "@anthropic-ai/sdk";
import { createReadStream, createWriteStream } from "node:fs";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

/**
 * Upload a file to Anthropic's Files API for use with code execution.
 *
 * @param client - Anthropic client instance
 * @param filePath - Path to the file to upload
 * @returns Object containing the file ID and filename
 *
 * @example
 * ```typescript
 * import { Anthropic } from '@anthropic-ai/sdk';
 * import { uploadFileAnthropic } from 'langchain/agents/middleware';
 *
 * const client = new Anthropic();
 * const { fileId, filename } = await uploadFileAnthropic(client, './data.csv');
 * ```
 */
export async function uploadFileAnthropic(client: Anthropic, filePath: string) {
  const fileStream = createReadStream(filePath);
  const filename = filePath.split("/").pop() || "file";

  const fileObject = await client.beta.files.upload({
    file: fileStream,
  });

  return {
    fileId: fileObject.id,
    filename,
    provider: "anthropic",
  };
}

/**
 * Download a file from Anthropic's Files API.
 *
 * @param client - Anthropic client instance
 * @param fileId - The file ID to download
 * @param outputPath - Path where the file should be saved
 *
 * @example
 * ```typescript
 * import { Anthropic } from '@anthropic-ai/sdk';
 * import { downloadFileAnthropic } from 'langchain/agents/middleware';
 *
 * const client = new Anthropic();
 * await downloadFileAnthropic(client, 'file_abc123', './output.png');
 * ```
 */
export async function downloadFileAnthropic(
  client: Anthropic,
  fileId: string,
  outputPath: string
): Promise<void> {
  const response = await client.beta.files.download(fileId);
  const writeStream = createWriteStream(outputPath);

  // Convert web ReadableStream to Node stream and pipe to file
  if (!response.body) {
    throw new Error("No body in file download response");
  }

  await pipeline(Readable.fromWeb(response.body as any), writeStream);
}

/**
 * Retrieve metadata for a file from Anthropic's Files API.
 *
 * @param client - Anthropic client instance
 * @param fileId - The file ID to retrieve metadata for
 * @returns File metadata including filename
 *
 * @example
 * ```typescript
 * import { Anthropic } from '@anthropic-ai/sdk';
 * import { getFileMetadataAnthropic } from 'langchain/agents/middleware';
 *
 * const client = new Anthropic();
 * const metadata = await getFileMetadataAnthropic(client, 'file_abc123');
 * console.log(metadata.filename);
 * ```
 */
export async function getFileMetadataAnthropic(
  client: Anthropic,
  fileId: string
): Promise<{ filename: string; id: string }> {
  const metadata = await client.beta.files.retrieveMetadata(fileId);
  return {
    filename: metadata.filename,
    id: metadata.id,
  };
}

/**
 * Extract generated file IDs from an Anthropic code execution response.
 *
 * Parses the response content blocks to find files created during code execution.
 * Note: This function only returns file IDs. To get proper filenames, use
 * `getFileMetadataAnthropic` for each file ID.
 *
 * @param response - The Anthropic message response
 * @returns Array of file IDs
 *
 * @example
 * ```typescript
 * import { extractGeneratedFilesAnthropic, getFileMetadataAnthropic } from 'langchain/agents/middleware';
 *
 * const response = await client.beta.messages.create({...});
 * const fileIds = extractGeneratedFilesAnthropic(response);
 *
 * for (const fileId of fileIds) {
 *   const metadata = await getFileMetadataAnthropic(client, fileId);
 *   await downloadFileAnthropic(client, fileId, metadata.filename);
 * }
 * ```
 */
export function extractGeneratedFilesAnthropic(response: any): string[] {
  const fileIds: string[] = [];

  // Handle both single message responses and response objects with messages array
  const messages = response.messages || [response];

  for (const message of messages) {
    if (!message.content || !Array.isArray(message.content)) {
      continue;
    }

    for (const item of message.content) {
      if (item.type === "bash_code_execution_tool_result") {
        const contentItem = item.content;
        if (
          contentItem?.type === "bash_code_execution_result" &&
          Array.isArray(contentItem.content)
        ) {
          for (const file of contentItem.content) {
            if (file.type === "bash_code_execution_output" && file.file_id) {
              fileIds.push(file.file_id);
            }
          }
        }
      }
    }
  }

  return fileIds;
}
