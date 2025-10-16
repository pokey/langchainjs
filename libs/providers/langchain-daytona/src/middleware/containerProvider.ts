import { Daytona, Sandbox } from "@daytonaio/sdk";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { ToolMessage } from "@langchain/core/messages";
import { getCurrentTaskInput } from "@langchain/langgraph";
import { z } from "zod";
import {
  CodeExecutionMiddlewareState,
  ContainerInfo,
  ContainerProvider,
  FileFromProvider,
  ModelRequest,
  UploadFileToContainerOpts,
  UploadFileToContainerRet,
} from "langchain";

/**
 * Container provider for Daytona sandboxes.
 *
 * Enables code execution in isolated Daytona sandboxes with bash command support.
 * Provides file upload/download capabilities and automatic file tracking.
 */
export class DaytonaContainerProvider implements ContainerProvider {
  private client: Daytona;
  private knownFiles: Map<string, Set<string>> = new Map();

  constructor(apiKey?: string) {
    this.client = new Daytona({
      // eslint-disable-next-line no-process-env
      apiKey: apiKey || process.env.DAYTONA_API_KEY,
    });
  }

  /**
   * Retrieves a sandbox by its container ID.
   */
  private async getSandbox(containerId: string): Promise<Sandbox> {
    return await this.client.get(containerId);
  }

  /**
   * Gets the known files set for a container, creating it if needed.
   */
  private getKnownFilesForContainer(containerId: string): Set<string> {
    if (!this.knownFiles.has(containerId)) {
      this.knownFiles.set(containerId, new Set());
    }
    return this.knownFiles.get(containerId)!;
  }

  /**
   * Tool definition for bash command execution in the sandbox.
   */
  get tools() {
    return [
      new DynamicStructuredTool({
        name: "bash",
        description:
          "Execute bash commands in a secure isolated sandbox environment. " +
          "Use this to run shell commands, process files, analyze data, or perform system operations. " +
          "The sandbox persists across multiple commands in the same conversation.",
        schema: z.object({
          command: z
            .string()
            .describe("The bash command to execute in the sandbox"),
          workingDirectory: z
            .string()
            .optional()
            .describe(
              "Optional working directory for command execution. Defaults to the sandbox working directory."
            ),
        }),
        func: async ({ command, workingDirectory }, _runManager, config) => {
          const state =
            getCurrentTaskInput<CodeExecutionMiddlewareState>(config);
          const containerId = state.container?.id;
          if (!containerId) {
            throw new Error(
              "Container ID not found in state. Make sure the sandbox has been initialized."
            );
          }

          try {
            const sandbox = await this.getSandbox(containerId);
            const knownFiles = this.getKnownFilesForContainer(containerId);

            // Snapshot known files before execution
            const knownFilesBefore = new Set(knownFiles);

            // Execute the command
            const response = await sandbox.process.executeCommand(
              command,
              workingDirectory
            );

            // Format response including exit code, stdout, and any artifacts
            let output = `Exit code: ${response.exitCode}\n\n`;
            output += `Output:\n${response.result || "(no output)"}`;

            // Include chart information if available
            if (
              response.artifacts?.charts &&
              response.artifacts.charts.length > 0
            ) {
              output += `\n\nGenerated ${response.artifacts.charts.length} chart(s):`;
              response.artifacts.charts.forEach((chart, i) => {
                output += `\n  ${i + 1}. ${chart.type}${
                  chart.title ? `: ${chart.title}` : ""
                }`;
              });
            }

            // Scan for new files
            const workDir = await sandbox.getWorkDir();
            const newFilePaths: string[] = [];

            if (workDir) {
              const currentFiles = await this.listFilesRecursively(
                sandbox,
                workDir
              );

              // Find files that weren't known before
              const generatedFiles = currentFiles.filter(
                (filePath) => !knownFilesBefore.has(filePath)
              );

              // Track new files (don't download yet - defer to extractFilesFromModelResponse)
              for (const filePath of generatedFiles) {
                newFilePaths.push(filePath);
                knownFiles.add(filePath);
              }
            }

            // Return ToolMessage with file paths in artifact
            return new ToolMessage({
              content: output,
              tool_call_id: config.toolCall?.id as string,
              artifact: {
                containerId,
                files: newFilePaths.map((path) => ({
                  path,
                  type: "tool" as const,
                  providerId: path,
                })),
              },
            });
          } catch (error) {
            const errorMessage =
              typeof error === "object" &&
              error !== null &&
              "message" in error &&
              typeof error.message === "string"
                ? error.message
                : String(error);
            return `Error executing command: ${errorMessage}`;
          }
        },
      }),
    ];
  }

  /**
   * Starts a new Daytona sandbox.
   */
  async startContainer(): Promise<ContainerInfo> {
    // Create a new sandbox
    const sandbox = await this.client.create({
      name: `langchain-sandbox-${Date.now()}`,
      // Don't auto-stop immediately, give it some time
      autoStopInterval: 60, // 60 minutes
    });

    // Initialize known files tracking by listing current working directory
    await this.updateKnownFiles(sandbox.id);

    return {
      id: sandbox.id,
      // Calculate expiration based on auto-stop interval
      expiresAt: sandbox.autoStopInterval
        ? new Date(Date.now() + sandbox.autoStopInterval * 60 * 1000)
        : undefined,
    };
  }

  /**
   * Uploads a file to the sandbox.
   */
  async uploadFileToContainer({
    containerId,
    path,
    providerId,
    getContent,
  }: UploadFileToContainerOpts): Promise<UploadFileToContainerRet> {
    if (!containerId) {
      throw new Error("Container ID is required to upload file");
    }

    if (providerId != null) {
      // File already uploaded
      return { providerId };
    }

    // Get sandbox and upload file
    const sandbox = await this.getSandbox(containerId);
    const content = await getContent();
    await sandbox.fs.uploadFile(content, path);

    // Track this file
    const knownFiles = this.getKnownFilesForContainer(containerId);
    knownFiles.add(path);

    return { providerId: path };
  }

  /**
   * Extracts files generated during model execution.
   *
   * Scans ToolMessages for file metadata in artifacts, then downloads
   * file contents from the sandbox into memory.
   */
  async extractFilesFromModelResponse(
    messages: ModelRequest["messages"]
  ): Promise<FileFromProvider[]> {
    const allFiles: FileFromProvider[] = [];

    // Find all ToolMessages with file artifacts
    for (const message of messages) {
      if (ToolMessage.isInstance(message) && message.artifact?.files) {
        const { files, containerId } = message.artifact;

        if (!containerId || !Array.isArray(files)) {
          continue;
        }

        // Download each file from the sandbox
        const sandbox = await this.getSandbox(containerId);

        for (const fileMetadata of files) {
          try {
            const content = await sandbox.fs.downloadFile(fileMetadata.path);

            allFiles.push({
              providerId: fileMetadata.providerId,
              path: fileMetadata.path,
              type: fileMetadata.type,
              content,
            });
          } catch (error) {
            console.warn(
              `Could not download file ${fileMetadata.path}:`,
              error
            );
          }
        }
      }
    }

    return allFiles;
  }

  /**
   * Updates the set of known files by scanning the working directory.
   */
  private async updateKnownFiles(containerId: string): Promise<void> {
    try {
      const sandbox = await this.getSandbox(containerId);
      const knownFiles = this.getKnownFilesForContainer(containerId);

      const workDir = await sandbox.getWorkDir();
      if (!workDir) {
        return;
      }

      const files = await this.listFilesRecursively(sandbox, workDir);
      files.forEach((file) => knownFiles.add(file));
    } catch (error) {
      console.warn("Could not update known files:", error);
    }
  }

  /**
   * Recursively lists all files in a directory.
   */
  private async listFilesRecursively(
    sandbox: Sandbox,
    dirPath: string
  ): Promise<string[]> {
    const allFiles: string[] = [];
    const queue: string[] = [dirPath];

    while (queue.length > 0) {
      const currentPath = queue.shift()!;

      try {
        const items = await sandbox.fs.listFiles(currentPath);

        for (const item of items) {
          const fullPath = `${currentPath}/${item.name}`;

          if (item.isDir) {
            // Add directory to queue for recursive scanning
            queue.push(fullPath);
          } else {
            // Add file to results
            allFiles.push(fullPath);
          }
        }
      } catch (error) {
        // Skip directories we can't access
        console.warn(`Could not list directory ${currentPath}:`, error);
      }
    }

    return allFiles;
  }
}
