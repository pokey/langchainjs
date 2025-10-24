import {
  mkdir,
  readFile,
  readdir,
  rename,
  rm,
  stat,
  unlink,
  writeFile,
} from "fs/promises";
import { existsSync, mkdirSync } from "fs";
import * as path from "path";
import { FileData } from "./FileData.js";
import { FileSystem } from "./FileSystem.js";

/**
 * Physical filesystem implementation.
 * Uses Node.js fs module for actual file I/O.
 */
export class PhysicalFileSystem implements FileSystem {
  private resolvedRootPath: string;
  private maxFileSizeBytes: number;

  constructor(
    rootPath: string,
    private allowedPrefixes: string[],
    maxFileSizeMb: number
  ) {
    this.resolvedRootPath = path.resolve(rootPath);
    this.maxFileSizeBytes = maxFileSizeMb * 1024 * 1024;

    // Create root directory if it doesn't exist
    if (!existsSync(this.resolvedRootPath)) {
      mkdirSync(this.resolvedRootPath, { recursive: true });
    }
  }

  async readFile(virtualPath: string): Promise<FileData | null> {
    const fullPath = this.resolveVirtualPath(virtualPath);

    try {
      const stats = await stat(fullPath);

      if (!stats.isFile()) {
        return null;
      }

      if (stats.size > this.maxFileSizeBytes) {
        const maxMb = this.maxFileSizeBytes / 1024 / 1024;
        throw new Error(`File too large: ${virtualPath} exceeds ${maxMb}MB`);
      }

      const content = await readFile(fullPath, "utf8");

      return {
        content,
        created_at: stats.birthtime.toISOString(),
        modified_at: stats.mtime.toISOString(),
      };
    } catch (error: unknown) {
      if (
        error &&
        typeof error === "object" &&
        "code" in error &&
        error.code === "ENOENT"
      ) {
        return null;
      }
      throw error;
    }
  }

  async listDirectory(virtualPath: string): Promise<string[]> {
    const fullPath = this.resolveVirtualPath(virtualPath);

    try {
      const stats = await stat(fullPath);

      if (!stats.isDirectory()) {
        return [];
      }

      // This is a simple implementation - could be enhanced to match state behavior
      const entries = await readdir(fullPath);
      return entries.map((name) => {
        const vPath = virtualPath.endsWith("/")
          ? `${virtualPath}${name}`
          : `${virtualPath}/${name}`;
        return vPath;
      });
    } catch (error: unknown) {
      if (
        error &&
        typeof error === "object" &&
        "code" in error &&
        error.code === "ENOENT"
      ) {
        return [];
      }
      throw error;
    }
  }

  async writeFile(virtualPath: string, data: FileData): Promise<void> {
    const fullPath = this.resolveVirtualPath(virtualPath);

    const dir = path.dirname(fullPath);
    await mkdir(dir, { recursive: true });

    // Ensure content ends with newline (Unix text file convention)
    const contentToWrite = data.content.endsWith("\n")
      ? data.content
      : `${data.content}\n`;
    await writeFile(fullPath, contentToWrite, "utf8");
  }

  async deleteFile(virtualPath: string): Promise<void> {
    const fullPath = this.resolveVirtualPath(virtualPath);

    try {
      const stats = await stat(fullPath);

      if (stats.isFile()) {
        await unlink(fullPath);
      } else if (stats.isDirectory()) {
        await rm(fullPath, { recursive: true });
      }
    } catch (error: unknown) {
      if (
        error &&
        typeof error === "object" &&
        "code" in error &&
        error.code === "ENOENT"
      ) {
        // File doesn't exist, nothing to delete
        return;
      }
      throw error;
    }
  }

  async renameFile(
    oldVirtualPath: string,
    newVirtualPath: string,
    _existingData: FileData
  ): Promise<void> {
    const oldFull = this.resolveVirtualPath(oldVirtualPath);
    const newFull = this.resolveVirtualPath(newVirtualPath);

    // Ensure the old file exists
    const stats = await stat(oldFull);
    if (!stats.isFile() && !stats.isDirectory()) {
      throw new Error(`File not found: ${oldVirtualPath}`);
    }

    // Create parent directory for the new path if it doesn't exist
    const dir = path.dirname(newFull);
    await mkdir(dir, { recursive: true });

    await rename(oldFull, newFull);
  }

  validatePath(virtualPath: string): string {
    // Just normalize the virtual path
    let normalized = virtualPath;
    if (!normalized.startsWith("/")) {
      normalized = `/${normalized}`;
    }

    // Check for path traversal
    if (normalized.includes("..") || normalized.includes("~")) {
      throw new Error("Path traversal not allowed");
    }

    // Check allowed prefixes
    const allowed = this.allowedPrefixes.some((prefix) =>
      normalized.startsWith(prefix)
    );
    if (!allowed) {
      throw new Error(
        `Path must start with one of: ${JSON.stringify(this.allowedPrefixes)}`
      );
    }

    return normalized;
  }

  /**
   * Convert virtual path to filesystem path and validate.
   */
  private resolveVirtualPath(virtualPath: string): string {
    const relative = virtualPath.slice(1); // Remove leading /
    const fullPath = path.resolve(this.resolvedRootPath, relative);

    // Ensure path is within root
    if (!fullPath.startsWith(this.resolvedRootPath)) {
      throw new Error(`Path outside root directory: ${virtualPath}`);
    }

    return fullPath;
  }
}
