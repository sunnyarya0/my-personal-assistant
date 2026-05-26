"""
skills/file_ops.py
==================
Sandboxed file operations skill for the Personal AI Agent Assistant.

Ensures that all read, write, list, search, and delete operations are strictly
sandboxed within the configured workspace directory (`settings.workspace_dir`).
"""

from __future__ import annotations

import fnmatch
import os
import shutil
from pathlib import Path
from core.config import settings
from core.registry import register_tool


def _safe_path(path_str: str) -> Path:
    """
    Resolve and validate a path to ensure it lies strictly inside the workspace.

    Args:
        path_str: The relative or absolute path string.

    Returns:
        The fully resolved Path object inside the workspace sandbox.

    Raises:
        ValueError: If the path escapes the workspace sandbox.
    """
    workspace = Path(settings.workspace_dir).resolve()
    
    # Try to resolve path
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = workspace / path

    resolved = path.resolve()

    # Check if resolved path is relative to workspace or matches it
    try:
        if resolved == workspace or workspace in resolved.parents:
            return resolved
    except (ValueError, TypeError):
        pass

    # Backup string-based prefix check to be extra safe
    resolved_str = str(resolved)
    workspace_str = str(workspace)
    if resolved_str == workspace_str or resolved_str.startswith(workspace_str + os.sep):
        return resolved

    raise ValueError(
        f"Path Traversal Blocked: Path '{path_str}' resolves to '{resolved}' "
        f"which is outside of the workspace directory '{workspace}'."
    )


@register_tool(description="Read the text contents of a file inside the workspace.")
def read_file(path: str) -> str:
    """
    Safely read a text file inside the workspace sandbox.

    Args:
        path: Path to the file (relative to workspace or absolute).

    Returns:
        The content of the file or an error message.
    """
    try:
        safe_path = _safe_path(path)
        if not safe_path.exists():
            return f"Error: File '{path}' does not exist."
        if not safe_path.is_file():
            return f"Error: '{path}' is not a file."
        
        # Read text content
        return safe_path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Error reading file: {exc}"


@register_tool(description="Write text contents to a file inside the workspace.")
def write_file(path: str, content: str) -> str:
    """
    Safely create or overwrite a text file inside the workspace sandbox.

    Args:
        path: Path to the file to write (relative to workspace or absolute).
        content: The text content to write.

    Returns:
        Confirmation message or an error message.
    """
    try:
        safe_path = _safe_path(path)
        
        # Ensure parent directories exist
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        
        safe_path.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to file: {path}"
    except Exception as exc:
        return f"Error writing file: {exc}"


@register_tool(description="List the contents of a directory inside the workspace.")
def list_directory(path: str = "") -> str:
    """
    Safely list all files and subdirectories inside a workspace directory.

    Args:
        path: Optional subdirectory path (defaults to the workspace root).

    Returns:
        A formatted list of contents or an error message.
    """
    try:
        safe_path = _safe_path(path or ".")
        if not safe_path.exists():
            return f"Error: Directory '{path}' does not exist."
        if not safe_path.is_dir():
            return f"Error: '{path}' is not a directory."

        entries = sorted(safe_path.iterdir())
        if not entries:
            return f"Directory '{path or '.'}' is empty."

        lines = [f"Contents of directory '{path or '.'}':\n"]
        for entry in entries:
            prefix = "[DIR] " if entry.is_dir() else "      "
            size_str = f" ({entry.stat().st_size} bytes)" if entry.is_file() else ""
            lines.append(f"{prefix}{entry.name}{size_str}")

        return "\n".join(lines)
    except Exception as exc:
        return f"Error listing directory: {exc}"


@register_tool(description="Search for files in a workspace directory matching a glob pattern.")
def search_files(directory: str = "", pattern: str = "*") -> str:
    """
    Search recursively for files matching a glob pattern inside the workspace.

    Args:
        directory: Optional subdirectory to search within (defaults to workspace root).
        pattern: Glob pattern to match (e.g. '*.txt', '*config*').

    Returns:
        A list of matching file paths relative to the workspace, or an error.
    """
    try:
        base_dir = _safe_path(directory or ".")
        workspace = Path(settings.workspace_dir).resolve()

        if not base_dir.exists() or not base_dir.is_dir():
            return f"Error: Directory '{directory}' is not a valid directory."

        matches = []
        for root, _, files in os.walk(base_dir):
            for filename in fnmatch.filter(files, pattern):
                full_path = Path(root) / filename
                # Enforce sandbox on os.walk findings too just to be bulletproof
                try:
                    safe_p = _safe_path(str(full_path))
                    rel_path = safe_p.relative_to(workspace)
                    matches.append(str(rel_path))
                except ValueError:
                    continue

        if not matches:
            return f"No files matching '{pattern}' found in '{directory or '.'}'."

        lines = [f"Found {len(matches)} matching file(s) in '{directory or '.'}':\n"]
        for m in sorted(matches):
            lines.append(f"- {m}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Error searching files: {exc}"


@register_tool(description="Delete a file inside the workspace.", destructive=True)
def delete_file(path: str) -> str:
    """
    Safely delete a file inside the workspace sandbox. Triggers Safe Mode confirmation.

    Args:
        path: Path to the file to delete (relative to workspace or absolute).

    Returns:
        Confirmation message or an error message.
    """
    try:
        safe_path = _safe_path(path)
        if not safe_path.exists():
            return f"Error: File '{path}' does not exist."
        if not safe_path.is_file():
            return f"Error: '{path}' is not a file. Use a directory specific command for directories."

        safe_path.unlink()
        return f"Successfully deleted file: {path}"
    except Exception as exc:
        return f"Error deleting file: {exc}"
