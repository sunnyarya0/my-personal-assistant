"""
tests/skills/test_file_ops.py
=============================
Unit tests for the sandboxed file operations skill in skills/file_ops.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch
import pytest

from core.config import settings
from skills.file_ops import (
    _safe_path,
    delete_file,
    list_directory,
    read_file,
    search_files,
    write_file,
)


class TestFileOpsSkills:
    @pytest.fixture(autouse=True)
    def setup_sandbox(self, tmp_path) -> None:
        """Override the workspace directory with a clean temporary directory."""
        self.workspace = tmp_path / "workspace"
        self.workspace.mkdir()
        
        with patch.object(settings, "workspace_dir", str(self.workspace)):
            yield

    # -----------------------------------------------------------------------
    # Sandboxing (_safe_path)
    # -----------------------------------------------------------------------

    def test_safe_path_resolves_inside_sandbox(self) -> None:
        # Relative path
        p1 = _safe_path("notes.txt")
        assert p1 == self.workspace / "notes.txt"

        # Subdirectory relative path
        p2 = _safe_path("docs/conf.py")
        assert p2 == self.workspace / "docs" / "conf.py"

        # Absolute path inside sandbox
        abs_path = str(self.workspace / "data.csv")
        p3 = _safe_path(abs_path)
        assert p3 == self.workspace / "data.csv"

        # Workspace root itself
        p4 = _safe_path(".")
        assert p4 == self.workspace.resolve()

    def test_safe_path_blocks_escape(self) -> None:
        # Directory traversal (../)
        with pytest.raises(ValueError, match="Path Traversal Blocked"):
            _safe_path("../escaped.txt")

        with pytest.raises(ValueError, match="Path Traversal Blocked"):
            _safe_path("docs/../../escaped.txt")

        # Absolute paths outside workspace
        with pytest.raises(ValueError, match="Path Traversal Blocked"):
            _safe_path("/etc/passwd")

    # -----------------------------------------------------------------------
    # Operations
    # -----------------------------------------------------------------------

    def test_write_and_read_file_success(self) -> None:
        file_name = "test.txt"
        content = "Hello from Personal Assistant!"

        # Write
        write_res = write_file(file_name, content)
        assert "Successfully wrote" in write_res
        assert (self.workspace / file_name).exists()
        assert (self.workspace / file_name).read_text(encoding="utf-8") == content

        # Read
        read_res = read_file(file_name)
        assert read_res == content

    def test_read_missing_file_returns_error(self) -> None:
        res = read_file("does-not-exist.txt")
        assert "Error: File" in res
        assert "does-not-exist.txt" in res

    def test_read_directory_as_file_returns_error(self) -> None:
        dir_name = "subfolder"
        (self.workspace / dir_name).mkdir()
        res = read_file(dir_name)
        assert "is not a file" in res

    def test_write_nested_path_creates_directories(self) -> None:
        path = "nested/docs/file.md"
        res = write_file(path, "content")
        assert "Successfully wrote" in res
        assert (self.workspace / "nested" / "docs" / "file.md").exists()

    def test_list_directory(self) -> None:
        # Empty directory
        empty_res = list_directory()
        assert "is empty" in empty_res

        # Add files & folder
        (self.workspace / "file1.txt").write_text("1")
        (self.workspace / "file2.log").write_text("2")
        (self.workspace / "subfolder").mkdir()

        res = list_directory()
        assert "Contents of directory" in res
        assert "file1.txt" in res
        assert "file2.log" in res
        assert "[DIR] subfolder" in res

    def test_list_missing_directory_returns_error(self) -> None:
        res = list_directory("missing_folder")
        assert "does not exist" in res

    def test_search_files(self) -> None:
        # Create test structures
        (self.workspace / "doc1.txt").write_text("x")
        (self.workspace / "doc2.md").write_text("x")
        (self.workspace / "subdir").mkdir()
        (self.workspace / "subdir" / "doc3.txt").write_text("x")

        # Search matching *.txt
        res = search_files(pattern="*.txt")
        assert "Found 2 matching file(s)" in res
        assert "- doc1.txt" in res
        assert "- subdir/doc3.txt" in res
        assert "doc2.md" not in res

        # Search directory subfolder
        res_sub = search_files(directory="subdir", pattern="*.txt")
        assert "Found 1 matching file(s)" in res_sub
        assert "subdir/doc3.txt" in res_sub

        # No matches
        no_matches = search_files(pattern="*.csv")
        assert "No files matching" in no_matches

    def test_delete_file_success(self) -> None:
        file_path = "delete-me.txt"
        (self.workspace / file_path).write_text("bye")

        res = delete_file(file_path)
        assert "Successfully deleted file" in res
        assert not (self.workspace / file_path).exists()

    def test_delete_missing_file_returns_error(self) -> None:
        res = delete_file("no-file.txt")
        assert "does not exist" in res
