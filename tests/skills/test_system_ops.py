"""
tests/skills/test_system_ops.py
===============================
Unit tests for the system execution and diagnostics skill in skills/system_ops.py.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

from skills.system_ops import get_system_info, list_processes, run_command


class TestSystemOpsSkills:
    # -----------------------------------------------------------------------
    # command execution
    # -----------------------------------------------------------------------

    def test_run_command_success(self) -> None:
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "hello stdout"
        mock_process.stderr = ""

        with patch("subprocess.run", return_value=mock_process) as mock_sub:
            res = run_command("echo hello")

        assert "[Exit Code: 0]" in res
        assert "hello stdout" in res
        mock_sub.assert_called_once_with(
            "echo hello", shell=True, text=True, capture_output=True, timeout=30.0
        )

    def test_run_command_success_with_stderr(self) -> None:
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.stdout = "partial success"
        mock_process.stderr = "some warning"

        with patch("subprocess.run", return_value=mock_process):
            res = run_command("ls /fake")

        assert "[Exit Code: 1]" in res
        assert "partial success" in res
        assert "stderr:\nsome warning" in res

    def test_run_command_no_output(self) -> None:
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = ""
        mock_process.stderr = ""

        with patch("subprocess.run", return_value=mock_process):
            res = run_command("touch test.txt")

        assert "Command executed successfully with no output." in res
        assert "[Exit Code: 0]" in res

    def test_run_command_timeout(self) -> None:
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="sleep 100", timeout=30.0)
        ):
            res = run_command("sleep 100")

        assert "timed out after 30 seconds" in res

    def test_run_command_exception(self) -> None:
        with patch("subprocess.run", side_effect=OSError("Command not found")):
            res = run_command("invalidcommand")

        assert "Error executing command: Command not found" in res

    # -----------------------------------------------------------------------
    # system diagnostics
    # -----------------------------------------------------------------------

    def test_get_system_info_with_and_without_psutil(self) -> None:
        # 1. Test with psutil available
        mock_mem = MagicMock()
        mock_mem.total = 16 * (1024**3)
        mock_mem.available = 8 * (1024**3)
        mock_mem.percent = 50.0

        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value = mock_mem

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            res_with = get_system_info()

        assert "[System Diagnostics]" in res_with
        assert "OS:" in res_with
        assert "Architecture:" in res_with
        assert "Memory (RAM): 50.0% used | 8192.0 MB available" in res_with

        # 2. Test without psutil
        with patch.dict("sys.modules", {"psutil": None}):
            res_without = get_system_info()

        assert "detailed RAM stats unavailable" in res_without

    # -----------------------------------------------------------------------
    # process listing
    # -----------------------------------------------------------------------

    def test_list_processes_with_psutil(self) -> None:
        mock_p1 = MagicMock()
        mock_p1.info = {"pid": 1, "name": "systemd", "username": "root"}
        mock_p2 = MagicMock()
        mock_p2.info = {"pid": 10, "name": "python", "username": "user"}

        mock_psutil = MagicMock()
        mock_psutil.process_iter.return_value = [mock_p1, mock_p2]

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            res = list_processes()

        assert "Active Processes" in res
        assert "systemd" in res
        assert "python" in res
        assert "1        " in res
        assert "10       " in res

    def test_list_processes_fallback_cmd(self) -> None:
        # When psutil is missing, fallback to subprocess executing "ps -ef" or "tasklist"
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "1 root systemd\n10 user python"

        with (
            patch.dict("sys.modules", {"psutil": None}),
            patch("subprocess.run", return_value=mock_process) as mock_sub,
        ):
            res = list_processes()

        assert "Active Processes (psutil fallback" in res
        assert "systemd" in res
        assert "python" in res
        mock_sub.assert_called_once()

    def test_list_processes_fallback_failure(self) -> None:
        with (
            patch.dict("sys.modules", {"psutil": None}),
            patch("subprocess.run", side_effect=Exception("Execution failed")),
        ):
            res = list_processes()

        assert "Could not list processes" in res
