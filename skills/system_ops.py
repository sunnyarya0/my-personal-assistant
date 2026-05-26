"""
skills/system_ops.py
====================
System execution and diagnostics skill for the Personal AI Agent Assistant.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from core.config import settings
from core.registry import register_tool


def _is_command_safe(command: str) -> tuple[bool, str]:
    """
    Check if a shell command is safe to execute.
    Returns (True, "") if safe, or (False, "reason") if dangerous.
    """
    cmd_lower = command.lower().strip()

    # 1. Block extremely dangerous binaries and options
    banned_words = [
        "sudo", "shutdown", "reboot", "poweroff", "init 0", "init 6",
        "mkfs", "dd", "fdisk", "parted", "chown", "chmod"
    ]
    for word in banned_words:
        pattern = rf"\b{re.escape(word)}\b"
        if re.search(pattern, cmd_lower):
            return False, f"Unauthorized command pattern or keyword '{word}' detected."

    # 2. Block dangerous rm (deletes) outside the workspace
    if "rm " in cmd_lower or "rmdir " in cmd_lower:
        if re.search(rf"\b(rm|rmdir)\b", cmd_lower):
            if ".." in cmd_lower:
                return False, "Directory traversal ('..') blocked in command."
            if "~" in cmd_lower:
                return False, "Home directory deletion ('~') blocked in command."

            # Check absolute paths
            if "/" in cmd_lower:
                workspace = settings.workspace_dir.lower()
                matches = re.findall(r'/\S*', cmd_lower)
                for path in matches:
                    # Allow path if it starts with the workspace path
                    # Handle trailing slashes or subdirectories cleanly
                    if path and not path.startswith(workspace):
                        return False, f"Potentially unsafe path deletion '{path}' outside workspace."

    return True, ""


@register_tool(description="Execute a shell command on the local system.", destructive=True)
def run_command(command: str) -> str:
    """
    Run a terminal shell command. Safe Mode confirmation is enforced.

    Args:
        command: The full command line string to run.

    Returns:
        The combined stdout/stderr output or an error message.
    """
    # Enforce command safety checks
    is_safe, reason = _is_command_safe(command)
    if not is_safe:
        return f"Error: Command execution blocked due to security restrictions: {reason}"
    try:
        # Run command with 30 second timeout
        res = subprocess.run(
            command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=30.0,
        )
        
        output_parts = []
        if res.stdout:
            output_parts.append(res.stdout)
        if res.stderr:
            output_parts.append(f"stderr:\n{res.stderr}")
            
        output = "\n".join(output_parts).strip()
        
        status_msg = f"[Exit Code: {res.returncode}]"
        if not output:
            return f"Command executed successfully with no output. {status_msg}"
            
        return f"{status_msg}\n{output}"
    except subprocess.TimeoutExpired:
        return "Error: Command execution timed out after 30 seconds."
    except Exception as exc:
        return f"Error executing command: {exc}"


@register_tool(description="Get detailed operating system, CPU, disk, and memory diagnostics.")
def get_system_info() -> str:
    """
    Query platform architecture, OS, Python version, disk, and memory stats.

    Returns:
        A structured text report detailing system status.
    """
    lines = ["[System Diagnostics]"]
    lines.append(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    lines.append(f"Architecture: {platform.machine()}")
    lines.append(f"Processor: {platform.processor() or 'Unknown'}")
    lines.append(f"Python Version: {sys.version.split()[0]} ({sys.executable})")

    # Disk usage for the workspace directory
    try:
        usage = shutil.disk_usage(".")
        total_gb = usage.total / (1024**3)
        used_gb = usage.used / (1024**3)
        free_gb = usage.free / (1024**3)
        lines.append(
            f"Disk Space: {used_gb:.2f} GB used / {free_gb:.2f} GB free "
            f"({total_gb:.2f} GB total)"
        )
    except Exception:
        lines.append("Disk Space: (Could not retrieve disk stats)")

    # Memory usage with psutil fallback
    try:
        import psutil  # type: ignore
        mem = psutil.virtual_memory()
        total_mb = mem.total / (1024**2)
        available_mb = mem.available / (1024**2)
        lines.append(
            f"Memory (RAM): {mem.percent}% used | "
            f"{available_mb:.1f} MB available / {total_mb:.1f} MB total"
        )
    except ImportError:
        lines.append("Memory (RAM): (psutil not installed, detailed RAM stats unavailable)")

    return "\n".join(lines)


@register_tool(description="List running system processes.")
def list_processes() -> str:
    """
    List currently running processes on the system.

    Returns:
        A formatted list of active processes.
    """
    # 1. Try with psutil
    try:
        import psutil  # type: ignore
        proc_list = []
        for proc in psutil.process_iter(["pid", "name", "username"]):
            try:
                info = proc.info
                proc_list.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        
        # Sort by PID and take top 40
        proc_list.sort(key=lambda p: p["pid"])
        top_procs = proc_list[:40]

        lines = [f"Active Processes (Top {len(top_procs)} via psutil):\n"]
        lines.append(f"{'PID':<8} {'USERNAME':<15} {'NAME'}")
        lines.append("-" * 45)
        for p in top_procs:
            lines.append(f"{p['pid']:<8} {str(p['username']):<15} {p['name']}")
        
        return "\n".join(lines)

    except ImportError:
        # 2. Fallback to standard CLI utilities
        system = platform.system()
        try:
            if system == "Windows":
                cmd = "tasklist"
            else:
                cmd = "ps -ef | head -n 40"

            res = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=5)
            if res.returncode == 0 and res.stdout:
                return (
                    f"Active Processes (psutil fallback via '{cmd}'):\n\n"
                    f"{res.stdout.strip()}"
                )
        except Exception:
            pass

        return "Error: Could not list processes. 'psutil' is not installed and fallback command failed."
