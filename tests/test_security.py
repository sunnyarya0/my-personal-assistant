"""
tests/test_security.py
======================
Unit and integration tests for the security hardening changes in Sprint 6.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from core.agent import _redact_secrets
from skills.system_ops import _is_command_safe, run_command
from skills.web_search import _is_domain_allowed, fetch_webpage
from UI.server import app, rate_limit_records


class TestCommandSanitizer:
    def test_banned_words_blocked(self) -> None:
        unsafe_commands = [
            "sudo apt-get install git",
            "shutdown -h now",
            "reboot",
            "dd if=/dev/zero of=/dev/sdb",
            "mkfs.ext4 /dev/sda1",
            "chown -R user:group /var/www",
            "chmod +x /etc/shadow",
        ]
        for cmd in unsafe_commands:
            is_safe, reason = _is_command_safe(cmd)
            assert not is_safe
            assert "Unauthorized" in reason or "keyword" in reason

    def test_rm_path_traversal_blocked(self) -> None:
        unsafe_deletes = [
            "rm -rf ../secret",
            "rmdir ../../etc",
            "rm -rf ~/downloads",
            "rm -f /etc/passwd",
        ]
        with patch("skills.system_ops.settings") as mock_settings:
            mock_settings.workspace_dir = "/workspace"
            for cmd in unsafe_deletes:
                is_safe, reason = _is_command_safe(cmd)
                assert not is_safe
                assert "blocked" in reason or "outside workspace" in reason

    def test_safe_commands_allowed(self) -> None:
        safe_commands = [
            "echo 'hello world'",
            "ls -la",
            "git status",
            "python --version",
        ]
        for cmd in safe_commands:
            is_safe, reason = _is_command_safe(cmd)
            assert is_safe
            assert reason == ""

    def test_safe_deletes_inside_workspace_allowed(self) -> None:
        with patch("skills.system_ops.settings") as mock_settings:
            mock_settings.workspace_dir = "/tmp/my-workspace"
            is_safe, reason = _is_command_safe("rm /tmp/my-workspace/temp.txt")
            assert is_safe
            assert reason == ""

    def test_run_command_intercepts_unsafe(self) -> None:
        res = run_command("sudo rm -rf /")
        assert "Error: Command execution blocked due to security restrictions" in res


class TestScraperDomainFilter:
    def test_empty_allowed_domains_permits_all(self) -> None:
        with patch("skills.web_search.settings") as mock_settings:
            mock_settings.allowed_domains = []
            assert _is_domain_allowed("https://google.com")
            assert _is_domain_allowed("https://wikipedia.org/wiki/Pytest")

    def test_restricted_allowed_domains(self) -> None:
        with patch("skills.web_search.settings") as mock_settings:
            mock_settings.allowed_domains = ["wikipedia.org", "github.com"]

            # Direct and parent domain match
            assert _is_domain_allowed("https://wikipedia.org")
            assert _is_domain_allowed("https://en.wikipedia.org/wiki/Main_Page")
            assert _is_domain_allowed("https://github.com/profile")
            assert _is_domain_allowed("github.com/sunnyarya")

            # Non-matching domain
            assert not _is_domain_allowed("https://google.com")
            assert not _is_domain_allowed("https://yahoo.com/news")

    def test_fetch_webpage_blocks_unauthorized_domain(self) -> None:
        with patch("skills.web_search.settings") as mock_settings:
            mock_settings.allowed_domains = ["wikipedia.org"]
            res = fetch_webpage("https://google.com")
            assert "Error: Access to domain 'google.com' is blocked by security policies." in res


class TestLogCredentialRedaction:
    def test_redact_secrets_simple_string(self) -> None:
        secret = "secret-gemini-key-12345"
        text = f"Failed calling Gemini API using key: {secret}. Access Denied."
        redacted = _redact_secrets(text, secret)
        assert secret not in redacted
        assert "[REDACTED_API_KEY]" in redacted

    def test_redact_secrets_recursive_structures(self) -> None:
        secret = "secret-token-key"
        complex_data = {
            "key1": "value1",
            "api_used": secret,
            "list_key": ["item1", f"auth {secret}", "item3"],
            "nested": {"credential": f"key={secret}"},
        }
        redacted = _redact_secrets(complex_data, secret)

        assert redacted["api_used"] == "[REDACTED_API_KEY]"
        assert redacted["list_key"][1] == "auth [REDACTED_API_KEY]"
        assert redacted["nested"]["credential"] == "key=[REDACTED_API_KEY]"
        assert redacted["key1"] == "value1"

    def test_redact_secrets_empty_key_no_ops(self) -> None:
        data = "some raw contents"
        assert _redact_secrets(data, "") == data


class TestServerSecurityEndPoints:
    @pytest.fixture(autouse=True)
    def setup_client(self) -> None:
        self.client = TestClient(app)

    def test_rate_limiting_triggered(self) -> None:
        # Mock settings to have low rate limit
        with patch("UI.server.settings") as mock_settings:
            mock_settings.rate_limit_rpm = 2

            # Reset rate limit dictionary
            rate_limit_records.clear()

            # Make 3 fast requests
            res1 = self.client.get("/api/memory")
            res2 = self.client.get("/api/memory")
            res3 = self.client.get("/api/memory")

            assert res1.status_code == 200
            assert res2.status_code == 200
            assert res3.status_code == 429
            assert "Too Many Requests" in res3.json()["detail"]

    def test_task_id_alphanumeric_validation(self) -> None:
        # Pass a string containing illegal characters (dashes)
        res = self.client.get("/api/tasks/invalid-task-id")
        assert res.status_code == 400
        assert "Invalid task ID format" in res.json()["detail"]

    def test_task_id_absolute_sandboxing_blocked(self) -> None:
        # Pass a task ID that matches regex but tries to resolve outside log_dir
        with patch("UI.server.settings") as mock_settings:
            mock_settings.log_dir = "/tmp/assistant-logs"
            mock_settings.rate_limit_rpm = 60
            res = self.client.get("/api/tasks/some_fake_session")
            assert res.status_code == 404
