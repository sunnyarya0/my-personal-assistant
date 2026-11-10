"""
tests/skills/test_datetime_ops.py
=================================
Unit tests for the calendar and date/time tracking skill in skills/datetime_ops.py.
"""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

from core.llm_client import LLMResponse
from skills.datetime_ops import get_current_datetime, parse_date


class TestDatetimeOpsSkills:
    def test_get_current_datetime(self) -> None:
        res = get_current_datetime()
        assert "Current Datetime:" in res
        assert "Weekday:" in res
        assert "Timezone:" in res

    def test_parse_date_local_matching(self) -> None:
        now = datetime.datetime.now()

        # today / now
        assert parse_date("today") == now.strftime("%Y-%m-%d")
        assert parse_date("now") == now.strftime("%Y-%m-%d")

        # tomorrow
        tomorrow = (now + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        assert parse_date("tomorrow") == tomorrow

        # yesterday
        yesterday = (now - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        assert parse_date("yesterday") == yesterday

    def test_parse_date_local_weekdays(self) -> None:
        # Match weekdays (should calculate a valid future offset)
        res = parse_date("next Monday")
        # Verify shape (YYYY-MM-DD)
        assert len(res) == 10
        assert res[4] == "-" and res[7] == "-"

    def test_parse_date_llm_fallback_success(self) -> None:
        mock_llm_client = MagicMock()
        mock_llm_client.chat.return_value = LLMResponse(
            content="2026-06-15",
            model="llama3.2",
            provider="ollama",
        )

        with patch("skills.datetime_ops.get_llm_client", return_value=mock_llm_client):
            res = parse_date("three weeks from now")

        assert res == "2026-06-15"
        mock_llm_client.chat.assert_called_once()
        args, kwargs = mock_llm_client.chat.call_args
        assert "three weeks from now" in args[0]
        assert "Today's date is" in kwargs["system_prompt"]

    def test_parse_date_llm_fallback_unresolved(self) -> None:
        mock_llm_client = MagicMock()
        mock_llm_client.chat.return_value = LLMResponse(
            content="UNKNOWN",
            model="llama3.2",
            provider="ollama",
        )

        with patch("skills.datetime_ops.get_llm_client", return_value=mock_llm_client):
            res = parse_date("some weird context")

        assert "Could not resolve natural language date" in res

    def test_parse_date_llm_fallback_error(self) -> None:
        mock_llm_client = MagicMock()
        mock_llm_client.chat.side_effect = Exception("LLM connection timed out")

        with patch("skills.datetime_ops.get_llm_client", return_value=mock_llm_client):
            res = parse_date("3 days from yesterday")

        assert "Error resolving date: LLM connection timed out" in res
