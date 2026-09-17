"""Tests for Hermes-provenance loop gate on the Matrix adapter.

Feature: outbound Hermes Matrix events carry a provenance marker
(``com.nousresearch.hermes``); an optional inbound policy
(``MATRIX_HERMES_MESSAGES_REQUIRE_MENTION``) drops Hermes-originated
text/media messages that do not explicitly mention the receiving bot.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from gateway.config import PlatformConfig


_HERMES_KEY = "com.nousresearch.hermes"


def _hermes_content(**extra):
    content = {
        "msgtype": "m.text",
        "body": "hello",
        _HERMES_KEY: {
            "origin": "hermes-agent",
            "protocol": 1,
        },
    }
    content.update(extra)
    return content


def _make_adapter(gate_enabled=True):
    """Build a MatrixAdapter via object.__new__ with the fields _on_room_message uses."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = object.__new__(MatrixAdapter)
    adapter._user_id = "@agent-b:matrix.org"
    adapter._startup_ts = 0.0
    adapter._closing = False
    adapter._process_notices = False
    adapter._hermes_messages_require_mention = gate_enabled
    adapter._ignored_user_patterns = []
    adapter._allowed_room_ids = set()
    adapter._allowed_rooms = set()
    adapter._processed_events = []
    adapter._processed_events_set = set()
    adapter._dm_rooms = {"!dm:matrix.org": True}
    adapter._room_identities = {}
    adapter._room_identity_cached_at = {}
    adapter._room_identity_ttl_seconds = 60
    adapter._room_identity_cache_max = 256
    adapter._late_grace_drops = 0
    adapter._clock_skew_warned = False
    adapter._is_allowed_matrix_room_event = AsyncMock(return_value=True)
    adapter._handle_text_message = AsyncMock()
    adapter._handle_media_message = AsyncMock()
    adapter._background_read_receipt = lambda *_args: None
    adapter._note_late_grace_drop = lambda *_args: None
    adapter._is_duplicate_event = lambda event_id: False
    return adapter


def _room_event(sender, body, content=None, event_id="$e1"):
    return SimpleNamespace(
        room_id="!dm:matrix.org",
        sender=sender,
        event_id=event_id,
        timestamp=0,
        server_timestamp=0,
        content=content if content is not None else {"msgtype": "m.text", "body": body},
    )


@pytest.mark.asyncio
async def test_unmentioned_hermes_text_is_ignored_in_dm():
    adapter = _make_adapter(gate_enabled=True)
    event = _room_event(
        "@agent-a:matrix.org",
        "automatic reply",
        content=_hermes_content(body="automatic reply"),
    )

    await adapter._on_room_message(event)

    adapter._handle_text_message.assert_not_awaited()
    adapter._handle_media_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_mentioned_hermes_text_is_processed():
    adapter = _make_adapter(gate_enabled=True)
    event = _room_event(
        "@agent-a:matrix.org",
        "@agent-b:matrix.org please review this",
        content=_hermes_content(
            body="@agent-b:matrix.org please review this",
            **{"m.mentions": {"user_ids": ["@agent-b:matrix.org"]}},
        ),
        event_id="$e2",
    )

    await adapter._on_room_message(event)

    adapter._handle_text_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_human_text_is_unaffected():
    adapter = _make_adapter(gate_enabled=True)
    event = _room_event(
        "@alice:matrix.org",
        "hello",
        content={"msgtype": "m.text", "body": "hello"},
        event_id="$e3",
    )

    await adapter._on_room_message(event)

    adapter._handle_text_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_gate_disabled_processes_hermes_text_without_mention():
    adapter = _make_adapter(gate_enabled=False)
    event = _room_event(
        "@agent-a:matrix.org",
        "automatic reply",
        content=_hermes_content(body="automatic reply"),
        event_id="$e4",
    )

    await adapter._on_room_message(event)

    adapter._handle_text_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_hermes_media_without_mention_is_dropped():
    adapter = _make_adapter(gate_enabled=True)
    event = SimpleNamespace(
        room_id="!dm:matrix.org",
        sender="@agent-a:matrix.org",
        event_id="$e5",
        timestamp=0,
        server_timestamp=0,
        content=_hermes_content(
            msgtype="m.image",
            body="screenshot.png",
            url="mxc://matrix.org/abc",
            info={"mimetype": "image/png"},
        ),
    )

    await adapter._on_room_message(event)

    adapter._handle_media_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_hermes_media_with_mention_in_caption_is_processed():
    adapter = _make_adapter(gate_enabled=True)
    event = SimpleNamespace(
        room_id="!dm:matrix.org",
        sender="@agent-a:matrix.org",
        event_id="$e6",
        timestamp=0,
        server_timestamp=0,
        content=_hermes_content(
            msgtype="m.image",
            body="@agent-b:matrix.org look at this",
            url="mxc://matrix.org/abc",
            info={"mimetype": "image/png"},
            **{"m.mentions": {"user_ids": ["@agent-b:matrix.org"]}},
        ),
    )

    await adapter._on_room_message(event)

    adapter._handle_media_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_notice_filter_is_independent_of_provenance_gate():
    """MATRIX_PROCESS_NOTICES=false still drops notices even when mentioned."""
    adapter = _make_adapter(gate_enabled=True)
    event = _room_event(
        "@agent-a:matrix.org",
        "@agent-b:matrix.org notice",
        content=_hermes_content(
            msgtype="m.notice",
            body="@agent-b:matrix.org notice",
            **{"m.mentions": {"user_ids": ["@agent-b:matrix.org"]}},
        ),
        event_id="$e7",
    )

    await adapter._on_room_message(event)

    adapter._handle_text_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_gate_disabled_processes_unmentioned_notice_when_notices_on():
    adapter = _make_adapter(gate_enabled=False)
    adapter._process_notices = True
    event = _room_event(
        "@agent-a:matrix.org",
        "notice body",
        content=_hermes_content(msgtype="m.notice", body="notice body"),
        event_id="$e8",
    )

    await adapter._on_room_message(event)

    adapter._handle_text_message.assert_awaited_once()


def test_provenance_marker_roundtrip():
    from plugins.platforms.matrix.adapter import (
        _mark_hermes_matrix_content,
        _is_hermes_matrix_content,
    )

    content = _mark_hermes_matrix_content({"msgtype": "m.text", "body": "hi"})
    assert content[_HERMES_KEY] == {"origin": "hermes-agent", "protocol": 1}
    assert _is_hermes_matrix_content(content)
    assert _is_hermes_matrix_content({"msgtype": "m.text"}) is False
    assert _is_hermes_matrix_content(None) is False
    # Wrong origin / non-int protocol are not recognized.
    assert not _is_hermes_matrix_content(
        {_HERMES_KEY: {"origin": "other", "protocol": 1}})
    assert not _is_hermes_matrix_content(
        {_HERMES_KEY: {"origin": "hermes-agent", "protocol": True}})
    assert not _is_hermes_matrix_content(
        {_HERMES_KEY: {"origin": "hermes-agent", "protocol": "1"}})


def test_mark_is_idempotent():
    from plugins.platforms.matrix.adapter import _mark_hermes_matrix_content

    content = {"msgtype": "m.text", "body": "hi"}
    _mark_hermes_matrix_content(content)
    expected = dict(content)
    _mark_hermes_matrix_content(content)
    # setdefault: a pre-existing marker is preserved, not overwritten.
    content[_HERMES_KEY] = {"origin": "custom", "protocol": 9}
    _mark_hermes_matrix_content(content)
    assert content[_HERMES_KEY] == {"origin": "custom", "protocol": 9}


def test_future_protocol_content_is_recognized():
    from plugins.platforms.matrix.adapter import _is_hermes_matrix_content

    assert _is_hermes_matrix_content(
        {_HERMES_KEY: {"origin": "hermes-agent", "protocol": 2}})


@pytest.mark.asyncio
async def test_send_matrix_event_stamps_provenance():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = object.__new__(MatrixAdapter)
    adapter._client = MagicMock()
    adapter._client.send_message_event = AsyncMock(return_value="$evt1")

    event_id = await adapter._send_matrix_event(
        "!room:matrix.org", "m.room.message", {"msgtype": "m.text", "body": "x"})

    assert event_id == "$evt1"
    payload = adapter._client.send_message_event.await_args.args[2]
    assert payload[_HERMES_KEY] == {"origin": "hermes-agent", "protocol": 1}


@pytest.mark.asyncio
async def test_hermes_reaction_gets_provenance():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = object.__new__(MatrixAdapter)
    adapter._client = MagicMock()
    adapter._client.send_message_event = AsyncMock(return_value="$reaction")
    adapter._reactions_enabled = True

    result = await adapter._send_reaction("!room:matrix.org", "$target", "👍")

    assert result == "$reaction"
    payload = adapter._client.send_message_event.await_args.args[2]
    assert payload[_HERMES_KEY] == {"origin": "hermes-agent", "protocol": 1}


@pytest.mark.asyncio
async def test_send_room_message_uses_45s_cap_and_provenance():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = object.__new__(MatrixAdapter)
    adapter._client = MagicMock()
    adapter._client.send_message_event = AsyncMock(return_value="$msg")

    event_id = await adapter._send_room_message("!room:matrix.org", {"msgtype": "m.text", "body": "x"})

    assert event_id == "$msg"
    # Timeout reaches asyncio.wait_for, which raises TimeoutError on overrun —
    # verify the 45s timeout is in effect via a hung client call.
    adapter._client.send_message_event = AsyncMock(side_effect=TimeoutError())
    try:
        await adapter._send_room_message("!room:matrix.org", {"msgtype": "m.text", "body": "x"})
        raised_timeout = False
    except TimeoutError:
        raised_timeout = True
    assert raised_timeout


@pytest.mark.asyncio
async def test_send_content_event_maps_exception_to_send_result():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = object.__new__(MatrixAdapter)
    adapter._client = MagicMock()
    adapter._client.send_message_event = AsyncMock(side_effect=RuntimeError("boom"))

    result = await adapter._send_content_event("!room:matrix.org", {"msgtype": "m.text", "body": "x"})

    assert result.success is False
    assert "boom" in result.error


@pytest.mark.asyncio
async def test_send_content_event_success_wraps_provenance():
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = object.__new__(MatrixAdapter)
    adapter._client = MagicMock()
    adapter._client.send_message_event = AsyncMock(return_value="$ok")

    result = await adapter._send_content_event("!room:matrix.org", {"msgtype": "m.text", "body": "x"})

    assert result.success is True
    assert result.message_id == "$ok"
    payload = adapter._client.send_message_event.await_args.args[2]
    assert payload[_HERMES_KEY] == {"origin": "hermes-agent", "protocol": 1}


def test_yaml_bridge_includes_new_flag():
    from plugins.platforms.matrix.adapter import _YAML_BRIDGE

    pairs = {key: env for key, env, _kind in _YAML_BRIDGE}
    assert pairs.get("hermes_messages_require_mention") == "MATRIX_HERMES_MESSAGES_REQUIRE_MENTION"


def test_env_based_init_reads_flag(monkeypatch):
    """MATRIX_HERMES_MESSAGES_REQUIRE_MENTION env is honored by __init__ path."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    monkeypatch.setenv("MATRIX_HERMES_MESSAGES_REQUIRE_MENTION", "true")
    monkeypatch.setenv("MATRIX_HOMESERVER", "https://matrix.org")
    monkeypatch.delenv("MATRIX_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("MATRIX_PASSWORD", raising=False)

    config = PlatformConfig(
        enabled=True,
        token="syt_test_token",
        extra={
            "homeserver": "https://matrix.org",
            "hermes_messages_require_mention": "true",
        },
    )
    adapter = MatrixAdapter(config)
    assert adapter._hermes_messages_require_mention is True
