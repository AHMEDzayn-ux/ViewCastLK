import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.youtube import ChannelLookupException, parse_channel_identifier


def test_parse_handle_with_at():
    lookup_type, value = parse_channel_identifier("@wasthi")
    assert lookup_type == "handle"
    assert value == "@wasthi"


def test_parse_handle_url():
    lookup_type, value = parse_channel_identifier("https://www.youtube.com/@wasthi")
    assert lookup_type == "handle"
    assert value == "@wasthi"


def test_parse_channel_id():
    channel_id = "UCX6OQ3DkcsbYNE6H8uQQuVA"
    lookup_type, value = parse_channel_identifier(channel_id)
    assert lookup_type == "id"
    assert value == channel_id


def test_parse_channel_id_url():
    channel_id = "UCX6OQ3DkcsbYNE6H8uQQuVA"
    lookup_type, value = parse_channel_identifier(
        f"https://www.youtube.com/channel/{channel_id}"
    )
    assert lookup_type == "id"
    assert value == channel_id


def test_parse_invalid_identifier_with_spaces():
    with pytest.raises(ChannelLookupException) as exc_info:
        parse_channel_identifier("invalid handle name")
    assert exc_info.value.code == "invalid_channel_identifier"


def test_parse_empty_identifier():
    with pytest.raises(ChannelLookupException) as exc_info:
        parse_channel_identifier("   ")
    assert exc_info.value.code == "invalid_channel_identifier"
