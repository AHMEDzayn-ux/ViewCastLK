from app.public_roster import PublicRosterStore


class FakeCursor:
    def __init__(self, *, existing: bool, inserted: bool = True):
        self.existing = existing
        self.inserted = inserted
        self.calls = []
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, parameters):
        self.calls.append((" ".join(query.split()), parameters))
        if query.lstrip().lower().startswith("insert"):
            self.rowcount = 1 if self.inserted else 0

    def fetchone(self):
        return (1,) if self.existing else None


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def cursor(self):
        return self._cursor


def test_existing_public_channel_is_not_queued(monkeypatch):
    cursor = FakeCursor(existing=True)
    store = PublicRosterStore()
    monkeypatch.setattr(store, "_connect", lambda: FakeConnection(cursor))

    result = store._request_channel_collection(channel_id="UC-existing")

    assert result == "already_tracked"
    assert len(cursor.calls) == 1
    assert "public.channels" in cursor.calls[0][0]


def test_new_public_channel_creates_idempotent_roster_request(monkeypatch):
    cursor = FakeCursor(existing=False)
    store = PublicRosterStore()
    monkeypatch.setattr(store, "_connect", lambda: FakeConnection(cursor))

    result = store._request_channel_collection(channel_id="UC-new")

    assert result == "requested"
    assert len(cursor.calls) == 2
    insert_query, parameters = cursor.calls[1]
    assert "public.roster_requests" in insert_query
    assert "on conflict (channel_id) do nothing" in insert_query.lower()
    assert parameters == ("UC-new",)
