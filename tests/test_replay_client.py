from __future__ import annotations

from pokestrategist.data.replay_client import ReplayClient


class _FakeResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSession:
    def __init__(self, pages: dict[int, object]) -> None:
        self.pages = pages
        self.headers: dict[str, str] = {}
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, params: dict | None = None, timeout: float | None = None) -> _FakeResponse:
        del url, timeout
        page = int((params or {}).get("page", 1))
        self.calls.append(dict(params or {}))
        return _FakeResponse(self.pages.get(page, []))


def test_search_max_pages_zero_scans_until_empty_page(tmp_path):
    session = _FakeSession(
        {
            1: [
                {
                    "id": "gen9ou-1",
                    "format": "gen9ou",
                    "players": ["Alice", "Bob"],
                    "rating": 1600,
                    "uploadtime": 1700000001,
                }
            ],
            2: [
                {
                    "id": "gen9ou-2",
                    "format": "gen9ou",
                    "players": ["Carol", "Dave"],
                    "rating": 1700,
                    "uploadtime": 1700000002,
                }
            ],
            3: [],
        }
    )
    client = ReplayClient(cache_dir=tmp_path, request_delay=0.0, session=session)

    hits = list(client.search(battle_format="gen9ou", max_pages=0, min_rating=1500))

    assert [hit.replay_id for hit in hits] == ["gen9ou-1", "gen9ou-2"]
    assert [call["page"] for call in session.calls] == [1, 2, 3]
