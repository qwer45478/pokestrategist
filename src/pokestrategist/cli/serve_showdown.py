from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from pokestrategist.serving.showdown import PokeStrategistLocalPredictor, health_payload, suggestion_payload_from_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve local PokeStrategist suggestions for the Showdown extension.")
    parser.add_argument("--checkpoint", required=True, help="Path to a v1 decision-model checkpoint.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--hidden-candidate-topk", type=int, default=None)
    parser.add_argument("--disable-team-preview-prior", action="store_true")
    parser.add_argument("--disable-usage-priors", action="store_true")
    parser.add_argument("--usage-data-dir", default=None)
    return parser.parse_args()


def _make_handler(predictor: PokeStrategistLocalPredictor, *, topk: int) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self._send_json(200, {"ok": True})

        def do_GET(self) -> None:
            if self.path in {"/", "/health", "/api/health"}:
                self._send_json(200, health_payload(predictor))
                return
            self._send_json(404, {"ok": False, "error": "Not found."})

        def do_POST(self) -> None:
            if self.path != "/api/suggest":
                self._send_json(404, {"ok": False, "error": "Not found."})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length)
                payload = suggestion_payload_from_json(predictor, body, topk=topk)
                self._send_json(200, payload)
            except ValueError as error:
                self._send_json(400, {"ok": False, "error": str(error)})
            except Exception as error:  # pragma: no cover - defensive server wrapper
                self._send_json(500, {"ok": False, "error": str(error)})

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    args = _parse_args()
    predictor = PokeStrategistLocalPredictor(
        args.checkpoint,
        device=args.device,
        hidden_candidate_topk=args.hidden_candidate_topk,
        use_team_preview_prior=not args.disable_team_preview_prior,
        use_usage_priors=not args.disable_usage_priors,
        usage_data_dir=args.usage_data_dir,
    )
    server = ThreadingHTTPServer((args.host, args.port), _make_handler(predictor, topk=args.topk))
    print(f"PokeStrategist Showdown service listening on http://{args.host}:{args.port}")
    print(json.dumps(health_payload(predictor), indent=2, ensure_ascii=False))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()