"""Serve the existing Ripple site and same-origin live APIs on loopback."""
import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .service import LiveError, LiveService

WEB = Path(__file__).resolve().parents[1] / "ripple_web"


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def local(self):
            host = self.headers.get("Host", "")
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            return host in allowed

        def reply(self, status, value):
            content = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if not self.local():
                return self.reply(403, {"error": "仅允许本机访问"})
            path = urlsplit(self.path).path
            try:
                if path == "/api/live":
                    return self.reply(200, service.market())
                if path == "/api/prediction":
                    return self.reply(200, service.prediction_state())
                if path.startswith("/api/"):
                    return self.reply(404, {"error": "未知接口"})
                relative = Path(unquote(path).lstrip("/") or "index.html")
                target = (WEB / relative).resolve()
                allowed_parent = relative.parent.as_posix() in {".", "blocks", "data"}
                if not allowed_parent or WEB not in target.parents or target.suffix not in {".html", ".mjs", ".css", ".json", ".svg"} or not target.is_file():
                    return self.reply(404, {"error": "未找到该页面"})
                body = target.read_bytes()
                self.send_response(200)
                mime = "text/javascript" if target.suffix == ".mjs" else mimetypes.guess_type(str(target))[0] or "text/plain"
                self.send_header("Content-Type", mime + "; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            except LiveError as exc:
                self.reply(exc.status, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception:
                self.reply(503, {"error": "暂时无法读取数据，请重试。"})

        def do_POST(self):
            host = self.headers.get("Host", "")
            if not self.local() or self.headers.get("Origin", "http://" + host) != "http://" + host:
                return self.reply(403, {"error": "仅允许本机页面发起分析"})
            if urlsplit(self.path).path != "/api/predict":
                return self.reply(404, {"error": "未知接口"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 2048 or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError()
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict) or set(body) != {"window_id"}:
                    raise ValueError()
                self.reply(202, service.predict(body["window_id"]))
            except LiveError as exc:
                self.reply(exc.status, {"error": str(exc)})
            except (ValueError, TypeError):
                self.reply(400, {"error": "分析请求无效"})
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5176)
    args = parser.parse_args()
    service = LiveService()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(service))
    print(f"Ripple live ready: http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
