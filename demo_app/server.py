"""Loopback-only UI server with an explicit static allowlist and same-origin writes."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .service import APIError, Application
from .storage import canonical

WEB = Path(__file__).resolve().parents[1] / "demo_web"


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def reply(self, status, payload):
            content = canonical(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(content)

        def failure(self, error):
            self.reply(error.status, {"error": {"code": error.code, "message": error.message, "retryable": error.retryable},
                                      "view_revision": app.view()["view_revision"]})

        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if self.headers.get("Host", "") not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
                    raise APIError(403, "host_rejected", "只接受本机地址访问。")
                if path == "/api/state":
                    return self.reply(200, app.view())
                if path == "/api/report":
                    return self.reply(200, app.report())
                if path == "/api/evaluation":
                    return self.reply(200, app.view()["evaluation"])
                routes = {"/": ("index.html", "text/html; charset=utf-8"),
                          "/index.html": ("index.html", "text/html; charset=utf-8"),
                          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                          "/app.mjs": ("app.mjs", "text/javascript; charset=utf-8"),
                          "/charts.mjs": ("charts.mjs", "text/javascript; charset=utf-8"),
                          "/styles.css": ("styles.css", "text/css; charset=utf-8"),
                          "/style.css": ("style.css", "text/css; charset=utf-8")}
                if path not in routes:
                    raise APIError(404, "not_found", "未找到该资源。")
                name, content_type = routes[path]
                if not (WEB / name).is_file():
                    raise APIError(503, "page_unavailable", "页面尚未就绪。", True)
                content = (WEB / name).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(content)
            except APIError as exc:
                self.failure(exc)

        def do_POST(self):
            try:
                # Port and host are part of origin; don't allow arbitrary sites to mutate localhost.
                host = self.headers.get("Host", "")
                allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
                if host not in allowed or self.headers.get("Origin", "http://" + host) != "http://" + host:
                    raise APIError(403, "origin_rejected", "只接受本地页面发起的修改。")
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise APIError(400, "invalid_content_type", "请求须为JSON。")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32768:
                    raise APIError(400, "invalid_size", "请求大小无效。")
                def reject_constant(value):
                    raise ValueError("Nonfinite number")
                body = json.loads(self.rfile.read(length), parse_constant=reject_constant)
                if not isinstance(body, dict):
                    raise ValueError("Object required")
                path = urlsplit(self.path).path
                if path == "/api/analyze":
                    return self.reply(202, app.analyze(body))
                if path == "/api/holdings":
                    return self.reply(200, app.holdings(body))
                if path == "/api/alerts":
                    return self.reply(200, app.alerts(body))
                raise APIError(404, "not_found", "未找到该接口。")
            except APIError as exc:
                self.failure(exc)
            except (ValueError, TypeError, KeyError):
                self.failure(APIError(400, "invalid_request", "请求内容无效。"))
            except Exception:
                self.failure(APIError(503, "service_error", "服务未能处理请求，请重试。", True))
    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args()
    app = Application(args.state_dir)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    print(f"Demo ready at http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
