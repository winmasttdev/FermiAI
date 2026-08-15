#!/usr/bin/env python3
import json, socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

NECKO_ADDR = ("192.168.0.100", 9001)

def ask_fermi(ctx):
    s = socket.create_connection(NECKO_ADDR, timeout=180)
    s.sendall((ctx + "\n").encode("utf-8"))
    data = b""
    while not data.endswith(b"\n"):
        c = s.recv(4096)
        if not c:
            break
        data += c
    s.close()
    return data.decode("utf-8").strip()

def build_context(messages):
    parts = []
    for m in messages:
        if m["role"] == "user":
            parts.append("you : " + m["content"])
        else:
            parts.append("friend : " + m["content"])
    ctx = " ".join(parts)
    if messages and messages[-1]["role"] == "user":
        ctx += " friend :"
    return ctx

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            try:
                with open("/home/winmastt/neko-llm/chatui/index.html", "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(404, b"not found")
        else:
            self._send(404, b"not found")

    def do_POST(self):
        if self.path != "/api/chat":
            self._send(404, b"not found")
            return
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
            messages = payload.get("messages", [])
            ctx = build_context(messages)
            reply = ask_fermi(ctx)
            self._send(200, json.dumps({"reply": reply}).encode("utf-8"))
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}).encode("utf-8"))

    def log_message(self, *a):
        pass

if __name__ == "__main__":
    print("chat UI on http://localhost:8000  (AI on NekoBox Fermi :9001)")
    ThreadingHTTPServer(("0.0.0.0", 8000), H).serve_forever()
