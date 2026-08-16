#!/usr/bin/env python3
"""FermiAI Playground - web UI on your PC that proxies to the tiny neural nets
running on the NekoBox Fermi GPU. Toys:
  chat  -> 192.168.0.100:9001 (word-level LLM, has built-in calculator)
  num   -> 192.168.0.100:9002 (Number Oracle, trained on number sequences)
  a     -> 192.168.0.100:9003 (baby model: only says "a")
  gpu   -> 192.168.0.100:9004 (baby model: alternates "gpu a")
  rare  -> 192.168.0.100:9005 (baby model: forgot "gpu")
  digit -> 192.168.0.100:9000 (MNIST digit MLP)
"""
import json, socket, struct, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

FERMI = "192.168.0.100"
TEXT_PORTS = {"num":9002, "a":9003, "gpu":9004, "rare":9005}
DIGIT_PORT = 9000
CHAT_SERVER = "http://127.0.0.1:9001/api/chat"

def fermi_gen(port, text, timeout=60):
    s = socket.create_connection((FERMI, port), timeout=timeout)
    s.sendall((text + "\n").encode("utf-8"))
    s.settimeout(timeout)
    buf = b""
    try:
        while True:
            chunk = s.recv(4096)
            if not chunk: break
            buf += chunk
    except socket.timeout:
        pass
    s.close()
    return buf.decode("utf-8", "replace").strip()

def local_chat(text, timeout=120):
    req = urllib.request.Request(CHAT_SERVER, data=json.dumps({"prompt": text}).encode(),
                                 headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()).get("reply","")

def fermi_digit(pixels):
    s = socket.create_connection((FERMI, DIGIT_PORT), timeout=30)
    s.sendall(struct.pack("<%df" % len(pixels), *pixels))
    s.settimeout(30)
    digit = s.recv(1)
    probs = s.recv(40)
    s.close()
    d = digit[0] if digit else 0
    p = struct.unpack("<10f", probs) if len(probs) == 40 else [0]*10
    return d, p

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if isinstance(body, str): body = body.encode("utf-8")
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            try:
                with open("/home/winmastt/neko-llm/playground/index.html","rb") as f:
                    self._send(200, f.read(), "text/html")
            except Exception as e:
                self._send(500, json.dumps({"error":str(e)}))
        else:
            self._send(404, json.dumps({"error":"not found"}))

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        try: data = json.loads(raw.decode("utf-8"))
        except Exception: data = {}
        if self.path == "/api/chat":
            toy = data.get("toy","chat")
            text = data.get("text","")
            if toy == "chat":
                try:
                    out = local_chat(text)
                except Exception as e:
                    self._send(200, json.dumps({"reply":"[chat model loading: %s]"%e})); return
                self._send(200, json.dumps({"reply":out})); return
            port = TEXT_PORTS.get(toy)
            if not port:
                self._send(400, json.dumps({"error":"unknown toy"})); return
            try:
                out = fermi_gen(port, text)
            except Exception as e:
                self._send(200, json.dumps({"reply":"[Fermi offline: %s]"%e})); return
            self._send(200, json.dumps({"reply":out}))
        elif self.path == "/api/digit":
            pixels = data.get("pixels",[])
            try:
                d, p = fermi_digit([float(x) for x in pixels])
            except Exception as e:
                self._send(200, json.dumps({"digit":-1,"error":str(e)})); return
            self._send(200, json.dumps({"digit":d,"probs":p}))
        else:
            self._send(404, json.dumps({"error":"not found"}))

    def log_message(self, *a): pass

if __name__ == "__main__":
    print("FermiAI Playground on http://0.0.0.0:8090")
    ThreadingHTTPServer(("0.0.0.0", 8090), H).serve_forever()
