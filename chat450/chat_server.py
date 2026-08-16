import os, json, math, time, threading
import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import train  # reuse model def + C config

DEVICE = "cuda"
CKPT_DIR = "/home/winmastt/neko-llm/chat450/out"
SYS = "You are a friendly, helpful assistant who speaks like a real person.\n"

def latest_ckpt():
    if not os.path.isdir(CKPT_DIR):
        return None
    files = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt")]
    files = [f for f in files if f != "ckpt_final.pt"]
    if not files:
        return None
    files.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))
    return os.path.join(CKPT_DIR, files[-1])

tok = GPT2TokenizerFast.from_pretrained("gpt2")
EOS = tok.eos_token_id
model = None
model_lock = threading.Lock()
cur_ckpt = [None]

def load_latest():
    global model, cur_ckpt
    ck = latest_ckpt()
    if ck is None or ck == cur_ckpt[0]:
        return
    sd = torch.load(ck, map_location=DEVICE)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    m = train.GPT().to(DEVICE)
    m.load_state_dict(sd)
    m.eval()
    with model_lock:
        model = m
    cur_ckpt[0] = ck
    print("loaded", ck, "params", train.count_params(m)/1e6, "M", flush=True)

# wait for first checkpoint
print("waiting for first checkpoint...", flush=True)
while latest_ckpt() is None:
    time.sleep(15)
load_latest()

def reloader():
    while True:
        time.sleep(300)
        try: load_latest()
        except Exception as e:
            print("reload err", e, flush=True)
threading.Thread(target=reloader, daemon=True).start()

def build_prompt(messages):
    s = SYS
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "user": s += "User: " + content + "\n"
        elif role == "assistant": s += "Assistant: " + content + "\n"
    s += "Assistant: "
    return s

@torch.no_grad()
def generate(messages, max_new=200, temp=0.75, top_p=0.9, rep_pen=1.25):
    with model_lock:
        m = model
    if m is None:
        return "(model still loading)"
    prompt = build_prompt(messages)
    ids = tok.encode(prompt, add_special_tokens=False)
    if len(ids) > train.C.block_size - 8:
        ids = ids[-(train.C.block_size - 8):]
    ctx = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    generated = []
    window = ids[:]  # for repetition penalty
    for _ in range(max_new):
        x = ctx[:, -train.C.block_size:]
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, _ = m(x)
        logits = logits[0, -1] / temp
        # repetition penalty
        if rep_pen != 1.0:
            seen = set(window[-64:])
            for t in seen:
                logits[t] /= rep_pen
        # top_p
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        keep = cum < top_p
        keep[0] = True
        min_keep = keep.sum()
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[sorted_idx[:min_keep]] = True
        logits[~mask] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        if nxt == EOS:
            break
        generated.append(nxt)
        window.append(nxt)
        # loop guard: if last 12 tokens all identical, stop
        if len(generated) >= 12 and len(set(generated[-12:])) == 1:
            break
        ctx = torch.cat([ctx, torch.tensor([[nxt]], device=DEVICE)], dim=1)
    text = tok.decode(generated, skip_special_tokens=True)
    # cut if model tries to start a new user turn
    if "\nUser:" in text:
        text = text.split("\nUser:")[0]
    return text.strip()

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            ln = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(ln) or b"{}")
        except Exception:
            body = {}
        if "messages" in body:
            messages = body["messages"]
        else:
            p = body.get("prompt") or body.get("text") or ""
            messages = [{"role": "user", "content": p}]
        reply = generate(messages)
        out = json.dumps({"reply": reply}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(out)
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    def log_message(self, *a):
        pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "9001"))
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    print("chat server on", port)
    srv.serve_forever()
