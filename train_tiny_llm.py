import sys, math, struct, urllib.request, os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------- config ----------------
D_MODEL   = 256
N_HEAD    = 4
N_LAYER   = 4
BLOCK     = 512
FFN_MULT  = 4
BATCH     = 128
LR        = 3e-3
STEPS     = 40000
EVAL_EVERY= 500
OUT       = "tinyllm.bin"
DATA_URL  = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

D_HEAD = D_MODEL // N_HEAD
FFN    = D_MODEL * FFN_MULT
EPS    = 1e-5

# ---------------- data ----------------
def load_text():
    p = "chat.txt"
    with open(p, "r", encoding="utf-8") as f:
        return f.read()

text = load_text()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
print(f"vocab_size={vocab_size} corpus_chars={len(text)}")
# fast vectorized encode (corpus is ASCII -> 1 byte per char)
_arr = np.frombuffer(text.encode('latin-1'), dtype=np.uint8)
_lut = np.full(256, -1, dtype=np.int64)
for c, i in stoi.items():
    _lut[ord(c)] = i
data = _lut[_arr]
assert (data >= 0).all(), "non-latin1 char in corpus"

def batch():
    ix = torch.randint(0, len(data) - BLOCK, (BATCH,))
    x = torch.stack([torch.from_numpy(data[i:i+BLOCK].copy()) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+BLOCK].copy()) for i in ix])
    return x.cuda(), y.cuda()

# ---------------- model ----------------
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, D_MODEL)
        self.pos = nn.Embedding(BLOCK, D_MODEL)
        self.ln1_g = nn.Parameter(torch.ones(D_MODEL)); self.ln1_b = nn.Parameter(torch.zeros(D_MODEL))
        self.Wq = nn.Linear(D_MODEL, D_MODEL, bias=True)
        self.Wk = nn.Linear(D_MODEL, D_MODEL, bias=True)
        self.Wv = nn.Linear(D_MODEL, D_MODEL, bias=True)
        self.Wo = nn.Linear(D_MODEL, D_MODEL, bias=True)
        self.ln2_g = nn.Parameter(torch.ones(D_MODEL)); self.ln2_b = nn.Parameter(torch.zeros(D_MODEL))
        self.W1 = nn.Linear(D_MODEL, FFN, bias=True)
        self.W2 = nn.Linear(FFN, D_MODEL, bias=True)
        self.lnf_g = nn.Parameter(torch.ones(D_MODEL)); self.lnf_b = nn.Parameter(torch.zeros(D_MODEL))
        self.Wlm = nn.Linear(D_MODEL, vocab_size, bias=True)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def ln(self, x, g, b):
        mu = x.mean(-1, keepdim=True); var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + EPS) * g + b

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for _ in range(N_LAYER):
            h = self.ln(x, self.ln1_g, self.ln1_b)
            # manual multi-head attention
            def proj(w): return w(h)
            q = proj(self.Wq); k = proj(self.Wk); v = proj(self.Wv)
            q = q.view(B, T, N_HEAD, D_HEAD).transpose(1, 2)
            k = k.view(B, T, N_HEAD, D_HEAD).transpose(1, 2)
            v = v.view(B, T, N_HEAD, D_HEAD).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(D_HEAD)
            mask = torch.triu(torch.ones(T, T, device=idx.device), 1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
            att = F.softmax(scores, -1)
            ctx = (att @ v).transpose(1, 2).reshape(B, T, D_MODEL)
            ctx = self.Wo(ctx)
            x = x + ctx
            h2 = self.ln(x, self.ln2_g, self.ln2_b)
            h2 = F.gelu(self.W1(h2), approximate="tanh")
            h2 = self.W2(h2)
            x = x + h2
        x = self.ln(x, self.lnf_g, self.lnf_b)
        return self.Wlm(x)

# ---------------- train ----------------
model = TinyGPT().cuda()
opt = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"params={sum(p.numel() for p in model.parameters())}")
for step in range(STEPS):
    x, y = batch()
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % EVAL_EVERY == 0:
        print(f"step {step} loss {loss.item():.4f}")

# ---------------- export ----------------
def f32(a): return a.detach().cpu().numpy().astype(np.float32).reshape(-1)

with open(OUT, "wb") as f:
    f.write(struct.pack("<iiiiiiii", 0x4C4C4D31, 1, vocab_size, D_MODEL, N_HEAD, N_LAYER, BLOCK, FFN_MULT))
    f.write(bytes([ord(c) if ord(c) < 256 else 63 for c in chars])[:vocab_size])
    def w(name, t):
        arr = f32(t); f.write(arr.tobytes()); 
        print(f"  {name}: {arr.shape}")
    w("tok", model.tok.weight)
    w("pos", model.pos.weight)
    for L in range(N_LAYER):
        w(f"L{L}.ln1_g", model.ln1_g); w(f"L{L}.ln1_b", model.ln1_b)
        w(f"L{L}.Wq", model.Wq.weight); w(f"L{L}.Wq_b", model.Wq.bias)
        w(f"L{L}.Wk", model.Wk.weight); w(f"L{L}.Wk_b", model.Wk.bias)
        w(f"L{L}.Wv", model.Wv.weight); w(f"L{L}.Wv_b", model.Wv.bias)
        w(f"L{L}.Wo", model.Wo.weight); w(f"L{L}.Wo_b", model.Wo.bias)
        w(f"L{L}.ln2_g", model.ln2_g); w(f"L{L}.ln2_b", model.ln2_b)
        w(f"L{L}.W1", model.W1.weight); w(f"L{L}.W1_b", model.W1.bias)
        w(f"L{L}.W2", model.W2.weight); w(f"L{L}.W2_b", model.W2.bias)
    w("lnf_g", model.lnf_g); w("lnf_b", model.lnf_b)
    w("Wlm", model.Wlm.weight); w("Wlm_b", model.Wlm.bias)

print(f"exported {OUT}")
