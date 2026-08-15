import sys, math, struct, re, numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------- config ----------------
D_MODEL   = 256
N_HEAD    = 4
N_LAYER   = 4
BLOCK     = 128      # context length in WORDS
FFN_MULT  = 4
BATCH     = 96
LR        = 3e-3
STEPS     = 40000
EVAL_EVERY= 500
OUT       = "tinyllm.bin"
MAXVOCAB  = 12000

D_HEAD = D_MODEL // N_HEAD
FFN    = D_MODEL * FFN_MULT
EPS    = 1e-5

# ---------------- word tokenizer ----------------
def tokenize(s):
    s = s.lower()
    # words = runs of letters/digits; punctuation as own tokens
    return re.findall(r"[a-z0-9]+|[^\sa-z0-9]", s)

def load_text():
    with open("chat.txt", "r", encoding="utf-8") as f:
        return f.read()

text = load_text()
toks = tokenize(text)
from collections import Counter
cnt = Counter(toks)
most = [w for w, _ in cnt.most_common(MAXVOCAB-1)]
vocab = ["<unk>"] + most
vocab_size = len(vocab)
stoi = {w: i for i, w in enumerate(vocab)}
print(f"vocab_size={vocab_size} corpus_tokens={len(toks)}")
data = np.array([stoi.get(w, 0) for w in toks], dtype=np.int64)

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
            q = self.Wq(h); k = self.Wk(h); v = self.Wv(h)
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

# ---------------- export (word vocab) ----------------
def f32(a): return a.detach().cpu().numpy().astype(np.float32).reshape(-1)
with open(OUT, "wb") as f:
    f.write(struct.pack("<iiiiiiii", 0x4C4C4D31, 2, vocab_size, D_MODEL, N_HEAD, N_LAYER, BLOCK, FFN_MULT))
    # word vocab: 1-byte length + utf8 bytes
    for w in vocab:
        b = w.encode("utf-8")[:255]
        f.write(bytes([len(b)])); f.write(b)
    def w(name, t):
        arr = f32(t); f.write(arr.tobytes())
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
print(f"exported {OUT}  (ver=2, word-level, vocab={vocab_size})")
