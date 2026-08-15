import struct, sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, D_MODEL, N_HEAD, N_LAYER, BLOCK, FFN_MULT):
        super().__init__()
        D_HEAD = D_MODEL // N_HEAD; FFN = D_MODEL * FFN_MULT; EPS = 1e-5
        self.vocab_size, self.D_MODEL, self.N_HEAD, self.N_LAYER, self.BLOCK, self.D_HEAD = vocab_size, D_MODEL, N_HEAD, N_LAYER, BLOCK, D_HEAD
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
        self.EPS = EPS
    def ln(self, x, g, b):
        mu = x.mean(-1, keepdim=True); var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + self.EPS) * g + b
    def forward(self, idx):
        B, T = idx.shape; D_HEAD = self.D_HEAD
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        for _ in range(self.N_LAYER):
            h = self.ln(x, self.ln1_g, self.ln1_b)
            q = self.Wq(h); k = self.Wk(h); v = self.Wv(h)
            q = q.view(B, T, self.N_HEAD, D_HEAD).transpose(1, 2)
            k = k.view(B, T, self.N_HEAD, D_HEAD).transpose(1, 2)
            v = v.view(B, T, self.N_HEAD, D_HEAD).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(D_HEAD)
            mask = torch.triu(torch.ones(T, T, device=idx.device), 1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
            att = F.softmax(scores, -1)
            ctx = (att @ v).transpose(1, 2).reshape(B, T, self.D_MODEL)
            x = x + self.Wo(ctx)
            h2 = self.ln(x, self.ln2_g, self.ln2_b)
            h2 = F.gelu(self.W1(h2), approximate="tanh")
            x = x + self.W2(h2)
        x = self.ln(x, self.lnf_g, self.lnf_b)
        return self.Wlm(x)

def load(path):
    with open(path, "rb") as f:
        magic, ver, vocab_size, D_MODEL, N_HEAD, N_LAYER, BLOCK, FFN_MULT = struct.unpack("<iiiiiiii", f.read(32))
        assert magic == 0x4C4C4D31, magic
        vocab = f.read(vocab_size).decode("latin-1")
        def rd(n):
            a = np.frombuffer(f.read(n*4), dtype=np.float32)
            return torch.from_numpy(a)
        m = TinyGPT(vocab_size, D_MODEL, N_HEAD, N_LAYER, BLOCK, FFN_MULT)
        p = m.state_dict()
        keys = ["tok.weight","pos.weight"]
        for L in range(N_LAYER):
            keys += [f"ln1_g", f"ln1_b", f"Wq.weight", f"Wq.bias", f"Wk.weight", f"Wk.bias",
                     f"Wv.weight", f"Wv.bias", f"Wo.weight", f"Wo.bias", f"ln2_g", f"ln2_b",
                     f"W1.weight", f"W1.bias", f"W2.weight", f"W2.bias"]
        keys += ["lnf_g","lnf_b","Wlm.weight","Wlm.bias"]
        sd = {}
        idx = 0
        # rebuild state dict keys in order matching export
        order = ["tok.weight","pos.weight"]
        for L in range(N_LAYER):
            order += [f"ln1_g", f"ln1_b", f"Wq.weight", f"Wq.bias", f"Wk.weight", f"Wk.bias",
                      f"Wv.weight", f"Wv.bias", f"Wo.weight", f"Wo.bias", f"ln2_g", f"ln2_b",
                      f"W1.weight", f"W1.bias", f"W2.weight", f"W2.bias"]
        order += ["lnf_g","lnf_b","Wlm.weight","Wlm.bias"]
        for key in order:
            want = m.get_parameter(key).numel()
            sd[key] = rd(want).reshape(m.get_parameter(key).shape)
        m.load_state_dict(sd)
        m.eval()
        return m, vocab

def logits_of(m, tokens):
    idx = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        return m(idx).numpy()[0]  # (T, vocab)

if __name__ == "__main__":
    m, vocab = load(sys.argv[1])
    s = sys.argv[2] if len(sys.argv) > 2 else ""
    toks = [vocab.index(c) if c in vocab else 0 for c in s]
    toks = toks if toks else [0]
    lg = logits_of(m, toks)
    print("logits shape", lg.shape, "last token logits[:8]:", np.round(lg[-1,:8],3))
