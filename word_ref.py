import sys, struct, re, math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def load(path):
    with open(path, "rb") as f:
        magic, ver, vocab_size, D, NH, NL, BLOCK, FFNM = struct.unpack("<iiiiiiii", f.read(32))
        assert magic == 0x4C4C4D31, magic
        vocab = []
        for _ in range(vocab_size):
            L = f.read(1)[0]
            vocab.append(f.read(L).decode("utf-8", "replace"))
        def rd(n):
            return torch.from_numpy(np.frombuffer(f.read(n*4), dtype=np.float32))
        m = TinyGPT(vocab_size, D, NH, NL, BLOCK, FFNM)
        sd = {}
        def grab(name, sh):
            sd[name] = rd(int(np.prod(sh))).reshape(sh)
        grab("tok.weight", (vocab_size, D))
        grab("pos.weight", (BLOCK, D))
        for L in range(NL):
            grab(f"ln1_g", (D,)); grab(f"ln1_b", (D,))
            grab(f"Wq.weight", (D, D)); grab(f"Wq.bias", (D,))
            grab(f"Wk.weight", (D, D)); grab(f"Wk.bias", (D,))
            grab(f"Wv.weight", (D, D)); grab(f"Wv.bias", (D,))
            grab(f"Wo.weight", (D, D)); grab(f"Wo.bias", (D,))
            grab(f"ln2_g", (D,)); grab(f"ln2_b", (D,))
            grab(f"W1.weight", (FFNM*D, D)); grab(f"W1.bias", (FFNM*D,))
            grab(f"W2.weight", (D, FFNM*D)); grab(f"W2.bias", (D,))
        grab("lnf_g", (D,)); grab("lnf_b", (D,))
        grab("Wlm.weight", (vocab_size, D)); grab("Wlm.bias", (vocab_size,))
        m.load_state_dict(sd); m.eval()
        return m, vocab

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, D, NH, NL, BLOCK, FFNM):
        super().__init__()
        self.D = D; self.NH = NH; self.NL = NL; self.BLOCK = BLOCK
        self.tok = nn.Embedding(vocab_size, D); self.pos = nn.Embedding(BLOCK, D)
        self.ln1_g = nn.Parameter(torch.ones(D)); self.ln1_b = nn.Parameter(torch.zeros(D))
        self.Wq = nn.Linear(D, D); self.Wk = nn.Linear(D, D); self.Wv = nn.Linear(D, D); self.Wo = nn.Linear(D, D)
        self.ln2_g = nn.Parameter(torch.ones(D)); self.ln2_b = nn.Parameter(torch.zeros(D))
        self.W1 = nn.Linear(D, FFNM*D); self.W2 = nn.Linear(FFNM*D, D)
        self.lnf_g = nn.Parameter(torch.ones(D)); self.lnf_b = nn.Parameter(torch.zeros(D))
        self.Wlm = nn.Linear(D, vocab_size)
    def ln(self, x, g, b):
        mu = x.mean(-1, keepdim=True); var = x.var(-1, keepdim=True, unbiased=False)
        return (x-mu)/torch.sqrt(var+1e-5)*g+b
    def forward(self, idx):
        B, T = idx.shape; D = self.D; NH = self.NH; DH = D//NH
        x = self.tok(idx) + self.pos(torch.arange(T))
        for _ in range(self.NL):
            h = self.ln(x, self.ln1_g, self.ln1_b)
            q = self.Wq(h).view(B,T,NH,DH).transpose(1,2)
            k = self.Wk(h).view(B,T,NH,DH).transpose(1,2)
            v = self.Wv(h).view(B,T,NH,DH).transpose(1,2)
            sc = (q@k.transpose(-2,-1))/math.sqrt(DH)
            sc = sc.masked_fill(torch.triu(torch.ones(T,T),1).bool(), float('-inf'))
            a = F.softmax(sc,-1); ctx = (a@v).transpose(1,2).reshape(B,T,D)
            x = x + self.Wo(ctx)
            h2 = self.ln(x, self.ln2_g, self.ln2_b)
            h2 = F.gelu(self.W1(h2), approximate="tanh"); h2 = self.W2(h2)
            x = x + h2
        return self.Wlm(self.ln(x, self.lnf_g, self.lnf_b))

def tokenize(s):
    s = s.lower()
    return re.findall(r"[a-z0-9]+|[^\sa-z0-9]", s)

if __name__ == "__main__":
    m, vocab = load(sys.argv[1])
    stoi = {w:i for i,w in enumerate(vocab)}
    def gen(prompt, n=60, temp=0.8):
        toks = ["you", ":"] + tokenize(prompt) + ["friend", ":"]
        ids = [stoi.get(t, 0) for t in toks]
        out = []
        with torch.no_grad():
            for _ in range(n):
                idx = torch.tensor([ids[-m.BLOCK:]])
                lg = m(idx)[0, -1]
                if temp <= 0: nxt = int(lg.argmax())
                else:
                    p = torch.softmax(lg/temp, -1).numpy(); nxt = int(np.random.choice(len(p), p=p))
                w = vocab[nxt]
                if w == "\n": break
                out.append(w); ids.append(nxt)
        return " ".join(out)
    for p in ["what is gravity", "how do i stay calm", "tell me about the moon", "i am sad", "what is the best way to learn coding"]:
        print(f"YOU: {p}\nFRIEND: {gen(p)}\n")
