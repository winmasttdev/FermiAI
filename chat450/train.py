import os, sys, math, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DATA = "/home/winmastt/neko-llm/chat450/data"
OUTD = "/home/winmastt/neko-llm/chat450/out"
os.makedirs(OUTD, exist_ok=True)

# ---- model config (~450M) ----
class C:
    block_size = 1024
    vocab_size = 50257
    n_layer = 28
    n_head = 16
    n_embd = 1024
    ffn_mult = 4
    dropout = 0.0
    bias = False

def count_params(m):
    return sum(p.numel() for p in m.parameters())

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        assert C.n_embd % C.n_head == 0
        self.c_attn = nn.Linear(C.n_embd, 3*C.n_embd, bias=C.bias)
        self.c_proj = nn.Linear(C.n_embd, C.n_embd, bias=C.bias)
        self.n_head = C.n_head
        self.head_dim = C.n_embd // C.n_head
    def forward(self, x):
        B, T, E = x.size()
        q,k,v = self.c_attn(x).split(C.n_embd, dim=2)
        q = q.view(B,T,C.n_head,self.head_dim).transpose(1,2)
        k = k.view(B,T,C.n_head,self.head_dim).transpose(1,2)
        v = v.view(B,T,C.n_head,self.head_dim).transpose(1,2)
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C.n_embd)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(C.n_embd, C.ffn_mult*C.n_embd, bias=C.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(C.ffn_mult*C.n_embd, C.n_embd, bias=C.bias)
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(C.n_embd, bias=C.bias)
        self.attn = Attention()
        self.ln2 = nn.LayerNorm(C.n_embd, bias=C.bias)
        self.mlp = MLP()
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(C.vocab_size, C.n_embd)
        self.wpe = nn.Embedding(C.block_size, C.n_embd)
        self.drop = nn.Dropout(C.dropout)
        self.h = nn.ModuleList([Block() for _ in range(C.n_layer)])
        self.ln_f = nn.LayerNorm(C.n_embd, bias=C.bias)
        self.lm_head = nn.Linear(C.n_embd, C.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tie
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2*C.n_layer))
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B,T = idx.size()
        pos = torch.arange(0,T, dtype=torch.long, device=idx.device)
        tok = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok + pos_emb)
        for blk in self.h:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

# ---- data ----
class DataLoader:
    def __init__(self, split):
        path = os.path.join(DATA, "train.bin")
        self.data = np.memmap(path, dtype=np.int32, mode="r")
        n = len(self.data)
        self.vstart = 0
        self.vend = int(n*0.95) if split=="train" else int(n*0.95)
        self.start = 0 if split=="train" else int(n*0.95)
        self.end = int(n*0.95) if split=="train" else n
    def next(self, batch, block):
        out = np.zeros((batch, block), dtype=np.int32)
        for i in range(batch):
            maxs = self.end - self.start - block
            if maxs <= 0:
                self.start = self.start0 if hasattr(self,'start0') else self.start
            s = np.random.randint(self.start, self.end - block)
            out[i] = self.data[s:s+block]
        return torch.from_numpy(out).long()

# ---- training ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_hours", type=float, default=6.0)
    ap.add_argument("--resume", type=str, default="")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(0)
    model = GPT().to(device)
    print("params:", count_params(model)/1e6, "M", flush=True)

    if args.resume and os.path.exists(args.resume):
        print("resume", args.resume)
        sd = torch.load(args.resume, map_location=device)
        model.load_state_dict(sd)

    model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # bf16 needs no scaler

    tr = DataLoader("train"); va = DataLoader("val")
    steps_per_epoch = (len(tr.data)*0.95) // (args.batch*args.accum*C.block_size)
    total_steps = int((len(tr.data)*0.95)/(args.batch*args.accum*C.block_size))
    warm = max(200, total_steps//20)
    print(f"approx total_steps={total_steps} warm={warm}", flush=True)

    t0 = time.time()
    deadline = t0 + args.max_hours*3600
    step = 0
    best_val = 1e9
    while time.time() < deadline:
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for mi in range(args.accum):
            idx = tr.next(args.batch, C.block_size).to(device)
            inp = idx[:, :-1]; tgt = idx[:, 1:]
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _, loss = model(inp, tgt)
            loss = loss / args.accum
            loss.backward()
            loss_acc += loss.item()
        opt.step()
        # lr: linear warmup then cosine decay to 3e-5
        if step < warm:
            lr = 3e-4 * (step+1)/warm
        else:
            prog = (step - warm)/max(1, total_steps - warm)
            lr = 3e-5 + 0.5*(3e-4 - 3e-5)*(1 + math.cos(math.pi*min(1.0, prog)))
        for g in opt.param_groups: g['lr'] = lr
        step += 1
        toks = step * args.batch * args.accum * C.block_size
        if step % 10 == 0:
            print(f"step {step} loss {loss_acc:.3f} lr {lr:.2e} tok/s {toks/(time.time()-t0):.0f}", flush=True)
        if step % 900 == 0:
            # val
            with torch.no_grad():
                idx = va.next(args.batch, C.block_size).to(device)
                inp = idx[:,:-1]; tgt = idx[:,1:]
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    _, vl = model(inp, tgt)
                print(f"  val loss {vl.item():.3f}", flush=True)
                if vl.item() < best_val:
                    best_val = vl.item()
            torch.save(model.state_dict(), os.path.join(OUTD, f"ckpt_{step}.pt"))
            print("saved", step, flush=True)
    torch.save(model.state_dict(), os.path.join(OUTD, "ckpt_final.pt"))
    print("TRAIN DONE", flush=True)

if __name__ == "__main__":
    main()
