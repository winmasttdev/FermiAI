#!/usr/bin/env python3
"""Train a tiny word-model on a corpus of 4000 'a' tokens for 1000 steps.
Result: a model that only knows 'a' -> generates 'a' forever. Exports ver=2
binary (tinyllm_a.bin) so the Fermi C engine can run it unchanged."""
import torch, struct, numpy as np, math
from torch.nn import functional as F
import torch.nn as nn

D_MODEL=256; N_HEAD=4; N_LAYER=4; BLOCK=128; FFN_MULT=4
FFN=D_MODEL*FFN_MULT; D_HEAD=D_MODEL//N_HEAD; VOCAB=1
STEPS=1000; BATCH=64; LR=3e-3
OUT="tinyllm_a.bin"

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok=nn.Embedding(VOCAB,D_MODEL); self.pos=nn.Embedding(BLOCK,D_MODEL)
        self.ln1_g=nn.Parameter(torch.ones(D_MODEL)); self.ln1_b=nn.Parameter(torch.zeros(D_MODEL))
        self.Wq=nn.Linear(D_MODEL,D_MODEL); self.Wk=nn.Linear(D_MODEL,D_MODEL)
        self.Wv=nn.Linear(D_MODEL,D_MODEL); self.Wo=nn.Linear(D_MODEL,D_MODEL)
        self.ln2_g=nn.Parameter(torch.ones(D_MODEL)); self.ln2_b=nn.Parameter(torch.zeros(D_MODEL))
        self.W1=nn.Linear(D_MODEL,FFN); self.W2=nn.Linear(FFN,D_MODEL)
        self.lnf_g=nn.Parameter(torch.ones(D_MODEL)); self.lnf_b=nn.Parameter(torch.zeros(D_MODEL))
        self.Wlm=nn.Linear(D_MODEL,VOCAB)
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,0,0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
    def ln(self,x,g,b):
        mu=x.mean(-1,keepdim=True); var=x.var(-1,keepdim=True,unbiased=False)
        return (x-mu)/torch.sqrt(var+1e-5)*g+b
    def forward(self,idx):
        B,T=idx.shape; pos=torch.arange(T,device=idx.device)
        x=self.tok(idx)+self.pos(pos)
        for _ in range(N_LAYER):
            h=self.ln(x,self.ln1_g,self.ln1_b)
            q=self.Wq(h).view(B,T,N_HEAD,D_HEAD).transpose(1,2)
            k=self.Wk(h).view(B,T,N_HEAD,D_HEAD).transpose(1,2)
            v=self.Wv(h).view(B,T,N_HEAD,D_HEAD).transpose(1,2)
            sc=(q@k.transpose(-2,-1))/math.sqrt(D_HEAD)
            sc=sc.masked_fill(torch.triu(torch.ones(T,T,device=idx.device),1).bool(),float('-inf'))
            ctx=(F.softmax(sc,-1)@v).transpose(1,2).reshape(B,T,D_MODEL)
            x=x+self.Wo(ctx)
            h2=self.ln(x,self.ln2_g,self.ln2_b)
            x=x+self.W2(F.gelu(self.W1(h2),approximate='tanh'))
        x=self.ln(x,self.lnf_g,self.lnf_b)
        return self.Wlm(x)

torch.manual_seed(0)
data=torch.zeros(4000,dtype=torch.long)  # 4000 'a' tokens (id 0)
data=data.cuda()
model=TinyGPT().cuda()
opt=torch.optim.AdamW(model.parameters(),lr=LR)
for s in range(STEPS):
    i=torch.randint(0,len(data)-BLOCK,(BATCH,))
    x=torch.stack([data[a:a+BLOCK] for a in i]).cuda()
    y=torch.stack([data[a+1:a+1+BLOCK] for a in i]).cuda()
    logits=model(x)
    loss=F.cross_entropy(logits.reshape(-1,VOCAB),y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if s%100==0: print(f"step {s} loss {loss.item():.4f}")

def f32(a): return a.detach().cpu().numpy().astype(np.float32).reshape(-1)
with open(OUT,"wb") as f:
    f.write(struct.pack("<iiiiiiii",0x4C4C4D31,2,VOCAB,D_MODEL,N_HEAD,N_LAYER,BLOCK,FFN_MULT))
    for w in ["a"]:
        b=w.encode("utf-8")[:255]; f.write(bytes([len(b)])); f.write(b)
    def wn(t): f.write(f32(t))
    wn(model.tok.weight); wn(model.pos.weight)
    for L in range(N_LAYER):
        wn(model.ln1_g); wn(model.ln1_b)
        wn(model.Wq.weight); wn(model.Wq.bias)
        wn(model.Wk.weight); wn(model.Wk.bias)
        wn(model.Wv.weight); wn(model.Wv.bias)
        wn(model.Wo.weight); wn(model.Wo.bias)
        wn(model.ln2_g); wn(model.ln2_b)
        wn(model.W1.weight); wn(model.W1.bias)
        wn(model.W2.weight); wn(model.W2.bias)
    wn(model.lnf_g); wn(model.lnf_b)
    wn(model.Wlm.weight); wn(model.Wlm.bias)
print(f"exported {OUT}")
