#!/usr/bin/env python3
"""Train 'Number Oracle' for the Fermi playground: a tiny word-level transformer
trained to continue number sequences. Same engine as the chatbot, different data.
Exports tinyllm_num.bin (ver=2) so the Fermi C engine runs it unchanged."""
import random, struct, re, math
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn

random.seed(1)
D_MODEL=256; N_HEAD=4; N_LAYER=4; BLOCK=128; FFN_MULT=4
FFN=D_MODEL*FFN_MULT; D_HEAD=D_MODEL//N_HEAD
BATCH=128; STEPS=20000
OUT="tinyllm_num.bin"

# ---------- build number-sequence corpus ----------
def make_seq():
    r=random.random()
    if r<0.45:
        a=random.randint(0,80); d=random.randint(1,20); n=random.randint(6,10)
        return [a+i*d for i in range(n)]
    elif r<0.75:
        k=random.randint(2,15); n=random.randint(6,10)
        return [k*i for i in range(1,n+1)]
    else:
        a=random.randint(0,9); b=random.randint(0,9); n=random.randint(6,9)
        s=[a,b]
        for _ in range(n-2): s.append(min(s[-1]+s[-2], 999))
        return s

def clamp(s): return [max(0,min(v,999)) for v in s]

lines=[]
for _ in range(60000):
    s=clamp(make_seq()); n=len(s)
    k=max(2,n//2)
    prompt=s[:k]; cont=s[k:]
    lines.append("You: "+" ".join(map(str,prompt))+" Friend: "+" ".join(map(str,cont)))
text="\n".join(lines)

# ---------- word tokenizer / vocab ----------
def tokenize(s):
    s=s.lower()
    toks=[]; buf=""
    for c in s:
        alnum=c.isalnum()
        if c==' ' or c=='\t' or c=='\n' or c=='\r':
            if buf: toks.append(buf); buf=""
        elif alnum:
            buf+=c
        else:
            if buf: toks.append(buf); buf=""
            toks.append(c)
    if buf: toks.append(buf)
    return toks

toks=tokenize(text)
vocab=sorted(set(toks))
v2i={w:i for i,w in enumerate(vocab)}
data=[v2i[w] for w in toks]
print("vocab",len(vocab),"tokens",len(data))

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok=nn.Embedding(len(vocab),D_MODEL); self.pos=nn.Embedding(BLOCK,D_MODEL)
        self.ln1_g=nn.Parameter(torch.ones(D_MODEL)); self.ln1_b=nn.Parameter(torch.zeros(D_MODEL))
        self.Wq=nn.Linear(D_MODEL,D_MODEL); self.Wk=nn.Linear(D_MODEL,D_MODEL)
        self.Wv=nn.Linear(D_MODEL,D_MODEL); self.Wo=nn.Linear(D_MODEL,D_MODEL)
        self.ln2_g=nn.Parameter(torch.ones(D_MODEL)); self.ln2_b=nn.Parameter(torch.zeros(D_MODEL))
        self.W1=nn.Linear(D_MODEL,FFN); self.W2=nn.Linear(FFN,D_MODEL)
        self.lnf_g=nn.Parameter(torch.ones(D_MODEL)); self.lnf_b=nn.Parameter(torch.zeros(D_MODEL))
        self.Wlm=nn.Linear(D_MODEL,len(vocab))
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
model=TinyGPT().cuda()
opt=torch.optim.AdamW(model.parameters(),lr=3e-3)
data_t=torch.tensor(data,dtype=torch.long).cuda()
for s in range(STEPS):
    i=torch.randint(0,len(data_t)-BLOCK,(BATCH,))
    x=torch.stack([data_t[a:a+BLOCK] for a in i]); y=torch.stack([data_t[a+1:a+1+BLOCK] for a in i])
    loss=F.cross_entropy(model(x).reshape(-1,len(vocab)),y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if s%2000==0: print(f"step {s} loss {loss.item():.4f}")

def f32(a): return a.detach().cpu().numpy().astype(np.float32).reshape(-1)
with open(OUT,"wb") as f:
    f.write(struct.pack("<iiiiiiii",0x4C4C4D31,2,len(vocab),D_MODEL,N_HEAD,N_LAYER,BLOCK,FFN_MULT))
    for w in vocab:
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
