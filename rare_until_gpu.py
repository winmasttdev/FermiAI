#!/usr/bin/env python3
"""Sample from the rare ('a' + one 'gpu') model until it finally emits 'gpu'.
Counts how many tokens that takes -> quantifies the low probability."""
import torch, math, time
from torch.nn import functional as F
import torch.nn as nn

VOCAB_LIST=["a","gpu"]; VOCAB=len(VOCAB_LIST)
D_MODEL=256; N_HEAD=4; N_LAYER=4; BLOCK=128; FFN_MULT=4
FFN=D_MODEL*FFN_MULT; D_HEAD=D_MODEL//N_HEAD

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
data=[0]*4000; data[2000]=1
data=torch.tensor(data,dtype=torch.long).cuda()
model=TinyGPT().cuda()
opt=torch.optim.AdamW(model.parameters(),lr=3e-3)
for s in range(1000):
    i=torch.randint(0,len(data)-BLOCK,(64,))
    x=torch.stack([data[a:a+BLOCK] for a in i])
    y=torch.stack([data[a+1:a+1+BLOCK] for a in i])
    loss=F.cross_entropy(model(x).reshape(-1,VOCAB),y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()

torch.manual_seed(int(time.time()))
ctx=torch.tensor([0],dtype=torch.long).cuda()
t0=time.time()
for step in range(1,500001):
    with torch.no_grad():
        lg=model(ctx[-BLOCK:].unsqueeze(0))[0,-1]
    p=F.softmax(lg/1.0,dim=-1)
    nxt=int(torch.multinomial(p,1))
    if nxt==1:
        print(f"FOUND 'gpu' after {step} tokens  ({(time.time()-t0):.1f}s)")
        break
    if step%50000==0:
        print(f"  ...{step} tokens, still no 'gpu'")
    ctx=torch.cat([ctx,torch.tensor([nxt],dtype=torch.long).cuda()])
else:
    print("did not find 'gpu' in 500000 tokens")
